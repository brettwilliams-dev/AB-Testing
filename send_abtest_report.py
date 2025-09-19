import os
import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import pandas as pd
from google.cloud import bigquery
from openai import OpenAI

# --------------------
# Config (env-driven)
# --------------------
BQ_PROJECT_ID = os.getenv("BQ_PROJECT_ID")
BQ_DATASET = os.getenv("BQ_DATASET")
BQ_TABLE = os.getenv("BQ_TABLE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_APP_PASSWORD")
EMAIL_RECIPIENTS = [r.strip() for r in os.getenv("EMAIL_RECIPIENTS", "").split(",") if r.strip()]

# Optional overrides
DATE_FIELD = os.getenv("BQ_DATE_FIELD", "max_test_date")  # table field used to filter recency
DAYS_BACK = int(os.getenv("DAYS_BACK", "2"))              # pull last N days incl. today
CONTROL_LABEL = os.getenv("CONTROL_LABEL", "Original")    # how control is labeled in your table
TEST_KEY_COL = os.getenv("TEST_KEY_COL", "abtasty_campaign_sp")      # test grouping key
VARIANT_COL = os.getenv("VARIANT_COL", "abtasty_variation_sp")       # variant/control column
PROPERTY_COL = os.getenv("PROPERTY_COL", "property")                 # optional metadata

# Rule thresholds for categories
GOOD_THRESH = float(os.getenv("GOOD_THRESH", "0.05"))   # +5% or greater → "Good"
BAD_THRESH  = float(os.getenv("BAD_THRESH", "-0.05"))   # -5% or less → "Bad"

# Subject line
EMAIL_SUBJECT = os.getenv("EMAIL_SUBJECT", "In-Flight A/B Tests: Summary & Lift")

# --------------------
# Helper functions
# --------------------
def write_key_to_tmp():
    """Materialize the SA JSON key from the secret into a temp file."""
    key_json = os.getenv("NB_BQ_SERVICE_ACCOUNT_KEY")
    if not key_json:
        raise RuntimeError("NB_BQ_SERVICE_ACCOUNT_KEY not found in env")
    key_path = "/tmp/bq_key.json"
    with open(key_path, "w") as f:
        f.write(key_json)
    return key_path

def fetch_data():
    """Query BigQuery for recent rows from your summary table/view."""
    key_path = write_key_to_tmp()
    client = bigquery.Client.from_service_account_json(key_path)

    query = f"""
    SELECT
      {PROPERTY_COL},
      {TEST_KEY_COL},
      {VARIANT_COL},
      min_test_date,
      max_test_date,
      sessions,
      submits,
      form_starts,
      start_rate,
      completion_rate,
      conversion_rate
    FROM `{BQ_PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}`
    WHERE DATE({DATE_FIELD}) >= DATE_SUB(CURRENT_DATE(), INTERVAL {DAYS_BACK} DAY)
    """

    df = client.query(query).to_dataframe()
    return df

def _fmt_pct(x, places=1):
    if pd.isna(x):
        return ""
    return f"{x*100:.{places}f}%"

def _fmt_int(x):
    if pd.isna(x):
        return ""
    return f"{int(x):,}"

def compute_lifts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a combined table with BOTH control and variant rows:
      - Variant rows include lifts vs the matching control.
      - Control rows have blank lift columns.
    Keeps original counts for context.
    """
    if df.empty:
        return df

    # Split control vs variants
    ctrl = df[df[VARIANT_COL] == CONTROL_LABEL].copy()
    var  = df[df[VARIANT_COL] != CONTROL_LABEL].copy()

    key_cols = [PROPERTY_COL, TEST_KEY_COL]

    # Index for joining
    ctrl_keyed = ctrl.set_index(key_cols)
    var_keyed  = var.set_index(key_cols)

    # Join variants to control (to bring in control rates)
    joined = var_keyed.join(
        ctrl_keyed[
            ["sessions","submits","form_starts",
             "start_rate","completion_rate","conversion_rate"]
        ],
        how="left",
        rsuffix="_ctrl",
        lsuffix="_var"
    ).reset_index()

    # Compute lifts on variants
    for metric in ["conversion_rate", "start_rate", "completion_rate"]:
        joined[f"{metric}_lift"] = (
            (joined.get(f"{metric}_var") - joined.get(f"{metric}_ctrl")) /
            joined.get(f"{metric}_ctrl")
        )

    # Category by CVR lift
    def categorize(l):
        if pd.isna(l): return "Neutral"
        if l >= GOOD_THRESH: return "Good"
        if l <= BAD_THRESH:  return "Bad"
        return "Neutral"

    joined["category"] = joined["conversion_rate_lift"].apply(categorize)

    # ---- Normalize columns for VARIANT rows (presentation names)
    var_out = joined[[
        PROPERTY_COL, TEST_KEY_COL, VARIANT_COL,
        "sessions_var","form_starts_var","submits_var",
        "conversion_rate_ctrl","conversion_rate_var","conversion_rate_lift",
        "start_rate_ctrl","start_rate_var","start_rate_lift",
        "completion_rate_ctrl","completion_rate_var","completion_rate_lift",
        "category"
    ]].copy()

    var_out = var_out.rename(columns={
        PROPERTY_COL: "property",
        TEST_KEY_COL: "test",
        VARIANT_COL: "variant"
    })

    # ---- Build CONTROL rows with blank lifts
    ctrl_tmp = ctrl_keyed.reset_index()
    ctrl_out = ctrl_tmp[[
        PROPERTY_COL, TEST_KEY_COL, VARIANT_COL,
        "sessions","form_starts","submits",
        "conversion_rate","start_rate","completion_rate"
    ]].copy()

    ctrl_out["conversion_rate_lift"] = None
    ctrl_out["start_rate_lift"] = None
    ctrl_out["completion_rate_lift"] = None
    ctrl_out["category"] = "Control"

    ctrl_out = ctrl_out.rename(columns={
        PROPERTY_COL: "property",
        TEST_KEY_COL: "test",
        VARIANT_COL: "variant",
        "sessions": "sessions_var",
        "form_starts": "form_starts_var",
        "submits": "submits_var",
        "conversion_rate": "conversion_rate_ctrl",
        "start_rate": "start_rate_ctrl",
        "completion_rate": "completion_rate_ctrl"
    })

    # For control rows, "var" columns should be blank
    ctrl_out["conversion_rate_var"] = None
    ctrl_out["start_rate_var"] = None
    ctrl_out["completion_rate_var"] = None

    # Reorder columns same as variants and combine
    ctrl_out = ctrl_out[var_out.columns]
    combined = pd.concat([ctrl_out, var_out], ignore_index=True)

    # ----- Formatting for the HTML table
    combined["sessions"]    = combined["sessions_var"].apply(_fmt_int)
    combined["form_starts"] = combined["form_starts_var"].apply(_fmt_int)
    combined["submits"]     = combined["submits_var"].apply(_fmt_int)

    combined["conv_ctrl"] = combined["conversion_rate_ctrl"].apply(_fmt_pct)
    combined["conv_var"]  = combined["conversion_rate_var"].apply(_fmt_pct)
    combined["conv_lift"] = combined["conversion_rate_lift"].apply(_fmt_pct)

    combined["start_ctrl"] = combined["start_rate_ctrl"].apply(_fmt_pct)
    combined["start_var"]  = combined["start_rate_var"].apply(_fmt_pct)
    combined["start_lift"] = combined["start_rate_lift"].apply(_fmt_pct)

    combined["comp_ctrl"] = combined["completion_rate_ctrl"].apply(_fmt_pct)
    combined["comp_var"]  = combined["completion_rate_var"].apply(_fmt_pct)
    combined["comp_lift"] = combined["completion_rate_lift"].apply(_fmt_pct)

    # Final ordered cols (keep both control + variant)
    combined = combined[[
        "property","test","variant",
        "sessions","form_starts","submits",
        "conv_ctrl","conv_var","conv_lift",
        "start_ctrl","start_var","start_lift",
        "comp_ctrl","comp_var","comp_lift",
        "category"
    ]]

    # Sort: Controls first, then Good/Neutral/Bad variants; within that by property/test
    order = pd.Categorical(combined["category"], ["Control","Good","Neutral","Bad"])
    combined = combined.assign(_sort=order).sort_values(
        ["property","test","_sort","variant"]
    ).drop(columns="_sort")

    return combined

def to_html_table(df: pd.DataFrame) -> str:
    """Render a compact HTML table with green/red lift colors (variants only)."""
    if df.empty:
        return "<p><em>No variant/control rows found in the selected window.</em></p>"

    def lift_class(val: str):
        if not val or not val.endswith("%"):
            return ""
        try:
            num = float(val.replace("%","")) / 100.0
            if num >= GOOD_THRESH: return "good"
            if num <= BAD_THRESH:  return "bad"
            return ""
        except Exception:
            return ""

    styled = df.copy()
    styled["_conv_cls"]  = styled["conv_lift"].apply(lift_class)
    styled["_start_cls"] = styled["start_lift"].apply(lift_class)
    styled["_comp_cls"]  = styled["comp_lift"].apply(lift_class)

    cols = [
        ("Property","property"),
        ("Test","test"),
        ("Variant","variant"),
        ("Sessions","sessions"),
        ("Form Starts","form_starts"),
        ("Submits","submits"),
        ("Conv (Ctrl)","conv_ctrl"),
        ("Conv (Var)","conv_var"),
        ("Conv Lift","conv_lift","_conv_cls"),
        ("Start (Ctrl)","start_ctrl"),
        ("Start (Var)","start_var"),
        ("Start Lift","start_lift","_start_cls"),
        ("Comp (Ctrl)","comp_ctrl"),
        ("Comp (Var)","comp_var"),
        ("Comp Lift","comp_lift","_comp_cls"),
        ("Category","category"),
    ]

    thead = "".join([f"<th>{lbl}</th>" for (lbl, *_) in cols])
    rows_html = []
    for _, r in styled.iterrows():
        tds = []
        for spec in cols:
            if len(spec) == 2:
                _, key = spec
                tds.append(f"<td>{r.get(key, '')}</td>")
            else:
                _, key, cls_key = spec
                cls = r.get(cls_key, "")
                tds.append(f"<td class='{cls}'>{r.get(key, '')}</td>")
        rows_html.append(f"<tr>{''.join(tds)}</tr>")

    css = """
    <style>
      table.ab { border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; font-size: 13px; }
      table.ab th, table.ab td { border: 1px solid #e6e6e6; padding: 6px 8px; text-align: right; }
      table.ab th:nth-child(1), table.ab td:nth-child(1),
      table.ab th:nth-child(2), table.ab td:nth-child(2),
      table.ab th:nth-child(3), table.ab td:nth-child(3) { text-align: left; }
      table.ab thead { background: #fafafa; }
      td.good { color: #0a7a0a; font-weight: 600; }
      td.bad  { color: #b00020; font-weight: 600; }
    </style>
    """
    html = f"""
    {css}
    <table class="ab">
      <thead><tr>{thead}</tr></thead>
      <tbody>
        {''.join(rows_html)}
      </tbody>
    </table>
    """
    return html

def build_gpt_summary(df_for_gpt: pd.DataFrame) -> str:
    """
    Build an executive HTML summary via GPT:
      - Analyze VARIANT rows only (ignore controls).
      - Always lead with conversion-rate (CVR) lift.
      - Group into Good / Bad / Neutral.
      - Return ready-to-embed HTML (headings + bullets).
    """
    if df_for_gpt.empty:
        return "<p>No in-flight variant data found in the selected window.</p>"

    client = OpenAI(api_key=OPENAI_API_KEY)

    # Only variants for analysis
    variants_only = df_for_gpt[df_for_gpt["category"] != "Control"].copy()
    if variants_only.empty:
        return "<p>No variant rows to analyze (only controls in the window).</p>"

    compact = variants_only[[
        "property","test","variant",
        "sessions","form_starts","submits",
        "conv_ctrl","conv_var","conv_lift",
        "start_ctrl","start_var","start_lift",
        "comp_ctrl","comp_var","comp_lift",
        "category"
    ]]
    records = compact.to_dict(orient="records")

    system_msg = "You are a senior CRO analyst. Write concise, executive-ready summaries."

    user_msg = f"""
You are given a list of A/B test VARIANT rows (controls are excluded in this list).
Rules:
- Group items by category using conversion-rate lift (CVR lift) thresholds:
  * Good: conv_lift >= {GOOD_THRESH*100:.1f}%
  * Bad:  conv_lift <= {BAD_THRESH*100:.1f}%
  * Neutral: between them.
- For each variant, ALWAYS start with CVR lift (e.g., '+8.2% CVR lift'), then succinctly explain the likely driver using start_lift and comp_lift (e.g., 'driven by completion rate').
- Keep bullets very short (≤ 1 line).
- End with a 'What’s Next' section with 2–3 bullets.
- Output pure HTML only with <h3> headings and <ul><li> bullets. No markdown.

Data (list of objects):
{records}
"""

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.3,
        max_tokens=900,
    )

    html = resp.choices[0].message.content.strip()
    return html

def send_email(summary_html: str, table_html: str):
    """Send the summary + table via Gmail (summary already in HTML)."""
    if not EMAIL_SENDER or not EMAIL_PASSWORD or not EMAIL_RECIPIENTS:
        raise RuntimeError("Email sender/password/recipients not configured.")

    html_body = f"""
    <html>
      <body>
        <div style="font-family: Arial, sans-serif; max-width: 980px;">
          <h2 style="margin-bottom: 6px;">In-Flight A/B Test Summary</h2>
          <div style="color:#555; font-size: 13px; margin-bottom: 14px;">
            Window: last {DAYS_BACK} day(s). Categorization: Good ≥ {GOOD_THRESH*100:.0f}%, Bad ≤ {BAD_THRESH*100:.0f}% (by conversion lift).
          </div>
          {summary_html}
          <div style="height:12px;"></div>
          {table_html}
        </div>
      </body>
    </html>
    """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = EMAIL_SUBJECT
    msg["From"] = EMAIL_SENDER
    msg["To"] = ", ".join(EMAIL_RECIPIENTS)
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECIPIENTS, msg.as_string())

def main():
    df = fetch_data()

    if df.empty:
        summary_html = "<p>No activity in the selected window.</p>"
        table_html   = "<p><em>No rows to display.</em></p>"
        send_email(summary_html, table_html)
        return

    agg = compute_lifts(df)               # includes BOTH control + variant rows
    summary_html = build_gpt_summary(agg) # analyzes variants only; outputs HTML
    table_html = to_html_table(agg)       # renders both control + variant rows
    send_email(summary_html, table_html)

if __name__ == "__main__":
    main()
