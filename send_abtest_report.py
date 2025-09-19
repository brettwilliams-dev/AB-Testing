import os
import json
import math
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
PROPERTY_COL = os.getenv("PROPERTY_COL", "property")                # optional metadata

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
    Pair each variant with control within a test, compute lifts for:
    - conversion_rate
    - start_rate
    - completion_rate
    Keep original counts for context (sessions, submits, form_starts).
    """
    if df.empty:
        return df

    # Split control vs variants
    ctrl = df[df[VARIANT_COL] == CONTROL_LABEL].copy()
    var  = df[df[VARIANT_COL] != CONTROL_LABEL].copy()

    # Use (property, test_key) as composite, so cross-property tests remain separate
    ctrl_index_cols = [PROPERTY_COL, TEST_KEY_COL]
    ctrl = ctrl.set_index(ctrl_index_cols)

    # Join variants to control on (property, test)
    var = var.set_index(ctrl_index_cols)
    joined = var.join(
        ctrl[[ "sessions", "submits", "form_starts",
               "start_rate", "completion_rate", "conversion_rate"]],
        how="left",
        rsuffix="_ctrl",
        lsuffix="_var"
    ).reset_index()

    # Compute lifts
    for metric in ["conversion_rate", "start_rate", "completion_rate"]:
        joined[f"{metric}_lift"] = (
            (joined[f"{metric}_var"] - joined[f"{metric}_ctrl"]) /
            joined[f"{metric}_ctrl"]
        )

    # Simple categorization by conversion_rate_lift
    def categorize(lift):
        if pd.isna(lift):
            return "Neutral"
        if lift >= GOOD_THRESH:
            return "Good"
        if lift <= BAD_THRESH:
            return "Bad"
        return "Neutral"

    joined["category"] = joined["conversion_rate_lift"].apply(categorize)

    # Nicely formatted columns for the HTML table
    out = joined[[
        PROPERTY_COL,
        TEST_KEY_COL,
        VARIANT_COL,
        "sessions_var", "submits_var", "form_starts_var",
        "conversion_rate_ctrl", "conversion_rate_var", "conversion_rate_lift",
        "start_rate_ctrl",      "start_rate_var",      "start_rate_lift",
        "completion_rate_ctrl", "completion_rate_var", "completion_rate_lift",
        "category"
    ]].copy()

    # Create presentation columns
    out["sessions"]      = out["sessions_var"].apply(_fmt_int)
    out["submits"]       = out["submits_var"].apply(_fmt_int)
    out["form_starts"]   = out["form_starts_var"].apply(_fmt_int)

    out["conv_ctrl"]     = out["conversion_rate_ctrl"].apply(_fmt_pct)
    out["conv_var"]      = out["conversion_rate_var"].apply(_fmt_pct)
    out["conv_lift"]     = out["conversion_rate_lift"].apply(_fmt_pct)

    out["start_ctrl"]    = out["start_rate_ctrl"].apply(_fmt_pct)
    out["start_var"]     = out["start_rate_var"].apply(_fmt_pct)
    out["start_lift"]    = out["start_rate_lift"].apply(_fmt_pct)

    out["comp_ctrl"]     = out["completion_rate_ctrl"].apply(_fmt_pct)
    out["comp_var"]      = out["completion_rate_var"].apply(_fmt_pct)
    out["comp_lift"]     = out["completion_rate_lift"].apply(_fmt_pct)

    # Final column order for the email table
    out = out.rename(columns={
        PROPERTY_COL: "property",
        TEST_KEY_COL: "test",
        VARIANT_COL: "variant"
    })

    out = out[[
        "property", "test", "variant",
        "sessions", "form_starts", "submits",
        "conv_ctrl", "conv_var", "conv_lift",
        "start_ctrl","start_var","start_lift",
        "comp_ctrl", "comp_var", "comp_lift",
        "category"
    ]].sort_values(["category", "property", "test", "variant"])

    return out

def to_html_table(df: pd.DataFrame) -> str:
    """Render a compact HTML table with light styling and green/red lift colors."""
    if df.empty:
        return "<p><em>No variant rows found in the selected window.</em></p>"

    # Add CSS classes for lift coloring
    def lift_class(val: str):
        if not val or not val.endswith("%"):
            return ""
        try:
            num = float(val.replace("%","")) / 100.0
            if num >= GOOD_THRESH:
                return "good"
            if num <= BAD_THRESH:
                return "bad"
            return ""
        except Exception:
            return ""

    styled = df.copy()
    styled["_conv_cls"]  = styled["conv_lift"].apply(lift_class)
    styled["_start_cls"] = styled["start_lift"].apply(lift_class)
    styled["_comp_cls"]  = styled["comp_lift"].apply(lift_class)

    # Build HTML rows manually so we can inject classes
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
    """Send categorized results to GPT with your rules for Good/Bad/Neutral."""
    if df_for_gpt.empty:
        return "No in-flight variant data found in the selected window."

    client = OpenAI(api_key=OPENAI_API_KEY)

    # Give GPT concise, structured input (don’t dump gigantic tables).
    # We pass just the essential fields per variant.
    compact = df_for_gpt[[
        "property","test","variant",
        "sessions","form_starts","submits",
        "conv_ctrl","conv_var","conv_lift",
        "start_ctrl","start_var","start_lift",
        "comp_ctrl","comp_var","comp_lift",
        "category"
    ]]

    # Convert to markdown table for readability inside the model
    md_table = compact.to_markdown(index=False)

    system_msg = (
        "You are a senior CRO analyst writing an executive summary. "
        "Be crisp, action-oriented, and avoid jargon."
    )

    user_msg = f"""
Data from in-flight A/B tests (variants vs control) is below as a table.
Use these rules to categorize by conversion rate lift:
- Good: conv_lift >= {GOOD_THRESH*100:.1f}%
- Bad:  conv_lift <= {BAD_THRESH*100:.1f}%
- Neutral: otherwise

For each category, list the most important tests (group by property → test → variant).
For each item, give a one-line takeaway that calls out where the lift seems to come from
(e.g., higher start rate, higher completion rate, or both) using the provided lifts.
Keep it short and executive-friendly. Then end with 2-3 “What’s next” bullets.
    
Table:
{md_table}
"""

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.3,
        max_tokens=700,
    )

    return resp.choices[0].message.content.strip()

def send_email(summary_md: str, table_html: str):
    """Send the summary + table via Gmail."""
    if not EMAIL_SENDER or not EMAIL_PASSWORD or not EMAIL_RECIPIENTS:
        raise RuntimeError("Email sender/password/recipients not configured.")

    # Convert markdown-style bullets to simple HTML (light touch)
    summary_html = (
        summary_md
        .replace("\n\n", "<br><br>")
        .replace("\n- ", "<br>• ")
        .replace("\n", "<br>")
    )

    html_body = f"""
    <html>
      <body>
        <div style="font-family: Arial, sans-serif; max-width: 980px;">
          <h2 style="margin-bottom: 6px;">In-Flight A/B Test Summary</h2>
          <div style="color:#555; font-size: 13px; margin-bottom: 14px;">
            Window: last {DAYS_BACK} day(s). Categorization: Good &ge; {GOOD_THRESH*100:.0f}%, Bad &le; {BAD_THRESH*100:.0f}% (by conversion lift).
          </div>
          <div style="font-size: 14px; line-height: 1.5; margin-bottom: 18px;">
            {summary_html}
          </div>
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
    # If your table occasionally has only control rows in the window, handle gracefully
    if df.empty or (df[VARIANT_COL] == CONTROL_LABEL).all():
        summary = "No variant activity in the selected window."
        table_html = "<p><em>No rows to display.</em></p>"
        send_email(summary, table_html)
        return

    agg = compute_lifts(df)
    summary = build_gpt_summary(agg)
    table_html = to_html_table(agg)
    send_email(summary, table_html)

if __name__ == "__main__":
    main()
