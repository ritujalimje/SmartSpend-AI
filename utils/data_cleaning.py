# ═══════════════════════════════════════════════════════════════════════════════
#  SmartSpend AI – Expense & Budget Analyzer
#  utils/data_cleaning.py  |  v2.1 — pandas 3.x compatible
#
#  Public API (imported by app.py):
#    load_sample_data()    → DataFrame                    (built-in fallback)
#    load_file(f)          → (DataFrame | None, err | None)
#    validate_columns(df)  → (DataFrame | None, err | None)
#    clean_data(df)        → (DataFrame, report_dict)
#    render_data_preview() → None  (Streamlit UI)
#
#  Changes in v2.1 vs v2.0:
#    • Removed infer_datetime_format (dropped in pandas 3.x)
#    • Added _parse_dates_robust() — tries 8 common formats sequentially
#    • Added load_sample_data() — loads bundled CSV from data_samples/
#    • render_data_preview() now shows a download button for the cleaned data
# ═══════════════════════════════════════════════════════════════════════════════

import io
import os
import pandas as pd
import numpy as np
import streamlit as st


# ───────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ───────────────────────────────────────────────────────────────────────────────

REQUIRED_COLUMNS = ["Date", "Amount", "Category"]

# Maps any reasonable raw column name (lower-cased) → internal standard name.
COLUMN_NAME_MAP = {
    "date": "Date", "dates": "Date", "transaction_date": "Date",
    "txn_date": "Date", "trans_date": "Date", "dt": "Date",
    "amount": "Amount", "price": "Amount", "cost": "Amount",
    "expense": "Amount", "debit": "Amount", "spend": "Amount",
    "spent": "Amount", "value": "Amount", "total": "Amount",
    "category": "Category", "categories": "Category", "type": "Category",
    "expense_type": "Category", "tag": "Category", "head": "Category",
    "description": "Description", "desc": "Description",
    "narration": "Description", "note": "Description",
    "details": "Description", "remarks": "Description",
    "payment_mode": "Payment_Mode", "mode": "Payment_Mode",
    "payment": "Payment_Mode", "method": "Payment_Mode",
    "paid_via": "Payment_Mode", "channel": "Payment_Mode",
}

# Maps any messy category value (lower-cased) → clean display name.
CATEGORY_ALIASES = {
    "food": "Food & Dining", "dining": "Food & Dining",
    "restaurant": "Food & Dining", "eating out": "Food & Dining",
    "meals": "Food & Dining", "swiggy": "Food & Dining",
    "zomato": "Food & Dining", "delivery": "Food & Dining",
    "groceries": "Groceries", "grocery": "Groceries",
    "supermarket": "Groceries", "vegetables": "Groceries",
    "dairy": "Groceries", "fruits": "Groceries",
    "transport": "Transport", "transportation": "Transport",
    "travel": "Transport", "cab": "Transport",
    "uber": "Transport", "ola": "Transport",
    "fuel": "Transport", "petrol": "Transport",
    "auto": "Transport", "metro": "Transport",
    "rent": "Housing & Rent", "housing": "Housing & Rent",
    "home": "Housing & Rent", "maintenance": "Housing & Rent",
    "electricity": "Utilities", "utilities": "Utilities",
    "water": "Utilities", "gas": "Utilities",
    "internet": "Utilities", "phone": "Utilities",
    "mobile": "Utilities", "bill": "Utilities", "wifi": "Utilities",
    "entertainment": "Entertainment", "movies": "Entertainment",
    "netflix": "Entertainment", "ott": "Entertainment",
    "streaming": "Entertainment", "hotstar": "Entertainment",
    "spotify": "Entertainment", "gaming": "Entertainment",
    "shopping": "Shopping", "clothes": "Shopping",
    "clothing": "Shopping", "fashion": "Shopping",
    "amazon": "Shopping", "flipkart": "Shopping",
    "health": "Health & Medical", "medical": "Health & Medical",
    "medicine": "Health & Medical", "pharmacy": "Health & Medical",
    "doctor": "Health & Medical", "hospital": "Health & Medical",
    "gym": "Health & Medical", "fitness": "Health & Medical",
    "education": "Education", "course": "Education",
    "books": "Education", "tuition": "Education",
    "udemy": "Education", "training": "Education",
    "investment": "Investments", "savings": "Investments",
    "mutual fund": "Investments", "stocks": "Investments",
    "insurance": "Insurance",
    "emi": "EMI / Loans", "loan": "EMI / Loans",
    "other": "Miscellaneous", "misc": "Miscellaneous",
    "miscellaneous": "Miscellaneous",
}

# Date formats to try, in priority order.
# Sequential trial is O(n×formats) but far more reliable than letting pandas guess.
_DATE_FORMATS = [
    "%Y-%m-%d",    # 2024-01-15  (ISO — most common in CSVs)
    "%d/%m/%Y",    # 15/01/2024  (Indian / European)
    "%m/%d/%Y",    # 01/15/2024  (US format)
    "%d-%m-%Y",    # 15-01-2024
    "%Y/%m/%d",    # 2024/01/15
    "%d %b %Y",    # 15 Jan 2024
    "%d-%b-%Y",    # 15-Jan-2024
    "%b %d, %Y",   # Jan 15, 2024
    "%d/%m/%y",    # 15/01/24  (2-digit year)
    "%m/%d/%y",    # 01/15/24
]

# Path to the bundled sample dataset (relative to project root)
_SAMPLE_CSV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data_samples", "sample_data.csv"
)


# ───────────────────────────────────────────────────────────────────────────────
#  PRIVATE HELPERS
# ───────────────────────────────────────────────────────────────────────────────

def _parse_dates_robust(series: pd.Series) -> pd.Series:
    """
    Try each format in _DATE_FORMATS in order. For each format, parse only
    the rows that are still NaT from previous attempts. This gives us correct
    parsing for mixed-format columns without ever calling the deprecated
    infer_datetime_format parameter (removed in pandas 3.0).

    Returns a pd.Series of dtype datetime64[ns].
    """
    # Work with string representations to avoid type issues
    raw = series.astype(str).str.strip()
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    for fmt in _DATE_FORMATS:
        # Only attempt rows not yet successfully parsed
        missing_mask = result.isna()
        if not missing_mask.any():
            break
        parsed = pd.to_datetime(
            raw[missing_mask], format=fmt, errors="coerce"
        )
        result[missing_mask] = parsed.values

    return result


def _map_category(raw_val) -> str:
    """Lower-case lookup in CATEGORY_ALIASES; fall back to title-case original."""
    if pd.isna(raw_val):
        return "Miscellaneous"
    key = str(raw_val).lower().strip()
    return CATEGORY_ALIASES.get(key, str(raw_val).strip().title())


# ───────────────────────────────────────────────────────────────────────────────
#  1. SAMPLE DATA LOADER
# ───────────────────────────────────────────────────────────────────────────────
def load_sample_data() -> pd.DataFrame:
    """
    Load the built-in sample expense CSV, run it through the full cleaning
    pipeline, and return a ready-to-use DataFrame.

    Called by app.py when no file has been uploaded yet, so the dashboard
    always shows live charts instead of an empty state.

    Returns:
        Cleaned DataFrame (same schema as user-uploaded data after clean_data())
    """
    df_raw = pd.read_csv(_SAMPLE_CSV_PATH)
    df_validated, err = validate_columns(df_raw)
    if err:
        raise RuntimeError(f"Sample data validation failed: {err}")
    df_clean, _ = clean_data(df_validated)
    return df_clean


# ───────────────────────────────────────────────────────────────────────────────
#  2. FILE LOADER
# ───────────────────────────────────────────────────────────────────────────────
def load_file(uploaded_file) -> tuple:
    """
    Read a Streamlit UploadedFile into a raw pandas DataFrame.

    Supports:
        .csv  — comma-separated; auto-retries with semicolon if only 1 col parsed
        .xlsx / .xls — reads sheet 0

    Returns:
        (DataFrame, None)      on success
        (None, error_string)   on any failure
    """
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            raw = uploaded_file.read()
            df = pd.read_csv(io.BytesIO(raw), sep=",")
            if df.shape[1] == 1:                          # likely semicolon-delimited
                df = pd.read_csv(io.BytesIO(raw), sep=";")

        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file, sheet_name=0)

        else:
            return None, (
                f"❌ Unsupported file type **'{uploaded_file.name}'**. "
                "Please upload a `.csv`, `.xlsx`, or `.xls` file."
            )

        if df.empty:
            return None, "❌ The uploaded file is empty — no rows found."

        return df, None

    except Exception as exc:
        return None, f"❌ Could not read file: {exc}"


# ───────────────────────────────────────────────────────────────────────────────
#  3. COLUMN VALIDATOR
# ───────────────────────────────────────────────────────────────────────────────
def validate_columns(df: pd.DataFrame) -> tuple:
    """
    Strip whitespace from column names, apply COLUMN_NAME_MAP renames,
    then verify that Date, Amount, and Category are all present.

    Returns:
        (renamed_df, None)          if all required columns found
        (None, error_message_str)   if any are missing
    """
    # Strip leading/trailing whitespace that Excel often adds
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Build rename map (lower-case lookup)
    rename_map = {
        col: COLUMN_NAME_MAP[col.lower()]
        for col in df.columns
        if col.lower() in COLUMN_NAME_MAP
    }
    df = df.rename(columns=rename_map)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return None, (
            f"Missing required column(s): **{', '.join(missing)}**.\n\n"
            f"Columns found in your file: `{list(df.columns)}`.\n\n"
            "Rename your columns to **Date**, **Amount**, and **Category** "
            "(or common variants — see README for the full alias list)."
        )

    return df, None


# ───────────────────────────────────────────────────────────────────────────────
#  4. CLEANING PIPELINE
# ───────────────────────────────────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> tuple:
    """
    Full preprocessing pipeline — transforms raw validated data into an
    analysis-ready DataFrame.

    Steps:
        A. Amount     — strip currency symbols, cast float, drop ≤ 0
        B. Date       — multi-format robust parse, drop NaT rows
        C. Duplicates — drop exact row duplicates
        D. Optionals  — create/fill Description and Payment_Mode if missing
        E. Category   — map aliases → canonical names, title-case unknowns
        F. Time cols  — derive Month, Year, Month_Name, Month_Year,
                        Day_of_Week, Week
        G. Sort       — chronological ascending

    Returns:
        (clean_df, report_dict)
        report_dict keys:
            rows_before, rows_after, nulls_dropped,
            duplicates_dropped, categories_standardised
    """
    report = {
        "rows_before": len(df),
        "nulls_dropped": 0,
        "duplicates_dropped": 0,
        "categories_standardised": 0,
        "rows_after": 0,
    }
    df = df.copy()

    # ── A. Amount ─────────────────────────────────────────────────────────
    if df["Amount"].dtype == object:
        df["Amount"] = (
            df["Amount"]
            .astype(str)
            .str.replace(r"[^\d\.]", "", regex=True)  # strip ₹ $ , etc.
            .replace("", np.nan)
        )
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["Amount"])
    df = df[df["Amount"] > 0]                         # exclude refunds/reversals
    report["nulls_dropped"] += before - len(df)

    # ── B. Date — robust multi-format parsing (pandas 3.x compatible) ─────
    before = len(df)
    df["Date"] = _parse_dates_robust(df["Date"])
    df = df.dropna(subset=["Date"])
    report["nulls_dropped"] += before - len(df)

    # ── C. Duplicates ─────────────────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates()
    report["duplicates_dropped"] = before - len(df)

    # ── D. Optional columns — ensure they always exist ────────────────────
    if "Description" not in df.columns:
        df["Description"] = "No description"
    else:
        df["Description"] = df["Description"].fillna("No description").astype(str)

    if "Payment_Mode" not in df.columns:
        df["Payment_Mode"] = "Unknown"
    else:
        df["Payment_Mode"] = (
            df["Payment_Mode"].fillna("Unknown").astype(str).str.strip()
        )

    # ── E. Category standardisation ───────────────────────────────────────
    original = df["Category"].copy()
    df["Category"] = df["Category"].apply(_map_category)
    report["categories_standardised"] = int((df["Category"] != original).sum())

    # ── F. Derived time columns ────────────────────────────────────────────
    df["Month"]       = df["Date"].dt.month                      # 1–12
    df["Year"]        = df["Date"].dt.year                       # e.g. 2024
    df["Month_Name"]  = df["Date"].dt.strftime("%b")             # Jan…Dec
    df["Month_Year"]  = df["Date"].dt.to_period("M").astype(str) # "2024-01"
    df["Day_of_Week"] = df["Date"].dt.day_name()                 # Monday…
    df["Week"]        = df["Date"].dt.isocalendar().week.astype(int)

    # ── G. Sort ───────────────────────────────────────────────────────────
    df = df.sort_values("Date").reset_index(drop=True)

    report["rows_after"] = len(df)
    return df, report


# ───────────────────────────────────────────────────────────────────────────────
#  5. DATA PREVIEW PAGE  (Streamlit UI — called from app.py router)
# ───────────────────────────────────────────────────────────────────────────────
def render_data_preview(df: pd.DataFrame, report: dict, currency: str = "₹") -> None:
    """
    Renders the Data Preview page:
        • Cleaning report metrics row
        • Column badge strip
        • Filterable / sortable data table (first 200 rows)
        • CSV download button for the cleaned data

    Args:
        df       : cleaned + filtered DataFrame
        report   : dict from clean_data() (may be None for sample data)
        currency : currency symbol for Amount formatting
    """
    if report is None:
        # Sample data path — supply placeholder report
        report = {
            "rows_after": len(df), "rows_before": len(df),
            "nulls_dropped": 0, "duplicates_dropped": 0,
            "categories_standardised": 0,
        }

    # ── Cleaning summary metrics ───────────────────────────────────────────
    st.markdown(
        '<div class="section-header">'
        '<h2>🧹 Cleaning Summary</h2>'
        '<p>What the preprocessing pipeline changed in your data</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows Loaded",          f"{report['rows_after']:,}")
    c2.metric("Rows Before",          f"{report['rows_before']:,}")
    c3.metric("Nulls / Bad Rows",     f"{report['nulls_dropped']:,}")
    c4.metric("Duplicates Removed",   f"{report['duplicates_dropped']:,}")
    c5.metric("Categories Mapped",    f"{report['categories_standardised']:,}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Column chip strip ──────────────────────────────────────────────────
    # Colour-code: required cols = teal, optional = blue, derived = gold
    required  = {"Date", "Amount", "Category"}
    optional  = {"Description", "Payment_Mode"}
    chips = []
    for col in df.columns:
        if col in required:
            cls = "badge-teal"
        elif col in optional:
            cls = "badge-blue"
        else:
            cls = "badge-gold"
        chips.append(f'<span class="badge {cls}">{col}</span>')

    st.markdown(
        f'<div style="margin-bottom:0.6rem;line-height:2.2;">'
        + " ".join(chips)
        + "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="font-size:0.7rem;color:#484F58;margin-bottom:1rem;">'
        '<span class="badge badge-teal" style="margin-right:4px;">■</span> Required &nbsp;'
        '<span class="badge badge-blue" style="margin-right:4px;">■</span> Optional &nbsp;'
        '<span class="badge badge-gold">■</span> Derived</p>',
        unsafe_allow_html=True,
    )

    # ── Data table ────────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">'
        '<h2>📋 Cleaned Records</h2>'
        f'<p>Showing first 200 of {len(df):,} rows · '
        f'{df["Category"].nunique()} categories · '
        f'{df["Month_Year"].nunique()} months</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    display = df.head(200).copy()
    display["Date"]   = display["Date"].dt.strftime("%d %b %Y")
    display["Amount"] = display["Amount"].apply(lambda x: f"{currency}{x:,.2f}")

    # Show only the most useful columns in the preview table
    show_cols = [c for c in
                 ["Date", "Amount", "Category", "Description",
                  "Payment_Mode", "Month_Year", "Day_of_Week"]
                 if c in display.columns]

    st.dataframe(display[show_cols], use_container_width=True, hide_index=True)

    if len(df) > 200:
        st.caption(f"Showing 200 of {len(df):,} records. Download below for full data.")

    # ── Download cleaned CSV ───────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️  Download Cleaned Data (CSV)",
        data=csv_bytes,
        file_name="smartspend_cleaned.csv",
        mime="text/csv",
        use_container_width=True,
    )
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# # ═══════════════════════════════════════════════════════════════════════════════
# #  SmartSpend AI – Expense & Budget Analyzer
# #  utils/data_cleaning.py  |  v2.1 — pandas 3.x compatible
# #
# #  Public API (imported by app.py):
# #    load_sample_data()    → DataFrame                    (built-in fallback)
# #    load_file(f)          → (DataFrame | None, err | None)
# #    validate_columns(df)  → (DataFrame | None, err | None)
# #    clean_data(df)        → (DataFrame, report_dict)
# #    render_data_preview() → None  (Streamlit UI)
# #
# #  Changes in v2.1 vs v2.0:
# #    • Removed infer_datetime_format (dropped in pandas 3.x)
# #    • Added _parse_dates_robust() — tries 8 common formats sequentially
# #    • Added load_sample_data() — loads bundled CSV from data_samples/
# #    • render_data_preview() now shows a download button for the cleaned data
# # ═══════════════════════════════════════════════════════════════════════════════

# import io
# import os
# import pandas as pd
# import numpy as np
# import streamlit as st


# # ───────────────────────────────────────────────────────────────────────────────
# #  CONSTANTS
# # ───────────────────────────────────────────────────────────────────────────────

# REQUIRED_COLUMNS = ["Date", "Amount", "Category"]

# # Maps any reasonable raw column name (lower-cased) → internal standard name.
# COLUMN_NAME_MAP = {
#     "date": "Date", "dates": "Date", "transaction_date": "Date",
#     "txn_date": "Date", "trans_date": "Date", "dt": "Date",
#     "amount": "Amount", "price": "Amount", "cost": "Amount",
#     "expense": "Amount", "debit": "Amount", "spend": "Amount",
#     "spent": "Amount", "value": "Amount", "total": "Amount",
#     "category": "Category", "categories": "Category", "type": "Category",
#     "expense_type": "Category", "tag": "Category", "head": "Category",
#     "description": "Description", "desc": "Description",
#     "narration": "Description", "note": "Description",
#     "details": "Description", "remarks": "Description",
#     "payment_mode": "Payment_Mode", "mode": "Payment_Mode",
#     "payment": "Payment_Mode", "method": "Payment_Mode",
#     "paid_via": "Payment_Mode", "channel": "Payment_Mode",
# }

# # Maps any messy category value (lower-cased) → clean display name.
# CATEGORY_ALIASES = {
#     "food": "Food & Dining", "dining": "Food & Dining",
#     "restaurant": "Food & Dining", "eating out": "Food & Dining",
#     "meals": "Food & Dining", "swiggy": "Food & Dining",
#     "zomato": "Food & Dining", "delivery": "Food & Dining",
#     "groceries": "Groceries", "grocery": "Groceries",
#     "supermarket": "Groceries", "vegetables": "Groceries",
#     "dairy": "Groceries", "fruits": "Groceries",
#     "transport": "Transport", "transportation": "Transport",
#     "travel": "Transport", "cab": "Transport",
#     "uber": "Transport", "ola": "Transport",
#     "fuel": "Transport", "petrol": "Transport",
#     "auto": "Transport", "metro": "Transport",
#     "rent": "Housing & Rent", "housing": "Housing & Rent",
#     "home": "Housing & Rent", "maintenance": "Housing & Rent",
#     "electricity": "Utilities", "utilities": "Utilities",
#     "water": "Utilities", "gas": "Utilities",
#     "internet": "Utilities", "phone": "Utilities",
#     "mobile": "Utilities", "bill": "Utilities", "wifi": "Utilities",
#     "entertainment": "Entertainment", "movies": "Entertainment",
#     "netflix": "Entertainment", "ott": "Entertainment",
#     "streaming": "Entertainment", "hotstar": "Entertainment",
#     "spotify": "Entertainment", "gaming": "Entertainment",
#     "shopping": "Shopping", "clothes": "Shopping",
#     "clothing": "Shopping", "fashion": "Shopping",
#     "amazon": "Shopping", "flipkart": "Shopping",
#     "health": "Health & Medical", "medical": "Health & Medical",
#     "medicine": "Health & Medical", "pharmacy": "Health & Medical",
#     "doctor": "Health & Medical", "hospital": "Health & Medical",
#     "gym": "Health & Medical", "fitness": "Health & Medical",
#     "education": "Education", "course": "Education",
#     "books": "Education", "tuition": "Education",
#     "udemy": "Education", "training": "Education",
#     "investment": "Investments", "savings": "Investments",
#     "mutual fund": "Investments", "stocks": "Investments",
#     "insurance": "Insurance",
#     "emi": "EMI / Loans", "loan": "EMI / Loans",
#     "other": "Miscellaneous", "misc": "Miscellaneous",
#     "miscellaneous": "Miscellaneous",
# }

# # Date formats to try, in priority order.
# # Sequential trial is O(n×formats) but far more reliable than letting pandas guess.
# _DATE_FORMATS = [
#     "%Y-%m-%d",    # 2024-01-15  (ISO — most common in CSVs)
#     "%d/%m/%Y",    # 15/01/2024  (Indian / European)
#     "%m/%d/%Y",    # 01/15/2024  (US format)
#     "%d-%m-%Y",    # 15-01-2024
#     "%Y/%m/%d",    # 2024/01/15
#     "%d %b %Y",    # 15 Jan 2024
#     "%d-%b-%Y",    # 15-Jan-2024
#     "%b %d, %Y",   # Jan 15, 2024
#     "%d/%m/%y",    # 15/01/24  (2-digit year)
#     "%m/%d/%y",    # 01/15/24
# ]

# # Path to the bundled sample dataset (relative to project root)
# _SAMPLE_CSV_PATH = os.path.join(
#     os.path.dirname(os.path.dirname(__file__)),
#     "data_samples", "sample_expenses.csv"
# )


# # ───────────────────────────────────────────────────────────────────────────────
# #  PRIVATE HELPERS
# # ───────────────────────────────────────────────────────────────────────────────

# def _parse_dates_robust(series: pd.Series) -> pd.Series:
#     """
#     Try each format in _DATE_FORMATS in order. For each format, parse only
#     the rows that are still NaT from previous attempts. This gives us correct
#     parsing for mixed-format columns without ever calling the deprecated
#     infer_datetime_format parameter (removed in pandas 3.0).

#     Returns a pd.Series of dtype datetime64[ns].
#     """
#     # Work with string representations to avoid type issues
#     raw = series.astype(str).str.strip()
#     result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

#     for fmt in _DATE_FORMATS:
#         # Only attempt rows not yet successfully parsed
#         missing_mask = result.isna()
#         if not missing_mask.any():
#             break
#         parsed = pd.to_datetime(
#             raw[missing_mask], format=fmt, errors="coerce"
#         )
#         result[missing_mask] = parsed.values

#     return result


# def _map_category(raw_val) -> str:
#     """Lower-case lookup in CATEGORY_ALIASES; fall back to title-case original."""
#     if pd.isna(raw_val):
#         return "Miscellaneous"
#     key = str(raw_val).lower().strip()
#     return CATEGORY_ALIASES.get(key, str(raw_val).strip().title())


# # ───────────────────────────────────────────────────────────────────────────────
# #  1. SAMPLE DATA LOADER
# # ───────────────────────────────────────────────────────────────────────────────
# def load_sample_data() -> pd.DataFrame:
#     """
#     Load the built-in sample expense CSV, run it through the full cleaning
#     pipeline, and return a ready-to-use DataFrame.

#     Called by app.py when no file has been uploaded yet, so the dashboard
#     always shows live charts instead of an empty state.

#     Returns:
#         Cleaned DataFrame (same schema as user-uploaded data after clean_data())
#     """
#     df_raw = pd.read_csv(_SAMPLE_CSV_PATH)
#     df_validated, err = validate_columns(df_raw)
#     if err:
#         raise RuntimeError(f"Sample data validation failed: {err}")
#     df_clean, _ = clean_data(df_validated)
#     return df_clean


# # ───────────────────────────────────────────────────────────────────────────────
# #  2. FILE LOADER
# # ───────────────────────────────────────────────────────────────────────────────
# def load_file(uploaded_file) -> tuple:
#     """
#     Read a Streamlit UploadedFile into a raw pandas DataFrame.

#     Supports:
#         .csv  — comma-separated; auto-retries with semicolon if only 1 col parsed
#         .xlsx / .xls — reads sheet 0

#     Returns:
#         (DataFrame, None)      on success
#         (None, error_string)   on any failure
#     """
#     name = uploaded_file.name.lower()
#     try:
#         if name.endswith(".csv"):
#             raw = uploaded_file.read()
#             df = pd.read_csv(io.BytesIO(raw), sep=",")
#             if df.shape[1] == 1:                          # likely semicolon-delimited
#                 df = pd.read_csv(io.BytesIO(raw), sep=";")

#         elif name.endswith((".xlsx", ".xls")):
#             df = pd.read_excel(uploaded_file, sheet_name=0)

#         else:
#             return None, (
#                 f"❌ Unsupported file type **'{uploaded_file.name}'**. "
#                 "Please upload a `.csv`, `.xlsx`, or `.xls` file."
#             )

#         if df.empty:
#             return None, "❌ The uploaded file is empty — no rows found."

#         return df, None

#     except Exception as exc:
#         return None, f"❌ Could not read file: {exc}"


# # ───────────────────────────────────────────────────────────────────────────────
# #  3. COLUMN VALIDATOR
# # ───────────────────────────────────────────────────────────────────────────────
# def validate_columns(df: pd.DataFrame) -> tuple:
#     """
#     Strip whitespace from column names, apply COLUMN_NAME_MAP renames,
#     then verify that Date, Amount, and Category are all present.

#     Returns:
#         (renamed_df, None)          if all required columns found
#         (None, error_message_str)   if any are missing
#     """
#     # Strip leading/trailing whitespace that Excel often adds
#     df = df.copy()
#     df.columns = [str(c).strip() for c in df.columns]

#     # Build rename map (lower-case lookup)
#     rename_map = {
#         col: COLUMN_NAME_MAP[col.lower()]
#         for col in df.columns
#         if col.lower() in COLUMN_NAME_MAP
#     }
#     df = df.rename(columns=rename_map)

#     missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
#     if missing:
#         return None, (
#             f"Missing required column(s): **{', '.join(missing)}**.\n\n"
#             f"Columns found in your file: `{list(df.columns)}`.\n\n"
#             "Rename your columns to **Date**, **Amount**, and **Category** "
#             "(or common variants — see README for the full alias list)."
#         )

#     return df, None


# # ───────────────────────────────────────────────────────────────────────────────
# #  4. CLEANING PIPELINE
# # ───────────────────────────────────────────────────────────────────────────────
# def clean_data(df: pd.DataFrame) -> tuple:
#     """
#     Full preprocessing pipeline — transforms raw validated data into an
#     analysis-ready DataFrame.

#     Steps:
#         A. Amount     — strip currency symbols, cast float, drop ≤ 0
#         B. Date       — multi-format robust parse, drop NaT rows
#         C. Duplicates — drop exact row duplicates
#         D. Optionals  — create/fill Description and Payment_Mode if missing
#         E. Category   — map aliases → canonical names, title-case unknowns
#         F. Time cols  — derive Month, Year, Month_Name, Month_Year,
#                         Day_of_Week, Week
#         G. Sort       — chronological ascending

#     Returns:
#         (clean_df, report_dict)
#         report_dict keys:
#             rows_before, rows_after, nulls_dropped,
#             duplicates_dropped, categories_standardised
#     """
#     report = {
#         "rows_before": len(df),
#         "nulls_dropped": 0,
#         "duplicates_dropped": 0,
#         "categories_standardised": 0,
#         "rows_after": 0,
#     }
#     df = df.copy()

#     # ── A. Amount ─────────────────────────────────────────────────────────
#     if df["Amount"].dtype == object:
#         df["Amount"] = (
#             df["Amount"]
#             .astype(str)
#             .str.replace(r"[^\d\.]", "", regex=True)  # strip ₹ $ , etc.
#             .replace("", np.nan)
#         )
#     df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")

#     before = len(df)
#     df = df.dropna(subset=["Amount"])
#     df = df[df["Amount"] > 0]                         # exclude refunds/reversals
#     report["nulls_dropped"] += before - len(df)

#     # ── B. Date — robust multi-format parsing (pandas 3.x compatible) ─────
#     before = len(df)
#     df["Date"] = _parse_dates_robust(df["Date"])
#     df = df.dropna(subset=["Date"])
#     report["nulls_dropped"] += before - len(df)

#     # ── C. Duplicates ─────────────────────────────────────────────────────
#     before = len(df)
#     df = df.drop_duplicates()
#     report["duplicates_dropped"] = before - len(df)

#     # ── D. Optional columns — ensure they always exist ────────────────────
#     if "Description" not in df.columns:
#         df["Description"] = "No description"
#     else:
#         df["Description"] = df["Description"].fillna("No description").astype(str)

#     if "Payment_Mode" not in df.columns:
#         df["Payment_Mode"] = "Unknown"
#     else:
#         df["Payment_Mode"] = (
#             df["Payment_Mode"].fillna("Unknown").astype(str).str.strip()
#         )

#     # ── E. Category standardisation ───────────────────────────────────────
#     original = df["Category"].copy()
#     df["Category"] = df["Category"].apply(_map_category)
#     report["categories_standardised"] = int((df["Category"] != original).sum())

#     # ── F. Derived time columns ────────────────────────────────────────────
#     df["Month"]       = df["Date"].dt.month                      # 1–12
#     df["Year"]        = df["Date"].dt.year                       # e.g. 2024
#     df["Month_Name"]  = df["Date"].dt.strftime("%b")             # Jan…Dec
#     df["Month_Year"]  = df["Date"].dt.to_period("M").astype(str) # "2024-01"
#     df["Day_of_Week"] = df["Date"].dt.day_name()                 # Monday…
#     df["Week"]        = df["Date"].dt.isocalendar().week.astype(int)

#     # ── G. Sort ───────────────────────────────────────────────────────────
#     df = df.sort_values("Date").reset_index(drop=True)

#     report["rows_after"] = len(df)
#     return df, report


# # ───────────────────────────────────────────────────────────────────────────────
# #  5. DATA PREVIEW PAGE  (Streamlit UI — called from app.py router)
# # ───────────────────────────────────────────────────────────────────────────────
# def render_data_preview(df: pd.DataFrame, report: dict, currency: str = "₹") -> None:
#     """
#     Renders the Data Preview page:
#         • Cleaning report metrics row
#         • Column badge strip
#         • Filterable / sortable data table (first 200 rows)
#         • CSV download button for the cleaned data

#     Args:
#         df       : cleaned + filtered DataFrame
#         report   : dict from clean_data() (may be None for sample data)
#         currency : currency symbol for Amount formatting
#     """
#     if report is None:
#         # Sample data path — supply placeholder report
#         report = {
#             "rows_after": len(df), "rows_before": len(df),
#             "nulls_dropped": 0, "duplicates_dropped": 0,
#             "categories_standardised": 0,
#         }

#     # ── Cleaning summary metrics ───────────────────────────────────────────
#     st.markdown(
#         '<div class="section-header">'
#         '<h2>🧹 Cleaning Summary</h2>'
#         '<p>What the preprocessing pipeline changed in your data</p>'
#         '</div>',
#         unsafe_allow_html=True,
#     )

#     c1, c2, c3, c4, c5 = st.columns(5)
#     c1.metric("Rows Loaded",          f"{report['rows_after']:,}")
#     c2.metric("Rows Before",          f"{report['rows_before']:,}")
#     c3.metric("Nulls / Bad Rows",     f"{report['nulls_dropped']:,}")
#     c4.metric("Duplicates Removed",   f"{report['duplicates_dropped']:,}")
#     c5.metric("Categories Mapped",    f"{report['categories_standardised']:,}")

#     st.markdown("<br>", unsafe_allow_html=True)

#     # ── Column chip strip ──────────────────────────────────────────────────
#     # Colour-code: required cols = teal, optional = blue, derived = gold
#     required  = {"Date", "Amount", "Category"}
#     optional  = {"Description", "Payment_Mode"}
#     chips = []
#     for col in df.columns:
#         if col in required:
#             cls = "badge-teal"
#         elif col in optional:
#             cls = "badge-blue"
#         else:
#             cls = "badge-gold"
#         chips.append(f'<span class="badge {cls}">{col}</span>')

#     st.markdown(
#         f'<div style="margin-bottom:0.6rem;line-height:2.2;">'
#         + " ".join(chips)
#         + "</div>",
#         unsafe_allow_html=True,
#     )
#     st.markdown(
#         '<p style="font-size:0.7rem;color:#484F58;margin-bottom:1rem;">'
#         '<span class="badge badge-teal" style="margin-right:4px;">■</span> Required &nbsp;'
#         '<span class="badge badge-blue" style="margin-right:4px;">■</span> Optional &nbsp;'
#         '<span class="badge badge-gold">■</span> Derived</p>',
#         unsafe_allow_html=True,
#     )

#     # ── Data table ────────────────────────────────────────────────────────
#     st.markdown(
#         '<div class="section-header">'
#         '<h2>📋 Cleaned Records</h2>'
#         f'<p>Showing first 200 of {len(df):,} rows · '
#         f'{df["Category"].nunique()} categories · '
#         f'{df["Month_Year"].nunique()} months</p>'
#         '</div>',
#         unsafe_allow_html=True,
#     )

#     display = df.head(200).copy()
#     display["Date"]   = display["Date"].dt.strftime("%d %b %Y")
#     display["Amount"] = display["Amount"].apply(lambda x: f"{currency}{x:,.2f}")

#     # Show only the most useful columns in the preview table
#     show_cols = [c for c in
#                  ["Date", "Amount", "Category", "Description",
#                   "Payment_Mode", "Month_Year", "Day_of_Week"]
#                  if c in display.columns]

#     st.dataframe(display[show_cols], use_container_width=True, hide_index=True)

#     if len(df) > 200:
#         st.caption(f"Showing 200 of {len(df):,} records. Download below for full data.")

#     # ── Download cleaned CSV ───────────────────────────────────────────────
#     st.markdown("<br>", unsafe_allow_html=True)
#     csv_bytes = df.to_csv(index=False).encode("utf-8")
#     st.download_button(
#         label="⬇️  Download Cleaned Data (CSV)",
#         data=csv_bytes,
#         file_name="smartspend_cleaned.csv",
#         mime="text/csv",
#         use_container_width=True,
#     )