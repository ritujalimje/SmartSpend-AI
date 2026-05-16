# ═══════════════════════════════════════════════════════════════════════════════
#  SmartSpend AI – Expense & Budget Analyzer
#  app.py  |  v3.0 — Real data workflow, sample fallback, dynamic filters
#
#  Execution order every Streamlit rerun:
#    1. set_page_config()      — must be first st.* call
#    2. inject_global_css()    — all custom CSS injected before content
#    3. init_session_state()   — declare all keys with safe defaults (no-op after first run)
#    4. render_sidebar()       — draw sidebar, return (uploaded_file, filters)
#    5. _run_data_pipeline()   — load sample OR uploaded file (once per new file)
#    6. apply_filters()        — slice clean_df by month/category/payment_mode
#    7. route_page()           — render the page the user navigated to
#
#  Module responsibilities:
#    app.py                  — orchestration, CSS, session-state, sidebar, routing
#    utils/data_cleaning.py  — load_file, validate_columns, clean_data,
#                              load_sample_data, render_data_preview
#    utils/analysis.py       — compute_kpis, charts, render_dashboard
# ═══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd

# ── Local imports ─────────────────────────────────────────────────────────────
from utils.data_cleaning import (
    load_file,            # read CSV / Excel → raw DataFrame
    validate_columns,     # rename cols + check required ones present
    clean_data,           # full preprocessing pipeline
    load_sample_data,     # load bundled sample_expenses.csv
    render_data_preview,  # Streamlit table + cleaning stats + download
)
from utils.analysis import (
    render_dashboard,     # KPI cards + budget bar + 3 charts + summary table
)
from utils.prediction import (
    render_predictions_page,   # full ML predictions page — Linear Regression
)
from utils.insights import generate_insights
from utils.scoring import calculate_health_score
from utils.report_generator import generate_pdf_report
# ═══════════════════════════════════════════════════════════════════════════════
#  1. PAGE CONFIG  — must be the very first Streamlit call in the file
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SmartSpend AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════════════════════════
#  2. GLOBAL CSS
#  Single source of truth for every visual style in the app.
#  CSS variables in :root mean one colour change rethemes everything.
# ═══════════════════════════════════════════════════════════════════════════════
def inject_global_css() -> None:
    """Inject the full fintech-dark CSS layer on top of Streamlit's theme."""
    st.markdown("""
    <style>
    /* ── Google Fonts ─────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Design tokens ────────────────────────────────────────────────── */
    :root {
        --bg-primary:   #0D1117;
        --bg-secondary: #161B22;
        --bg-card:      #1C2333;
        --border:       #30363D;
        --border-hi:    #444C56;
        --teal:         #00C9A7;
        --gold:         #F5A623;
        --red:          #FF6B6B;
        --blue:         #58A6FF;
        --purple:       #A78BFA;
        --text-hi:      #E6EDF3;
        --text-lo:      #8B949E;
        --text-muted:   #484F58;
        --font-sans:    'Inter', sans-serif;
        --font-mono:    'IBM Plex Mono', monospace;
        --radius:       10px;
        --radius-sm:    6px;
        --shadow:       0 4px 24px rgba(0,0,0,0.45);
    }

    /* ── Reset / base ─────────────────────────────────────────────────── */
    html, body, [class*="css"]   { font-family: var(--font-sans); color: var(--text-hi); }
    .block-container             { padding-top: 1.5rem; padding-bottom: 3rem; }
    h1, h2, h3                   { font-family: var(--font-sans); }

    /* ── Sidebar ──────────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border);
    }
    /* Override Streamlit's default label size for all sidebar widgets */
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stMultiSelect label,
    section[data-testid="stSidebar"] .stNumberInput label {
        color: var(--text-lo) !important;
        font-size: 0.7rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
    }

    /* ── KPI card ─────────────────────────────────────────────────────── */
    .kpi-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.15rem 1.35rem 1.2rem;
        box-shadow: var(--shadow);
        transition: border-color 0.18s ease, box-shadow 0.18s ease;
        height: 100%;
        box-sizing: border-box;
    }
    .kpi-card:hover {
        border-color: var(--teal);
        box-shadow: 0 4px 28px rgba(0,201,167,0.12);
    }
    .kpi-label {
        font-size: 0.66rem;
        font-weight: 700;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        color: var(--text-lo);
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.35rem;
    }
    .kpi-value {
        font-family: var(--font-mono);
        font-size: 1.6rem;
        font-weight: 600;
        color: var(--text-hi);
        line-height: 1.1;
        word-break: break-word;
        letter-spacing: -0.01em;
    }
    .kpi-sub {
        font-size: 0.69rem;
        color: var(--text-muted);
        margin-top: 0.45rem;
        line-height: 1.4;
    }

    /* ── Section header ───────────────────────────────────────────────── */
    .section-header {
        border-left: 3px solid var(--teal);
        padding-left: 0.8rem;
        margin: 1.8rem 0 1rem;
    }
    .section-header h2 {
        font-size: 0.93rem;
        font-weight: 600;
        color: var(--text-hi);
        margin: 0;
        letter-spacing: 0.01em;
    }
    .section-header p {
        font-size: 0.76rem;
        color: var(--text-lo);
        margin: 0.18rem 0 0;
    }

    /* ── Badge / pill ─────────────────────────────────────────────────── */
    .badge {
        display: inline-block;
        padding: 0.17rem 0.52rem;
        border-radius: 20px;
        font-size: 0.67rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        margin: 0.1rem;
        line-height: 1.6;
    }
    .badge-teal   { background: rgba(0,201,167,.14); color: var(--teal); }
    .badge-blue   { background: rgba(88,166,255,.14); color: var(--blue); }
    .badge-gold   { background: rgba(245,166,35,.14); color: var(--gold); }
    .badge-red    { background: rgba(255,107,107,.14); color: var(--red); }
    .badge-purple { background: rgba(167,139,250,.14); color: var(--purple); }

    /* ── Sample data notice banner ────────────────────────────────────── */
    .sample-banner {
        background: rgba(88,166,255,0.07);
        border: 1px solid rgba(88,166,255,0.25);
        border-radius: var(--radius-sm);
        padding: 0.6rem 1rem;
        font-size: 0.78rem;
        color: var(--blue);
        margin-bottom: 1.25rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ── Sidebar navigation radio ─────────────────────────────────────── */
    div[data-testid="stRadio"] > label { display: none; }
    div[data-testid="stRadio"] div[role="radiogroup"] {
        gap: 0.2rem;
        display: flex;
        flex-direction: column;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label {
        background: transparent;
        border: 1px solid transparent;
        border-radius: var(--radius-sm);
        padding: 0.5rem 0.8rem;
        cursor: pointer;
        font-size: 0.85rem;
        font-weight: 500;
        color: var(--text-lo);
        transition: all 0.14s ease;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
        background: rgba(0,201,167,0.07);
        color: var(--text-hi);
        border-color: var(--border);
    }

    /* ── File uploader ────────────────────────────────────────────────── */
    [data-testid="stFileUploader"] {
        border: 1px dashed var(--border);
        border-radius: var(--radius);
        padding: 0.4rem;
        background: var(--bg-card);
    }
    [data-testid="stFileUploader"]:hover {
        border-color: var(--teal);
    }

    /* ── Plotly wrapper ───────────────────────────────────────────────── */
    .js-plotly-plot { border-radius: var(--radius); }

    /* ── Streamlit dataframe ──────────────────────────────────────────── */
    .dataframe thead tr th {
        background: var(--bg-card) !important;
        color: var(--text-lo) !important;
        font-size: 0.69rem !important;
        text-transform: uppercase;
        letter-spacing: 0.07em;
    }

    /* ── Download / primary buttons ───────────────────────────────────── */
    div[data-testid="stDownloadButton"] button,
    div[data-testid="stButton"] button[kind="primary"] {
        background: var(--teal);
        color: #0D1117;
        border: none;
        font-weight: 600;
        font-size: 0.82rem;
        border-radius: var(--radius-sm);
    }
    div[data-testid="stDownloadButton"] button:hover {
        background: #00b394;
    }

    /* ── Metric widget overrides ──────────────────────────────────────── */
    [data-testid="stMetricValue"] {
        font-family: var(--font-mono) !important;
        color: var(--text-hi) !important;
    }
    [data-testid="stMetricLabel"] {
        color: var(--text-lo) !important;
        font-size: 0.72rem !important;
    }

    /* ── Alert / info ─────────────────────────────────────────────────── */
    div[data-testid="stAlert"] { border-radius: var(--radius); font-size: 0.82rem; }

    /* ── Scrollbar ────────────────────────────────────────────────────── */
    ::-webkit-scrollbar       { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: var(--bg-primary); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--teal); }

    /* ── Divider ──────────────────────────────────────────────────────── */
    hr { border-color: var(--border) !important; margin: 0.75rem 0; }

    /* ── Spinner ──────────────────────────────────────────────────────── */
    .stSpinner > div { border-top-color: var(--teal) !important; }
    </style>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  3. SESSION STATE
#  Declare every key once with a safe default.
#  Pattern: check-then-set so reruns never overwrite existing values.
# ═══════════════════════════════════════════════════════════════════════════════
def init_session_state() -> None:
    """
    Initialise all session-state keys. Called every rerun; is a no-op
    for keys already set (Streamlit preserves state across reruns).
    """
    defaults: dict = {
        # ── Data pipeline ─────────────────────────────────────────────────
        "raw_df":           None,   # Unprocessed DataFrame from the file
        "clean_df":         None,   # Output of clean_data() — the source of truth
        "filtered_df":      None,   # Output of apply_filters()
        "cleaning_report":  None,   # Stats dict from clean_data()
        "file_name":        None,   # Name of the currently loaded file (str)
        "using_sample":     False,  # True when no real file is uploaded
        "upload_error":     None,   # Error message string or None

        # ── User preferences ──────────────────────────────────────────────
        "currency":         "₹",
        "monthly_budget":   60000,  # sane default for Indian users (~₹60k/mo)

        # ── Navigation ────────────────────────────────────────────────────
        "active_page":      "Dashboard",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ═══════════════════════════════════════════════════════════════════════════════
#  4. DATA PIPELINE
#  Two entry points:
#    _load_sample()          — called on first run when no file is uploaded
#    _process_upload(file)   — called when a new file arrives from the uploader
#  Both write to the same session_state keys so the rest of the app is agnostic.
# ═══════════════════════════════════════════════════════════════════════════════

def _load_sample() -> None:
    """
    Load the bundled sample_expenses.csv through the same pipeline as a real
    upload, storing results in session_state. Sets using_sample = True so the
    UI can show an informational banner prompting the user to upload their data.
    """
    try:
        clean_df = load_sample_data()          # already cleaned inside data_cleaning.py
        st.session_state["clean_df"]        = clean_df
        st.session_state["raw_df"]          = clean_df   # sample has no separate raw
        st.session_state["cleaning_report"] = None       # no cleaning stats for sample
        st.session_state["file_name"]       = "sample_expenses.csv"
        st.session_state["using_sample"]    = True
        st.session_state["upload_error"]    = None
    except Exception as exc:
        st.session_state["upload_error"] = f"Failed to load sample data: {exc}"


def _process_upload(uploaded_file) -> None:
    """
    Run the full 3-step pipeline on a user-uploaded file and persist results.

    Steps:
        1. load_file()        — read CSV/Excel bytes → raw DataFrame
        2. validate_columns() — rename + check Date, Amount, Category exist
        3. clean_data()       — preprocess → clean DataFrame + cleaning report

    Any failure is stored in upload_error and the function returns early.
    The UI reads upload_error and shows _render_upload_error().
    """
    # Reset all data state before processing the new file
    for key in ("raw_df", "clean_df", "filtered_df", "cleaning_report", "upload_error"):
        st.session_state[key] = None
    st.session_state["using_sample"] = False

    # Step 1 — read bytes into DataFrame
    raw_df, err = load_file(uploaded_file)
    if err:
        st.session_state["upload_error"] = err
        return

    # Step 2 — validate + standardise column names
    validated_df, err = validate_columns(raw_df)
    if err:
        st.session_state["upload_error"] = err
        return

    # Step 3 — full cleaning pipeline
    try:
        clean_df, report = clean_data(validated_df)
    except Exception as exc:
        st.session_state["upload_error"] = f"Preprocessing failed unexpectedly: {exc}"
        return

    # Persist
    st.session_state["raw_df"]          = raw_df
    st.session_state["clean_df"]        = clean_df
    st.session_state["cleaning_report"] = report
    st.session_state["file_name"]       = uploaded_file.name
    st.session_state["upload_error"]    = None


def _run_data_pipeline(uploaded_file) -> None:
    """
    Decide whether to load sample data or process a real upload.

    Called once per rerun from main(). Logic:
        • No file uploaded + no data loaded yet  → load sample
        • New file uploaded (name differs)        → process the upload
        • Same file as before                     → do nothing (use cached state)
    """
    current_name = st.session_state.get("file_name")

    if uploaded_file is None:
        # No file uploaded by the user
        if st.session_state["clean_df"] is None:
            # First visit — auto-load sample so dashboard is never empty
            _load_sample()
        # else: sample or previously uploaded data already in state — keep it
        return

    # A file has been uploaded — only re-process if it's a new file
    if uploaded_file.name != current_name:
        with st.spinner("🔄  Reading and cleaning your data…"):
            _process_upload(uploaded_file)
        if st.session_state["upload_error"] is None:
            n = len(st.session_state["clean_df"])
            st.toast(f"✅ {uploaded_file.name} — {n:,} records ready.", icon="📂")
        else:
            st.toast("❌ Upload failed. See error details in the main panel.", icon="⚠️")


# ═══════════════════════════════════════════════════════════════════════════════
#  5. FILTER APPLICATION
#  Applies sidebar filter selections to clean_df every rerun.
#  Always re-filters from clean_df (never from a previous filtered_df)
#  so filter changes are independent and never cumulative.
# ═══════════════════════════════════════════════════════════════════════════════
def apply_filters(filters: dict) -> pd.DataFrame | None:
    """
    Slice the cleaned DataFrame by the sidebar filter selections.

    Args:
        filters : dict returned by render_sidebar() with keys:
                  "months"        (list[str]) — Month_Year values e.g. ["2024-01"]
                  "categories"    (list[str]) — Category names
                  "payment_modes" (list[str]) — Payment_Mode values  (optional key)

    Returns:
        Filtered DataFrame (may be empty), or None if no clean data is loaded.
    """
    df = st.session_state.get("clean_df")
    if df is None or df.empty:
        return None

    result = df.copy()

    # Month filter — Month_Year is "YYYY-MM" strings derived in clean_data()
    if "months" in filters and filters["months"]:
        result = result[result["Month_Year"].isin(filters["months"])]

    # Category filter
    if "categories" in filters and filters["categories"]:
        result = result[result["Category"].isin(filters["categories"])]

    # Payment mode filter — column always exists (clean_data() ensures it)
    if "payment_modes" in filters and filters["payment_modes"]:
        result = result[result["Payment_Mode"].isin(filters["payment_modes"])]

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  6. SIDEBAR
#  Renders the full sidebar and returns (uploaded_file, filters_dict).
#  Sections: brand → upload → status → nav → filters → settings → footer
# ═══════════════════════════════════════════════════════════════════════════════
def render_sidebar() -> tuple:
    """
    Render sidebar UI. Returns (uploaded_file, filters_dict).

    Filters section is only rendered when clean data is available, so new
    users see a clear call-to-action rather than empty widgets.
    """
    with st.sidebar:

        # ── Brand ──────────────────────────────────────────────────────────
        st.markdown("""
        <div style="padding:0.5rem 0 1.1rem;">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:1.08rem;
                        font-weight:600;color:#00C9A7;letter-spacing:-0.01em;">
                💳 SmartSpend<span style="color:#E6EDF3;">AI</span>
            </div>
            <div style="font-size:0.64rem;color:#484F58;margin-top:0.2rem;
                        letter-spacing:0.04em;">
                Expense &amp; Budget Analyzer · v3.0
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # ── File uploader ───────────────────────────────────────────────────
        _sb_label("📂 Data Source")
        uploaded_file = st.file_uploader(
            label="Upload expense file",
            type=["csv", "xlsx", "xls"],
            help="Required columns: Date, Amount, Category. Optional: Description, Payment_Mode.",
            label_visibility="collapsed",
        )

        # Status badge below the uploader
        file_name  = st.session_state.get("file_name")
        has_error  = bool(st.session_state.get("upload_error"))
        is_sample  = st.session_state.get("using_sample", False)

        if file_name and not has_error:
            badge_color  = "#8B949E" if is_sample else "#00C9A7"
            badge_prefix = "📋 Sample" if is_sample else "✓"
            st.markdown(
                f'<div style="font-size:0.7rem;color:{badge_color};'
                f'margin-top:0.3rem;">{badge_prefix} {file_name}</div>',
                unsafe_allow_html=True,
            )
        elif has_error:
            st.markdown(
                '<div style="font-size:0.7rem;color:#FF6B6B;margin-top:0.3rem;">'
                '❌ Upload failed — see main panel</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # ── Navigation ─────────────────────────────────────────────────────
        _sb_label("🗂 Navigation")
        nav_options = [
            "📊  Dashboard",
            "🔍  Data Preview",
            "🎯  Budget Tracker",
            "📈  Predictions",
            "🤖  AI Insights",
            "❤️  Health Score",
            "📄  PDF Report",
        ]
        selected = st.radio(
            "Navigation",
            options=nav_options,
            index=0,
            label_visibility="collapsed",
        )
        # Strip "📊  " prefix → clean page name stored in session_state
        st.session_state["active_page"] = selected.split("  ", 1)[-1].strip()

        st.markdown("---")

        # ── Filters (only shown when data is available) ─────────────────────
        filters: dict = {}
        df = st.session_state.get("clean_df")

        if df is not None and not df.empty:

            _sb_label("🔧 Filters")

            # Month — sorted "YYYY-MM" period strings
            all_months  = sorted(df["Month_Year"].unique().tolist())
            sel_months  = st.multiselect(
                "Month",
                options=all_months,
                default=all_months,
                placeholder="All months",
            )
            # Empty selection = user cleared it = treat as "all"
            filters["months"] = sel_months or all_months

            # Category
            all_cats = sorted(df["Category"].unique().tolist())
            sel_cats = st.multiselect(
                "Category",
                options=all_cats,
                default=all_cats,
                placeholder="All categories",
            )
            filters["categories"] = sel_cats or all_cats

            # Payment Mode — always present (clean_data() guarantees the column)
            all_modes = sorted(df["Payment_Mode"].dropna().unique().tolist())
            sel_modes = st.multiselect(
                "Payment Mode",
                options=all_modes,
                default=all_modes,
                placeholder="All modes",
            )
            filters["payment_modes"] = sel_modes or all_modes

            st.markdown("---")

            # ── Settings ────────────────────────────────────────────────────
            _sb_label("💰 Settings")

            st.session_state["currency"] = st.selectbox(
                "Currency Symbol",
                options=["₹", "$", "€", "£"],
                index=["₹", "$", "€", "£"].index(st.session_state["currency"]),
                label_visibility="visible",
            )
            st.session_state["monthly_budget"] = st.number_input(
                "Monthly Budget",
                min_value=100,
                max_value=10_000_000,
                value=int(st.session_state["monthly_budget"]),
                step=500,
                help="Used to calculate budget utilisation and estimated savings.",
            )

        else:
            # Data not loaded yet — show a helpful placeholder
            st.markdown("""
            <div style="background:#1C2333;border:1px dashed #30363D;border-radius:8px;
                        padding:1rem 0.9rem;text-align:center;color:#484F58;
                        font-size:0.77rem;line-height:1.75;">
                ⬆️ Upload your expense file above<br>
                to unlock filters and analysis.<br>
                <span style="color:#30363D;">——————————————</span><br>
                <span style="color:#8B949E;font-size:0.7rem;">
                    A sample dataset is loaded by default.
                </span>
            </div>
            """, unsafe_allow_html=True)

        # ── Footer ─────────────────────────────────────────────────────────
        st.markdown("""
        <div style="margin-top:2.5rem;font-size:0.6rem;color:#30363D;text-align:center;
                    letter-spacing:0.04em;">
            SmartSpend AI © 2025 · Built with Streamlit
        </div>
        """, unsafe_allow_html=True)

    return uploaded_file, filters


def _sb_label(text: str) -> None:
    """Render a small all-caps section label inside the sidebar."""
    st.markdown(
        f'<p style="font-size:0.66rem;font-weight:700;letter-spacing:0.12em;'
        f'text-transform:uppercase;color:#8B949E;margin:0 0 0.45rem;">{text}</p>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  7. PAGE RENDERERS
#  Each page is a self-contained function. Heavy logic lives in utils/.
#  Inline pages here are lightweight (budget tracker, stubs, error screens).
# ═══════════════════════════════════════════════════════════════════════════════

# ── Upload error ───────────────────────────────────────────────────────────────
def _render_upload_error() -> None:
    """Structured error card with actionable fix instructions."""
    error_msg = st.session_state.get("upload_error", "Unknown error")
    _page_title("Upload Failed", "Review the details below and re-upload your file.")
    st.markdown(f"""
    <div style="background:rgba(255,107,107,0.07);border:1px solid rgba(255,107,107,0.4);
                border-radius:10px;padding:1.25rem 1.5rem;margin-bottom:1rem;">
        <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.1em;
                    text-transform:uppercase;color:#FF6B6B;margin-bottom:0.6rem;">
            ❌ Error Details
        </div>
        <div style="font-size:0.84rem;color:#E6EDF3;line-height:1.8;">{error_msg}</div>
    </div>
    <div style="background:#1C2333;border:1px solid #30363D;border-radius:10px;
                padding:1.2rem 1.5rem;">
        <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.1em;
                    text-transform:uppercase;color:#8B949E;margin-bottom:0.75rem;">
            💡 How to Fix
        </div>
        <ul style="font-size:0.82rem;color:#8B949E;line-height:2.2;margin:0;
                   padding-left:1.2rem;">
            <li>Ensure columns named <b style="color:#E6EDF3;">Date</b>,
                <b style="color:#E6EDF3;">Amount</b>, and
                <b style="color:#E6EDF3;">Category</b> exist.</li>
            <li>Date formats accepted: <code>2024-01-15</code>, <code>15/01/2024</code>,
                <code>01/15/2024</code>, <code>15 Jan 2024</code>, and more.</li>
            <li>Amount must be numeric — remove currency symbols (₹ $ €) if present.</li>
            <li>File must be <code>.csv</code>, <code>.xlsx</code>, or <code>.xls</code>.</li>
            <li>File must not be empty (at least one data row required).</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# ── Budget Tracker ─────────────────────────────────────────────────────────────
def _render_budget_tracker(df: pd.DataFrame) -> None:
    """
    Budget Tracker page: period KPI summary + per-category spend bars.
    Full charted version ships in the next phase.
    """
    cur    = st.session_state["currency"]
    budget = float(st.session_state["monthly_budget"])
    months = max(int(df["Month_Year"].nunique()), 1)

    period_budget = budget * months
    total_spent   = float(df["Amount"].sum())
    remaining     = max(period_budget - total_spent, 0.0)
    pct_used      = min((total_spent / period_budget) * 100, 100.0) if period_budget > 0 else 0.0

    _page_title("Budget Tracker",
                f"Spending vs your {cur}{budget:,.0f}/month budget — {months} month(s) visible")

    # Summary row
    c1, c2, c3 = st.columns(3, gap="small")
    _kpi_card(c1, "Period Budget",    f"{cur}{period_budget:,.0f}",
              f"{months} month(s)", "#58A6FF", "💼")
    _kpi_card(c2, "Total Spent",      f"{cur}{total_spent:,.0f}",
              f"{pct_used:.1f}% of period budget", "#F5A623", "💸")
    _kpi_card(c3, "Est. Remaining",   f"{cur}{remaining:,.0f}",
              "budget left", "#00C9A7" if remaining > 0 else "#FF6B6B", "🏦")

    st.markdown("<div style='margin-top:0.75rem;'></div>", unsafe_allow_html=True)

    # Per-category progress bars
    st.markdown(
        '<div class="section-header"><h2>📊 Spend per Category</h2>'
        f'<p>Each bar = actual spend vs equal budget slice '
        f'({cur}{budget/max(df["Category"].nunique(),1):,.0f} each)</p></div>',
        unsafe_allow_html=True,
    )

    cat_totals = (
        df.groupby("Category")["Amount"].sum()
        .reset_index()
        .sort_values("Amount", ascending=False)
    )
    n_cats       = max(len(cat_totals), 1)
    slice_budget = budget / n_cats   # equal slice of one month's budget

    for _, row in cat_totals.iterrows():
        cat   = row["Category"]
        spent = float(row["Amount"])
        pct   = min((spent / slice_budget * 100), 100) if slice_budget > 0 else 0
        over  = spent > slice_budget
        color = "#FF6B6B" if over else ("#F5A623" if pct >= 75 else "#00C9A7")
        flag  = "⚠ Over budget" if over else f"{pct:.0f}%"

        st.markdown(f"""
        <div style="margin-bottom:0.9rem;">
            <div style="display:flex;justify-content:space-between;margin-bottom:0.25rem;">
                <span style="font-size:0.82rem;color:#E6EDF3;font-weight:500;">{cat}</span>
                <span style="font-size:0.75rem;color:{color};font-weight:600;">
                    {cur}{spent:,.0f}
                    <span style="color:#484F58;font-weight:400;">/ {cur}{slice_budget:,.0f}</span>
                    &nbsp;<span style="opacity:.75;">({flag})</span>
                </span>
            </div>
            <div style="background:#21262D;border-radius:3px;height:5px;">
                <div style="background:{color};width:{pct:.1f}%;
                            height:5px;border-radius:3px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Stub (pages not yet built) ─────────────────────────────────────────────────
def _render_stub(page_name: str) -> None:
    """Placeholder shown for pages not yet implemented."""
    _page_title(page_name, "Coming in the next phase.")
    st.markdown(f"""
    <div style="background:#1C2333;border:1px dashed #30363D;border-radius:10px;
                padding:3rem 2rem;text-align:center;margin-top:1rem;">
        <div style="font-size:2.5rem;margin-bottom:0.75rem;">🔧</div>
        <div style="font-size:0.88rem;color:#8B949E;line-height:1.8;">
            <b style="color:#E6EDF3;">{page_name}</b> is under construction.<br>
            Explore the <b style="color:#00C9A7;">Dashboard</b> and
            <b style="color:#00C9A7;">Budget Tracker</b> in the meantime.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Empty filter result ────────────────────────────────────────────────────────
def _render_no_filter_results() -> None:
    """Warning shown when active filters produce zero matching rows."""
    st.markdown("""
    <div style="background:rgba(245,166,35,0.07);border:1px solid rgba(245,166,35,0.3);
                border-radius:10px;padding:1.25rem 1.5rem;margin-top:1rem;">
        <div style="font-size:0.82rem;color:#F5A623;font-weight:600;
                    margin-bottom:0.4rem;">⚠ No Matching Transactions</div>
        <div style="font-size:0.8rem;color:#8B949E;line-height:1.7;">
            The current filters (Month, Category, or Payment Mode) returned
            zero rows. Adjust the filters in the sidebar to see data.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Sample data notice ─────────────────────────────────────────────────────────
def _render_sample_banner() -> None:
    """Informational banner shown when sample data is active."""
    st.markdown("""
    <div class="sample-banner">
        📋&nbsp; <b>Sample dataset loaded.</b>&nbsp;
        Upload your own CSV or Excel file from the sidebar to analyse your real expenses.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  SHARED UI HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _page_title(title: str, subtitle: str = "") -> None:
    """Standard page heading block used by every page renderer."""
    sub_html = (
        f'<p style="color:#8B949E;font-size:0.81rem;margin-top:0.3rem;">{subtitle}</p>'
        if subtitle else ""
    )
    st.markdown(
        f'<div style="margin-bottom:1.5rem;">'
        f'<h1 style="font-size:1.35rem;font-weight:700;color:#E6EDF3;'
        f'margin:0;letter-spacing:-0.01em;">{title}</h1>'
        f'{sub_html}</div>',
        unsafe_allow_html=True,
    )


def _kpi_card(col, label: str, value: str,
              subtitle: str = "", accent: str = "#00C9A7", icon: str = "") -> None:
    """Render one KPI card into a Streamlit column object."""
    with col:
        st.markdown(
            f'<div class="kpi-card" style="border-left:3px solid {accent};">'
            f'  <div class="kpi-label">{icon}&nbsp;{label}</div>'
            f'  <div class="kpi-value">{value}</div>'
            f'  <div class="kpi-sub">{subtitle}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  8. PAGE ROUTER
#  Maps page name → renderer. Applies three guards in order before routing:
#    Guard A — upload error present          → show error screen
#    Guard B — no data in session_state      → should not happen (sample fallback)
#    Guard C — filters returned 0 rows       → show "no results" warning
# ═══════════════════════════════════════════════════════════════════════════════
def route_page(page: str, filtered_df: pd.DataFrame | None) -> None:
    """
    Route to the correct page renderer.

    Args:
        page        : Clean page name from session_state["active_page"]
        filtered_df : DataFrame after sidebar filters applied, or None
    """
    cur    = st.session_state["currency"]
    budget = float(st.session_state["monthly_budget"])

    # Guard A — pipeline failed
    if st.session_state.get("upload_error"):
        _render_upload_error()
        return

    # Guard B — no data (sample load also failed, rare)
    if st.session_state.get("clean_df") is None:
        st.info("⏳ Loading data… If this persists, try uploading a file manually.")
        return

    # Guard C — filters eliminated all rows
    if filtered_df is not None and len(filtered_df) == 0:
        _render_no_filter_results()
        return

    # Sample data banner — shown on every page when sample is active
    if st.session_state.get("using_sample"):
        _render_sample_banner()

    # ── Route ──────────────────────────────────────────────────────────────
    if page == "Dashboard":
        render_dashboard(filtered_df, budget, cur)

    elif page == "Data Preview":
        _page_title("Data Preview", "Cleaned and validated expense records.")
        render_data_preview(
            filtered_df,
            st.session_state.get("cleaning_report"),
            cur,
        )

    elif page == "Budget Tracker":
        _render_budget_tracker(filtered_df)

    elif page == "Predictions":
        # Use clean_df (full history) for predictions — NOT filtered_df.
        # Filters like "only show March" would give the model too few months.
        # Predictions need the complete chronological record to fit the trend.
        prediction_df = st.session_state.get("clean_df")
        render_predictions_page(prediction_df, budget, cur)

    # elif page == "Predictions":
    #     _render_stub("Expense Predictions")
    elif page == "AI Insights":

        st.title("🧠 AI Financial Insights")

        insights = generate_insights(filtered_df)

        for insight in insights:
            st.info(insight)

    # elif page == "AI Insights":
    #     _render_stub("AI Insights")
    elif page == "Health Score":

        st.title("❤️ Financial Health Score")

        score, status = calculate_health_score(filtered_df)

        st.metric(
            label="Financial Score",
            value=f"{score}/100"
        )

        st.success(f"Financial Status: {status}")

        if score >= 80:
            st.info("Excellent financial management.")

        elif score >= 60:
            st.warning("Your finances are stable but can improve.")

        else:
            st.error("High spending detected. Consider reducing expenses.")

    # elif page == "Health Score":
    #     _render_stub("Financial Health Score")

    elif page == "PDF Report":

        st.title("📄 PDF Financial Report")

        score, status = calculate_health_score(filtered_df)

        insights = generate_insights(filtered_df)

        if st.button("Generate PDF Report"):

            generate_pdf_report(
                "SmartSpend_Report.pdf",
                score,
                status,
                insights
            )

            with open("SmartSpend_Report.pdf", "rb") as f:

                st.download_button(
                    label="Download Report",
                    data=f,
                    file_name="SmartSpend_Report.pdf",
                    mime="application/pdf"
                )
    # elif page == "PDF Report":
    #     _render_stub("PDF Report")


# ═══════════════════════════════════════════════════════════════════════════════
#  9. MAIN — App Orchestrator
#
#  Streamlit re-executes main() top-to-bottom on every user interaction.
#  Order matters: CSS → state → sidebar → pipeline → filters → route.
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    """
    Primary entry point. Wires all phases together on every Streamlit rerun.

    Phase order:
        1. inject_global_css()   — apply styles before anything renders
        2. init_session_state()  — declare keys with defaults (no-op after first run)
        3. render_sidebar()      — draw sidebar, capture uploaded_file + filters
        4. _run_data_pipeline()  — load sample or process upload (skips if unchanged)
        5. apply_filters()       — filter clean_df → filtered_df
        6. route_page()          — render the selected page with filtered data
    """
    # 1. Styles
    inject_global_css()

    # 2. State (safe no-op after the first run)
    init_session_state()

    # 3. Sidebar — must come before pipeline so upload widget exists
    uploaded_file, filters = render_sidebar()

    # 4. Data pipeline — loads sample on first visit; processes new uploads
    _run_data_pipeline(uploaded_file)

    # 5. Apply sidebar filters to the clean DataFrame
    filtered_df = apply_filters(filters)
    if filtered_df is not None:
        st.session_state["filtered_df"] = filtered_df

    # 6. Route to the active page
    route_page(st.session_state["active_page"], filtered_df)


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# # ═══════════════════════════════════════════════════════════════════════════════
# #  SmartSpend AI – Expense & Budget Analyzer
# #  app.py  |  Main Entry Point  |  Version 2.0
# #
# #  What this file does (and ONLY this file does):
# #    1. Streamlit page config                        → set_page_config()
# #    2. Global CSS injection                         → inject_global_css()
# #    3. Session-state initialisation                 → init_session_state()
# #    4. Sidebar: upload + filters + nav              → render_sidebar()
# #    5. Data pipeline: load → validate → clean       → handle_file_upload()
# #    6. Filter application                           → apply_filters()
# #    7. Page routing to the right renderer           → route_page()
# #
# #  All heavy logic lives in utils/:
# #    utils/data_cleaning.py  — load_file, validate_columns, clean_data,
# #                               render_data_preview
# #    utils/analysis.py       — compute_kpis, render_dashboard
# #
# #  To add a new page: import its renderer here and add one elif to route_page().
# # ═══════════════════════════════════════════════════════════════════════════════

# import streamlit as st
# import pandas as pd

# # ── Utility module imports ────────────────────────────────────────────────────
# from utils.data_cleaning import (
#     load_file,           # reads CSV / Excel → raw DataFrame
#     validate_columns,    # renames & checks required cols
#     clean_data,          # full preprocessing pipeline
#     render_data_preview, # Streamlit UI: cleaned table + cleaning report
# )
# from utils.analysis import (
#     render_dashboard,    # full dashboard: KPIs + budget bar + 3 charts + table
# )


# # ═══════════════════════════════════════════════════════════════════════════════
# #  1. PAGE CONFIG
# #  Must be the FIRST Streamlit call — before any other st.* usage.
# # ═══════════════════════════════════════════════════════════════════════════════
# st.set_page_config(
#     page_title="SmartSpend AI",
#     page_icon="💳",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )


# # ═══════════════════════════════════════════════════════════════════════════════
# #  2. GLOBAL CSS
# #  Injected once per page load. Covers: fonts, KPI cards, badges, section
# #  headers, nav pills, table headers, scrollbar, upload area, and dividers.
# #  All colours reference the CSS variables defined in :root so one edit
# #  rethemes the entire app.
# # ═══════════════════════════════════════════════════════════════════════════════
# def inject_global_css() -> None:
#     """Inject fintech-grade CSS on top of the Streamlit dark theme."""
#     st.markdown(
#         """
#         <style>
#         /* ── Google Fonts ──────────────────────────────────────────── */
#         @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@300;400;500;600;700&display=swap');

#         /* ── Design tokens ─────────────────────────────────────────── */
#         :root {
#             --bg-primary:    #0D1117;
#             --bg-secondary:  #161B22;
#             --bg-card:       #1C2333;
#             --border:        #30363D;
#             --teal:          #00C9A7;
#             --gold:          #F5A623;
#             --red:           #FF6B6B;
#             --blue:          #58A6FF;
#             --text-hi:       #E6EDF3;
#             --text-lo:       #8B949E;
#             --text-muted:    #484F58;
#             --mono:          'IBM Plex Mono', monospace;
#             --sans:          'Inter', sans-serif;
#             --radius:        10px;
#             --shadow:        0 4px 24px rgba(0,0,0,0.45);
#         }

#         /* ── Base ───────────────────────────────────────────────────── */
#         html, body, [class*="css"] { font-family: var(--sans); color: var(--text-hi); }
#         .block-container { padding-top: 1.5rem; padding-bottom: 2.5rem; }

#         /* ── Sidebar ────────────────────────────────────────────────── */
#         section[data-testid="stSidebar"] {
#             background: var(--bg-secondary);
#             border-right: 1px solid var(--border);
#         }
#         section[data-testid="stSidebar"] label {
#             color: var(--text-lo) !important;
#             font-size: 0.72rem !important;
#             font-weight: 600 !important;
#             letter-spacing: 0.09em !important;
#             text-transform: uppercase !important;
#         }

#         /* ── KPI card ───────────────────────────────────────────────── */
#         .kpi-card {
#             background: var(--bg-card);
#             border: 1px solid var(--border);
#             border-radius: var(--radius);
#             padding: 1.2rem 1.4rem;
#             box-shadow: var(--shadow);
#             transition: border-color 0.2s;
#             height: 100%;
#         }
#         .kpi-card:hover { border-color: var(--teal); }
#         .kpi-label {
#             font-size: 0.68rem;
#             font-weight: 700;
#             letter-spacing: 0.12em;
#             text-transform: uppercase;
#             color: var(--text-lo);
#             margin-bottom: 0.45rem;
#         }
#         .kpi-value {
#             font-family: var(--mono);
#             font-size: 1.65rem;
#             font-weight: 600;
#             color: var(--text-hi);
#             line-height: 1.1;
#             word-break: break-word;
#         }
#         .kpi-sub {
#             font-size: 0.7rem;
#             color: var(--text-muted);
#             margin-top: 0.45rem;
#         }

#         /* ── Section header ─────────────────────────────────────────── */
#         .section-header {
#             border-left: 3px solid var(--teal);
#             padding-left: 0.75rem;
#             margin: 1.75rem 0 1rem;
#         }
#         .section-header h2 {
#             font-size: 0.95rem;
#             font-weight: 600;
#             color: var(--text-hi);
#             margin: 0;
#             letter-spacing: 0.01em;
#         }
#         .section-header p {
#             font-size: 0.78rem;
#             color: var(--text-lo);
#             margin: 0.15rem 0 0;
#         }

#         /* ── Badge / pill ───────────────────────────────────────────── */
#         .badge {
#             display: inline-block;
#             padding: 0.18rem 0.55rem;
#             border-radius: 20px;
#             font-size: 0.68rem;
#             font-weight: 600;
#             letter-spacing: 0.04em;
#             margin: 0.15rem 0.1rem;
#         }
#         .badge-teal { background: rgba(0,201,167,.15); color: var(--teal); }
#         .badge-gold { background: rgba(245,166,35,.15); color: var(--gold); }
#         .badge-red  { background: rgba(255,107,107,.15); color: var(--red); }
#         .badge-blue { background: rgba(88,166,255,.15);  color: var(--blue); }

#         /* ── Sidebar navigation radio ────────────────────────────────── */
#         div[data-testid="stRadio"] > label { display: none; }
#         div[data-testid="stRadio"] div[role="radiogroup"] {
#             gap: 0.25rem; display: flex; flex-direction: column;
#         }
#         div[data-testid="stRadio"] div[role="radiogroup"] label {
#             background: transparent;
#             border: 1px solid transparent;
#             border-radius: 6px;
#             padding: 0.52rem 0.8rem;
#             cursor: pointer;
#             font-size: 0.86rem;
#             font-weight: 500;
#             color: var(--text-lo);
#             transition: all 0.15s;
#         }
#         div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
#             background: rgba(0,201,167,.07);
#             color: var(--text-hi);
#             border-color: var(--border);
#         }

#         /* ── Upload drop-zone ───────────────────────────────────────── */
#         [data-testid="stFileUploader"] {
#             border: 1px dashed var(--border);
#             border-radius: var(--radius);
#             padding: 0.5rem;
#             background: var(--bg-card);
#         }

#         /* ── Plotly chart wrapper ────────────────────────────────────── */
#         .js-plotly-plot { border-radius: var(--radius); }

#         /* ── Dataframe header ────────────────────────────────────────── */
#         .dataframe thead tr th {
#             background: var(--bg-card) !important;
#             color: var(--text-lo) !important;
#             font-size: 0.7rem;
#             text-transform: uppercase;
#             letter-spacing: 0.07em;
#         }

#         /* ── Scrollbar ──────────────────────────────────────────────── */
#         ::-webkit-scrollbar       { width: 5px; height: 5px; }
#         ::-webkit-scrollbar-track { background: var(--bg-primary); }
#         ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
#         ::-webkit-scrollbar-thumb:hover { background: var(--teal); }

#         /* ── Divider ────────────────────────────────────────────────── */
#         hr { border-color: var(--border) !important; margin: 0.8rem 0; }

#         /* ── Alert / info overrides ─────────────────────────────────── */
#         div[data-testid="stAlert"] {
#             border-radius: var(--radius);
#             font-size: 0.83rem;
#         }
#         </style>
#         """,
#         unsafe_allow_html=True,
#     )


# # ═══════════════════════════════════════════════════════════════════════════════
# #  3. SESSION STATE
# #  All keys are declared here with safe defaults.
# #  Every module reads/writes via st.session_state["key"].
# #  Declaring here prevents KeyError on first load.
# # ═══════════════════════════════════════════════════════════════════════════════
# def init_session_state() -> None:
#     """Initialise all session-state keys with defaults (no-op if already set)."""
#     defaults = {
#         # ── Data pipeline ─────────────────────────────────────────────
#         "raw_df":          None,   # Original DataFrame straight from the file
#         "clean_df":        None,   # After clean_data() — the source of truth
#         "filtered_df":     None,   # After sidebar filters are applied
#         "cleaning_report": None,   # Dict returned by clean_data() for display
#         "file_name":       None,   # Uploaded filename (string, for display)
#         "upload_error":    None,   # Last upload error message or None

#         # ── User preferences ──────────────────────────────────────────
#         "currency":        "₹",
#         "monthly_budget":  5000,

#         # ── Navigation ────────────────────────────────────────────────
#         "active_page": "Dashboard",
#     }
#     for key, val in defaults.items():
#         if key not in st.session_state:
#             st.session_state[key] = val


# # ═══════════════════════════════════════════════════════════════════════════════
# #  4. FILE UPLOAD HANDLER
# #  Runs whenever a new file object appears from the sidebar uploader.
# #  Pipeline: load → validate columns → clean → store in session_state.
# #  Errors are stored in session_state["upload_error"] for the UI to display.
# # ═══════════════════════════════════════════════════════════════════════════════
# def handle_file_upload(uploaded_file) -> None:
#     """
#     Full data ingestion pipeline triggered on new file upload.

#     Steps:
#         1. load_file()        → raw DataFrame or error string
#         2. validate_columns() → rename + check required cols or error string
#         3. clean_data()       → preprocessed DataFrame + cleaning report dict
#         4. Persist results to session_state

#     On any failure the error message is stored and the function returns early —
#     the UI reads upload_error and shows _render_upload_error().

#     Args:
#         uploaded_file : Streamlit UploadedFile object from st.file_uploader()
#     """
#     # Clear stale state from any previous upload
#     for key in ("raw_df", "clean_df", "filtered_df", "cleaning_report", "upload_error"):
#         st.session_state[key] = None

#     # ── Step 1: Read file into DataFrame ─────────────────────────────────
#     raw_df, load_err = load_file(uploaded_file)
#     if load_err:
#         st.session_state["upload_error"] = load_err
#         return

#     # ── Step 2: Rename & validate required columns ────────────────────────
#     validated_df, val_err = validate_columns(raw_df)
#     if val_err:
#         st.session_state["upload_error"] = val_err
#         return

#     # ── Step 3: Full preprocessing ────────────────────────────────────────
#     try:
#         clean_df, report = clean_data(validated_df)
#     except Exception as exc:
#         st.session_state["upload_error"] = f"Preprocessing failed: {exc}"
#         return

#     # ── Step 4: Persist ───────────────────────────────────────────────────
#     st.session_state["raw_df"]          = raw_df
#     st.session_state["clean_df"]        = clean_df
#     st.session_state["cleaning_report"] = report
#     st.session_state["file_name"]       = uploaded_file.name


# # ═══════════════════════════════════════════════════════════════════════════════
# #  5. SIDEBAR
# #  Renders the complete left panel and returns:
# #    • uploaded_file — raw Streamlit UploadedFile object (or None)
# #    • filters       — dict of active filter selections
# #
# #  Layout sections (top → bottom):
# #    Brand header → File uploader → Navigation → Filters → Settings → Footer
# # ═══════════════════════════════════════════════════════════════════════════════
# def render_sidebar():
#     """
#     Draw the sidebar and return (uploaded_file, filters_dict).

#     The Filters section only renders after clean data exists in session_state,
#     so first-time users see a helpful placeholder prompt instead of empty widgets.
#     """
#     with st.sidebar:

#         # ── Brand header ──────────────────────────────────────────────────
#         st.markdown(
#             """
#             <div style="padding:0.4rem 0 1.2rem;">
#                 <div style="font-family:'IBM Plex Mono',monospace;
#                             font-size:1.1rem;font-weight:600;
#                             color:#00C9A7;letter-spacing:-0.01em;">
#                     💳 SmartSpend<span style="color:#E6EDF3;">AI</span>
#                 </div>
#                 <div style="font-size:0.67rem;color:#484F58;margin-top:0.2rem;">
#                     Expense &amp; Budget Analyzer · v2.0
#                 </div>
#             </div>
#             """,
#             unsafe_allow_html=True,
#         )
#         st.markdown("---")

#         # ── File uploader ─────────────────────────────────────────────────
#         _sidebar_label("📂 Data Source")
#         uploaded_file = st.file_uploader(
#             label="Upload expense file",
#             type=["csv", "xlsx", "xls"],
#             help=(
#                 "Supported: CSV, Excel (.xlsx / .xls). "
#                 "Required columns: Date, Amount, Category."
#             ),
#             label_visibility="collapsed",
#         )

#         # Confirm badge: shows filename when data is loaded successfully
#         if st.session_state["file_name"] and st.session_state["upload_error"] is None:
#             st.markdown(
#                 f'<div style="font-size:0.72rem;color:#00C9A7;margin-top:0.35rem;">'
#                 f'✓ {st.session_state["file_name"]}</div>',
#                 unsafe_allow_html=True,
#             )

#         st.markdown("---")

#         # ── Navigation ────────────────────────────────────────────────────
#         _sidebar_label("🗂 Navigation")
#         pages = [
#             "📊  Dashboard",
#             "🔍  Data Preview",
#             "🎯  Budget Tracker",
#             "📈  Predictions",
#             "🤖  AI Insights",
#             "❤️  Health Score",
#             "📄  PDF Report",
#         ]
#         selected = st.radio(
#             "Navigation",
#             options=pages,
#             index=0,
#             label_visibility="collapsed",
#         )
#         # Store clean name (strip emoji prefix "📊  ")
#         st.session_state["active_page"] = selected.split("  ", 1)[-1].strip()

#         st.markdown("---")

#         # ── Filters — only appear after data is loaded ────────────────────
#         filters = {}
#         df = st.session_state.get("clean_df")

#         if df is not None and not df.empty:

#             _sidebar_label("🔧 Filters")

#             # Month filter — multi-select over "YYYY-MM" period strings
#             all_months = sorted(df["Month_Year"].unique().tolist())
#             sel_months = st.multiselect(
#                 "Month",
#                 options=all_months,
#                 default=all_months,
#                 placeholder="All months",
#             )
#             # Empty selection → treat as "all selected"
#             filters["months"] = sel_months if sel_months else all_months

#             # Category filter
#             all_cats = sorted(df["Category"].unique().tolist())
#             sel_cats = st.multiselect(
#                 "Category",
#                 options=all_cats,
#                 default=all_cats,
#                 placeholder="All categories",
#             )
#             filters["categories"] = sel_cats if sel_cats else all_cats

#             # Payment mode filter — only shown if column exists in the file
#             if "Payment_Mode" in df.columns:
#                 all_modes = sorted(df["Payment_Mode"].dropna().unique().tolist())
#                 sel_modes = st.multiselect(
#                     "Payment Mode",
#                     options=all_modes,
#                     default=all_modes,
#                     placeholder="All modes",
#                 )
#                 filters["payment_modes"] = sel_modes if sel_modes else all_modes

#             st.markdown("---")

#             # ── Budget & currency settings ─────────────────────────────────
#             _sidebar_label("💰 Settings")

#             st.session_state["currency"] = st.selectbox(
#                 "Currency",
#                 options=["₹", "$", "€", "£"],
#                 index=0,
#             )
#             st.session_state["monthly_budget"] = st.number_input(
#                 "Monthly Budget",
#                 min_value=100,
#                 max_value=10_000_000,
#                 value=int(st.session_state["monthly_budget"]),
#                 step=500,
#             )

#         else:
#             # No data yet — friendly placeholder
#             st.markdown(
#                 """
#                 <div style="background:#1C2333;border:1px dashed #30363D;
#                             border-radius:8px;padding:1rem;text-align:center;
#                             color:#484F58;font-size:0.78rem;line-height:1.7;">
#                     ⬆️ Upload an expense file above to unlock filters and analysis.
#                 </div>
#                 """,
#                 unsafe_allow_html=True,
#             )

#         # ── Sidebar footer ─────────────────────────────────────────────────
#         st.markdown(
#             """
#             <div style="margin-top:2rem;font-size:0.62rem;
#                         color:#30363D;text-align:center;">
#                 SmartSpend AI © 2025 · Built with Streamlit
#             </div>
#             """,
#             unsafe_allow_html=True,
#         )

#     return uploaded_file, filters


# def _sidebar_label(text: str) -> None:
#     """Render a small uppercase section heading in the sidebar."""
#     st.markdown(
#         f'<p style="font-size:0.68rem;font-weight:700;letter-spacing:0.1em;'
#         f'text-transform:uppercase;color:#8B949E;margin:0 0 0.4rem;">{text}</p>',
#         unsafe_allow_html=True,
#     )


# # ═══════════════════════════════════════════════════════════════════════════════
# #  6. FILTER APPLICATION
# #  Applies the sidebar filter dict to clean_df every rerun.
# #  Always re-applies to clean_df (not filtered_df) so changes are
# #  independent and non-cumulative.
# # ═══════════════════════════════════════════════════════════════════════════════
# def apply_filters(filters: dict):
#     """
#     Filter the cleaned DataFrame according to sidebar selections.

#     Args:
#         filters : dict with keys "months", "categories", "payment_modes"

#     Returns:
#         Filtered pandas DataFrame, or None if no clean data is loaded.
#         Returns an empty DataFrame (not None) when filters produce zero rows
#         so the router can display a "no results" warning instead of landing.
#     """
#     df = st.session_state.get("clean_df")
#     if df is None:
#         return None

#     df = df.copy()   # always work on a copy — never mutate clean_df

#     # Apply each filter only if the key is present in the filters dict
#     if "months" in filters:
#         df = df[df["Month_Year"].isin(filters["months"])]

#     if "categories" in filters:
#         df = df[df["Category"].isin(filters["categories"])]

#     if "payment_modes" in filters and "Payment_Mode" in df.columns:
#         df = df[df["Payment_Mode"].isin(filters["payment_modes"])]

#     # Return empty DataFrame instead of None so the router distinguishes
#     # "no data loaded" from "filters matched nothing"
#     return df


# # ═══════════════════════════════════════════════════════════════════════════════
# #  7. PAGE RENDERERS (inline, minimal — heavy logic lives in utils/)
# # ═══════════════════════════════════════════════════════════════════════════════

# def _render_landing() -> None:
#     """Full-page welcome screen shown before any file is uploaded."""
#     st.markdown(
#         """
#         <div style="display:flex;flex-direction:column;align-items:center;
#                     justify-content:center;min-height:62vh;text-align:center;
#                     padding:2rem;">
#             <div style="font-size:3.5rem;margin-bottom:1.25rem;">💳</div>
#             <h1 style="color:#E6EDF3;font-size:1.55rem;font-weight:700;
#                        margin:0;letter-spacing:-0.02em;">
#                 Welcome to SmartSpend AI
#             </h1>
#             <p style="color:#8B949E;font-size:0.88rem;margin:0.75rem 0 1.75rem;
#                       max-width:440px;line-height:1.75;">
#                 Upload your expense CSV or Excel file from the sidebar to get
#                 instant KPI insights, interactive Plotly charts, budget tracking,
#                 and AI-powered spending analysis.
#             </p>
#             <div style="display:flex;gap:0.6rem;flex-wrap:wrap;
#                         justify-content:center;margin-bottom:2rem;">
#                 <span class="badge badge-teal">📊 Dashboard</span>
#                 <span class="badge badge-blue">📈 Predictions</span>
#                 <span class="badge badge-gold">🤖 AI Insights</span>
#                 <span class="badge badge-red">❤️ Health Score</span>
#                 <span class="badge badge-teal">📄 PDF Report</span>
#             </div>
#             <div style="background:#1C2333;border:1px solid #30363D;
#                         border-radius:10px;padding:1.2rem 1.75rem;
#                         text-align:left;max-width:420px;width:100%;">
#                 <div style="font-size:0.7rem;font-weight:700;letter-spacing:0.1em;
#                             text-transform:uppercase;color:#8B949E;margin-bottom:0.7rem;">
#                     Required CSV columns
#                 </div>
#                 <div style="line-height:2.2;">
#                     <span class="badge badge-teal">Date</span>
#                     <span class="badge badge-blue">Amount</span>
#                     <span class="badge badge-gold">Category</span>
#                 </div>
#                 <div style="font-size:0.72rem;color:#484F58;margin-top:0.5rem;">
#                     Optional: Description &nbsp;·&nbsp; Payment_Mode
#                 </div>
#             </div>
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )


# def _render_upload_error(error_msg: str) -> None:
#     """Structured error card shown when the upload/validation pipeline fails."""
#     _page_header("Upload Failed", "Review the issue below and re-upload your file.")
#     st.markdown(
#         f"""
#         <div style="background:rgba(255,107,107,0.07);border:1px solid #FF6B6B;
#                     border-radius:10px;padding:1.25rem 1.5rem;margin-top:0.5rem;">
#             <div style="font-size:0.7rem;font-weight:700;letter-spacing:0.1em;
#                         text-transform:uppercase;color:#FF6B6B;margin-bottom:0.5rem;">
#                 ❌ Error Details
#             </div>
#             <div style="font-size:0.84rem;color:#E6EDF3;line-height:1.75;">
#                 {error_msg}
#             </div>
#         </div>
#         <div style="margin-top:1rem;background:#1C2333;border:1px solid #30363D;
#                     border-radius:10px;padding:1.2rem 1.5rem;">
#             <div style="font-size:0.7rem;font-weight:700;letter-spacing:0.1em;
#                         text-transform:uppercase;color:#8B949E;margin-bottom:0.7rem;">
#                 💡 Common Fixes
#             </div>
#             <ul style="font-size:0.82rem;color:#8B949E;line-height:2.1;
#                        margin:0;padding-left:1.2rem;">
#                 <li>Ensure columns named <b style="color:#E6EDF3;">Date</b>,
#                     <b style="color:#E6EDF3;">Amount</b>, and
#                     <b style="color:#E6EDF3;">Category</b> exist.</li>
#                 <li>Dates: use <code>2024-01-15</code> or <code>15/01/2024</code>.</li>
#                 <li>Amount: numeric only — remove currency symbols if needed.</li>
#                 <li>File must be <code>.csv</code>, <code>.xlsx</code>,
#                     or <code>.xls</code>.</li>
#             </ul>
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )


# def _render_empty_filter() -> None:
#     """Warning shown when active filters produce zero matching rows."""
#     st.warning(
#         "⚠️ No transactions match the current filters. "
#         "Adjust Month, Category, or Payment Mode in the sidebar.",
#     )


# def _render_budget_tracker(df: pd.DataFrame) -> None:
#     """
#     Budget Tracker page — period summary cards + per-category progress bars.
#     Full dedicated module (with charts) ships in the next phase.
#     """
#     _page_header(
#         "Budget Tracker",
#         "Monthly spending vs your set budget, broken down by category.",
#     )

#     budget = st.session_state["monthly_budget"]
#     cur    = st.session_state["currency"]
#     months = max(df["Month_Year"].nunique(), 1)

#     period_budget = budget * months
#     total_spent   = df["Amount"].sum()
#     remaining     = max(period_budget - total_spent, 0)
#     pct_used      = min((total_spent / period_budget) * 100, 100) if period_budget else 0

#     # ── Summary KPI row ───────────────────────────────────────────────────
#     c1, c2, c3 = st.columns(3)
#     _kpi_html(c1, "Period Budget",  f"{cur}{period_budget:,.0f}",
#               f"over {months} month(s)", "#58A6FF", "💼")
#     _kpi_html(c2, "Total Spent",    f"{cur}{total_spent:,.0f}",
#               f"{pct_used:.1f}% utilised", "#F5A623", "💸")
#     _kpi_html(c3, "Est. Remaining", f"{cur}{remaining:,.0f}",
#               "budget left in period",
#               "#00C9A7" if remaining > 0 else "#FF6B6B", "🏦")

#     st.markdown("<br>", unsafe_allow_html=True)

#     # ── Category progress bars ─────────────────────────────────────────────
#     st.markdown(
#         '<div class="section-header"><h2>📊 Per-Category Spend</h2>'
#         '<p>Actual spend vs equal-share budget slice per category</p></div>',
#         unsafe_allow_html=True,
#     )

#     cat_totals = (
#         df.groupby("Category")["Amount"]
#         .sum()
#         .reset_index()
#         .sort_values("Amount", ascending=False)
#     )
#     # Each category gets an equal share of one month's budget
#     slice_budget = budget / max(len(cat_totals), 1)

#     for _, row in cat_totals.iterrows():
#         cat   = row["Category"]
#         spent = row["Amount"]
#         pct   = min((spent / slice_budget) * 100, 100) if slice_budget else 0
#         over  = spent > slice_budget
#         color = "#FF6B6B" if over else ("#F5A623" if pct >= 75 else "#00C9A7")
#         flag  = "⚠ Over" if over else f"{pct:.0f}%"

#         st.markdown(
#             f"""
#             <div style="margin-bottom:0.95rem;">
#                 <div style="display:flex;justify-content:space-between;
#                             margin-bottom:0.28rem;">
#                     <span style="font-size:0.82rem;color:#E6EDF3;">{cat}</span>
#                     <span style="font-size:0.76rem;color:{color};font-weight:600;">
#                         {cur}{spent:,.0f}
#                         <span style="color:#484F58;font-weight:400;">
#                             / {cur}{slice_budget:,.0f}
#                         </span>
#                         &nbsp;({flag})
#                     </span>
#                 </div>
#                 <div style="background:#21262D;border-radius:3px;height:5px;">
#                     <div style="background:{color};width:{pct:.1f}%;
#                                 height:5px;border-radius:3px;"></div>
#                 </div>
#             </div>
#             """,
#             unsafe_allow_html=True,
#         )


# def _render_stub(page_name: str) -> None:
#     """Temporary placeholder for pages not yet implemented."""
#     _page_header(page_name, "This module is coming in the next phase.")
#     st.markdown(
#         f"""
#         <div style="background:#1C2333;border:1px dashed #30363D;
#                     border-radius:10px;padding:2.5rem;text-align:center;
#                     margin-top:1rem;">
#             <div style="font-size:2rem;margin-bottom:0.75rem;">🔧</div>
#             <div style="font-size:0.88rem;color:#8B949E;">
#                 <b style="color:#E6EDF3;">{page_name}</b> is being built.<br>
#                 Explore the <b style="color:#00C9A7;">Dashboard</b>
#                 and <b style="color:#00C9A7;">Budget Tracker</b> in the meantime.
#             </div>
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )


# # ═══════════════════════════════════════════════════════════════════════════════
# #  SHARED UI HELPERS
# # ═══════════════════════════════════════════════════════════════════════════════

# def _page_header(title: str, subtitle: str = "") -> None:
#     """Standard page title + optional subtitle used by every page."""
#     sub = (
#         f'<p style="color:#8B949E;font-size:0.82rem;margin-top:0.3rem;">'
#         f'{subtitle}</p>'
#         if subtitle else ""
#     )
#     st.markdown(
#         f'<div style="margin-bottom:1.5rem;">'
#         f'<h1 style="font-size:1.35rem;font-weight:700;color:#E6EDF3;margin:0;">'
#         f'{title}</h1>{sub}</div>',
#         unsafe_allow_html=True,
#     )


# def _kpi_html(col, label: str, value: str,
#               subtitle: str = "", accent: str = "#00C9A7", icon: str = "") -> None:
#     """Render a KPI card into a Streamlit column object."""
#     with col:
#         st.markdown(
#             f"""
#             <div class="kpi-card" style="border-left:3px solid {accent};">
#                 <div class="kpi-label">{icon} {label}</div>
#                 <div class="kpi-value">{value}</div>
#                 <div class="kpi-sub">{subtitle}</div>
#             </div>
#             """,
#             unsafe_allow_html=True,
#         )


# # ═══════════════════════════════════════════════════════════════════════════════
# #  8. PAGE ROUTER
# #  Single switch that maps page name → renderer function.
# #
# #  Guard order (checked before routing):
# #    1. No data loaded at all  → landing or error screen
# #    2. Filters gave 0 rows    → "no results" warning
# #    3. Otherwise              → route to the correct page renderer
# # ═══════════════════════════════════════════════════════════════════════════════
# def route_page(page: str, filtered_df) -> None:
#     """
#     Route to the correct page based on the sidebar navigation selection.

#     Args:
#         page        : Clean page name (e.g. "Dashboard", "Budget Tracker")
#         filtered_df : pandas DataFrame after filters applied, or None
#     """
#     cur    = st.session_state["currency"]
#     budget = st.session_state["monthly_budget"]

#     # Guard 1 — nothing loaded yet (or pipeline failed)
#     if st.session_state.get("clean_df") is None:
#         if st.session_state.get("upload_error"):
#             _render_upload_error(st.session_state["upload_error"])
#         else:
#             _render_landing()
#         return

#     # Guard 2 — filters returned an empty result set
#     if filtered_df is not None and len(filtered_df) == 0:
#         _render_empty_filter()
#         return

#     # ── Route ─────────────────────────────────────────────────────────────
#     if page == "Dashboard":
#         # render_dashboard is the full page from utils/analysis.py
#         render_dashboard(filtered_df, budget, cur)

#     elif page == "Data Preview":
#         _page_header("Data Preview", "Cleaned and validated expense records.")
#         render_data_preview(
#             filtered_df,
#             st.session_state["cleaning_report"],
#             cur,
#         )

#     elif page == "Budget Tracker":
#         _render_budget_tracker(filtered_df)

#     elif page == "Predictions":
#         _render_stub("Expense Predictions")

#     elif page == "AI Insights":
#         _render_stub("AI Insights")

#     elif page == "Health Score":
#         _render_stub("Financial Health Score")

#     elif page == "PDF Report":
#         _render_stub("PDF Report")


# # ═══════════════════════════════════════════════════════════════════════════════
# #  9. MAIN — App Orchestrator
# #
# #  Streamlit calls main() on every user interaction (widget change, button
# #  click, navigation, etc.).  Execution is always top-to-bottom:
# #
# #    inject_global_css()    → styles ready before any content
# #    init_session_state()   → safe defaults; no-op after first load
# #    render_sidebar()       → draws sidebar, returns file + filter dict
# #    handle_file_upload()   → runs pipeline only when a NEW file appears
# #    apply_filters()        → produces filtered_df from clean_df + filters
# #    route_page()           → renders the selected page
# # ═══════════════════════════════════════════════════════════════════════════════
# def main() -> None:
#     """Entry point — orchestrates all app phases on every Streamlit rerun."""

#     # ── 1. Styles first (before any content renders) ──────────────────────
#     inject_global_css()

#     # ── 2. Session state (safe no-op after first load) ────────────────────
#     init_session_state()

#     # ── 3. Sidebar — draws UI and returns controls ─────────────────────────
#     uploaded_file, filters = render_sidebar()

#     # ── 4. Data pipeline — only when a new file is detected ───────────────
#     # Compare the new filename to what's currently in session_state.
#     # This prevents re-running the expensive pipeline on every widget click.
#     if uploaded_file is not None:
#         is_new_file = (uploaded_file.name != st.session_state.get("file_name"))
#         no_data_yet = (st.session_state["clean_df"] is None)

#         if is_new_file or no_data_yet:
#             with st.spinner("🔄 Loading and cleaning your data…"):
#                 handle_file_upload(uploaded_file)

#             # Show a success toast if the pipeline completed without errors
#             if st.session_state["upload_error"] is None:
#                 n = len(st.session_state["clean_df"])
#                 st.toast(f"✅ {uploaded_file.name} — {n:,} records loaded.", icon="📂")

#     # ── 5. Apply sidebar filters to produce the working DataFrame ──────────
#     filtered_df = apply_filters(filters)
#     if filtered_df is not None:
#         st.session_state["filtered_df"] = filtered_df

#     # ── 6. Route to the selected page ──────────────────────────────────────
#     route_page(st.session_state["active_page"], filtered_df)


# # ═══════════════════════════════════════════════════════════════════════════════
# if __name__ == "__main__":
#     main()