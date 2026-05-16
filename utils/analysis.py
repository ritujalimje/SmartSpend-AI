# ═══════════════════════════════════════════════════════════════════════════════
#  SmartSpend AI – Expense & Budget Analyzer
#  utils/analysis.py  |  v2.1 — currency-aware, real-data connected
#
#  Public API (imported by app.py):
#    compute_kpis(df, budget, currency)  → dict of KPI values
#    get_category_summary(df)            → aggregated DataFrame
#    get_monthly_trend(df)               → time-series DataFrame
#    chart_category_pie(df, currency)    → Plotly Figure
#    chart_monthly_trend(df, currency)   → Plotly Figure
#    chart_category_bar(df, currency)    → Plotly Figure
#    render_dashboard(df, budget, cur)   → None  (full Streamlit page)
#
#  Changes in v2.1:
#    • All chart hover templates now use the `currency` parameter (no hardcoded ₹)
#    • chart_category_pie: centre annotation also uses currency
#    • render_dashboard: budget bar and summary table use currency throughout
#    • get_monthly_trend: Month_Year sorted correctly as period string
#    • compute_kpis: returns raw numeric values alongside formatted strings
#      so callers can do further computation if needed
# ═══════════════════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st


# ───────────────────────────────────────────────────────────────────────────────
#  SHARED PLOTLY THEME
#  Every chart calls `fig.update_layout(**CHART_LAYOUT)` as its first step,
#  giving the app a consistent dark-fintech look without repeating style code.
# ───────────────────────────────────────────────────────────────────────────────
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",   # transparent — CSS card bg shows through
    plot_bgcolor ="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#8B949E", size=12),
    margin=dict(l=10, r=10, t=44, b=10),
    legend=dict(
        bgcolor="rgba(28,35,51,0.85)",
        bordercolor="#30363D",
        borderwidth=1,
        font=dict(size=11, color="#E6EDF3"),
    ),
    # 10-colour palette shared across all charts
    colorway=[
        "#00C9A7",  # teal
        "#58A6FF",  # blue
        "#F5A623",  # gold
        "#FF6B6B",  # coral/red
        "#A78BFA",  # purple
        "#34D399",  # green
        "#FB923C",  # orange
        "#F472B6",  # pink
        "#94A3B8",  # slate
        "#FBBF24",  # amber
    ],
)

# Reusable axis style dict (spread into xaxis / yaxis dicts)
_AXIS = dict(
    gridcolor="#21262D",
    linecolor="#30363D",
    tickfont=dict(size=11, color="#8B949E"),
    title_font=dict(size=11, color="#8B949E"),
    zeroline=False,
)


# ───────────────────────────────────────────────────────────────────────────────
#  1. KPI COMPUTATION
# ───────────────────────────────────────────────────────────────────────────────
def compute_kpis(df: pd.DataFrame, monthly_budget: float, currency: str) -> dict:
    """
    Compute the four headline KPIs from the filtered DataFrame.

    Args:
        df             : cleaned + filtered DataFrame (real data or sample)
        monthly_budget : user-configured monthly budget (sidebar number_input)
        currency       : symbol string — "₹", "$", "€", or "£"

    Returns dict with keys:
        total_expense       (str)  — formatted total spend
        total_raw           (float)— raw total for further computation
        avg_daily_expense   (str)  — formatted daily average
        top_category        (str)  — name of highest-spend category
        top_category_amount (str)  — formatted spend in top category
        estimated_savings   (str)  — formatted savings vs budget
        budget_used_pct     (float)— 0–100+ percentage
        budget_status       (str)  — "on_track" | "caution" | "over"
        total_transactions  (int)
        unique_days         (int)
        date_range_str      (str)  — "01 Jan 2024 → 30 Jun 2024"
        months_in_range     (int)
    """
    # Safe empty-data defaults — prevents any crash when filters produce 0 rows
    if df is None or df.empty:
        return {
            "total_expense": f"{currency}0",
            "total_raw": 0.0,
            "avg_daily_expense": f"{currency}0",
            "top_category": "—",
            "top_category_amount": f"{currency}0",
            "estimated_savings": f"{currency}0",
            "budget_used_pct": 0.0,
            "budget_status": "on_track",
            "total_transactions": 0,
            "unique_days": 0,
            "date_range_str": "No data",
            "months_in_range": 1,
        }

    total = float(df["Amount"].sum())

    # Average daily: total ÷ distinct calendar days in the filtered period
    unique_days = int(df["Date"].dt.date.nunique())
    avg_daily   = total / unique_days if unique_days > 0 else 0.0

    # Top spending category
    cat_totals  = df.groupby("Category")["Amount"].sum()
    top_cat     = str(cat_totals.idxmax())
    top_cat_amt = float(cat_totals.max())

    # Savings vs budget
    months_in_range = max(int(df["Month_Year"].nunique()), 1)
    if monthly_budget > 0:
        total_budget    = monthly_budget * months_in_range
        est_savings     = max(total_budget - total, 0.0)
        budget_used_pct = round((total / total_budget) * 100, 1)
        savings_str     = f"{currency}{est_savings:,.0f}"
    else:
        est_savings     = 0.0
        budget_used_pct = 0.0
        savings_str     = "N/A"

    # Budget status label used for colour coding
    if budget_used_pct < 75:
        budget_status = "on_track"
    elif budget_used_pct < 90:
        budget_status = "caution"
    else:
        budget_status = "over"

    # Readable date range
    min_d = df["Date"].min().strftime("%d %b %Y")
    max_d = df["Date"].max().strftime("%d %b %Y")

    return {
        "total_expense":       f"{currency}{total:,.0f}",
        "total_raw":           total,
        "avg_daily_expense":   f"{currency}{avg_daily:,.0f}",
        "top_category":        top_cat,
        "top_category_amount": f"{currency}{top_cat_amt:,.0f}",
        "estimated_savings":   savings_str,
        "budget_used_pct":     budget_used_pct,
        "budget_status":       budget_status,
        "total_transactions":  len(df),
        "unique_days":         unique_days,
        "date_range_str":      f"{min_d} → {max_d}",
        "months_in_range":     months_in_range,
    }


# ───────────────────────────────────────────────────────────────────────────────
#  2. AGGREGATION HELPERS
# ───────────────────────────────────────────────────────────────────────────────
def get_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by Category → total, mean, count, % of total.
    Returned sorted by Total descending (highest-spend first).
    """
    summary = (
        df.groupby("Category")["Amount"]
        .agg(Total="sum", Average="mean", Transactions="count")
        .reset_index()
        .sort_values("Total", ascending=False)
        .reset_index(drop=True)
    )
    grand_total = summary["Total"].sum()
    summary["Pct"] = (summary["Total"] / grand_total * 100).round(1)
    return summary


def get_monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate Amount by Month_Year, sorted chronologically.
    Also computes a 3-month rolling average for the trend overlay line.
    """
    trend = (
        df.groupby("Month_Year")["Amount"]
        .agg(Total="sum", Transactions="count")
        .reset_index()
    )
    # Sort as period string "YYYY-MM" — lexicographic = chronological
    trend = trend.sort_values("Month_Year").reset_index(drop=True)
    trend["Rolling_Avg"] = (
        trend["Total"].rolling(window=3, min_periods=1).mean()
    )
    return trend


# ───────────────────────────────────────────────────────────────────────────────
#  3. PLOTLY CHARTS
#  All three charts accept a `currency` param so hover labels are correct
#  regardless of which symbol the user has chosen in the sidebar.
# ───────────────────────────────────────────────────────────────────────────────

def chart_category_pie(df: pd.DataFrame, currency: str = "₹") -> go.Figure:
    """
    Donut chart: proportion of total spend per category.

    Visual design:
    • hole=0.58 — leaves clean space for a centre total label
    • Dark (#0D1117) gap between slices to separate on dark background
    • Hover shows amount + % using the caller-provided currency symbol
    • Centre annotation shows grand total
    """
    summary = get_category_summary(df)
    grand_total = summary["Total"].sum()

    fig = go.Figure(
        go.Pie(
            labels=summary["Category"],
            values=summary["Total"],
            hole=0.58,
            textinfo="label+percent",
            textfont=dict(size=11, color="#E6EDF3"),
            hovertemplate=(
                "<b>%{label}</b><br>"
                f"Amount: {currency}%{{value:,.0f}}<br>"
                "Share: %{percent}<extra></extra>"
            ),
            marker=dict(
                colors=CHART_LAYOUT["colorway"],
                line=dict(color="#0D1117", width=2),
            ),
            sort=True,       # largest slice at top
        )
    )

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="Spending by Category", font=dict(size=13, color="#E6EDF3"), x=0.01),
        showlegend=True,
        height=380,
    )

    # Centre label: total spend in selected currency
    fig.add_annotation(
        text=f"<b>{currency}{grand_total:,.0f}</b><br>"
             f"<span style='font-size:10px;color:#8B949E'>Total</span>",
        x=0.5, y=0.5,
        font=dict(size=13, color="#E6EDF3"),
        showarrow=False,
        align="center",
    )

    return fig


def chart_monthly_trend(df: pd.DataFrame, currency: str = "₹") -> go.Figure:
    """
    Area + line chart of monthly spending with a 3-month rolling average overlay.

    Visual design:
    • Teal filled area under line — "wealth chart" aesthetic
    • Dashed gold line for rolling average — clearer trend signal
    • Unified hover: all series show values on the same x-axis position
    """
    trend = get_monthly_trend(df)

    fig = go.Figure()

    # Main spend line with filled area underneath
    fig.add_trace(go.Scatter(
        x=trend["Month_Year"],
        y=trend["Total"],
        mode="lines+markers",
        name="Monthly Spend",
        line=dict(color="#00C9A7", width=2.5),
        marker=dict(size=5, color="#00C9A7", line=dict(color="#0D1117", width=1)),
        fill="tozeroy",
        fillcolor="rgba(0,201,167,0.08)",
        hovertemplate=f"<b>%{{x}}</b><br>Spent: {currency}%{{y:,.0f}}<extra></extra>",
    ))

    # Rolling average — only meaningful with 3+ months
    if len(trend) >= 3:
        fig.add_trace(go.Scatter(
            x=trend["Month_Year"],
            y=trend["Rolling_Avg"],
            mode="lines",
            name="3-Month Avg",
            line=dict(color="#F5A623", width=1.8, dash="dash"),
            hovertemplate=f"3-Mo Avg: {currency}%{{y:,.0f}}<extra></extra>",
        ))

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="Monthly Spending Trend", font=dict(size=13, color="#E6EDF3"), x=0.01),
        xaxis=dict(**_AXIS, title="Month", tickangle=-30),
        yaxis=dict(**_AXIS, title=f"Amount ({currency})", tickformat=",.0f"),
        height=380,
        hovermode="x unified",
    )

    return fig


def chart_category_bar(df: pd.DataFrame, currency: str = "₹") -> go.Figure:
    """
    Horizontal bar chart ranking categories by total spend (highest at top).

    Visual design:
    • Ascending sort so highest bar appears at the top of the chart
    • Teal opacity gradient: most-spent = full opacity, least = 35%
    • Spend value labels inside each bar
    • Dynamic height: taller when there are more categories
    """
    summary = get_category_summary(df)
    # Ascending so the highest-spend category ends up at the top
    summary_asc = summary.sort_values("Total", ascending=True).reset_index(drop=True)

    n = len(summary_asc)
    # Opacity gradient: 0.30 (least spend) → 1.0 (most spend)
    opacities  = [0.30 + 0.70 * (i / max(n - 1, 1)) for i in range(n)]
    bar_colors = [f"rgba(0,201,167,{o:.2f})" for o in opacities]

    fig = go.Figure(go.Bar(
        x=summary_asc["Total"],
        y=summary_asc["Category"],
        orientation="h",
        marker=dict(color=bar_colors, line=dict(width=0)),
        text=[f"{currency}{v:,.0f}" for v in summary_asc["Total"]],
        textposition="inside",
        textfont=dict(size=10, color="#E6EDF3"),
        hovertemplate=(
            "<b>%{y}</b><br>"
            f"Total: {currency}%{{x:,.0f}}<br>"
            "Share: %{customdata:.1f}%<extra></extra>"
        ),
        customdata=summary_asc["Pct"],
    ))

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="Category Comparison", font=dict(size=13, color="#E6EDF3"), x=0.01),
        xaxis=dict(**_AXIS, title=f"Total Spend ({currency})", tickformat=",.0f"),
        yaxis=dict(title=""),
        # yaxis={**_AXIS, "title": ""}
        # yaxis=dict(**_AXIS, title="", tickfont=dict(size=11, color="#E6EDF3")),
        height=max(320, n * 44),   # grow with number of categories
        showlegend=False,
    )

    return fig


# ───────────────────────────────────────────────────────────────────────────────
#  4. SHARED HTML COMPONENT
# ───────────────────────────────────────────────────────────────────────────────
def _kpi_card_html(label: str, value: str, subtitle: str,
                   accent: str, icon: str) -> str:
    """Return the HTML string for one KPI card (rendered via st.markdown)."""
    return (
        f'<div class="kpi-card" style="border-left:3px solid {accent};">'
        f'  <div class="kpi-label">{icon}&nbsp;{label}</div>'
        f'  <div class="kpi-value">{value}</div>'
        f'  <div class="kpi-sub">{subtitle}</div>'
        f'</div>'
    )


# ───────────────────────────────────────────────────────────────────────────────
#  5. FULL DASHBOARD PAGE  (called from app.py router)
# ───────────────────────────────────────────────────────────────────────────────
def render_dashboard(df: pd.DataFrame, monthly_budget: float, currency: str) -> None:
    """
    Render the complete Dashboard page inside the Streamlit main area.

    Layout:
        ① Page title + date range subtitle
        ② 4 KPI cards  (Total Expense | Avg Daily | Top Category | Est. Savings)
        ③ Budget utilisation progress bar
        ④ Charts row   (Donut pie LEFT  |  Monthly trend RIGHT)
        ⑤ Full-width category bar chart
        ⑥ Top-10 category summary table with download button

    Args:
        df             : filtered DataFrame (what's currently visible)
        monthly_budget : from session_state["monthly_budget"]
        currency       : from session_state["currency"]
    """

    # ── ① Page title ──────────────────────────────────────────────────────
    kpis = compute_kpis(df, monthly_budget, currency)

    st.markdown(
        f"""
        <div style="margin-bottom:1.5rem;">
            <h1 style="font-size:1.4rem;font-weight:700;color:#E6EDF3;margin:0;
                       letter-spacing:-0.01em;">Financial Dashboard</h1>
            <p style="color:#8B949E;font-size:0.82rem;margin-top:0.3rem;">
                {kpis['date_range_str']}
                &nbsp;·&nbsp; {kpis['total_transactions']:,} transactions
                &nbsp;·&nbsp; {kpis['months_in_range']} month(s)
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── ② KPI cards ───────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4, gap="small")

    with col1:
        st.markdown(
            _kpi_card_html(
                "Total Expense", kpis["total_expense"],
                subtitle=f"{kpis['unique_days']} active days",
                accent="#00C9A7", icon="💸",
            ), unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            _kpi_card_html(
                "Avg Daily Expense", kpis["avg_daily_expense"],
                subtitle="per day average",
                accent="#58A6FF", icon="📅",
            ), unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            _kpi_card_html(
                "Top Category", kpis["top_category"],
                subtitle=kpis["top_category_amount"],
                accent="#F5A623", icon="🏆",
            ), unsafe_allow_html=True,
        )
    with col4:
        savings_accent = (
            "#00C9A7" if kpis["budget_used_pct"] < 90
            else "#FF6B6B"
        )
        pct_sub = (
            f"{kpis['budget_used_pct']:.1f}% of budget used"
            if kpis["budget_used_pct"] > 0
            else "Set a budget in the sidebar"
        )
        st.markdown(
            _kpi_card_html(
                "Est. Savings", kpis["estimated_savings"],
                subtitle=pct_sub,
                accent=savings_accent, icon="🏦",
            ), unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

    # ── ③ Budget utilisation bar ──────────────────────────────────────────
    if monthly_budget > 0 and kpis["budget_used_pct"] > 0:
        pct   = min(kpis["budget_used_pct"], 100)
        color = {"on_track": "#00C9A7", "caution": "#F5A623", "over": "#FF6B6B"}[
            kpis["budget_status"]
        ]
        label = {"on_track": "✓ On Track", "caution": "⚠ Nearing Limit",
                 "over": "✗ Over Budget"}[kpis["budget_status"]]

        months      = kpis["months_in_range"]
        total_bgt   = monthly_budget * months
        remaining   = max(total_bgt - kpis["total_raw"], 0)

        st.markdown(
            f"""
            <div style="background:#1C2333;border:1px solid #30363D;border-radius:10px;
                        padding:1rem 1.25rem;margin-bottom:0.5rem;">
                <div style="display:flex;justify-content:space-between;align-items:center;
                            margin-bottom:0.55rem;">
                    <span style="font-size:0.7rem;font-weight:700;letter-spacing:0.1em;
                                 text-transform:uppercase;color:#8B949E;">
                        Budget Utilisation — {months} month(s)
                    </span>
                    <span style="font-size:0.75rem;color:{color};font-weight:700;">
                        {label}
                    </span>
                </div>
                <div style="background:#30363D;border-radius:4px;height:7px;">
                    <div style="background:{color};width:{pct:.1f}%;height:7px;
                                border-radius:4px;"></div>
                </div>
                <div style="display:flex;justify-content:space-between;margin-top:0.45rem;">
                    <span style="font-size:0.7rem;color:#484F58;">
                        Spent: {currency}{kpis['total_raw']:,.0f}
                    </span>
                    <span style="font-size:0.7rem;color:#484F58;">
                        {pct:.1f}% used
                    </span>
                    <span style="font-size:0.7rem;color:#484F58;">
                        Budget: {currency}{total_bgt:,.0f}
                        &nbsp;·&nbsp; Remaining: {currency}{remaining:,.0f}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── ④ Charts row: pie LEFT, trend RIGHT ──────────────────────────────
    st.markdown(
        '<div class="section-header"><h2>📊 Spending Breakdown</h2>'
        '<p>Category split and month-over-month trend</p></div>',
        unsafe_allow_html=True,
    )

    pie_col, trend_col = st.columns([1, 1.4], gap="medium")

    with pie_col:
        st.plotly_chart(
            chart_category_pie(df, currency),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with trend_col:
        st.plotly_chart(
            chart_monthly_trend(df, currency),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    # ── ⑤ Full-width category bar chart ───────────────────────────────────
    st.markdown(
        '<div class="section-header"><h2>📊 Category Comparison</h2>'
        '<p>All categories ranked by total spend — highest at top</p></div>',
        unsafe_allow_html=True,
    )

    st.plotly_chart(
        chart_category_bar(df, currency),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    # ── ⑥ Category summary table ──────────────────────────────────────────
    st.markdown(
        '<div class="section-header"><h2>📋 Category Summary</h2>'
        '<p>Detailed breakdown: total, average, transaction count, share</p></div>',
        unsafe_allow_html=True,
    )

    summary = get_category_summary(df)
    display = summary.copy()
    display["Total"]   = display["Total"].apply(lambda x: f"{currency}{x:,.0f}")
    display["Average"] = display["Average"].apply(lambda x: f"{currency}{x:,.0f}")
    display["Pct"]     = display["Pct"].apply(lambda x: f"{x:.1f}%")
    display = display.rename(columns={
        "Total": "Total Spend",
        "Average": "Avg / Transaction",
        "Transactions": "# Transactions",
        "Pct": "% of Total",
    })

    st.dataframe(display, use_container_width=True, hide_index=True)

    # Download button for filtered data
    st.markdown("<div style='margin-top:0.5rem;'></div>", unsafe_allow_html=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Filtered Data (CSV)",
        data=csv,
        file_name="smartspend_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# # ═══════════════════════════════════════════════════════════════════════════════
# #  SmartSpend AI – Expense & Budget Analyzer
# #  utils/analysis.py  |  Expense Analysis, KPIs & Interactive Charts
# #
# #  Responsibilities:
# #    1. compute_kpis()            — Return dict of dashboard KPI values
# #    2. get_category_summary()    — Spending totals grouped by Category
# #    3. get_monthly_trend()       — Monthly spending aggregated over time
# #    4. chart_category_pie()      — Plotly donut chart (category split)
# #    5. chart_monthly_trend()     — Plotly line + area chart (trend over months)
# #    6. chart_category_bar()      — Plotly horizontal bar chart (category rank)
# #    7. render_dashboard()        — Full dashboard page: KPIs + charts combined
# # ═══════════════════════════════════════════════════════════════════════════════

# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# import plotly.express as px
# import streamlit as st


# # ───────────────────────────────────────────────────────────────────────────────
# #  PLOTLY BASE THEME
# #  All charts share this layout dict — ensures a consistent dark fintech look
# #  without repeating the same styling code in every chart function.
# # ───────────────────────────────────────────────────────────────────────────────
# CHART_LAYOUT = dict(
#     paper_bgcolor="rgba(0,0,0,0)",   # transparent background (CSS handles it)
#     plot_bgcolor="rgba(0,0,0,0)",
#     font=dict(
#         family="Inter, sans-serif",
#         color="#8B949E",
#         size=12,
#     ),
#     margin=dict(l=10, r=10, t=40, b=10),
#     legend=dict(
#         bgcolor="rgba(28,35,51,0.8)",
#         bordercolor="#30363D",
#         borderwidth=1,
#         font=dict(size=11, color="#E6EDF3"),
#     ),
#     colorway=[
#         "#00C9A7",  # teal
#         "#58A6FF",  # blue
#         "#F5A623",  # gold
#         "#FF6B6B",  # red/coral
#         "#A78BFA",  # purple
#         "#34D399",  # green
#         "#FB923C",  # orange
#         "#F472B6",  # pink
#         "#94A3B8",  # slate
#         "#FBBF24",  # amber
#     ],
# )

# # Axis style shorthand (reused across bar/line charts)
# AXIS_STYLE = dict(
#     gridcolor="#21262D",
#     linecolor="#30363D",
#     tickfont=dict(size=11, color="#8B949E"),
#     title_font=dict(size=11, color="#8B949E"),
#     zeroline=False,
# )


# # ───────────────────────────────────────────────────────────────────────────────
# #  1. KPI COMPUTATION
# # ───────────────────────────────────────────────────────────────────────────────
# def compute_kpis(df: pd.DataFrame, monthly_budget: float, currency: str) -> dict:
#     """
#     Derive the four headline KPIs displayed in the dashboard cards.

#     Args:
#         df             : cleaned + filtered DataFrame
#         monthly_budget : user-set monthly budget (from sidebar)
#         currency       : currency symbol string (e.g. "₹")

#     Returns:
#         dict with keys:
#             total_expense, avg_daily_expense, top_category,
#             estimated_savings, budget_used_pct, total_transactions,
#             date_range_str
#     """
#     if df.empty:
#         # Return safe defaults so UI never crashes on empty filter result
#         return {
#             "total_expense":     f"{currency}0",
#             "avg_daily_expense": f"{currency}0",
#             "top_category":      "—",
#             "estimated_savings": f"{currency}0",
#             "budget_used_pct":   0.0,
#             "total_transactions": 0,
#             "date_range_str":    "No data",
#         }

#     total = df["Amount"].sum()

#     # Average Daily Expense — total spend ÷ number of unique days in the period
#     unique_days = df["Date"].dt.date.nunique()
#     avg_daily   = total / unique_days if unique_days > 0 else 0

#     # Highest spending category
#     top_cat = (
#         df.groupby("Category")["Amount"]
#         .sum()
#         .idxmax()
#     )

#     # Estimated Savings — budget × months in range minus actual spend
#     # If no budget set, show "N/A"
#     if monthly_budget > 0:
#         months_in_range = max(
#             df["Date"].dt.to_period("M").nunique(), 1
#         )
#         total_budget   = monthly_budget * months_in_range
#         est_savings    = max(total_budget - total, 0)
#         savings_str    = f"{currency}{est_savings:,.0f}"
#         budget_used_pct = round((total / total_budget) * 100, 1)
#     else:
#         savings_str     = "N/A"
#         budget_used_pct = 0.0

#     # Human-readable date range
#     min_d = df["Date"].min().strftime("%d %b %Y")
#     max_d = df["Date"].max().strftime("%d %b %Y")
#     date_range_str = f"{min_d} → {max_d}"

#     return {
#         "total_expense":      f"{currency}{total:,.0f}",
#         "avg_daily_expense":  f"{currency}{avg_daily:,.0f}",
#         "top_category":       top_cat,
#         "estimated_savings":  savings_str,
#         "budget_used_pct":    budget_used_pct,
#         "total_transactions": len(df),
#         "date_range_str":     date_range_str,
#     }


# # ───────────────────────────────────────────────────────────────────────────────
# #  2. AGGREGATION HELPERS
# # ───────────────────────────────────────────────────────────────────────────────
# def get_category_summary(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Group by Category and compute total, average, and transaction count.

#     Returns:
#         DataFrame with columns: Category, Total, Average, Transactions, Pct
#         Sorted by Total descending.
#     """
#     summary = (
#         df.groupby("Category")["Amount"]
#         .agg(Total="sum", Average="mean", Transactions="count")
#         .reset_index()
#         .sort_values("Total", ascending=False)
#     )
#     total_spend = summary["Total"].sum()
#     summary["Pct"] = (summary["Total"] / total_spend * 100).round(1)
#     return summary


# def get_monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Aggregate total spending per calendar month.

#     Returns:
#         DataFrame with columns: Month_Year (str), Total, Transactions
#         Sorted chronologically (oldest first).
#     """
#     trend = (
#         df.groupby("Month_Year")["Amount"]
#         .agg(Total="sum", Transactions="count")
#         .reset_index()
#         .sort_values("Month_Year")
#     )
#     # Add a rolling 3-month average for the trend overlay
#     trend["Rolling_Avg"] = trend["Total"].rolling(window=3, min_periods=1).mean()
#     return trend


# # ───────────────────────────────────────────────────────────────────────────────
# #  3. PLOTLY CHARTS
# # ───────────────────────────────────────────────────────────────────────────────

# def chart_category_pie(df: pd.DataFrame) -> go.Figure:
#     """
#     Donut chart — proportion of total spend per category.

#     Design choices:
#     • Donut (hole=0.55) leaves room for a centre label
#     • Thin white border between slices for separation on dark bg
#     • Custom hover template shows amount and % on same line
#     """
#     summary = get_category_summary(df)

#     fig = go.Figure(
#         go.Pie(
#             labels=summary["Category"],
#             values=summary["Total"],
#             hole=0.55,
#             textinfo="label+percent",
#             textfont=dict(size=11, color="#E6EDF3"),
#             hovertemplate=(
#                 "<b>%{label}</b><br>"
#                 "Amount: ₹%{value:,.0f}<br>"
#                 "Share: %{percent}<extra></extra>"
#             ),
#             marker=dict(
#                 colors=CHART_LAYOUT["colorway"],
#                 line=dict(color="#0D1117", width=2),  # dark gap between slices
#             ),
#         )
#     )

#     fig.update_layout(
#         **CHART_LAYOUT,
#         title=dict(
#             text="Spending by Category",
#             font=dict(size=13, color="#E6EDF3"),
#             x=0.01,
#         ),
#         showlegend=True,
#         height=360,
#     )

#     # Centre annotation showing total spend
#     total = summary["Total"].sum()
#     fig.add_annotation(
#         text=f"<b>₹{total:,.0f}</b><br><span style='font-size:10px'>Total</span>",
#         x=0.5, y=0.5,
#         font=dict(size=13, color="#E6EDF3"),
#         showarrow=False,
#         align="center",
#     )

#     return fig


# def chart_monthly_trend(df: pd.DataFrame) -> go.Figure:
#     """
#     Line + shaded area chart showing monthly spending over time.

#     Design:
#     • Filled area under the line gives a "wealth chart" visual
#     • Rolling average overlay as a dashed line (smoothed trend signal)
#     • Markers on data points for precision reading
#     """
#     trend = get_monthly_trend(df)

#     fig = go.Figure()

#     # ── Shaded area under the spending line ──────────────────────────────
#     fig.add_trace(
#         go.Scatter(
#             x=trend["Month_Year"],
#             y=trend["Total"],
#             mode="lines+markers",
#             name="Monthly Spend",
#             line=dict(color="#00C9A7", width=2.5),
#             marker=dict(size=5, color="#00C9A7"),
#             fill="tozeroy",
#             fillcolor="rgba(0,201,167,0.08)",   # subtle teal fill
#             hovertemplate=(
#                 "<b>%{x}</b><br>"
#                 "Spent: ₹%{y:,.0f}<extra></extra>"
#             ),
#         )
#     )

#     # ── Rolling average overlay ───────────────────────────────────────────
#     if len(trend) >= 3:
#         fig.add_trace(
#             go.Scatter(
#                 x=trend["Month_Year"],
#                 y=trend["Rolling_Avg"],
#                 mode="lines",
#                 name="3-Month Avg",
#                 line=dict(color="#F5A623", width=1.5, dash="dash"),
#                 hovertemplate=(
#                     "3-Month Avg: ₹%{y:,.0f}<extra></extra>"
#                 ),
#             )
#         )

#     fig.update_layout(
#         **CHART_LAYOUT,
#         title=dict(
#             text="Monthly Spending Trend",
#             font=dict(size=13, color="#E6EDF3"),
#             x=0.01,
#         ),
#         xaxis=dict(
#             **AXIS_STYLE,
#             title="Month",
#             tickangle=-30,
#         ),
#         yaxis=dict(
#             **AXIS_STYLE,
#             title="Amount (₹)",
#             tickformat=",.0f",
#         ),
#         height=360,
#         hovermode="x unified",
#     )

#     return fig


# def chart_category_bar(df: pd.DataFrame) -> go.Figure:
#     """
#     Horizontal bar chart ranking categories by total spend (highest → lowest).

#     Design:
#     • Horizontal layout so long category names are readable
#     • Teal gradient from most spent (vibrant) to least (muted)
#     • Spend labels inside bars for at-a-glance reading
#     """
#     summary = get_category_summary(df)

#     # Reverse order so highest spend appears at TOP of horizontal bars
#     summary_sorted = summary.sort_values("Total", ascending=True)

#     # Colour gradient: highest bar is full teal, lowest is muted
#     n = len(summary_sorted)
#     # Opacity steps from 0.35 (least spend) to 1.0 (most spend)
#     opacities = [0.35 + (0.65 * i / max(n - 1, 1)) for i in range(n)]
#     bar_colors = [f"rgba(0,201,167,{o:.2f})" for o in opacities]

#     fig = go.Figure(
#         go.Bar(
#             x=summary_sorted["Total"],
#             y=summary_sorted["Category"],
#             orientation="h",
#             marker=dict(
#                 color=bar_colors,
#                 line=dict(color="rgba(0,0,0,0)", width=0),
#             ),
#             text=[f"₹{v:,.0f}" for v in summary_sorted["Total"]],
#             textposition="inside",
#             textfont=dict(size=10, color="#E6EDF3"),
#             hovertemplate=(
#                 "<b>%{y}</b><br>"
#                 "Total: ₹%{x:,.0f}<br>"
#                 "Share: %{customdata:.1f}%<extra></extra>"
#             ),
#             customdata=summary_sorted["Pct"],
#         )
#     )

#     fig.update_layout(
#         **CHART_LAYOUT,
#         title=dict(
#             text="Category Comparison",
#             font=dict(size=13, color="#E6EDF3"),
#             x=0.01,
#         ),
#         xaxis=dict(
#             **AXIS_STYLE,
#             title="Total Spend (₹)",
#             tickformat=",.0f",
#         ),
#         yaxis=dict(
#             **AXIS_STYLE,
#             title="",
#             tickfont=dict(size=11, color="#E6EDF3"),
#         ),
#         height=max(300, n * 42),   # dynamic height based on category count
#         showlegend=False,
#     )

#     return fig


# # ───────────────────────────────────────────────────────────────────────────────
# #  4. KPI CARD RENDERER  (Streamlit HTML helper)
# # ───────────────────────────────────────────────────────────────────────────────
# def render_kpi_card(
#     label: str,
#     value: str,
#     subtitle: str = "",
#     accent_color: str = "#00C9A7",
#     icon: str = "",
# ) -> str:
#     """
#     Returns the HTML string for a single KPI card.
#     Called by render_dashboard() inside st.markdown(..., unsafe_allow_html=True).

#     Args:
#         label        : Top micro-label text (e.g. "TOTAL EXPENSE")
#         value        : Large displayed value (e.g. "₹24,500")
#         subtitle     : Small grey note below value (e.g. "Apr 2024 – Dec 2024")
#         accent_color : Left-border colour (default teal)
#         icon         : Optional emoji prefix before the label
#     """
#     return f"""
#     <div class="kpi-card" style="border-left: 3px solid {accent_color};">
#         <div class="kpi-label">{icon} {label}</div>
#         <div class="kpi-value">{value}</div>
#         <div style="font-size:0.72rem;color:#8B949E;margin-top:0.5rem;">
#             {subtitle}
#         </div>
#     </div>
#     """


# # ───────────────────────────────────────────────────────────────────────────────
# #  5. FULL DASHBOARD RENDERER  (called from app.py router)
# # ───────────────────────────────────────────────────────────────────────────────
# def render_dashboard(df: pd.DataFrame, monthly_budget: float, currency: str):
#     """
#     Renders the complete Dashboard page:
#         Row 1 — Page title + date range subtitle
#         Row 2 — 4 KPI cards (Total Expense, Avg Daily, Top Category, Est. Savings)
#         Row 3 — Pie chart (left) + Monthly trend (right)
#         Row 4 — Full-width category bar chart
#         Row 5 — Top 5 category breakdown table

#     Args:
#         df             : filtered DataFrame (from sidebar filters)
#         monthly_budget : float from session_state
#         currency       : symbol string from session_state
#     """

#     # ── Page header ───────────────────────────────────────────────────────
#     kpis = compute_kpis(df, monthly_budget, currency)

#     st.markdown(
#         f"""
#         <div style="margin-bottom:1.5rem;">
#             <h1 style="font-size:1.4rem;font-weight:700;color:#E6EDF3;margin:0;">
#                 Financial Dashboard
#             </h1>
#             <p style="color:#8B949E;font-size:0.83rem;margin-top:0.3rem;">
#                 {kpis['date_range_str']} &nbsp;·&nbsp;
#                 {kpis['total_transactions']:,} transactions
#             </p>
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )

#     # ── KPI Cards row ─────────────────────────────────────────────────────
#     c1, c2, c3, c4 = st.columns(4)

#     with c1:
#         st.markdown(
#             render_kpi_card(
#                 "Total Expense",
#                 kpis["total_expense"],
#                 subtitle=kpis["date_range_str"],
#                 accent_color="#00C9A7",
#                 icon="💸",
#             ),
#             unsafe_allow_html=True,
#         )

#     with c2:
#         st.markdown(
#             render_kpi_card(
#                 "Avg Daily Expense",
#                 kpis["avg_daily_expense"],
#                 subtitle="per day average",
#                 accent_color="#58A6FF",
#                 icon="📅",
#             ),
#             unsafe_allow_html=True,
#         )

#     with c3:
#         st.markdown(
#             render_kpi_card(
#                 "Top Category",
#                 kpis["top_category"],
#                 subtitle="highest spend area",
#                 accent_color="#F5A623",
#                 icon="🏆",
#             ),
#             unsafe_allow_html=True,
#         )

#     with c4:
#         savings_color = "#00C9A7" if kpis["estimated_savings"] != "N/A" else "#484F58"
#         pct = kpis["budget_used_pct"]
#         pct_label = f"{pct}% of budget used" if pct > 0 else "Set a budget to track"
#         st.markdown(
#             render_kpi_card(
#                 "Est. Savings",
#                 kpis["estimated_savings"],
#                 subtitle=pct_label,
#                 accent_color=savings_color,
#                 icon="🏦",
#             ),
#             unsafe_allow_html=True,
#         )

#     st.markdown("<br>", unsafe_allow_html=True)

#     # ── Budget usage progress bar ─────────────────────────────────────────
#     if monthly_budget > 0 and kpis["budget_used_pct"] > 0:
#         pct = min(kpis["budget_used_pct"], 100)
#         bar_color = (
#             "#00C9A7" if pct < 75 else
#             "#F5A623" if pct < 90 else
#             "#FF6B6B"
#         )
#         status_label = (
#             "On Track" if pct < 75 else
#             "Caution — nearing budget" if pct < 90 else
#             "⚠ Over Budget"
#         )
#         st.markdown(
#             f"""
#             <div style="background:#1C2333;border:1px solid #30363D;
#                         border-radius:10px;padding:1rem 1.25rem;margin-bottom:1rem;">
#                 <div style="display:flex;justify-content:space-between;
#                             margin-bottom:0.5rem;">
#                     <span style="font-size:0.75rem;font-weight:600;
#                                  letter-spacing:0.08em;text-transform:uppercase;
#                                  color:#8B949E;">Budget Utilisation</span>
#                     <span style="font-size:0.75rem;color:{bar_color};
#                                  font-weight:600;">{status_label}</span>
#                 </div>
#                 <div style="background:#30363D;border-radius:4px;height:6px;">
#                     <div style="background:{bar_color};width:{pct}%;
#                                 height:6px;border-radius:4px;
#                                 transition:width 0.4s ease;"></div>
#                 </div>
#                 <div style="display:flex;justify-content:space-between;
#                             margin-top:0.4rem;">
#                     <span style="font-size:0.7rem;color:#484F58;">0%</span>
#                     <span style="font-size:0.7rem;color:#484F58;">
#                         {pct:.1f}% used · {100-pct:.1f}% remaining
#                     </span>
#                     <span style="font-size:0.7rem;color:#484F58;">100%</span>
#                 </div>
#             </div>
#             """,
#             unsafe_allow_html=True,
#         )

#     # ── Charts row 1: Pie (left) + Monthly trend (right) ─────────────────
#     st.markdown(
#         '<div class="section-header"><h2>📊 Spending Breakdown</h2>'
#         '<p>Visual overview of your expense patterns</p></div>',
#         unsafe_allow_html=True,
#     )

#     chart_col1, chart_col2 = st.columns([1, 1.4])

#     with chart_col1:
#         fig_pie = chart_category_pie(df)
#         st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

#     with chart_col2:
#         fig_trend = chart_monthly_trend(df)
#         st.plotly_chart(fig_trend, use_container_width=True, config={"displayModeBar": False})

#     # ── Chart row 2: Full-width category bar ─────────────────────────────
#     st.markdown(
#         '<div class="section-header"><h2>📊 Category Comparison</h2>'
#         '<p>Ranked by total spending — highest to lowest</p></div>',
#         unsafe_allow_html=True,
#     )

#     fig_bar = chart_category_bar(df)
#     st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

#     # ── Top 5 category summary table ─────────────────────────────────────
#     st.markdown(
#         '<div class="section-header"><h2>📋 Category Summary</h2>'
#         '<p>Top spending categories with averages and transaction counts</p></div>',
#         unsafe_allow_html=True,
#     )

#     summary = get_category_summary(df).head(5)
#     summary["Total"]   = summary["Total"].apply(lambda x: f"{currency}{x:,.0f}")
#     summary["Average"] = summary["Average"].apply(lambda x: f"{currency}{x:,.0f}")
#     summary["Pct"]     = summary["Pct"].apply(lambda x: f"{x:.1f}%")
#     summary = summary.rename(columns={
#         "Total": "Total Spend",
#         "Average": "Avg per Txn",
#         "Transactions": "# Transactions",
#         "Pct": "% of Total",
#     })

#     st.dataframe(summary, use_container_width=True, hide_index=True)