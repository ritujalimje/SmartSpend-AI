# ═══════════════════════════════════════════════════════════════════════════════
#  SmartSpend AI – Expense & Budget Analyzer
#  utils/prediction.py  |  ML Prediction Module
#
#  What this module does:
#    Given a cleaned expense DataFrame it:
#      1. Aggregates spending into a monthly time-series
#      2. Encodes months as integers (0, 1, 2 …) — the ML "feature"
#      3. Trains a Linear Regression model on that series
#      4. Predicts the next unseen month's expense
#      5. Detects the spending trend (increasing / stable / decreasing)
#      6. Estimates savings against the user's monthly budget
#      7. Builds a Plotly chart overlaying actuals, the regression line,
#         and the single forecast point
#      8. Renders a full Streamlit page combining all of the above
#
#  Public API (imported by app.py):
#    render_predictions_page(df, monthly_budget, currency) → None
#
#  Internal helpers (not imported externally but documented for learning):
#    prepare_monthly_data(df)          → DataFrame
#    train_prediction_model(monthly)   → (model, X, y, r2)
#    predict_next_month(model, n)      → (predicted_value, next_label)
#    detect_trend(model, mean_spend)   → (trend_label, trend_color, slope)
#    generate_prediction_summary(...)  → dict
#    chart_prediction(...)             → Plotly Figure
#
#  ML concepts used (beginner-friendly explanations inline):
#    • Feature engineering : converting month labels to integers
#    • Linear Regression   : fits y = mx + b through historical spend points
#    • R² score            : 0–1 measure of how well the line fits (1 = perfect)
#    • Slope               : direction and magnitude of the fitted line
#    • Extrapolation       : using the fitted line to predict beyond known data
#
#  Minimum data requirement: 3 months of data.
#  The model gives meaningful results from 4+ months.
# ═══════════════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# ───────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ───────────────────────────────────────────────────────────────────────────────

# Minimum months needed to train a meaningful regression model.
# With fewer than this we cannot reliably fit a straight line.
MIN_MONTHS_REQUIRED = 3

# Percentage of mean spend used to decide if the trend is "stable".
# If the monthly change is < ±STABLE_THRESHOLD_PCT of the mean, call it stable.
STABLE_THRESHOLD_PCT = 0.05   # 5 %

# Plotly / colour constants shared with analysis.py aesthetic
_CHART_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#8B949E", size=12),
    margin=dict(l=10, r=10, t=50, b=10),
    legend=dict(
        bgcolor="rgba(28,35,51,0.85)",
        bordercolor="#30363D",
        borderwidth=1,
        font=dict(size=11, color="#E6EDF3"),
    ),
)
_AXIS = dict(
    gridcolor="#21262D",
    linecolor="#30363D",
    tickfont=dict(size=11, color="#8B949E"),
    title_font=dict(size=11, color="#8B949E"),
    zeroline=False,
)


# ───────────────────────────────────────────────────────────────────────────────
#  STEP 1 — DATA PREPARATION
# ───────────────────────────────────────────────────────────────────────────────
def prepare_monthly_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw expense DataFrame into a monthly time-series suitable
    for Linear Regression.

    What happens here:
        1. Group every transaction by its "Month_Year" period string (e.g. "2024-01").
        2. Sum the Amount within each period → one row per month.
        3. Sort chronologically (oldest first) — important for regression!
        4. Add a numeric "Month_Index" column: 0 for the first month,
           1 for the second, 2 for the third, etc.

    Why integer index?
        Linear Regression needs numbers, not strings. "2024-01" is meaningless
        to a model; the integer 0 tells it "this is the starting point".
        The model then learns: "each step of 1 in Month_Index corresponds to
        some change in spend" — that relationship is the regression line.

    Args:
        df : cleaned expense DataFrame from data_cleaning.clean_data()

    Returns:
        DataFrame with columns:
            Month_Year   (str)   — "2024-01"
            Total        (float) — sum of expenses in that month
            Month_Index  (int)   — 0-based integer for ML feature
            Month_Label  (str)   — human-readable "Jan 2024" for chart axis

    Raises:
        ValueError : if df is empty or has no valid Amount / Month_Year data
    """
    if df is None or df.empty:
        raise ValueError("Cannot prepare data: DataFrame is empty.")

    required = {"Amount", "Month_Year", "Date"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for prediction: {missing}")

    # ── Aggregate: one row per month ──────────────────────────────────────
    monthly = (
        df.groupby("Month_Year")["Amount"]
        .sum()
        .reset_index()
        .rename(columns={"Amount": "Total"})
    )

    # ── Sort chronologically ──────────────────────────────────────────────
    # Month_Year strings like "2024-01" sort lexicographically = chronologically
    monthly = monthly.sort_values("Month_Year").reset_index(drop=True)

    # ── Numeric index (the ML "feature" X) ───────────────────────────────
    # Month_Index = 0 for oldest month, 1 for next, 2 for next, …
    monthly["Month_Index"] = np.arange(len(monthly))

    # ── Human-readable label for chart axis ──────────────────────────────
    # Convert "2024-01" → "Jan 2024" for display
    monthly["Month_Label"] = pd.to_datetime(
        monthly["Month_Year"] + "-01"
    ).dt.strftime("%b %Y")

    return monthly


# ───────────────────────────────────────────────────────────────────────────────
#  STEP 2 — MODEL TRAINING
# ───────────────────────────────────────────────────────────────────────────────
def train_prediction_model(monthly: pd.DataFrame) -> tuple:
    """
    Fit a Linear Regression model on the monthly expense time-series.

    ML Concept — Linear Regression:
        The model tries to find the best-fit straight line through the data:
            Spend = slope × Month_Index + intercept
        It minimises the sum of squared differences between real spend and the
        line's predicted values (Ordinary Least Squares / OLS).

    The fitted "slope" tells us:
        • Positive slope → spending increases each month
        • Near-zero slope → spending is stable
        • Negative slope → spending decreases each month

    R² (R-squared):
        Measures how well the fitted line explains the variation in the data.
        • R² = 1.0 → perfect fit (line goes through every data point)
        • R² = 0.0 → the line is no better than predicting the mean every time
        • R² < 0   → the line is worse than the mean (can happen with small n)
        For financial data with 3–12 months, values of 0.4–0.8 are typical.

    Args:
        monthly : DataFrame from prepare_monthly_data()

    Returns:
        (model, X, y, r2_score)
            model    : trained LinearRegression object
            X        : numpy array of Month_Index values, shape (n, 1)
            y        : numpy array of Total spend values, shape (n,)
            r2       : float, coefficient of determination

    Raises:
        ValueError : if fewer than MIN_MONTHS_REQUIRED rows available
    """
    n = len(monthly)
    if n < MIN_MONTHS_REQUIRED:
        raise ValueError(
            f"Need at least {MIN_MONTHS_REQUIRED} months of data to train the model. "
            f"Only {n} month(s) found. Upload more historical data."
        )

    # ── Prepare X (feature) and y (target) ───────────────────────────────
    # X must be 2-D for sklearn: shape (n_samples, n_features)
    # We have 1 feature (Month_Index), so shape is (n, 1)
    X = monthly["Month_Index"].values.reshape(-1, 1)  # shape: (n, 1)
    y = monthly["Total"].values                        # shape: (n,)

    # ── Fit the model ─────────────────────────────────────────────────────
    model = LinearRegression()
    model.fit(X, y)

    # ── Evaluate — R² on the training data ───────────────────────────────
    y_pred = model.predict(X)
    r2     = float(r2_score(y, y_pred))

    return model, X, y, r2


# ───────────────────────────────────────────────────────────────────────────────
#  STEP 3 — NEXT-MONTH PREDICTION
# ───────────────────────────────────────────────────────────────────────────────
def predict_next_month(model: LinearRegression, n_months: int,
                       last_month_year: str) -> tuple:
    """
    Use the trained model to predict the next month's total expense.

    ML Concept — Extrapolation:
        The model has learned the pattern from months 0 … n-1.
        We now ask: "what does the line predict at month n?"
        This is simply: prediction = slope × n + intercept

    Args:
        model           : trained LinearRegression object from train_prediction_model()
        n_months        : number of months already seen (= next Month_Index to predict)
        last_month_year : "YYYY-MM" string of the last known month

    Returns:
        (predicted_value, next_label)
            predicted_value : float, predicted total expense for next month
            next_label      : str, human-readable label e.g. "Jul 2024"
    """
    # Next index = one step beyond the last known index
    next_index = np.array([[n_months]])          # shape (1, 1) for sklearn
    predicted  = float(model.predict(next_index)[0])

    # Clamp: spend cannot be negative even if the regression line dips below 0
    predicted  = max(predicted, 0.0)

    # Build human-readable label for the next month
    next_period    = pd.Period(last_month_year, freq="M") + 1
    next_label     = pd.Timestamp(str(next_period)).strftime("%b %Y")

    return predicted, next_label


# ───────────────────────────────────────────────────────────────────────────────
#  STEP 4 — TREND DETECTION
# ───────────────────────────────────────────────────────────────────────────────
def detect_trend(model: LinearRegression, mean_spend: float) -> tuple:
    """
    Classify the spending trend based on the regression line's slope.

    Logic:
        slope > 0  means the line goes upward  → spending is increasing
        slope < 0  means the line goes downward → spending is decreasing
        |slope| < STABLE_THRESHOLD_PCT × mean_spend → negligible change → stable

    Using slope / mean normalises the threshold across different spending scales:
        A slope of ₹1,000/month means very different things if mean spend is
        ₹10,000 (10% change — significant) vs ₹200,000 (0.5% — negligible).

    Args:
        model       : trained LinearRegression (we read model.coef_[0] = slope)
        mean_spend  : float, average monthly spend used to normalise the threshold

    Returns:
        (trend_label, trend_color, slope)
            trend_label : "📈 Increasing" | "📉 Decreasing" | "➡️ Stable"
            trend_color : hex colour string for UI display
            slope       : raw slope value in currency units per month
    """
    slope     = float(model.coef_[0])          # change in spend per month (₹/month)
    threshold = STABLE_THRESHOLD_PCT * max(mean_spend, 1.0)

    if slope > threshold:
        return "📈 Increasing", "#FF6B6B", slope     # red — spending rising
    elif slope < -threshold:
        return "📉 Decreasing", "#00C9A7", slope     # teal — spending falling (good)
    else:
        return "➡️ Stable", "#F5A623", slope         # gold — roughly flat


# ───────────────────────────────────────────────────────────────────────────────
#  STEP 5 — PREDICTION SUMMARY DICT
# ───────────────────────────────────────────────────────────────────────────────
def generate_prediction_summary(
    monthly: pd.DataFrame,
    model: LinearRegression,
    r2: float,
    predicted_value: float,
    next_label: str,
    monthly_budget: float,
    currency: str,
) -> dict:
    """
    Collect all computed ML results into a single dict that the UI layer
    can read without needing to understand any ML internals.

    This separation keeps the rendering functions clean — they just read
    keys from this dict; they don't do any computation themselves.

    Args:
        monthly         : monthly DataFrame from prepare_monthly_data()
        model           : trained LinearRegression object
        r2              : R² score (float)
        predicted_value : next month predicted spend (float)
        next_label      : next month human-readable label (str)
        monthly_budget  : user's configured budget (float)
        currency        : symbol string e.g. "₹"

    Returns:
        dict with keys:
            predicted_value        (float)  — raw predicted spend
            predicted_fmt          (str)    — formatted e.g. "₹72,400"
            next_month_label       (str)    — e.g. "Jul 2024"
            trend_label            (str)    — "📈 Increasing" etc.
            trend_color            (str)    — hex colour
            slope                  (float)  — monthly change in spend
            slope_fmt              (str)    — formatted with sign e.g. "+₹1,200"
            r2_score               (float)  — model fit quality 0–1
            r2_label               (str)    — "Good" / "Fair" / "Low"
            mean_monthly_spend     (float)
            mean_monthly_fmt       (str)
            n_months               (int)    — months used for training
            est_savings            (float)  — budget − predicted (may be negative)
            est_savings_fmt        (str)    — formatted with sign
            est_savings_color      (str)    — teal if positive, red if over
            budget_gap_pct         (float)  — predicted / budget × 100
    """
    mean_spend  = float(monthly["Total"].mean())
    n_months    = len(monthly)
    slope       = float(model.coef_[0])

    trend_label, trend_color, _ = detect_trend(model, mean_spend)

    # R² quality label — helps non-technical users interpret the fit score
    if r2 >= 0.75:
        r2_label = "Strong fit"
    elif r2 >= 0.45:
        r2_label = "Moderate fit"
    elif r2 >= 0.0:
        r2_label = "Weak fit — treat forecast as indicative"
    else:
        r2_label = "Very weak fit — insufficient data pattern"

    # Estimated savings = budget − predicted (negative means over-budget forecast)
    est_savings       = monthly_budget - predicted_value
    est_savings_color = "#00C9A7" if est_savings >= 0 else "#FF6B6B"
    savings_sign      = "+" if est_savings >= 0 else ""

    # Budget gap percentage
    budget_gap_pct = (predicted_value / monthly_budget * 100) if monthly_budget > 0 else 0.0

    # Slope formatted with directional sign
    slope_sign = "+" if slope >= 0 else ""

    return {
        "predicted_value":     predicted_value,
        "predicted_fmt":       f"{currency}{predicted_value:,.0f}",
        "next_month_label":    next_label,
        "trend_label":         trend_label,
        "trend_color":         trend_color,
        "slope":               slope,
        "slope_fmt":           f"{slope_sign}{currency}{abs(slope):,.0f}/mo",
        "r2_score":            r2,
        "r2_label":            r2_label,
        "mean_monthly_spend":  mean_spend,
        "mean_monthly_fmt":    f"{currency}{mean_spend:,.0f}",
        "n_months":            n_months,
        "est_savings":         est_savings,
        "est_savings_fmt":     f"{savings_sign}{currency}{abs(est_savings):,.0f}",
        "est_savings_color":   est_savings_color,
        "budget_gap_pct":      budget_gap_pct,
    }


# ───────────────────────────────────────────────────────────────────────────────
#  STEP 6 — PREDICTION CHART
# ───────────────────────────────────────────────────────────────────────────────
def chart_prediction(
    monthly: pd.DataFrame,
    model: LinearRegression,
    predicted_value: float,
    next_label: str,
    currency: str = "₹",
) -> go.Figure:
    """
    Build a Plotly figure with three layers:
        1. Bar chart     — actual monthly spend (historical truth)
        2. Dashed line   — regression line fitted over history
        3. Single marker — the forecast point for the next month

    Why show the regression line over history?
        It lets the user visually judge how well the model fits.
        A line that closely tracks the bars = high R², trustworthy forecast.
        A line that misses most bars = low R², treat forecast cautiously.

    Args:
        monthly         : DataFrame from prepare_monthly_data()
        model           : trained LinearRegression object
        predicted_value : float, next month forecast
        next_label      : str, label for forecast point on x-axis
        currency        : symbol string

    Returns:
        Plotly Figure object (rendered by st.plotly_chart in the UI layer)
    """
    # Labels for x-axis: history + one future point
    x_labels = monthly["Month_Label"].tolist() + [next_label]

    # Regression line values for each historical month
    X_hist   = monthly["Month_Index"].values.reshape(-1, 1)
    y_fitted = model.predict(X_hist).tolist()

    # The regression line extended one step to the forecast month
    n        = len(monthly)
    X_next   = np.array([[n]])
    y_future = float(model.predict(X_next)[0])

    fig = go.Figure()

    # ── Layer 1: Actual spend bars (history) ─────────────────────────────
    fig.add_trace(go.Bar(
        x=monthly["Month_Label"],
        y=monthly["Total"],
        name="Actual Spend",
        marker=dict(
            color="rgba(0,201,167,0.55)",
            line=dict(color="#00C9A7", width=1.5),
        ),
        hovertemplate=(
            "<b>%{x}</b><br>"
            f"Actual: {currency}%{{y:,.0f}}<extra></extra>"
        ),
    ))

    # ── Layer 2: Regression line over history ────────────────────────────
    fig.add_trace(go.Scatter(
        x=monthly["Month_Label"],
        y=y_fitted,
        mode="lines",
        name="Trend Line (LR)",
        line=dict(color="#58A6FF", width=2),#, dash="dot"
        hovertemplate=(
            "<b>%{x}</b><br>"
            f"Trend: {currency}%{{y:,.0f}}<extra></extra>"
        ),
    ))

    # ── Layer 3: Forecast bar (next month) ───────────────────────────────
    fig.add_trace(go.Bar(
        x=[next_label],
        y=[max(predicted_value, 0)],
        name="Forecast",
        marker=dict(
            color="rgba(245,166,35,0.25)",
            line=dict(color="#F5A623", width=2),  # dashed border , dash="solid"
        ),
        hovertemplate=(
            f"<b>{next_label}</b><br>"
            f"Forecast: {currency}{predicted_value:,.0f}<extra></extra>"
        ),
    ))

    # ── Forecast annotation arrow ─────────────────────────────────────────
    fig.add_annotation(
        x=next_label,
        y=max(predicted_value, 0),
        text=f"<b>{currency}{predicted_value:,.0f}</b>",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#F5A623",
        arrowsize=1,
        arrowwidth=1.5,
        ax=0,
        ay=-40,
        font=dict(size=11, color="#F5A623"),
        bgcolor="rgba(28,35,51,0.85)",
        bordercolor="#F5A623",
        borderwidth=1,
        borderpad=4,
    )

    # ── Layout ────────────────────────────────────────────────────────────
    fig.update_layout(
        **_CHART_BASE,
        title=dict(
            text="Monthly Spend · Trend Line · Next Month Forecast",
            font=dict(size=13, color="#E6EDF3"),
            x=0.01,
        ),
        xaxis=dict(**_AXIS, title="Month", tickangle=-25),
        yaxis=dict(**_AXIS, title=f"Amount ({currency})", tickformat=",.0f"),
        barmode="group",
        height=420,
        hovermode="x unified",
        bargap=0.25,
        showlegend=True,
    )

    return fig


# ───────────────────────────────────────────────────────────────────────────────
#  STEP 7 — AI FORECAST NARRATIVE
# ───────────────────────────────────────────────────────────────────────────────
def _build_ai_narrative(summary: dict, currency: str) -> list[str]:
    """
    Generate 4–5 human-readable insight sentences from the prediction summary.

    This function uses rule-based logic (if/else on computed values) to produce
    sentences that feel like analyst commentary. No LLM or external API needed.

    Returns:
        List of insight strings (rendered as bullet points in the UI)
    """
    insights = []
    pred     = summary["predicted_value"]
    mean     = summary["mean_monthly_spend"]
    slope    = summary["slope"]
    budget   = summary["budget_gap_pct"]
    r2       = summary["r2_score"]
    savings  = summary["est_savings"]
    trend    = summary["trend_label"]

    # Insight 1 — forecast vs historical mean
    change_pct = ((pred - mean) / max(mean, 1)) * 100
    if abs(change_pct) < 3:
        insights.append(
            f"📊 Forecast of **{summary['predicted_fmt']}** is in line with your "
            f"historical average of **{summary['mean_monthly_fmt']}** — "
            "spending pattern is consistent."
        )
    elif change_pct > 0:
        insights.append(
            f"📊 Forecast of **{summary['predicted_fmt']}** is "
            f"**{change_pct:.1f}% higher** than your historical average of "
            f"**{summary['mean_monthly_fmt']}** — consider where the extra spend is going."
        )
    else:
        insights.append(
            f"📊 Forecast of **{summary['predicted_fmt']}** is "
            f"**{abs(change_pct):.1f}% lower** than your historical average of "
            f"**{summary['mean_monthly_fmt']}** — your cost-cutting is showing results."
        )

    # Insight 2 — trend interpretation
    if "Increasing" in trend:
        insights.append(
            f"📈 Your spending has been rising by roughly "
            f"**{currency}{abs(slope):,.0f} per month**. "
            "If unchecked, this will erode your savings capacity over time."
        )
    elif "Decreasing" in trend:
        insights.append(
            f"📉 Spending is trending downward by about "
            f"**{currency}{abs(slope):,.0f} per month** — "
            "a positive sign that your financial habits are improving."
        )
    else:
        insights.append(
            "➡️ Your spending is **stable** with no significant upward or downward trend. "
            "Consistent habits make future planning more reliable."
        )

    # Insight 3 — budget outlook
    if savings >= 0:
        insights.append(
            f"💰 At this forecast, you are on track to stay **within budget**, "
            f"with an estimated **{summary['est_savings_fmt']}** to spare next month."
        )
    else:
        pct_over = budget - 100
        insights.append(
            f"⚠️ The forecast exceeds your budget by "
            f"**{summary['est_savings_fmt']}** ({pct_over:.1f}% over). "
            "Consider reducing discretionary spending in the coming weeks."
        )

    # Insight 4 — model confidence
    if r2 >= 0.75:
        insights.append(
            f"🎯 Model confidence is **high** (R² = {r2:.2f}). "
            "Your spending follows a clear pattern, making this forecast reliable."
        )
    elif r2 >= 0.45:
        insights.append(
            f"🎯 Model confidence is **moderate** (R² = {r2:.2f}). "
            "The forecast is directionally useful but may not be precise — "
            f"add more months of data to improve accuracy."
        )
    else:
        insights.append(
            f"🎯 Model confidence is **low** (R² = {r2:.2f}). "
            "Your spending doesn't yet show a clear linear pattern. "
            "The forecast is best treated as a rough estimate — "
            f"upload more historical months to improve it."
        )

    # Insight 5 — actionable tip based on top category
    insights.append(
        "💡 **Tip:** Review your highest-spend category on the Dashboard. "
        "Reducing it by even 10–15% could meaningfully shift your savings trajectory."
    )

    return insights


# ───────────────────────────────────────────────────────────────────────────────
#  STREAMLIT UI HELPERS  (internal)
# ───────────────────────────────────────────────────────────────────────────────
def _pred_kpi(col, label: str, value: str, subtitle: str,
              accent: str, icon: str) -> None:
    """Render one KPI card into a Streamlit column (matches app.py style)."""
    with col:
        st.markdown(
            f'<div class="kpi-card" style="border-left:3px solid {accent};">'
            f'  <div class="kpi-label">{icon}&nbsp;{label}</div>'
            f'  <div class="kpi-value">{value}</div>'
            f'  <div class="kpi-sub">{subtitle}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def _section(title: str, subtitle: str = "") -> None:
    """Render a teal-bordered section header."""
    sub = (f'<p style="font-size:0.76rem;color:#8B949E;margin:0.15rem 0 0;">'
           f'{subtitle}</p>') if subtitle else ""
    st.markdown(
        f'<div class="section-header"><h2>{title}</h2>{sub}</div>',
        unsafe_allow_html=True,
    )


# ───────────────────────────────────────────────────────────────────────────────
#  PUBLIC ENTRY POINT — Predictions Page
# ───────────────────────────────────────────────────────────────────────────────
def render_predictions_page(
    df: pd.DataFrame,
    monthly_budget: float,
    currency: str,
) -> None:
    """
    Render the complete Expense Predictions page inside Streamlit.

    Layout:
        ① Page title + data context subtitle
        ② ML pipeline explanation banner
        ③ 4 KPI cards  (Predicted Spend | Trend | Est. Savings | Model Fit)
        ④ Prediction chart (actuals + regression line + forecast bar)
        ⑤ Monthly data table used for training
        ⑥ AI forecast summary (narrative insights)

    Handles all error states internally:
        • Fewer than MIN_MONTHS_REQUIRED months → friendly error card
        • Empty DataFrame → friendly error card
        • Any unexpected exception → show error + traceback hint

    Args:
        df             : filtered expense DataFrame from app.py
        monthly_budget : user's monthly budget float from session_state
        currency       : symbol string from session_state
    """

    # ── ① Page title ──────────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-bottom:1.5rem;">
        <h1 style="font-size:1.35rem;font-weight:700;color:#E6EDF3;
                   margin:0;letter-spacing:-0.01em;">
            Expense Predictions
        </h1>
        <p style="color:#8B949E;font-size:0.82rem;margin-top:0.3rem;">
            Linear Regression forecast · scikit-learn · trained on your expense history
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── ② ML pipeline explanation banner ─────────────────────────────────
    st.markdown("""
    <div style="background:rgba(88,166,255,0.07);border:1px solid rgba(88,166,255,0.2);
                border-radius:10px;padding:0.9rem 1.2rem;margin-bottom:1.5rem;
                font-size:0.8rem;color:#8B949E;line-height:1.8;">
        <b style="color:#58A6FF;">🤖 How this works:</b>&nbsp;
        Monthly expenses are grouped and encoded as integers (month 0, 1, 2…).
        A <b style="color:#E6EDF3;">Linear Regression</b> model learns the best-fit
        line through those points.
        The line is then extended one step ahead to forecast next month's spend.
        <b style="color:#E6EDF3;">R²</b> measures how well the line fits your data (higher = more reliable).
    </div>
    """, unsafe_allow_html=True)

    # ── Run the ML pipeline — with graceful error handling ─────────────────

    try:
        # STEP 1 — prepare data
        monthly = prepare_monthly_data(df)

    except ValueError as e:
        # Not enough data or wrong columns — show user-friendly card
        st.markdown(f"""
        <div style="background:rgba(245,166,35,0.07);border:1px solid rgba(245,166,35,0.3);
                    border-radius:10px;padding:1.5rem;text-align:center;margin-top:1rem;">
            <div style="font-size:1.8rem;margin-bottom:0.6rem;">📅</div>
            <div style="font-size:0.88rem;color:#F5A623;font-weight:600;
                        margin-bottom:0.4rem;">Insufficient Data for Prediction</div>
            <div style="font-size:0.82rem;color:#8B949E;line-height:1.75;">{e}</div>
        </div>
        """, unsafe_allow_html=True)
        return

    try:
        # STEP 2 — train model
        model, X, y, r2 = train_prediction_model(monthly)

    except ValueError as e:
        st.markdown(f"""
        <div style="background:rgba(255,107,107,0.07);border:1px solid rgba(255,107,107,0.3);
                    border-radius:10px;padding:1.5rem;text-align:center;margin-top:1rem;">
            <div style="font-size:1.8rem;margin-bottom:0.6rem;">⚠️</div>
            <div style="font-size:0.88rem;color:#FF6B6B;font-weight:600;
                        margin-bottom:0.4rem;">Model Training Failed</div>
            <div style="font-size:0.82rem;color:#8B949E;line-height:1.75;">{e}</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # STEP 3 — predict next month
    predicted_value, next_label = predict_next_month(
        model,
        n_months=len(monthly),
        last_month_year=monthly["Month_Year"].iloc[-1],
    )

    # STEP 4 — assemble summary dict
    summary = generate_prediction_summary(
        monthly, model, r2,
        predicted_value, next_label,
        monthly_budget, currency,
    )

    # ── ③ KPI cards ───────────────────────────────────────────────────────
    _section("📊 Forecast Summary", f"Based on {summary['n_months']} months of data")

    c1, c2, c3, c4 = st.columns(4, gap="small")

    _pred_kpi(c1,
        label="Predicted Spend",
        value=summary["predicted_fmt"],
        subtitle=f"for {summary['next_month_label']}",
        accent="#F5A623", icon="🔮",
    )
    _pred_kpi(c2,
        label="Spending Trend",
        value=summary["trend_label"].split(" ", 1)[1],   # strip emoji
        subtitle=f"slope: {summary['slope_fmt']}",
        accent=summary["trend_color"], icon=summary["trend_label"][0],
    )
    _pred_kpi(c3,
        label="Est. Savings",
        value=summary["est_savings_fmt"],
        subtitle="predicted budget − forecast",
        accent=summary["est_savings_color"], icon="🏦",
    )
    _pred_kpi(c4,
        label="Model Fit (R²)",
        value=f"{r2:.2f}",
        subtitle=summary["r2_label"],
        accent="#A78BFA", icon="🎯",
    )

    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

    # Budget utilisation bar for forecast
    if monthly_budget > 0:
        pct   = min(summary["budget_gap_pct"], 100)
        color = (
            "#00C9A7" if pct < 75 else
            "#F5A623" if pct < 100 else
            "#FF6B6B"
        )
        status = (
            "✓ Within budget" if pct < 100 else
            f"✗ {summary['budget_gap_pct'] - 100:.1f}% over budget"
        )
        st.markdown(f"""
        <div style="background:#1C2333;border:1px solid #30363D;border-radius:10px;
                    padding:0.9rem 1.25rem;margin-bottom:0.5rem;">
            <div style="display:flex;justify-content:space-between;margin-bottom:0.5rem;">
                <span style="font-size:0.7rem;font-weight:700;letter-spacing:0.1em;
                             text-transform:uppercase;color:#8B949E;">
                    Forecast vs Budget ({summary['next_month_label']})
                </span>
                <span style="font-size:0.75rem;color:{color};font-weight:700;">{status}</span>
            </div>
            <div style="background:#30363D;border-radius:4px;height:7px;">
                <div style="background:{color};width:{pct:.1f}%;height:7px;border-radius:4px;">
                </div>
            </div>
            <div style="display:flex;justify-content:space-between;margin-top:0.4rem;">
                <span style="font-size:0.7rem;color:#484F58;">
                    Forecast: {summary['predicted_fmt']}
                </span>
                <span style="font-size:0.7rem;color:#484F58;">
                    {pct:.1f}% of budget
                </span>
                <span style="font-size:0.7rem;color:#484F58;">
                    Budget: {currency}{monthly_budget:,.0f}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── ④ Prediction chart ────────────────────────────────────────────────
    _section(
        "📈 Actual vs Trend vs Forecast",
        "Bars = actual spend · Dotted line = regression · Gold bar = next month forecast"
    )

    st.plotly_chart(
        chart_prediction(monthly, model, predicted_value, next_label, currency),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    # ── ⑤ Monthly training data table ────────────────────────────────────
    _section(
        "🗃 Training Data",
        f"{summary['n_months']} months used to train the Linear Regression model"
    )

    display = monthly[["Month_Label", "Month_Index", "Total"]].copy()
    display["Total"] = display["Total"].apply(lambda x: f"{currency}{x:,.0f}")

    # Add the regression-fitted value and residual for each month
    fitted_vals = model.predict(X).tolist()
    display["Trend (Fitted)"]  = [f"{currency}{v:,.0f}" for v in fitted_vals]
    display["Residual"]        = [
        f"{currency}{(a - f):+,.0f}"
        for a, f in zip(y, fitted_vals)
    ]

    display = display.rename(columns={
        "Month_Label":  "Month",
        "Month_Index":  "Index (X)",
        "Total":        "Actual Spend",
    })

    st.dataframe(display, use_container_width=True, hide_index=True)

    st.markdown("""
    <p style="font-size:0.7rem;color:#484F58;margin-top:0.3rem;">
        <b>Index (X)</b> — the numeric feature fed to the model &nbsp;·&nbsp;
        <b>Trend (Fitted)</b> — what the regression line predicts for each known month &nbsp;·&nbsp;
        <b>Residual</b> — difference between actual and fitted (smaller = better fit)
    </p>
    """, unsafe_allow_html=True)

    # ── ⑥ AI forecast narrative ───────────────────────────────────────────
    _section(
        "🤖 AI Forecast Summary",
        "Rule-based insights derived from the regression model output"
    )

    insights = _build_ai_narrative(summary, currency)

    st.markdown("""
    <div style="background:#1C2333;border:1px solid #30363D;border-radius:10px;
                padding:1.25rem 1.5rem;">
    """, unsafe_allow_html=True)

    for insight in insights:
        st.markdown(
            f'<div style="font-size:0.83rem;color:#E6EDF3;line-height:1.85;'
            f'margin-bottom:0.65rem;padding-left:0.1rem;">'
            f'• {insight}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # Model metadata footer
    st.markdown(f"""
    <div style="margin-top:0.75rem;padding:0.75rem 1rem;background:#161B22;
                border:1px solid #30363D;border-radius:8px;
                font-size:0.69rem;color:#484F58;line-height:1.9;">
        <b style="color:#8B949E;">Model details:</b> &nbsp;
        Algorithm: Linear Regression (OLS) &nbsp;·&nbsp;
        Library: scikit-learn &nbsp;·&nbsp;
        Feature: Month Index (integer) &nbsp;·&nbsp;
        Target: Monthly Spend ({currency}) &nbsp;·&nbsp;
        Training samples: {summary['n_months']} &nbsp;·&nbsp;
        R²: {r2:.4f} &nbsp;·&nbsp;
        Slope: {summary['slope_fmt']} &nbsp;·&nbsp;
        Intercept: {currency}{float(model.intercept_):,.0f}
    </div>
    """, unsafe_allow_html=True)