def calculate_health_score(df):

    total_expense = df["Amount"].sum()

    category_spend = df.groupby("Category")["Amount"].sum()

    top_category_percent = (
        category_spend.max() / total_expense
    ) * 100

    avg_daily = df.groupby("Date")["Amount"].sum().mean()

    score = 100

    if top_category_percent > 50:
        score -= 25

    elif top_category_percent > 35:
        score -= 15

    if avg_daily > 5000:
        score -= 20

    elif avg_daily > 3000:
        score -= 10

    if total_expense > 100000:
        score -= 15

    score = max(score, 0)

    if score >= 80:
        status = "Excellent"

    elif score >= 60:
        status = "Good"

    elif score >= 40:
        status = "Average"

    else:
        status = "Poor"

    return score, status