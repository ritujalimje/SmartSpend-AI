import pandas as pd

def generate_insights(df):
    insights = []

    total_expense = df["Amount"].sum()

    category_spend = df.groupby("Category")["Amount"].sum()

    top_category = category_spend.idxmax()
    top_amount = category_spend.max()

    percent = (top_amount / total_expense) * 100

    insights.append(
        f"Highest spending is on {top_category} ({percent:.1f}% of total expenses)."
    )

    if percent > 40:
        insights.append(
            f"Warning: Spending on {top_category} is very high."
        )

    avg_daily = df.groupby("Date")["Amount"].sum().mean()

    if avg_daily > 3000:
        insights.append(
            "Your average daily spending is high. Consider reducing unnecessary expenses."
        )

    weekend_spend = df[df["Date"].dt.dayofweek >= 5]["Amount"].sum()

    weekday_spend = df[df["Date"].dt.dayofweek < 5]["Amount"].sum()

    if weekend_spend > weekday_spend * 0.5:
        insights.append(
            "Weekend spending is significantly high."
        )

    shopping_spend = category_spend.get("Shopping", 0)

    if shopping_spend > total_expense * 0.25:
        insights.append(
            "Shopping expenses are higher than recommended levels."
        )

    insights.append(
        "Reducing unnecessary expenses by 10% can improve monthly savings."
    )

    return insights