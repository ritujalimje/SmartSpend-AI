from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

def generate_pdf_report(filename, score, status, insights):

    doc = SimpleDocTemplate(filename)

    styles = getSampleStyleSheet()

    content = []

    title = Paragraph(
        "SmartSpend AI - Financial Report",
        styles["Title"]
    )

    content.append(title)

    content.append(Spacer(1, 20))

    score_text = Paragraph(
        f"<b>Financial Health Score:</b> {score}/100",
        styles["BodyText"]
    )

    content.append(score_text)

    status_text = Paragraph(
        f"<b>Financial Status:</b> {status}",
        styles["BodyText"]
    )

    content.append(status_text)

    content.append(Spacer(1, 20))

    insights_title = Paragraph(
        "<b>AI Financial Insights</b>",
        styles["Heading2"]
    )

    content.append(insights_title)

    for insight in insights:

        p = Paragraph(f"• {insight}", styles["BodyText"])

        content.append(p)

    doc.build(content)