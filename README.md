# 💳 SmartSpend AI — Expense & Budget Analyzer

> A professional fintech-grade personal finance dashboard built with Python, Streamlit, Pandas, Plotly, and Scikit-learn.

---
## 🔗 Project Link
https://smartspend-ai-urwwrjzq89adhpvycxlvkw.streamlit.app/
---

## 📁 Project Structure

```
smartspend_ai/
│
├── app.py                        ← Main entry point (UI orchestrator)
├── requirements.txt              ← All Python dependencies
├── README.md                     ← This file
│
├── .streamlit/
│   └── config.toml               ← Theme colours, server settings
│
├── utils/                        ← All business logic (modular)
│   ├── __init__.py
│   ├── data_loader.py            ← File upload, format detection, column validation
│   ├── preprocessor.py           ← Data cleaning, type casting, null handling
│   ├── analyzer.py               ← Summary stats, aggregations, trend logic
│   ├── predictor.py              ← Linear Regression expense forecasting
│   ├── insights.py               ← Rule-based + AI spending insights engine
│   ├── health_score.py           ← Financial health score calculator (0–100)
│   ├── pdf_report.py             ← ReportLab PDF generation
│   └── ui_components.py          ← Reusable HTML/CSS Streamlit components
│
├── assets/                       ← Static files
│   └── logo.png                  ← App logo (optional)
│
├── reports/                      ← Generated PDF reports saved here
│
└── data_samples/                 ← Sample CSV files for testing
    └── sample_expenses.csv
```

---

## 🚀 Quick Start

```bash
# 1. Clone / download the project
cd smartspend_ai

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

---

## 📊 Features (Phase Overview)

| # | Feature | Module | Status |
|---|---------|--------|--------|
| 1 | CSV/Excel Upload | `data_loader.py` | 🔜 Next |
| 2 | Data Cleaning | `preprocessor.py` | 🔜 Next |
| 3 | Expense Analysis | `analyzer.py` | 🔜 Next |
| 4 | Interactive Dashboard | `app.py` + Plotly | 🔜 Next |
| 5 | Budget Tracking | `analyzer.py` | 🔜 Next |
| 6 | Expense Prediction | `predictor.py` | 🔜 Next |
| 7 | AI Insights | `insights.py` | 🔜 Next |
| 8 | Health Score | `health_score.py` | 🔜 Next |
| 9 | PDF Report | `pdf_report.py` | 🔜 Next |
| 10 | Pro UI | `ui_components.py` | ✅ Base done |

---

## 📋 Expected CSV Format

Your expense CSV should contain these columns (flexible naming supported):

| Column | Description | Example |
|--------|-------------|---------|
| `Date` | Transaction date | `2024-01-15` |
| `Amount` | Expense amount | `1500.00` |
| `Category` | Spending category | `Food`, `Rent`, `Transport` |
| `Description` | Transaction note | `Swiggy order` |
| `Payment_Mode` | How paid (optional) | `UPI`, `Card`, `Cash` |

---

## 🎨 Tech Stack

- **Frontend**: Streamlit + Custom CSS (dark fintech theme)
- **Data**: Pandas + NumPy
- **Charts**: Plotly (interactive, dark-themed)
- **ML**: Scikit-learn (Linear Regression)
- **PDF**: ReportLab / FPDF2
- **Fonts**: IBM Plex Mono + Inter (Google Fonts)
