# FinTracker AI - Python + FastAPI + Streamlit + SQLite + Local ML

This directory contains the **complete Python-based rebuild** of the FinTracker AI platform using:
- **Frontend:** Streamlit (`app.py`) with Plotly interactive charts.
- **Backend:** FastAPI (`main.py`) with full CRUD REST API endpoints.
- **Database:** SQLite (`finance.db`) managed via SQLAlchemy ORM.
- **Local Machine Learning:** Scikit-Learn TF-IDF + Multinomial Naive Bayes (`train_model.py`) running 100% offline with zero cloud API keys.

---

## 📁 Directory Structure
```text
python_app/
├── app.py              # Streamlit Interactive Web Application
├── main.py             # FastAPI REST Backend Server
├── database.py         # SQLite connection & SQLAlchemy session manager
├── models.py           # Database tables (Transactions, Investments, Goals)
├── schemas.py          # Pydantic request/response schemas
├── train_model.py      # Local NLP Training & Inference Pipeline
├── requirements.txt    # Python dependencies
└── README.md           # Instructions and documentation
```

---

## 🚀 How to Run Locally

### 1. Create a Virtual Environment & Install Dependencies
```bash
cd python_app

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Train the Local ML Model
```bash
python train_model.py
```
*Output: Generates `expense_model.pkl` with training accuracy score.*

---

### 3. Launch the Streamlit Frontend Web App
```bash
streamlit run app.py
```
*Opens in your browser at `http://localhost:8501`.*

---

### 4. (Optional) Run the FastAPI REST Backend Separately
If you want to use the backend via REST API or inspect the interactive Swagger docs:
```bash
uvicorn main:app --reload --port 8000
```
*Access interactive API documentation at `http://localhost:8000/docs`.*

---

## 🎯 Key Features Implemented in Python
1. **Executive Dashboard**: Real-time KPI metrics (Net Savings, Income, Expenses, Portfolio Gain) with Plotly doughnut & bar visual charts.
2. **Smart Expense Tracker**: Live local ML category prediction on typing descriptions (e.g., Swiggy, Uber, Electricity, SIP).
3. **Multi-Asset Portfolio**: Real-time tracking of Mutual Funds, Stocks, Gold, PPF, and Crypto with P&L and ROI% calculations.
4. **Savings Goals**: Milestone progress tracker with visual progress bars.
5. **Local ML Studio**: Visual model evaluation sandbox displaying probability distributions and one-click model retraining.
