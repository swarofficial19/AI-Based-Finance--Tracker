"""
FinTracker AI - Streamlit Web Application
Frontend & Interactive UI powered by Streamlit, Plotly, Pandas, SQLite, and Local Scikit-Learn ML.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import os
import sys

# Add python_app directory to Python path for direct imports
sys.path.append(os.path.dirname(__file__))

from database import engine, SessionLocal, Base
import models
from train_model import predict_category, train_and_save_model, MODEL_FILE, TRAINING_DATA

# Initialize SQLite tables
Base.metadata.create_all(bind=engine)

# Streamlit Page Setup
st.set_page_config(
    page_title="FinTracker AI (Python + SQLite + Local ML)",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling for a Clean, Modern Look
st.markdown("""
<style>
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .stMetric label {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        color: #64748b !important;
    }
    .stMetric .css-1wivap2 {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# --- DB Helper Functions ---
def get_db_session():
    return SessionLocal()

def load_transactions():
    db = get_db_session()
    try:
        txs = db.query(models.Transaction).order_by(models.Transaction.date.desc(), models.Transaction.id.desc()).all()
        data = [{
            "ID": t.id,
            "Title": t.title,
            "Amount (₹)": t.amount,
            "Type": t.type.capitalize(),
            "Category": t.category,
            "Date": t.date,
            "Payment Method": t.payment_method,
            "Notes": t.notes or ""
        } for t in txs]
        return pd.DataFrame(data)
    finally:
        db.close()

def load_investments():
    db = get_db_session()
    try:
        invs = db.query(models.Investment).order_by(models.Investment.id.desc()).all()
        data = [{
            "ID": i.id,
            "Name": i.name,
            "Category": i.category,
            "Invested (₹)": i.invested_amount,
            "Current Value (₹)": i.current_value,
            "Gain/Loss (₹)": i.current_value - i.invested_amount,
            "ROI (%)": round(((i.current_value - i.invested_amount) / i.invested_amount * 100), 2) if i.invested_amount > 0 else 0,
            "Date": i.purchase_date or ""
        } for i in invs]
        return pd.DataFrame(data)
    finally:
        db.close()

def load_goals():
    db = get_db_session()
    try:
        goals = db.query(models.Goal).all()
        data = [{
            "ID": g.id,
            "Goal Title": g.title,
            "Target (₹)": g.target_amount,
            "Saved (₹)": g.current_amount,
            "Progress (%)": round((g.current_amount / g.target_amount * 100), 1) if g.target_amount > 0 else 0,
            "Target Date": g.target_date,
            "Category": g.category
        } for g in goals]
        return pd.DataFrame(data)
    finally:
        db.close()


# Seed default records if completely blank
def seed_if_empty():
    db = get_db_session()
    try:
        if db.query(models.Transaction).count() == 0:
            sample_txs = [
                models.Transaction(title="Monthly Salary Credit", amount=85000, type="income", category="Salary & Income", date=str(date.today()), payment_method="Net Banking"),
                models.Transaction(title="Apartment Rent", amount=22000, type="expense", category="Housing & Rent", date=str(date.today()), payment_method="Net Banking"),
                models.Transaction(title="Swiggy Weekend Dinner", amount=950, type="expense", category="Food & Dining", date=str(date.today()), payment_method="UPI"),
                models.Transaction(title="Blinkit Weekly Essentials", amount=1650, type="expense", category="Groceries", date=str(date.today()), payment_method="UPI"),
                models.Transaction(title="Mutual Fund SIP", amount=15000, type="expense", category="Investments", date=str(date.today()), payment_method="UPI"),
                models.Transaction(title="Broadband & Jio Bill", amount=1499, type="expense", category="Utilities", date=str(date.today()), payment_method="UPI"),
            ]
            db.add_all(sample_txs)
            
        if db.query(models.Investment).count() == 0:
            sample_invs = [
                models.Investment(name="Nifty 50 Index Fund", category="Mutual Funds", invested_amount=120000, current_value=145000, purchase_date="2024-03-01"),
                models.Investment(name="PPF Government Scheme", category="PPF / EPF", invested_amount=150000, current_value=168000, purchase_date="2023-04-10"),
                models.Investment(name="Sovereign Gold Bonds", category="Gold", invested_amount=60000, current_value=78500, purchase_date="2023-10-15")
            ]
            db.add_all(sample_invs)
            
        if db.query(models.Goal).count() == 0:
            sample_goals = [
                models.Goal(title="Emergency Safety Fund", target_amount=200000, current_amount=140000, target_date="2026-12-31", category="Safety"),
                models.Goal(title="New Electric Car Downpayment", target_amount=300000, current_amount=115000, target_date="2027-08-31", category="Asset")
            ]
            db.add_all(sample_goals)
        db.commit()
    finally:
        db.close()

seed_if_empty()

# --- TOP NAVIGATION & HEADER ---
st.title("💼 FinTracker AI")
st.caption("🚀 Built in pure Python (Streamlit + FastAPI + SQLite + Local Scikit-Learn Model)")

tabs = st.tabs([
    "📊 Executive Dashboard",
    "📝 Smart Expense Tracker (with ML)",
    "📈 Investments & Portfolio",
    "🎯 Savings Goals",
    "🤖 Local ML Model Studio"
])


# ==========================================
# TAB 1: EXECUTIVE DASHBOARD
# ==========================================
with tabs[0]:
    df_tx = load_transactions()
    df_inv = load_investments()
    df_goals = load_goals()

    total_income = df_tx[df_tx["Type"] == "Income"]["Amount (₹)"].sum() if not df_tx.empty else 0
    total_expense = df_tx[df_tx["Type"] == "Expense"]["Amount (₹)"].sum() if not df_tx.empty else 0
    net_savings = total_income - total_expense
    savings_rate = (net_savings / total_income * 100) if total_income > 0 else 0

    total_invested = df_inv["Invested (₹)"].sum() if not df_inv.empty else 0
    total_portfolio_val = df_inv["Current Value (₹)"].sum() if not df_inv.empty else 0
    portfolio_gain = total_portfolio_val - total_invested

    # Top KPI Metrics Row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Income (Inflow)", f"₹{total_income:,.2f}")
    c2.metric("Total Expenses (Outflow)", f"₹{total_expense:,.2f}", delta=f"-₹{total_expense:,.2f}", delta_color="inverse")
    c3.metric("Net Cash Flow / Savings", f"₹{net_savings:,.2f}", delta=f"{savings_rate:.1f}% Savings Rate")
    c4.metric("Total Investment Value", f"₹{total_portfolio_val:,.2f}", delta=f"+₹{portfolio_gain:,.2f} ({((portfolio_gain/total_invested)*100 if total_invested else 0):.1f}%)")

    st.divider()

    # Visual Charts
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Spending by Category")
        df_exp = df_tx[df_tx["Type"] == "Expense"]
        if not df_exp.empty:
            cat_summary = df_exp.groupby("Category")["Amount (₹)"].sum().reset_index()
            fig_pie = px.pie(
                cat_summary,
                values="Amount (₹)",
                names="Category",
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            fig_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No expense data logged yet.")

    with col_right:
        st.subheader("Portfolio Asset Allocation")
        if not df_inv.empty:
            inv_summary = df_inv.groupby("Category")["Current Value (₹)"].sum().reset_index()
            fig_bar = px.bar(
                inv_summary,
                x="Category",
                y="Current Value (₹)",
                color="Category",
                text_auto='.2s'
            )
            fig_bar.update_layout(margin=dict(t=20, b=20, l=20, r=20), showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No investments added yet.")


# ==========================================
# TAB 2: SMART EXPENSE TRACKER (WITH ML)
# ==========================================
with tabs[1]:
    st.subheader("Log a New Transaction")
    
    with st.container():
        form_col1, form_col2 = st.columns([2, 1])
        
        with form_col1:
            title_input = st.text_input("Transaction Description / Merchant", placeholder="e.g. Swiggy gourmet burger meal or Uber cab")
            
            # Interactive Machine Learning Auto-Categorization
            predicted_cat = "Food & Dining"
            confidence = 1.0
            if title_input.strip():
                ml_res = predict_category(title_input)
                predicted_cat = ml_res["predicted_category"]
                confidence = ml_res["confidence"]
                st.info(f"🤖 **Local ML Prediction:** Categorized as **{predicted_cat}** with **{confidence * 100:.1f}%** confidence.")

        with form_col2:
            amount_input = st.number_input("Amount (₹)", min_value=1.0, step=50.0, value=250.0)

        r2_c1, r2_c2, r2_c3, r2_c4 = st.columns(4)
        tx_type = r2_c1.selectbox("Type", ["Expense", "Income"])
        
        categories = [
            "Food & Dining", "Groceries", "Transportation", "Utilities",
            "Shopping", "Entertainment", "Healthcare", "Housing & Rent",
            "Investments", "Salary & Income", "Other"
        ]
        default_cat_idx = categories.index(predicted_cat) if predicted_cat in categories else 0
        category_choice = r2_c2.selectbox("Category", categories, index=default_cat_idx)
        
        tx_date = r2_c3.date_input("Date", value=date.today())
        payment_method = r2_c4.selectbox("Payment Method", ["UPI", "Credit Card", "Debit Card", "Net Banking", "Cash"])

        if st.button("➕ Save Transaction to SQLite", type="primary"):
            if title_input.strip():
                db = get_db_session()
                try:
                    new_tx = models.Transaction(
                        title=title_input.strip(),
                        amount=float(amount_input),
                        type=tx_type.lower(),
                        category=category_choice,
                        date=str(tx_date),
                        payment_method=payment_method
                    )
                    db.add(new_tx)
                    db.commit()
                    st.success("✅ Transaction successfully saved into SQLite Database!")
                    st.rerun()
                finally:
                    db.close()
            else:
                st.warning("Please enter a description.")

    st.divider()
    st.subheader("Financial Ledger")
    df_tx = load_transactions()
    if not df_tx.empty:
        st.dataframe(df_tx, use_container_width=True, hide_index=True)
        
        # Delete Transaction Option
        del_col1, del_col2 = st.columns([3, 1])
        tx_to_del = del_col1.selectbox("Select Transaction ID to Delete", df_tx["ID"].tolist())
        if del_col2.button("🗑️ Delete Selected"):
            db = get_db_session()
            try:
                tx_obj = db.query(models.Transaction).filter(models.Transaction.id == tx_to_del).first()
                if tx_obj:
                    db.delete(tx_obj)
                    db.commit()
                    st.success(f"Deleted transaction ID {tx_to_del}")
                    st.rerun()
            finally:
                db.close()
    else:
        st.info("Ledger is empty.")


# ==========================================
# TAB 3: INVESTMENTS & PORTFOLIO
# ==========================================
with tabs[2]:
    st.subheader("Multi-Asset Portfolio Tracker")
    
    with st.expander("➕ Add New Investment Asset", expanded=False):
        inv_c1, inv_c2 = st.columns(2)
        asset_name = inv_c1.text_input("Asset Name", placeholder="e.g. Parag Parikh Flexi Cap Fund")
        asset_cat = inv_c2.selectbox("Asset Category", ["Mutual Funds", "Stocks", "Fixed Deposits", "PPF / EPF", "Gold", "Crypto", "Real Estate"])
        
        inv_r2_1, inv_r2_2, inv_r2_3 = st.columns(3)
        inv_amt = inv_r2_1.number_input("Invested Amount (₹)", min_value=100.0, step=1000.0, value=10000.0)
        curr_val = inv_r2_2.number_input("Current Value (₹)", min_value=100.0, step=1000.0, value=11500.0)
        p_date = inv_r2_3.date_input("Investment Date", value=date.today())

        if st.button("Save Investment"):
            if asset_name.strip():
                db = get_db_session()
                try:
                    new_inv = models.Investment(
                        name=asset_name.strip(),
                        category=asset_cat,
                        invested_amount=float(inv_amt),
                        current_value=float(curr_val),
                        purchase_date=str(p_date)
                    )
                    db.add(new_inv)
                    db.commit()
                    st.success("Investment asset added!")
                    st.rerun()
                finally:
                    db.close()

    df_inv = load_investments()
    if not df_inv.empty:
        st.dataframe(df_inv, use_container_width=True, hide_index=True)
    else:
        st.info("No investments recorded.")


# ==========================================
# TAB 4: SAVINGS GOALS
# ==========================================
with tabs[3]:
    st.subheader("Milestone Savings Goals")
    
    with st.expander("➕ Create New Financial Goal"):
        g_c1, g_c2 = st.columns(2)
        goal_title = g_c1.text_input("Goal Name", placeholder="e.g. Emergency Fund or Vacation")
        goal_target = g_c2.number_input("Target Amount (₹)", min_value=1000.0, step=5000.0, value=50000.0)
        
        g_r2_1, g_r2_2 = st.columns(2)
        goal_initial = g_r2_1.number_input("Current Saved Amount (₹)", min_value=0.0, step=1000.0, value=10000.0)
        goal_date = g_r2_2.date_input("Target Date", value=date(2027, 1, 1))

        if st.button("Create Goal"):
            if goal_title.strip():
                db = get_db_session()
                try:
                    new_g = models.Goal(
                        title=goal_title.strip(),
                        target_amount=float(goal_target),
                        current_amount=float(goal_initial),
                        target_date=str(goal_date)
                    )
                    db.add(new_g)
                    db.commit()
                    st.success("Goal successfully created!")
                    st.rerun()
                finally:
                    db.close()

    df_goals = load_goals()
    if not df_goals.empty:
        for idx, row in df_goals.iterrows():
            with st.container():
                st.markdown(f"#### 🎯 {row['Goal Title']}")
                col_g1, col_g2 = st.columns([3, 1])
                progress_val = min(float(row['Saved (₹)']) / float(row['Target (₹)']), 1.0)
                col_g1.progress(progress_val)
                col_g2.write(f"**₹{row['Saved (₹)']:,.0f}** / ₹{row['Target (₹)']:,.0f} ({row['Progress (%)']}%)")
                st.caption(f"Target Completion Date: **{row['Target Date']}**")
                st.divider()
    else:
        st.info("No active goals created yet.")


# ==========================================
# TAB 5: LOCAL ML MODEL STUDIO
# ==========================================
with tabs[4]:
    st.subheader("🤖 Local Scikit-Learn Model Training & Evaluation Studio")
    st.write("This tab demonstrates your **locally trained NLP model** (TF-IDF + Multinomial Naive Bayes). It runs 100% offline with zero cloud API keys.")

    st.markdown("### 1. Interactive Classification Sandbox")
    sample_text = st.text_input("Enter any test transaction text:", value="Swiggy paneer butter masala delivery")
    if sample_text:
        pred_res = predict_category(sample_text)
        
        c_pred1, c_pred2 = st.columns([1, 2])
        c_pred1.metric("Predicted Category", pred_res["predicted_category"])
        c_pred1.metric("Model Confidence", f"{pred_res['confidence'] * 100:.1f}%")
        
        with c_pred2:
            prob_df = pd.DataFrame(list(pred_res["all_probabilities"].items()), columns=["Category", "Probability"])
            prob_df = prob_df.sort_values(by="Probability", ascending=True)
            fig_prob = px.bar(prob_df, x="Probability", y="Category", orientation='h', title="Class Probability Distribution")
            fig_prob.update_layout(margin=dict(t=30, b=10, l=10, r=10), height=300)
            st.plotly_chart(fig_prob, use_container_width=True)

    st.divider()
    st.markdown("### 2. Model Training Dataset Preview")
    train_df = pd.DataFrame(TRAINING_DATA, columns=["Text Sample", "True Category"])
    st.dataframe(train_df, use_container_width=True)

    st.divider()
    st.markdown("### 3. Retrain Model")
    if st.button("🚀 Retrain Local Model & Save to Disk"):
        with st.spinner("Training TF-IDF + Naive Bayes Pipeline..."):
            pipe = train_and_save_model()
            st.success("🎉 Local model trained and saved as `expense_model.pkl`!")
