"""
FinTracker AI - Streamlit Web Application
Monolithic Cloud-Optimized Version (Direct SQLite + Local ML)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import os
import sys

# Add python_app directory to Python path for direct imports
sys.path.append(os.path.dirname(__file__))

from database import engine, SessionLocal, Base
import models
import config
from train_model import predict_category, train_and_save_model, TRAINING_DATA
from analytics import get_spending_anomalies, simulate_what_if_scenario

# Initialize SQLite tables
Base.metadata.create_all(bind=engine)

# Streamlit Page Setup
st.set_page_config(
    page_title="FinTracker AI",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
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


# --- TOP NAVIGATION & HEADER ---
st.title("💼 FinTracker AI")
st.caption("🚀 Monolithic Cloud Dashboard (Streamlit + SQLite + Local ML)")

tabs = st.tabs([
    "📊 Executive Dashboard",
    "📝 Smart Expense Tracker",
    "📈 Investments & Portfolio",
    "🎯 Savings Goals",
    "🤖 Local ML Model Studio",
    "🎛️ AI What-If Simulator"
])


# ==========================================
# TAB 1: EXECUTIVE DASHBOARD
# ==========================================
with tabs[0]:
    df_tx = load_transactions()
    df_inv = load_investments()
    df_goals = load_goals()

    if not df_tx.empty and "Type" in df_tx.columns:
        total_income = df_tx[df_tx["Type"] == "Income"]["Amount (₹)"].sum()
        total_expense = df_tx[df_tx["Type"] == "Expense"]["Amount (₹)"].sum()
    else:
        total_income = 0
        total_expense = 0
        
    net_savings = total_income - total_expense
    savings_rate = (net_savings / total_income * 100) if total_income > 0 else 0

    if not df_inv.empty and "Invested (₹)" in df_inv.columns:
        total_invested = df_inv["Invested (₹)"].sum()
        total_portfolio_val = df_inv["Current Value (₹)"].sum()
    else:
        total_invested = 0
        total_portfolio_val = 0
        
    portfolio_gain = total_portfolio_val - total_invested

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Income (Inflow)", f"₹{total_income:,.2f}")
    c2.metric("Total Expenses (Outflow)", f"₹{total_expense:,.2f}", delta=f"-₹{total_expense:,.2f}", delta_color="inverse")
    c3.metric("Net Cash Flow / Savings", f"₹{net_savings:,.2f}", delta=f"{savings_rate:.1f}% Savings Rate")
    c4.metric("Total Investment Value", f"₹{total_portfolio_val:,.2f}", delta=f"+₹{portfolio_gain:,.2f} ({((portfolio_gain/total_invested)*100 if total_invested else 0):.1f}%)")

    st.divider()

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Spending by Category")
        if not df_tx.empty and "Type" in df_tx.columns:
            df_exp = df_tx[df_tx["Type"] == "Expense"]
            if not df_exp.empty:
                cat_summary = df_exp.groupby("Category")["Amount (₹)"].sum().reset_index()
                fig_pie = px.pie(
                    cat_summary,
                    values="Amount (₹)",
                    names="Category",
                    hole=0.6,
                    color_discrete_sequence=px.colors.sequential.Tealgrn
                )
                fig_pie.update_traces(
                    textposition='inside', 
                    textinfo='percent+label',
                    hovertemplate="<b>%{label}</b><br>Amount: ₹%{value:,.2f}<extra></extra>"
                )
                fig_pie.update_layout(
                    showlegend=False,
                    margin=dict(t=10, b=10, l=10, r=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    annotations=[dict(text='Expenses', x=0.5, y=0.5, font_size=20, showarrow=False)]
                )
                st.plotly_chart(fig_pie, width="stretch")
            else:
                st.info("No expense data logged yet.")
        else:
            st.info("No expense data logged yet.")

    with col_right:
        st.subheader("Portfolio Asset Allocation")
        if not df_inv.empty and "Category" in df_inv.columns:
            inv_summary = df_inv.groupby("Category")["Current Value (₹)"].sum().reset_index()
            fig_bar = px.bar(
                inv_summary,
                x="Category",
                y="Current Value (₹)",
                color="Category",
                color_discrete_sequence=px.colors.qualitative.Prism,
                text_auto='$.2s'
            )
            fig_bar.update_traces(
                textfont_size=12, 
                textangle=0, 
                textposition="outside", 
                cliponaxis=False,
                hovertemplate="<b>%{x}</b><br>Value: ₹%{y:,.2f}<extra></extra>"
            )
            fig_bar.update_layout(
                xaxis_title=None,
                yaxis_title=None,
                showlegend=False,
                margin=dict(t=20, b=10, l=10, r=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='#e2e8f0')
            )
            st.plotly_chart(fig_bar, width="stretch")
        else:
            st.info("No investments added yet.")

    # --- AI ANOMALY DETECTION ---
    st.divider()
    st.subheader("🚨 AI Anomaly Detection")
    st.write("Using *Isolation Forest* machine learning to automatically flag unusual spending behavior based on your historical baseline.")
    
    anomalies_df = get_spending_anomalies(df_tx)
    
    if not anomalies_df.empty:
        st.warning(f"**Attention:** We detected {len(anomalies_df)} unusual transactions that fall outside your normal spending patterns.")
        st.dataframe(anomalies_df, width="stretch", hide_index=True)
    elif not df_tx.empty and "Type" in df_tx.columns and len(df_tx[df_tx["Type"] == "Expense"]) < 10:
        st.info("🤖 The AI needs at least 10 expense transactions to establish a baseline before detecting anomalies.")
    else:
        st.success("✅ No unusual spending detected. Your recent transactions align with your historical baseline.")


# ==========================================
# TAB 2: SMART EXPENSE TRACKER (WITH ML)
# ==========================================
with tabs[1]:
    st.subheader("Log a New Transaction")
    
    with st.container():
        form_col1, form_col2 = st.columns([2, 1])
        
        with form_col1:
            title_input = st.text_input("Transaction Description / Merchant", placeholder="e.g. Swiggy gourmet burger meal or Uber cab")
            
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
        tx_type = r2_c1.selectbox("Type", [t.capitalize() for t in config.TRANSACTION_TYPES])
        
        categories = config.TRANSACTION_CATEGORIES
        default_cat_idx = categories.index(predicted_cat) if predicted_cat in categories else 0
        category_choice = r2_c2.selectbox("Category", categories, index=default_cat_idx)
        
        tx_date = r2_c3.date_input("Date", value=date.today())
        payment_method = r2_c4.selectbox("Payment Method", config.PAYMENT_METHODS)

        if st.button("➕ Save Transaction", type="primary"):
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
                    st.success("✅ Transaction successfully saved!")
                    st.rerun()
                finally:
                    db.close()
            else:
                st.warning("Please enter a description.")

    st.divider()
    st.subheader("Financial Ledger")
    df_tx = load_transactions()
    if not df_tx.empty and "ID" in df_tx.columns:
        st.dataframe(df_tx, width="stretch", hide_index=True)
        
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
        asset_cat = inv_c2.selectbox("Asset Category", config.INVESTMENT_CATEGORIES)
        
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
        st.dataframe(df_inv, width="stretch", hide_index=True)
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
                        target_date=str(goal_date),
                        category="Savings"
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
    st.write("This tab demonstrates your **locally trained NLP model** (TF-IDF + Multinomial Naive Bayes).")

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
            st.plotly_chart(fig_prob, width="stretch")

    st.divider()
    st.markdown("### 2. Model Training Dataset Preview")
    train_df = pd.DataFrame(TRAINING_DATA, columns=["Text Sample", "True Category"])
    st.dataframe(train_df, width="stretch")

    st.divider()
    st.markdown("### 3. Retrain Model")
    if st.button("🚀 Retrain Local Model & Save to Disk"):
        with st.spinner("Training TF-IDF + Naive Bayes Pipeline..."):
            pipe = train_and_save_model()
            st.success("🎉 Local model trained and saved as `expense_model.pkl`!")


# ==========================================
# TAB 6: AI WHAT-IF SIMULATOR
# ==========================================
with tabs[5]:
    st.subheader("🎛️ AI What-If Counterfactual Simulator")
    st.write("Adjust your spending behavior in key categories to predict your future financial trajectory. Discover how small behavioral shifts impact your monthly savings.")
    
    df_tx_sim = load_transactions()
    
    if df_tx_sim.empty or "Type" not in df_tx_sim.columns or len(df_tx_sim[df_tx_sim["Type"] == "Expense"]) == 0:
        st.info("Log some expenses in the Smart Expense Tracker to use the simulator.")
    else:
        # Extract the top 4 expense categories for the interactive sliders
        df_exp_sim = df_tx_sim[df_tx_sim["Type"] == "Expense"]
        top_cats = df_exp_sim.groupby("Category")["Amount (₹)"].sum().sort_values(ascending=False).head(4).index.tolist()
        
        st.markdown("### Adjust Your Habits")
        adjustments = {}
        
        sim_cols = st.columns(len(top_cats))
        for i, cat in enumerate(top_cats):
            with sim_cols[i]:
                # Sliders allow +/- 50% adjustment in spending habits
                val = st.slider(f"{cat} Adjustment (%)", min_value=-50, max_value=50, value=0, step=5, key=f"slider_{cat}")
                if val != 0:
                    adjustments[cat] = val
                    
        # Run the research-grade counterfactual simulation
        sim_results = simulate_what_if_scenario(df_tx_sim, adjustments)
        
        if sim_results:
            st.divider()
            res_c1, res_c2 = st.columns([1, 2])
            
            with res_c1:
                st.markdown("### Simulation Impact")
                st.metric("Baseline Monthly Forecast", f"₹{sim_results['baseline_total']:,.2f}")
                
                delta_val = sim_results['savings_impact']
                delta_str = f"+₹{abs(delta_val):,.2f} Saved" if delta_val > 0 else f"-₹{abs(delta_val):,.2f} Extra Spent"
                delta_color = "normal" if delta_val > 0 else "inverse"
                
                if delta_val == 0:
                    st.metric("Simulated Monthly Forecast", f"₹{sim_results['simulated_total']:,.2f}", delta="No Change", delta_color="off")
                else:
                    st.metric("Simulated Monthly Forecast", f"₹{sim_results['simulated_total']:,.2f}", delta=delta_str, delta_color=delta_color)
            
            with res_c2:
                # Prepare visual comparison data
                plot_data = []
                for cat in sim_results['baseline_breakdown']:
                    plot_data.append({"Category": cat, "Amount": sim_results['baseline_breakdown'][cat], "Scenario": "Baseline Forecast"})
                    plot_data.append({"Category": cat, "Amount": sim_results['simulated_breakdown'][cat], "Scenario": "Simulated Reality"})
                
                df_plot = pd.DataFrame(plot_data)
                
                fig_sim = px.bar(
                    df_plot, 
                    x="Category", 
                    y="Amount", 
                    color="Scenario", 
                    barmode="group",
                    color_discrete_map={"Baseline Forecast": "#cbd5e1", "Simulated Reality": "#0f172a"}
                )
                fig_sim.update_layout(
                    margin=dict(t=20, b=10, l=10, r=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    legend_title_text=""
                )
                st.plotly_chart(fig_sim, width="stretch")
