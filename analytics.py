import pandas as pd
from sklearn.ensemble import IsolationForest

def get_spending_anomalies(df_tx: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    """
    Uses Isolation Forest to detect unusual spending patterns.
    """
    if df_tx.empty or "Type" not in df_tx.columns:
        return pd.DataFrame()
        
    df_exp = df_tx[df_tx["Type"] == "Expense"].copy()
    
    if len(df_exp) < 10:
        return pd.DataFrame()

    features = df_exp[['Amount (₹)']]
    
    model = IsolationForest(contamination=contamination, random_state=42)
    df_exp['Anomaly_Score'] = model.fit_predict(features)
    
    anomalies = df_exp[df_exp['Anomaly_Score'] == -1]
    
    return anomalies.drop(columns=['Anomaly_Score']).sort_values(by="Date", ascending=False)


def simulate_what_if_scenario(df_tx: pd.DataFrame, category_adjustments: dict) -> dict:
    """
    Research Component: Counterfactual Financial Simulator.
    Calculates the baseline future prediction and applies 'what-if' behavioral changes.
    """
    if df_tx.empty or "Type" not in df_tx.columns:
        return {}

    df_exp = df_tx[df_tx["Type"] == "Expense"].copy()
    if df_exp.empty:
        return {}

    # Standardize dates and extract Year-Month for monthly grouping
    df_exp['Date'] = pd.to_datetime(df_exp['Date'])
    df_exp['YearMonth'] = df_exp['Date'].dt.to_period('M')

    # Group by category and calculate historical monthly averages (the baseline prediction)
    monthly_cat = df_exp.groupby(['Category', 'YearMonth'])['Amount (₹)'].sum().reset_index()
    
    baseline_forecast = {}
    for cat in df_exp['Category'].unique():
        cat_data = monthly_cat[monthly_cat['Category'] == cat]['Amount (₹)']
        if not cat_data.empty:
            baseline_forecast[cat] = cat_data.mean()

    total_baseline = sum(baseline_forecast.values())

    # Apply Counterfactuals (What-If adjustments from the user UI)
    simulated_forecast = {}
    for cat, amount in baseline_forecast.items():
        if cat in category_adjustments:
            # Adjustment is a percentage slider value (e.g., -20 for 20% reduction)
            multiplier = 1 + (category_adjustments[cat] / 100.0)
            simulated_forecast[cat] = amount * multiplier
        else:
            simulated_forecast[cat] = amount

    total_simulated = sum(simulated_forecast.values())
    savings_impact = total_baseline - total_simulated

    return {
        "baseline_total": total_baseline,
        "simulated_total": total_simulated,
        "savings_impact": savings_impact,
        "baseline_breakdown": baseline_forecast,
        "simulated_breakdown": simulated_forecast
    }
