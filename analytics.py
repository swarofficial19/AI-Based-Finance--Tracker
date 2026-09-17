import pandas as pd
from sklearn.ensemble import IsolationForest

def get_spending_anomalies(df_tx: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    """
    Uses Isolation Forest to detect unusual spending patterns.
    Contamination sets the percentage of outliers to flag (default 5%).
    """
    if df_tx.empty or "Type" not in df_tx.columns:
        return pd.DataFrame()
        
    # We only care about detecting anomalous expenses, not income
    df_exp = df_tx[df_tx["Type"] == "Expense"].copy()
    
    # We need a minimum number of transactions to establish a "normal" baseline
    if len(df_exp) < 10:
        return pd.DataFrame()

    # Feature Engineering: For V1, we focus on the raw amount. 
    # (In V2, we will add One-Hot Encoded Categories and day-of-week)
    features = df_exp[['Amount (₹)']]
    
    # Train the Isolation Forest model
    model = IsolationForest(contamination=contamination, random_state=42)
    df_exp['Anomaly_Score'] = model.fit_predict(features)
    
    # Filter for anomalies (-1 indicates an outlier in Isolation Forest)
    anomalies = df_exp[df_exp['Anomaly_Score'] == -1]
    
    # Return sorted by date (newest first), dropping the internal ML score column
    return anomalies.drop(columns=['Anomaly_Score']).sort_values(by="Date", ascending=False)
