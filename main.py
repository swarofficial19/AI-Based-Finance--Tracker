from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional
import os

from .database import engine, Base, get_db
from . import models, schemas
from .train_model import predict_category, train_and_save_model

# Create database tables automatically
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="FinTracker Python FastAPI Backend",
    description="Full-featured REST API with SQLite Persistence & Local Scikit-Learn ML Model",
    version="2.0.0"
)

# CORS middleware for Streamlit and external web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    """Seed initial sample data if the SQLite database is empty."""
    db = next(get_db())
    try:
        if db.query(models.Transaction).count() == 0:
            sample_txs = [
                models.Transaction(title="Salary Credit", amount=85000, type="income", category="Salary & Income", date="2026-08-01", payment_method="Net Banking"),
                models.Transaction(title="Apartment Rent", amount=24000, type="expense", category="Housing & Rent", date="2026-08-02", payment_method="Net Banking"),
                models.Transaction(title="Swiggy Gourmet Dinner", amount=840, type="expense", category="Food & Dining", date="2026-08-05", payment_method="UPI"),
                models.Transaction(title="Blinkit Quick Grocery", amount=1420, type="expense", category="Groceries", date="2026-08-07", payment_method="UPI"),
                models.Transaction(title="Monthly Mutual Fund SIP", amount=15000, type="expense", category="Investments", date="2026-08-10", payment_method="UPI"),
                models.Transaction(title="Electricity & Broadband", amount=2900, type="expense", category="Utilities", date="2026-08-12", payment_method="UPI"),
                models.Transaction(title="Uber Cab Commute", amount=620, type="expense", category="Transportation", date="2026-08-15", payment_method="UPI"),
            ]
            db.add_all(sample_txs)

        if db.query(models.Investment).count() == 0:
            sample_invs = [
                models.Investment(name="Nifty 50 Index Mutual Fund", category="Mutual Funds", invested_amount=120000, current_value=148500, purchase_date="2024-04-15"),
                models.Investment(name="Parag Parikh Flexi Cap Fund", category="Mutual Funds", invested_amount=80000, current_value=98200, purchase_date="2024-06-10"),
                models.Investment(name="Public Provident Fund (PPF)", category="PPF / EPF", invested_amount=150000, current_value=168400, purchase_date="2023-03-31"),
                models.Investment(name="Sovereign Gold Bonds (SGB)", category="Gold", invested_amount=50000, current_value=67800, purchase_date="2023-11-20"),
            ]
            db.add_all(sample_invs)

        if db.query(models.Goal).count() == 0:
            sample_goals = [
                models.Goal(title="Emergency Fund (6 Months)", target_amount=200000, current_amount=135000, target_date="2026-12-31", category="Safety"),
                models.Goal(title="Europe Vacation Trip", target_amount=180000, current_amount=60000, target_date="2027-06-30", category="Travel"),
            ]
            db.add_all(sample_goals)

        db.commit()
    finally:
        db.close()


# ----------------- SYSTEM & HEALTH ----------------- #
@app.get("/")
def root():
    return {
        "status": "online",
        "app": "FinTracker Python Backend",
        "database": "SQLite (finance.db)",
        "ai_engine": "Local Scikit-Learn TF-IDF Naive Bayes"
    }


# ----------------- LOCAL ML ENDPOINTS ----------------- #
@app.post("/api/ai/predict-category", response_model=schemas.PredictCategoryResponse)
def predict_expense_category(req: schemas.PredictCategoryRequest):
    """Classifies raw transaction string using the locally trained Scikit-Learn model."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    result = predict_category(req.text.strip())
    return result


@app.post("/api/ai/train-model")
def retrain_model():
    """Trigger local model training."""
    train_and_save_model()
    return {"message": "Local Scikit-Learn model retrained successfully and saved to disk."}


@app.get("/api/ai/spending-audit")
def generate_spending_audit(db: Session = Depends(get_db)):
    """Generates an algorithmic and statistical spending audit from SQLite data."""
    txs = db.query(models.Transaction).all()
    total_income = sum(t.amount for t in txs if t.type == "income")
    total_expense = sum(t.amount for t in txs if t.type == "expense")
    net_savings = total_income - total_expense
    savings_rate = (net_savings / total_income * 100) if total_income > 0 else 0

    category_breakdown = {}
    for t in txs:
        if t.type == "expense":
            category_breakdown[t.category] = category_breakdown.get(t.category, 0) + t.amount

    # Deterministic Local Financial Reasoning Rules
    recommendations = []
    if savings_rate < 20:
        recommendations.append("⚠️ Your savings rate is below the recommended 20% threshold. Consider trimming discretionary categories.")
    elif savings_rate >= 40:
        recommendations.append("🌟 Outstanding savings rate (>40%)! You have strong surplus cash flow suitable for systematic equity investments.")
    else:
        recommendations.append("✅ Healthy savings rate (20-40%). Maintain consistent allocation towards emergency reserves.")

    top_expense = max(category_breakdown.items(), key=lambda x: x[1]) if category_breakdown else ("None", 0)
    if top_expense[0] != "None":
        recommendations.append(f"📌 Largest spending category: **{top_expense[0]}** (₹{top_expense[1]:,.2f}).")

    return {
        "total_income": total_income,
        "total_expense": total_expense,
        "net_savings": net_savings,
        "savings_rate_percent": round(savings_rate, 2),
        "top_expense_category": top_expense[0],
        "category_breakdown": category_breakdown,
        "recommendations": recommendations
    }


# ----------------- TRANSACTIONS (CRUD) ----------------- #
@app.get("/api/transactions", response_model=List[schemas.TransactionOut])
def get_transactions(
    type: Optional[str] = None,
    category: Optional[str] = None,
    db: Session = Depends(get_db)
):
    query = db.query(models.Transaction)
    if type:
        query = query.filter(models.Transaction.type == type)
    if category:
        query = query.filter(models.Transaction.category == category)
    return query.order_by(models.Transaction.date.desc(), models.Transaction.id.desc()).all()


@app.post("/api/transactions", response_model=schemas.TransactionOut, status_code=status.HTTP_201_CREATED)
def create_transaction(tx_in: schemas.TransactionCreate, db: Session = Depends(get_db)):
    tx = models.Transaction(**tx_in.dict())
    db.add(tx)
    db.commit()
    db.refresh(tx)
    return tx


@app.delete("/api/transactions/{tx_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_transaction(tx_id: int, db: Session = Depends(get_db)):
    tx = db.query(models.Transaction).filter(models.Transaction.id == tx_id).first()
    if not tx:
        raise HTTPException(status_code=404, detail="Transaction not found.")
    db.delete(tx)
    db.commit()
    return None


# ----------------- INVESTMENTS (CRUD) ----------------- #
@app.get("/api/investments", response_model=List[schemas.InvestmentOut])
def get_investments(db: Session = Depends(get_db)):
    return db.query(models.Investment).order_by(models.Investment.id.desc()).all()


@app.post("/api/investments", response_model=schemas.InvestmentOut, status_code=status.HTTP_201_CREATED)
def create_investment(inv_in: schemas.InvestmentCreate, db: Session = Depends(get_db)):
    inv = models.Investment(**inv_in.dict())
    db.add(inv)
    db.commit()
    db.refresh(inv)
    return inv


@app.delete("/api/investments/{inv_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_investment(inv_id: int, db: Session = Depends(get_db)):
    inv = db.query(models.Investment).filter(models.Investment.id == inv_id).first()
    if not inv:
        raise HTTPException(status_code=404, detail="Investment not found.")
    db.delete(inv)
    db.commit()
    return None


# ----------------- GOALS (CRUD) ----------------- #
@app.get("/api/goals", response_model=List[schemas.GoalOut])
def get_goals(db: Session = Depends(get_db)):
    return db.query(models.Goal).order_by(models.Goal.id.desc()).all()


@app.post("/api/goals", response_model=schemas.GoalOut, status_code=status.HTTP_201_CREATED)
def create_goal(goal_in: schemas.GoalCreate, db: Session = Depends(get_db)):
    goal = models.Goal(**goal_in.dict())
    db.add(goal)
    db.commit()
    db.refresh(goal)
    return goal


@app.put("/api/goals/{goal_id}", response_model=schemas.GoalOut)
def update_goal(goal_id: int, goal_up: schemas.GoalUpdate, db: Session = Depends(get_db)):
    goal = db.query(models.Goal).filter(models.Goal.id == goal_id).first()
    if not goal:
        raise HTTPException(status_code=404, detail="Goal not found.")
    if goal_up.current_amount is not None:
        goal.current_amount = goal_up.current_amount
    if goal_up.target_amount is not None:
        goal.target_amount = goal_up.target_amount
    db.commit()
    db.refresh(goal)
    return goal


@app.delete("/api/goals/{goal_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_goal(goal_id: int, db: Session = Depends(get_db)):
    goal = db.query(models.Goal).filter(models.Goal.id == goal_id).first()
    if not goal:
        raise HTTPException(status_code=404, detail="Goal not found.")
    db.delete(goal)
    db.commit()
    return None
