from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

# Transaction Schemas
class TransactionBase(BaseModel):
    title: str = Field(..., example="Swiggy Dinner")
    amount: float = Field(..., gt=0, example=450.0)
    type: str = Field(..., example="expense") # expense | income
    category: str = Field(..., example="Food & Dining")
    date: str = Field(..., example="2026-08-25")
    payment_method: Optional[str] = Field("UPI", example="UPI")
    notes: Optional[str] = None

class TransactionCreate(TransactionBase):
    pass

class TransactionOut(TransactionBase):
    id: int
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# Investment Schemas
class InvestmentBase(BaseModel):
    name: str = Field(..., example="Nifty 50 Index Fund")
    category: str = Field(..., example="Mutual Funds")
    invested_amount: float = Field(..., gt=0, example=50000.0)
    current_value: float = Field(..., gt=0, example=62000.0)
    purchase_date: Optional[str] = Field(None, example="2025-01-10")
    notes: Optional[str] = None

class InvestmentCreate(InvestmentBase):
    pass

class InvestmentOut(InvestmentBase):
    id: int
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# Goal Schemas
class GoalBase(BaseModel):
    title: str = Field(..., example="Emergency Fund (6 Months)")
    target_amount: float = Field(..., gt=0, example=150000.0)
    current_amount: float = Field(0.0, ge=0, example=45000.0)
    target_date: str = Field(..., example="2026-12-31")
    category: Optional[str] = "Savings"

class GoalCreate(GoalBase):
    pass

class GoalUpdate(BaseModel):
    current_amount: Optional[float] = None
    target_amount: Optional[float] = None

class GoalOut(GoalBase):
    id: int
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# AI / ML Prediction Schemas
class PredictCategoryRequest(BaseModel):
    text: str = Field(..., example="Ola cab ride to office")

class PredictCategoryResponse(BaseModel):
    text: str
    predicted_category: str
    confidence: float
    all_probabilities: dict


class TrainModelRequest(BaseModel):
    epochs: Optional[int] = 10
