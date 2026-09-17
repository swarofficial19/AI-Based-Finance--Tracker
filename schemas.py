from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from datetime import datetime
import config

# ----------------- TRANSACTIONS ----------------- #
class TransactionBase(BaseModel):
    title: str = Field(..., example="Swiggy Dinner")
    amount: float = Field(..., gt=0, example=450.0)
    type: str = Field(..., example="expense") 
    category: str = Field(..., example="Food & Dining")
    date: str = Field(..., example="2026-08-25")
    payment_method: Optional[str] = Field("UPI", example="UPI")
    notes: Optional[str] = None

    @field_validator('type')
    def validate_type(cls, v):
        if v.lower() not in config.TRANSACTION_TYPES:
            raise ValueError(f"Type must be one of {config.TRANSACTION_TYPES}")
        return v.lower()

    @field_validator('category')
    def validate_category(cls, v):
        if v not in config.TRANSACTION_CATEGORIES:
            raise ValueError(f"Category must be one of {config.TRANSACTION_CATEGORIES}")
        return v

    @field_validator('payment_method')
    def validate_payment(cls, v):
        if v not in config.PAYMENT_METHODS:
            raise ValueError(f"Payment method must be one of {config.PAYMENT_METHODS}")
        return v

class TransactionCreate(TransactionBase):
    pass

class TransactionOut(TransactionBase):
    id: int
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ----------------- INVESTMENTS ----------------- #
class InvestmentBase(BaseModel):
    name: str = Field(..., example="Nifty 50 Index Fund")
    category: str = Field(..., example="Mutual Funds")
    invested_amount: float = Field(..., gt=0, example=50000.0)
    current_value: float = Field(..., gt=0, example=62000.0)
    purchase_date: Optional[str] = Field(None, example="2025-01-10")
    notes: Optional[str] = None

    @field_validator('category')
    def validate_investment_category(cls, v):
        if v not in config.INVESTMENT_CATEGORIES:
            raise ValueError(f"Category must be one of {config.INVESTMENT_CATEGORIES}")
        return v

class InvestmentCreate(InvestmentBase):
    pass

class InvestmentOut(InvestmentBase):
    id: int
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ----------------- GOALS ----------------- #
class GoalBase(BaseModel):
    title: str = Field(..., example="Emergency Fund (6 Months)")
    target_amount: float = Field(..., gt=0, example=150000.0)
    current_amount: float = Field(0.0, ge=0, example=45000.0)
    target_date: str = Field(..., example="2026-12-31")
    category: Optional[str] = Field("Savings", example="Savings")
    
    @field_validator('category')
    def validate_goal_category(cls, v):
        if v not in config.GOAL_CATEGORIES:
            raise ValueError(f"Category must be one of {config.GOAL_CATEGORIES}")
        return v

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


# ----------------- AI / ML ----------------- #
class PredictCategoryRequest(BaseModel):
    text: str = Field(..., example="Ola cab ride to office")

class PredictCategoryResponse(BaseModel):
    text: str
    predicted_category: str
    confidence: float
    all_probabilities: dict

class TrainModelRequest(BaseModel):
    epochs: Optional[int] = 10
