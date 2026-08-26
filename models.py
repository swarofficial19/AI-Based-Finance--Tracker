from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.sql import func
from database import Base

class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    title = Column(String(255), nullable=False)
    amount = Column(Float, nullable=False)
    type = Column(String(50), nullable=False) # 'expense' or 'income'
    category = Column(String(100), nullable=False) # e.g. Food, Rent, Salary
    date = Column(String(50), nullable=False) # YYYY-MM-DD
    payment_method = Column(String(50), default="UPI") # UPI, Cash, Card, NetBanking
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Investment(Base):
    __tablename__ = "investments"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    category = Column(String(100), nullable=False) # Mutual Funds, Stocks, Gold, PPF, Crypto
    invested_amount = Column(Float, nullable=False)
    current_value = Column(Float, nullable=False)
    purchase_date = Column(String(50), nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Goal(Base):
    __tablename__ = "goals"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    title = Column(String(255), nullable=False)
    target_amount = Column(Float, nullable=False)
    current_amount = Column(Float, default=0.0)
    target_date = Column(String(50), nullable=False) # YYYY-MM-DD
    category = Column(String(100), default="Savings")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
