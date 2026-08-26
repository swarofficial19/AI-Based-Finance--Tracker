from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# SQLite local database file
DB_FILE = os.path.join(os.path.dirname(__file__), "finance.db")
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_FILE}"

# Connect args needed for SQLite in multithreaded FastAPI context
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Dependency to get DB session per request
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
