"""
Train Local Machine Learning Model for Automatic Expense & Income Categorization
Zero external API calls: Pure Scikit-Learn TF-IDF + Multinomial Naive Bayes / Logistic Regression.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

MODEL_FILE = os.path.join(os.path.dirname(__file__), "expense_model.pkl")

# Comprehensive Financial Training Dataset
TRAINING_DATA = [
    # Food & Dining
    ("Swiggy food order biryani", "Food & Dining"),
    ("Zomato pizza delivery lunch", "Food & Dining"),
    ("Starbucks coffee latte cappuccino", "Food & Dining"),
    ("Dominos pizza cheesy burst", "Food & Dining"),
    ("McDonalds burger meal fries", "Food & Dining"),
    ("Restaurant dinner with family", "Food & Dining"),
    ("Chai and snacks tea stall", "Food & Dining"),
    ("KFC fried chicken bucket", "Food & Dining"),
    ("Breakfast at Sagar Ratna", "Food & Dining"),
    ("Cafe coffee day cold coffee", "Food & Dining"),
    ("Dineout table reservation food", "Food & Dining"),
    ("Subway sandwich combo", "Food & Dining"),

    # Groceries & Supplies
    ("Blinkit grocery vegetables fruits", "Groceries"),
    ("Zepto instant delivery milk curd bread", "Groceries"),
    ("Instamart eggs butter cheese", "Groceries"),
    ("BigBasket monthly grocery order", "Groceries"),
    ("Local vegetable vendor sabzi", "Groceries"),
    ("DMart monthly supermarket shopping", "Groceries"),
    ("Nature basket organic pulses spices", "Groceries"),
    ("Supermarket ration rice atta oil", "Groceries"),
    ("Dairy milk bread paneer daily purchase", "Groceries"),

    # Travel & Transportation
    ("Uber cab ride to airport", "Transportation"),
    ("Ola auto ride to metro station", "Transportation"),
    ("Rapido bike taxi ride", "Transportation"),
    ("Petrol pump fuel refilling Indian Oil", "Transportation"),
    ("Diesel refuel Bharat Petroleum", "Transportation"),
    ("Metro card smart recharge", "Transportation"),
    ("IRCTC train ticket booking reservation", "Transportation"),
    ("IndiGo flight air tickets", "Transportation"),
    ("Toll plaza Fastag auto recharge", "Transportation"),
    ("Bus ticket redBus booking", "Transportation"),
    ("Car wash and periodic servicing", "Transportation"),

    # Utilities & Bills
    ("Electricity bill payment BESCOM Tata Power", "Utilities"),
    ("Water supply bill municipal corporation", "Utilities"),
    ("Jio prepaid mobile recharge plan", "Utilities"),
    ("Airtel postpaid bill broadband fiber", "Utilities"),
    ("ACT Fibernet broadband wifi internet bill", "Utilities"),
    ("Indane HP Bharat gas cylinder booking refill", "Utilities"),
    ("DTH Tata Play dish TV recharge", "Utilities"),
    ("House maintenance society charges", "Utilities"),

    # Shopping & Lifestyle
    ("Amazon prime electronics gadget order", "Shopping"),
    ("Flipkart fashion shoes clothing order", "Shopping"),
    ("Myntra casual shirts jeans t-shirt", "Shopping"),
    ("Zara clothing retail store purchase", "Shopping"),
    ("Nykaa cosmetics skincare fragrance", "Shopping"),
    ("Croma electronics laptop accessory", "Shopping"),
    ("Decathlon sports shoes gym activewear", "Shopping"),
    ("H&M cotton apparel trousers", "Shopping"),
    ("Titan watch sunglasses purchase", "Shopping"),

    # Entertainment & Subscriptions
    ("Netflix monthly 4k subscription", "Entertainment"),
    ("Spotify premium music subscription", "Entertainment"),
    ("BookMyShow movie cinema tickets PVR", "Entertainment"),
    ("Amazon Prime annual membership", "Entertainment"),
    ("Disney Hotstar cricket streaming subscription", "Entertainment"),
    ("PlayStation Sony gaming pass store", "Entertainment"),
    ("Concert live show entry tickets", "Entertainment"),
    ("Bowling amusement theme park tickets", "Entertainment"),

    # Health & Medical
    ("Apollo pharmacy medicines prescription", "Healthcare"),
    ("1mg health checkup blood test diagnostics", "Healthcare"),
    ("Doctor consultation fee clinic hospital", "Healthcare"),
    ("Dentist teeth cleaning dental root canal", "Healthcare"),
    ("Netmeds multivitamin protein supplement", "Healthcare"),
    ("Max hospital health insurance copay", "Healthcare"),
    ("Eye test spec lens Lenskart frame", "Healthcare"),

    # Housing & Rent
    ("Monthly apartment flat house rent payment", "Housing & Rent"),
    ("Home loan EMI monthly deduction SBI HDFC", "Housing & Rent"),
    ("Security deposit for house leasing", "Housing & Rent"),
    ("Brokerage fee for rental house", "Housing & Rent"),

    # Investments & Savings
    ("Zerodha stock equity shares purchase", "Investments"),
    ("Groww mutual fund monthly SIP deduction", "Investments"),
    ("Public Provident Fund PPF contribution", "Investments"),
    ("Fixed Deposit FD creation bank tenure", "Investments"),
    ("Digital Sovereign Gold SGB bonds buy", "Investments"),
    ("NPS National Pension Scheme deposit", "Investments"),

    # Income (Inflow)
    ("Monthly corporate salary credit payroll", "Salary & Income"),
    ("Freelance client project payment invoice", "Salary & Income"),
    ("Quarterly stock dividend payout", "Salary & Income"),
    ("Interest credited savings account bank", "Salary & Income"),
    ("Bonus performance incentive received", "Salary & Income"),
    ("Consulting advisory payout remuneration", "Salary & Income")
]


def train_and_save_model():
    """Trains the NLP Classification pipeline and evaluates accuracy."""
    df = pd.DataFrame(TRAINING_DATA, columns=["text", "category"])
    
    X = df["text"]
    y = df["category"]

    # Split dataset for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scikit-Learn Pipeline: TF-IDF Vectorizer + Multinomial Naive Bayes
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), stop_words='english', lowercase=True)),
        ('clf', MultinomialNB(alpha=0.1))
    ])

    pipeline.fit(X_train, y_train)

    # Evaluation
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Local Model Trained Successfully!")
    print(f"📊 Test Accuracy: {acc * 100:.2f}%")

    # Retrain on full dataset for maximum production coverage
    pipeline.fit(X, y)

    # Save to disk
    joblib.dump(pipeline, MODEL_FILE)
    print(f"💾 Model saved to: {MODEL_FILE}")
    return pipeline


def load_model():
    """Loads the trained model from disk or trains it if missing."""
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    return train_and_save_model()


def predict_category(text: str):
    """Predicts category and probability distribution for a text."""
    model = load_model()
    pred = model.predict([text])[0]
    probs = model.predict_proba([text])[0]
    classes = model.classes_
    
    prob_dict = {cls: round(float(prob), 4) for cls, prob in zip(classes, probs)}
    confidence = round(float(np.max(probs)), 4)
    
    return {
        "text": text,
        "predicted_category": pred,
        "confidence": confidence,
        "all_probabilities": prob_dict
    }


if __name__ == "__main__":
    train_and_save_model()
    
    # Test queries
    test_queries = [
        "Swiggy lunch combo",
        "Uber cab to office",
        "Electricity bill BESCOM",
        "Groww SIP mutual fund installment",
        "Monthly salary credited by employer"
    ]
    print("\n--- Test Predictions ---")
    for q in test_queries:
        res = predict_category(q)
        print(f"'{q}' ➡️ {res['predicted_category']} (Conf: {res['confidence'] * 100:.1f}%)")
