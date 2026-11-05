"""Feature preparation shared by training (index.ipynb) and the app.

The app must prepare data exactly as training did, so both import from here.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Public dataset: IBM Telco Customer Churn (7,043 customers, one row each).
DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

ID_COL = "customerID"
TARGET = "Churn"

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUMERIC + CATEGORICAL
REQUIRED_COLUMNS = [ID_COL] + FEATURES

# Friendly names used when explaining a prediction in plain words.
LABELS = {
    "tenure": "Months as a customer",
    "MonthlyCharges": "Monthly bill",
    "TotalCharges": "Total billed to date",
    "gender": "Gender",
    "SeniorCitizen": "Senior citizen",
    "Partner": "Has a partner",
    "Dependents": "Has dependents",
    "PhoneService": "Phone service",
    "MultipleLines": "Multiple lines",
    "InternetService": "Internet service",
    "OnlineSecurity": "Online security add-on",
    "OnlineBackup": "Online backup add-on",
    "DeviceProtection": "Device protection add-on",
    "TechSupport": "Tech support add-on",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming movies",
    "Contract": "Contract",
    "PaperlessBilling": "Paperless billing",
    "PaymentMethod": "Payment method",
}


def validate(df: pd.DataFrame) -> list[str]:
    """Return a list of human-readable problems with an uploaded table (empty = OK)."""
    problems = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        problems.append("Missing columns: " + ", ".join(missing))
    if df.empty:
        problems.append("The file has no rows.")
    for col in NUMERIC:
        if col in df.columns:
            bad = pd.to_numeric(df[col], errors="coerce").isna() & df[col].astype(str).str.strip().ne("")
            if bad.any():
                problems.append(f"Column '{col}' must be numeric ({int(bad.sum())} bad values).")
    return problems


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix the raw dataset's quirks. Safe to call on raw or already-clean data."""
    df = df.copy()
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Brand-new customers (tenure 0) have a blank TotalCharges: they have been billed nothing yet.
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    # Raw data codes SeniorCitizen as 0/1; use Yes/No like every other flag.
    if pd.api.types.is_numeric_dtype(df["SeniorCitizen"]):
        df["SeniorCitizen"] = df["SeniorCitizen"].map({1: "Yes", 0: "No"})
    for col in CATEGORICAL:
        df[col] = df[col].astype(str).str.strip()
    return df


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Turn a customer table into the model's input matrix."""
    return clean(df)[FEATURES]


def build_pipeline() -> Pipeline:
    """Preprocessing + model in one object, so it is saved and loaded together."""
    prep = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ]
    )
    return Pipeline(
        [
            ("prep", prep),
            ("clf", LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")),
        ]
    )
