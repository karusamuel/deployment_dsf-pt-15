"""Load the trained model, score customers and explain each score."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features import CATEGORICAL, LABELS, NUMERIC, prepare

ROOT = Path(__file__).resolve().parents[1]          # repo root: relative paths work anywhere
MODEL_PATH = ROOT / "models" / "churn_v1.joblib"
SAMPLE_PATH = ROOT / "data" / "sample.csv"


def load_model(path: Path = MODEL_PATH) -> dict:
    """The saved file is a dict: pipeline + baseline + metadata (see index.ipynb)."""
    return joblib.load(path)


def predict_risk(bundle: dict, df: pd.DataFrame) -> np.ndarray:
    """Probability that each customer churns in the next period."""
    return bundle["pipeline"].predict_proba(prepare(df))[:, 1]


def risk_band(p: float, high: float = 0.6, medium: float = 0.3) -> str:
    return "High" if p >= high else "Medium" if p >= medium else "Low"


def _feature_groups(pipeline) -> list[str]:
    """Map every transformed column back to the original feature it came from."""
    ohe = pipeline.named_steps["prep"].named_transformers_["cat"]
    groups = list(NUMERIC)
    for col, cats in zip(CATEGORICAL, ohe.categories_):
        groups += [col] * len(cats)
    return groups


def contributions(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Per-customer, per-feature push on the log-odds of churn vs. an average customer.

    For a logistic regression this is exact: coef * (x - training mean),
    summed over the one-hot columns of each original feature.
    Positive = pushes risk up, negative = pushes risk down.
    """
    pipe = bundle["pipeline"]
    xt = pipe.named_steps["prep"].transform(prepare(df))
    coef = pipe.named_steps["clf"].coef_[0]
    raw = (xt - bundle["baseline"]) * coef
    out = pd.DataFrame(raw, columns=_feature_groups(pipe), index=df.index)
    return out.T.groupby(level=0).sum().T


def top_reasons(bundle: dict, row: pd.Series, contrib: pd.Series, n: int = 3) -> list[str]:
    """The n strongest risk drivers for one customer, in plain words."""
    out = []
    for feat in contrib.sort_values(ascending=False).index[:n]:
        if contrib[feat] <= 0:
            break
        value = row[feat]
        if feat in NUMERIC:
            value = f"{float(value):,.0f}" if feat == "tenure" else f"${float(value):,.2f}"
        out.append(f"{LABELS[feat]}: {value}")
    return out or ["No strong risk drivers"]
