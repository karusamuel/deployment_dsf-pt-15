"""Turn saved test-set predictions into the curves shown on the About page.

index.ipynb stores the held-out labels and scores in the model bundle
(bundle["meta"]["evaluation"]); these functions are pure, so the notebook and app agree.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def test_frame(bundle: dict) -> pd.DataFrame | None:
    """Held-out customers with their true outcome (`churn`) and model score (`risk`)."""
    ev = bundle["meta"].get("evaluation")
    return None if ev is None else pd.DataFrame(ev["test"])


def gain_curve(y, p, points: int = 101) -> pd.DataFrame:
    """Call the riskiest X% of customers: what share of all churners do we reach?"""
    y = np.asarray(y)[np.argsort(-np.asarray(p), kind="stable")]
    reached = np.concatenate([[0], np.cumsum(y) / y.sum()])
    share = np.linspace(0, 1, points)
    idx = np.round(share * len(y)).astype(int)
    return pd.DataFrame({"share_called": share, "model": reached[idx], "random": share})


def threshold_table(y, p, thresholds=None) -> pd.DataFrame:
    """Precision, recall and list size for each call-list threshold."""
    y, p = np.asarray(y), np.asarray(p)
    if thresholds is None:
        thresholds = np.round(np.arange(0.1, 0.91, 0.05), 2)
    rows = []
    for t in thresholds:
        flag = p >= t
        rows.append({
            "threshold": float(t),
            "share_called": flag.mean(),
            "precision": y[flag].mean() if flag.any() else np.nan,
            "recall": flag[y == 1].mean(),
        })
    return pd.DataFrame(rows)


def calibration_bins(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Average score vs. real churn rate, in equal-width score bins."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["bin"] = pd.cut(df["p"], np.linspace(0, 1, n_bins + 1), include_lowest=True)
    out = df.groupby("bin", observed=True).agg(predicted=("p", "mean"), actual=("y", "mean"), n=("y", "size"))
    return out.reset_index(drop=True)


def segment_table(test: pd.DataFrame, col: str, threshold: float) -> pd.DataFrame:
    """Per-segment churn rate, share flagged, recall and ranking quality."""
    flagged = test["risk"] >= threshold
    rows = []
    for value, d in test.groupby(col):
        f = flagged.loc[d.index]
        rows.append({
            "segment": str(value),
            "customers": len(d),
            "churn_rate": d["churn"].mean(),
            "share_flagged": f.mean(),
            "recall": f[d["churn"] == 1].mean() if d["churn"].any() else np.nan,
            "roc_auc": roc_auc_score(d["churn"], d["risk"]) if d["churn"].nunique() == 2 else np.nan,
        })
    return pd.DataFrame(rows)
