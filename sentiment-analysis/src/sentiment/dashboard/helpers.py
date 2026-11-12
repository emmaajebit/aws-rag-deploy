"""Pure data helpers for the dashboard (importable and testable without Streamlit)."""
from __future__ import annotations

import pandas as pd

from ..config import LABELS, RATING_COL, TEXT_COL
from ..predictor import SentimentPredictor


def score_reviews(df: pd.DataFrame, predictor: SentimentPredictor, batch: int = 512) -> pd.DataFrame:
    """Adds `pred_sentiment`, `confidence`, `p_negative/neutral/positive` to a review DataFrame."""
    out = df.copy()
    probas = []
    texts = out[TEXT_COL].astype(str).tolist()
    for i in range(0, len(texts), batch):
        probas.append(predictor.predict_proba(texts[i:i + batch]))
    import numpy as np
    P = np.vstack(probas) if probas else np.zeros((0, 3))
    for j, lab in enumerate(LABELS):
        out[f"p_{lab}"] = P[:, j]
    out["pred_sentiment"] = [LABELS[k] for k in P.argmax(1)] if len(P) else []
    out["confidence"] = P.max(1) if len(P) else []
    return out


def top_reviews(scored: pd.DataFrame, label: str, n: int = 10) -> pd.DataFrame:
    cols = [c for c in (TEXT_COL, RATING_COL, "product", "confidence") if c in scored.columns]
    sub = scored[scored["pred_sentiment"] == label].sort_values("confidence", ascending=False)
    return sub[cols].head(n).reset_index(drop=True)


def product_sentiment(scored: pd.DataFrame, min_reviews: int = 5) -> pd.DataFrame:
    """Share of each predicted sentiment per product (rows sum to 1) + net sentiment score."""
    if "product" not in scored.columns or scored["product"].isna().all():
        return pd.DataFrame()
    g = scored.groupby("product")["pred_sentiment"]
    counts = g.value_counts().unstack(fill_value=0).reindex(columns=LABELS, fill_value=0)
    counts["n"] = counts.sum(axis=1)
    counts = counts[counts["n"] >= min_reviews]
    shares = counts[LABELS].div(counts["n"], axis=0)
    shares["n"] = counts["n"]
    shares["net_sentiment"] = shares["positive"] - shares["negative"]
    return shares.sort_values("net_sentiment")


def monthly_trend(scored: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in scored.columns:
        return pd.DataFrame()
    t = pd.to_datetime(scored["timestamp"], errors="coerce")
    d = scored.assign(month=t.dt.to_period("M").dt.to_timestamp()).dropna(subset=["month"])
    if d.empty:
        return pd.DataFrame()
    m = d.groupby(["month", "pred_sentiment"]).size().unstack(fill_value=0).reindex(columns=LABELS, fill_value=0)
    return m.div(m.sum(axis=1), axis=0).reset_index()
