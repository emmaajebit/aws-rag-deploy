"""Loaders that normalise any source to: review, rating, product, timestamp."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import RATING_COL, TEXT_COL

STANDARD_COLS = [TEXT_COL, RATING_COL, "product", "timestamp"]


def _finalise(df: pd.DataFrame) -> pd.DataFrame:
    for c in ("product", "timestamp"):
        if c not in df.columns:
            df[c] = pd.NA
    df = df[STANDARD_COLS].copy()
    df[RATING_COL] = pd.to_numeric(df[RATING_COL], errors="coerce")
    df = df.dropna(subset=[TEXT_COL, RATING_COL])
    df[TEXT_COL] = df[TEXT_COL].astype(str)
    df = df[df[TEXT_COL].str.strip().str.len() > 0]
    df = df[df[RATING_COL].between(1, 5)]
    df[RATING_COL] = df[RATING_COL].round().astype(int)
    return df.reset_index(drop=True)


def load_csv(path, columns: dict | None = None, nrows: int | None = None, **kw) -> pd.DataFrame:
    """Generic CSV. `columns` maps source column -> standard name."""
    df = pd.read_csv(path, nrows=nrows, **kw)
    if columns:
        df = df.rename(columns=columns)
    missing = {TEXT_COL, RATING_COL} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns {sorted(missing)}; pass columns={{src: dst}}")
    return _finalise(df)


def load_jsonl(path, columns: dict | None = None, nrows: int | None = None) -> pd.DataFrame:
    df = pd.read_json(path, lines=True, nrows=nrows)
    if columns:
        df = df.rename(columns=columns)
    return _finalise(df)


def load_amazon_fine_food(path, nrows: int | None = None) -> pd.DataFrame:
    """Kaggle / SNAP 'Amazon Fine Food Reviews' Reviews.csv (Text, Score, ProductId, Time[unix])."""
    df = pd.read_csv(path, nrows=nrows)
    df = df.rename(columns={"Text": TEXT_COL, "Score": RATING_COL, "ProductId": "product"})
    if "Time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["Time"], unit="s").dt.strftime("%Y-%m-%d")
    return _finalise(df)


def load_hf(name: str, split: str = "train", text_col: str = "text", rating_col: str = "label",
            label_offset: int = 0, nrows: int | None = None) -> pd.DataFrame:
    """Hugging Face `datasets` source (needs `pip install datasets`). label_offset=+1 for 0-4 star labels."""
    from datasets import load_dataset  # lazy

    ds = load_dataset(name, split=split)
    df = ds.to_pandas()
    if nrows:
        df = df.head(nrows)
    df = df.rename(columns={text_col: TEXT_COL, rating_col: RATING_COL})
    df[RATING_COL] = pd.to_numeric(df[RATING_COL]) + label_offset
    return _finalise(df)


def load_reviews(source: str, path: str | Path | None = None, n: int = 5000, seed: int = 42,
                 nrows: int | None = None, **kw) -> pd.DataFrame:
    """source: synthetic | amazon | csv | jsonl | hf"""
    if source == "synthetic":
        from .synthetic import generate_reviews
        return _finalise(generate_reviews(n=n, seed=seed))
    if source in {"amazon", "csv", "jsonl"} and path is None:
        raise ValueError(f"--path is required for source '{source}'")
    if source == "amazon":
        return load_amazon_fine_food(path, nrows=nrows)
    if source == "csv":
        return load_csv(path, nrows=nrows, **kw)
    if source == "jsonl":
        return load_jsonl(path, nrows=nrows, **kw)
    if source == "hf":
        return load_hf(path, nrows=nrows, **kw)
    raise ValueError(f"unknown source: {source}")
