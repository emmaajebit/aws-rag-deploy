"""Central configuration: paths, label scheme, defaults."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(os.getenv("SENTIMENT_HOME", Path(__file__).resolve().parents[2]))
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
BEST_MODEL_DIR = ARTIFACTS_DIR / "best_model"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Alphabetical so that class id order == sorted label order (sklearn convention).
LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

SEED = 42
TEXT_COL = "review"
RATING_COL = "rating"
CLEAN_COL = "cleaned_review"
LABEL_COL = "sentiment"

MAX_REVIEW_CHARS = 5000


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)
