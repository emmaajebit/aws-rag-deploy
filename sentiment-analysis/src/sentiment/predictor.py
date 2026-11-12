"""One inference interface over sklearn pipelines, DL models and fine-tuned BERT."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import BEST_MODEL_DIR, LABELS, MAX_REVIEW_CHARS
import warnings

from .preprocessing import clean_text, resources_status


class SentimentPredictor:
    def __init__(self, backend, meta: dict, kind: str):
        self._backend, self.meta, self.kind = backend, meta, kind

    @classmethod
    def load(cls, model_dir: str | Path = BEST_MODEL_DIR) -> "SentimentPredictor":
        model_dir = Path(model_dir)
        meta_path = model_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"No model at {model_dir}. Train one first: `sentiment train`.")
        meta = json.loads(meta_path.read_text())
        kind = meta["kind"]
        trained_with = meta.get("preprocessing")
        if trained_with and trained_with != resources_status():
            warnings.warn(f"Preprocessing resources differ from training time (trained with {trained_with}, now {resources_status()}); predictions may degrade.", RuntimeWarning)
        if kind == "sklearn":
            import joblib
            return cls(joblib.load(model_dir / "model.joblib"), meta, kind)
        if kind == "bert":
            from .bert import BertPredictor
            return cls(BertPredictor(model_dir), meta, kind)
        if kind == "dl":
            from .models_dl import DLPredictor
            return cls(DLPredictor(model_dir), meta, kind)
        raise ValueError(f"unknown model kind '{kind}'")

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        texts = [(t or "")[:MAX_REVIEW_CHARS] for t in texts]
        if self.kind == "sklearn":  # same clean_text() as training -> no train/serve skew
            return self._backend.predict_proba([clean_text(t) for t in texts])
        return self._backend.predict_proba(texts)

    def predict(self, texts: list[str]) -> list[dict]:
        proba = self.predict_proba(texts)
        out = []
        for p in proba:
            i = int(np.argmax(p))
            out.append({"sentiment": LABELS[i], "confidence": round(float(p[i]), 4),
                        "probabilities": {l: round(float(x), 4) for l, x in zip(LABELS, p)}})
        return out

    @property
    def description(self) -> str:
        m = self.meta
        return m.get("model") or m.get("arch") or m.get("base_model") or self.kind
