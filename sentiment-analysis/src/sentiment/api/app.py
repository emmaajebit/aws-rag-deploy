"""FastAPI service (spec step 9): POST a review, get its sentiment.

    SENTIMENT_MODEL_DIR=artifacts/best_model uvicorn sentiment.api.app:app --port 8000
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from .. import __version__
from ..config import BEST_MODEL_DIR, MAX_REVIEW_CHARS
from ..predictor import SentimentPredictor

MAX_BATCH = 100


class ReviewIn(BaseModel):
    review: str = Field(..., min_length=1, max_length=MAX_REVIEW_CHARS)


class BatchIn(BaseModel):
    reviews: list[str] = Field(..., min_length=1, max_length=MAX_BATCH)


class Prediction(BaseModel):
    sentiment: str
    confidence: float
    probabilities: dict[str, float]


class PredictOut(Prediction):
    review: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_dir = os.getenv("SENTIMENT_MODEL_DIR", str(BEST_MODEL_DIR))
    try:
        app.state.predictor = SentimentPredictor.load(model_dir)
    except FileNotFoundError:
        app.state.predictor = None  # /health reports it; /predict returns 503
    yield


app = FastAPI(title="Review Sentiment API", version=__version__, lifespan=lifespan)


def _predictor(request: Request) -> SentimentPredictor:
    p = request.app.state.predictor
    if p is None:
        raise HTTPException(503, "Model not loaded. Train one (`sentiment train`) and set SENTIMENT_MODEL_DIR.")
    return p


@app.get("/health")
def health(request: Request):
    p = request.app.state.predictor
    return {"status": "ok" if p else "degraded", "model_loaded": p is not None}


@app.get("/model-info")
def model_info(request: Request):
    p = _predictor(request)
    return {"kind": p.kind, "model": p.description, "labels": p.meta.get("labels"),
            "test_metrics": p.meta.get("test_metrics"), "version": __version__}


@app.post("/predict", response_model=PredictOut)
def predict(body: ReviewIn, request: Request):
    if not body.review.strip():
        raise HTTPException(422, "review must contain non-whitespace text")
    res = _predictor(request).predict([body.review])[0]
    return {"review": body.review, **res}


@app.post("/predict/batch", response_model=list[PredictOut])
def predict_batch(body: BatchIn, request: Request):
    texts = [t for t in body.reviews]
    if any(not t.strip() or len(t) > MAX_REVIEW_CHARS for t in texts):
        raise HTTPException(422, f"each review must be non-empty and <= {MAX_REVIEW_CHARS} chars")
    return [{"review": t, **r} for t, r in zip(texts, _predictor(request).predict(texts))]
