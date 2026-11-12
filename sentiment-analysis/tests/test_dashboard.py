import pandas as pd
import pytest

from sentiment.dashboard.helpers import monthly_trend, product_sentiment, score_reviews, top_reviews
from sentiment.data.loaders import load_reviews
from sentiment.predictor import SentimentPredictor


@pytest.fixture(scope="module")
def scored(trained_dir):
    return score_reviews(load_reviews("synthetic", n=600, seed=3), SentimentPredictor.load(trained_dir))


def test_score_and_top(scored):
    assert {"pred_sentiment", "confidence", "p_positive"} <= set(scored.columns)
    top = top_reviews(scored, "negative", 5)
    assert len(top) == 5 and (top["confidence"].diff().dropna() <= 1e-9).all()  # sorted desc


def test_product_and_trend(scored):
    ps = product_sentiment(scored, min_reviews=5)
    assert ((ps[["negative", "neutral", "positive"]].sum(axis=1) - 1).abs() < 1e-9).all()
    tr = monthly_trend(scored)
    assert ((tr[["negative", "neutral", "positive"]].sum(axis=1) - 1).abs() < 1e-9).all()


def test_helpers_handle_missing_columns(scored):
    bare = scored.drop(columns=["product", "timestamp"])
    assert product_sentiment(bare).empty and monthly_trend(bare).empty


def test_streamlit_app_runs_and_predicts(trained_dir, monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("SENTIMENT_MODEL_DIR", str(trained_dir))
    app_path = Path(__file__).parents[1] / "src/sentiment/dashboard/app.py"
    at = AppTest.from_file(str(app_path), default_timeout=120).run()
    assert not at.exception, at.exception
    assert len(at.metric) == 4
    at.text_area[0].input("Burger was cold and soggy").run()
    at.button[0].click().run()
    assert not at.exception and any("NEGATIVE" in m.value for m in at.markdown)


def test_streamlit_app_without_model(tmp_path, monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("SENTIMENT_MODEL_DIR", str(tmp_path))
    at = AppTest.from_file(str(Path(__file__).parents[1] / "src/sentiment/dashboard/app.py")).run()
    assert at.error and "No trained model" in at.error[0].value
