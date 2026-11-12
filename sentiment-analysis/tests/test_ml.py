import json

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from sentiment.config import LABELS
from sentiment.evaluation import evaluate
from sentiment.features import make_vectorizer
from sentiment.models_ml import MODEL_NAMES, build_pipeline, param_grid
from sentiment.predictor import SentimentPredictor
from sentiment.train_ml import as_text_array, select_best, split_data


def test_split_stratified_and_disjoint(small_df):
    tr, te, ytr, yte = split_data(small_df)
    assert not set(tr.index) & set(te.index)
    assert not set(tr["review"].str.lower()) & set(te["review"].str.lower())   # raw dupes removed before split
    assert abs(ytr.mean() - yte.mean()) < 0.03


@pytest.mark.parametrize("model", MODEL_NAMES)
@pytest.mark.parametrize("imb", ["none", "class_weight", "smote"])
def test_every_model_and_imbalance_mode_fits(small_df, model, imb):
    sub = small_df.sample(600, random_state=0)
    X, y = as_text_array(sub["cleaned_review"]), sub["sentiment"].map({l: i for i, l in enumerate(LABELS)}).values
    p = build_pipeline(model, "tfidf", imb).fit(X, y)
    proba = p.predict_proba(X[:20])
    assert proba.shape == (20, 3) and np.allclose(proba.sum(1), 1, atol=1e-5)


@pytest.mark.parametrize("model", MODEL_NAMES)
def test_param_grid_keys_are_valid(model):
    p = build_pipeline(model)
    p.set_params(**{k: v[0] for k, v in param_grid(model).items()})   # raises on a bad key


def test_nb_rejects_w2v():
    with pytest.raises(ValueError):
        build_pipeline("nb", "w2v")


def test_smote_not_applied_at_predict(small_df):
    sub = small_df.sample(500, random_state=1)
    X, y = as_text_array(sub["cleaned_review"]), sub["sentiment"].map({l: i for i, l in enumerate(LABELS)}).values
    p = build_pipeline("lr", imbalance="smote").fit(X, y)
    assert p.predict(X[:7]).shape == (7,)   # would error/resize if resampling leaked into inference


def test_w2v_vectorizer(small_df):
    pytest.importorskip("gensim")
    X = as_text_array(small_df["cleaned_review"].head(300))
    v = make_vectorizer("w2v").fit(X)
    out = v.transform(X)
    assert out.shape == (300, 100) and np.isfinite(out).all()
    assert (v.transform(["zzzunknownword"]) == 0).all()


def test_evaluate_perfect_and_auc():
    y = np.array([0, 1, 2, 2, 0, 1])
    proba = np.eye(3)[y]
    m = evaluate(y, y, proba)
    assert m["accuracy"] == 1 and m["f1_macro"] == 1 and m["roc_auc_ovr_macro"] == 1 and m["roc_auc_pos_vs_neg"] == 1


def test_experiment_outputs_and_predictor(trained_dir):
    meta = json.loads((trained_dir / "metadata.json").read_text())
    assert meta["selected_by"].startswith("cv_f1_macro") and meta["preprocessing"] == {"nltk_stopwords": True, "nltk_wordnet": True}
    assert (trained_dir.parent / "model_comparison.csv").exists()
    pred = SentimentPredictor.load(trained_dir)
    res = pred.predict(["Absolutely loved the pizza!", "Burger was cold and soggy", "It was okay, nothing special"])
    assert [r["sentiment"] for r in res] == ["positive", "negative", "neutral"]
    assert abs(sum(res[0]["probabilities"].values()) - 1) < 1e-3


def test_predictor_missing_model(tmp_path):
    with pytest.raises(FileNotFoundError):
        SentimentPredictor.load(tmp_path)


def test_select_best_uses_cv_not_test():
    t = pd.DataFrame({"model": ["a", "b"], "cv_f1_macro": [0.90, 0.80], "test_f1_macro": [0.50, 0.99]})
    assert select_best(t) == "a"


def test_smote_resamples_only_inside_fit(small_df):
    """The classifier must see a balanced training set under SMOTE (and the raw skew otherwise)."""
    from sklearn.base import BaseEstimator, ClassifierMixin
    seen = {}

    class Spy(BaseEstimator, ClassifierMixin):
        def fit(self, X, y):
            seen["counts"] = np.bincount(y); self.classes_ = np.unique(y); return self
        def predict_proba(self, X):
            return np.full((X.shape[0], 3), 1 / 3)

    X, y = as_text_array(small_df["cleaned_review"]), small_df["sentiment"].map({l: i for i, l in enumerate(LABELS)}).values
    build_pipeline("lr", imbalance="smote").set_params(clf=Spy()).fit(X, y)
    assert len(set(seen["counts"])) == 1 and seen["counts"][0] == np.bincount(y).max()
    build_pipeline("lr", imbalance="class_weight").set_params(clf=Spy()).fit(X, y)
    assert not len(set(seen["counts"])) == 1    # no resampling unless asked


def test_predictor_cleans_text_exactly_like_training():
    from sentiment.preprocessing import clean_text
    got = {}

    class Stub:
        def predict_proba(self, X):
            got["x"] = list(X); return np.array([[0.1, 0.1, 0.8]] * len(X))

    SentimentPredictor(Stub(), {"labels": LABELS}, "sklearn").predict(["The Burger was <b>NOT</b> good!!!"])
    assert got["x"] == [clean_text("The Burger was <b>NOT</b> good!!!")]
