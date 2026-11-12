"""Classical ML models (spec step 6) with imbalance handling and tuning grids (spec step 8)."""
from __future__ import annotations

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.class_weight import compute_sample_weight

from .config import SEED
from .features import make_vectorizer

MODEL_NAMES = ["lr", "nb", "rf", "xgb"]
IMBALANCE_MODES = ["none", "class_weight", "smote"]


class WeightedFit(BaseEstimator, ClassifierMixin):
    """Gives estimators without a `class_weight` option (XGBoost) balanced sample weights."""

    def __init__(self, estimator=None, balanced: bool = True):
        self.estimator, self.balanced = estimator, balanced

    def fit(self, X, y):
        est = clone(self.estimator)
        if self.balanced:
            est.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
        else:
            est.fit(X, y)
        self.estimator_ = est
        self.classes_ = est.classes_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


def make_classifier(name: str, imbalance: str = "class_weight", seed: int = SEED):
    cw = imbalance == "class_weight"
    if name == "lr":
        return LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced" if cw else None, random_state=seed)
    if name == "nb":
        # No class_weight in MultinomialNB: a uniform class prior is the equivalent re-balancing.
        return MultinomialNB(alpha=0.5, fit_prior=not cw)
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample" if cw else None,
                                      n_jobs=1, random_state=seed)
    if name == "xgb":
        from xgboost import XGBClassifier

        base = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.9,
                             colsample_bytree=0.8, objective="multi:softprob", tree_method="hist",
                             eval_metric="mlogloss", n_jobs=1, random_state=seed)
        return WeightedFit(base, balanced=cw)
    raise ValueError(f"unknown model '{name}' ({MODEL_NAMES})")


def build_pipeline(model: str, vectorizer: str = "tfidf", imbalance: str = "class_weight",
                   seed: int = SEED, max_features: int = 5000):
    """vec -> [SMOTE (train folds only)] -> clf. Everything is fitted inside CV folds, so no leakage."""
    if imbalance not in IMBALANCE_MODES:
        raise ValueError(f"imbalance must be one of {IMBALANCE_MODES}")
    if model == "nb" and vectorizer == "w2v":
        raise ValueError("MultinomialNB cannot use Word2Vec features (negative values)")
    steps = [("vec", make_vectorizer(vectorizer, max_features=max_features))]
    if imbalance == "smote":
        steps.append(("smote", SMOTE(random_state=seed)))
    steps.append(("clf", make_classifier(model, imbalance, seed)))
    return ImbPipeline(steps)


def param_grid(model: str, vectorizer: str = "tfidf", fast: bool = False) -> dict:
    grids = {
        "lr": {"clf__C": [0.3, 1, 3, 10]},
        "nb": {"clf__alpha": [0.1, 0.5, 1.0]},
        "rf": {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 40]},
        "xgb": {"clf__estimator__max_depth": [4, 6], "clf__estimator__n_estimators": [150, 300]},
    }
    g = dict(grids[model])
    if vectorizer in ("tfidf", "bow"):
        g["vec__ngram_range"] = [(1, 1), (1, 2)]
    if fast:  # smoke-test sized
        g = {k: v[:2] for k, v in g.items()}
    return g
