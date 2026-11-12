"""Train / compare / tune classical models; persist the best one (spec steps 6-8)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split  # noqa: E402

from . import __version__  # noqa: E402
from .config import (ARTIFACTS_DIR, CLEAN_COL, FIGURES_DIR, LABEL2ID, LABEL_COL, LABELS, SEED,  # noqa: E402
                     TEXT_COL)
from .evaluation import evaluate, plot_confusion_matrix, top_features  # noqa: E402
from .models_ml import MODEL_NAMES, build_pipeline, param_grid  # noqa: E402
from .preprocessing import resources_status  # noqa: E402


def as_text_array(series: pd.Series) -> np.ndarray:
    """Plain object ndarray. pandas>=3 string columns are Arrow-backed; `.values` then breaks sklearn CV slicing."""
    return series.astype(object).to_numpy()


def split_data(df: pd.DataFrame, test_size: float = 0.2, seed: int = SEED):
    """Stratified train/test split; the test set is touched exactly once, for final reporting."""
    y = df[LABEL_COL].map(LABEL2ID)
    return train_test_split(df, y, test_size=test_size, stratify=y, random_state=seed)


def run_experiment(df: pd.DataFrame, models=tuple(MODEL_NAMES), vectorizer="tfidf", imbalance="class_weight",
                   tune=False, cv=5, test_size=0.2, fast=False, out_dir: Path = ARTIFACTS_DIR,
                   fig_dir: Path = FIGURES_DIR, seed=SEED, verbose=True) -> dict:
    out_dir, fig_dir = Path(out_dir), Path(fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True); fig_dir.mkdir(parents=True, exist_ok=True)
    tr, te, y_tr, y_te = split_data(df, test_size, seed)
    X_tr, X_te = as_text_array(tr[CLEAN_COL]), as_text_array(te[CLEAN_COL])
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)

    results, fitted = [], {}
    for name in models:
        t0 = time.time()
        pipe = build_pipeline(name, vectorizer, imbalance, seed)
        if tune:
            gs = GridSearchCV(pipe, param_grid(name, vectorizer, fast), scoring="f1_macro", cv=skf, n_jobs=-1, refit=True)
            gs.fit(X_tr, y_tr)
            model, cv_f1, best_params = gs.best_estimator_, float(gs.best_score_), {k: str(v) for k, v in gs.best_params_.items()}
        else:
            scores = cross_val_score(pipe, X_tr, y_tr, scoring="f1_macro", cv=skf, n_jobs=-1)
            cv_f1, best_params = float(scores.mean()), {}
            model = pipe.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)
        pred = proba.argmax(1)
        m = evaluate(y_te, pred, proba)
        row = {"model": name, "vectorizer": vectorizer, "imbalance": imbalance, "tuned": tune,
               "cv_f1_macro": cv_f1, "test_accuracy": m["accuracy"], "test_precision_macro": m["precision_macro"],
               "test_recall_macro": m["recall_macro"], "test_f1_macro": m["f1_macro"],
               "test_roc_auc_ovr": m.get("roc_auc_ovr_macro"), "fit_seconds": round(time.time() - t0, 1),
               "best_params": best_params}
        results.append(row); fitted[name] = (model, m)
        plot_confusion_matrix(m["confusion_matrix"], f"{name} ({vectorizer}, {imbalance})", fig_dir / f"cm_{name}.png")
        if verbose:
            print(f"[{name:>3}] cv_f1={cv_f1:.4f}  test: acc={m['accuracy']:.4f} f1={m['f1_macro']:.4f} "
                  f"auc={m.get('roc_auc_ovr_macro', float('nan')):.4f}  ({row['fit_seconds']}s)")

    table = pd.DataFrame(results)
    table.to_csv(out_dir / "model_comparison.csv", index=False)
    _plot_comparison(table, fig_dir / "model_comparison.png")

    # Model selection uses CV score on TRAIN data only (never the test set).
    best_name = select_best(table)
    best_model, best_metrics = fitted[best_name]
    best_dir = out_dir / "best_model"; best_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, best_dir / "model.joblib")
    meta = {"kind": "sklearn", "model": best_name, "vectorizer": vectorizer, "imbalance": imbalance, "tuned": tune,
            "labels": LABELS, "version": __version__, "n_train": int(len(tr)), "n_test": int(len(te)),
            "test_metrics": {k: best_metrics[k] for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro")},
            "top_features": top_features(best_model), "selected_by": "cv_f1_macro (train only)", "preprocessing": resources_status()}
    (best_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    (out_dir / "ml_results.json").write_text(json.dumps(
        {"results": results, "best": best_name, "best_report": best_metrics["report"]}, indent=2, default=str))
    te.assign(pred=[LABELS[i] for i in best_model.predict(X_te)]).to_csv(out_dir / "test_predictions.csv", index=False)
    if verbose:
        print(f"\nBest by CV macro-F1: {best_name}\n")
    return {"table": table, "best": best_name, "metrics": best_metrics, "best_dir": best_dir}


def select_best(table: pd.DataFrame) -> str:
    """Pick the model with the best CROSS-VALIDATED macro-F1 on training data. Never use test metrics here:
    choosing on the test set would leak it into model selection and inflate the reported score."""
    return str(table.sort_values("cv_f1_macro", ascending=False).iloc[0]["model"])


def _plot_comparison(table: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(table)); w = 0.38
    ax.bar(x - w / 2, table["cv_f1_macro"], w, label="CV macro-F1 (train)")
    ax.bar(x + w / 2, table["test_f1_macro"], w, label="Test macro-F1")
    ax.set_xticks(x, table["model"]); ax.set_ylim(0, 1.0); ax.legend(loc="lower right")
    ax.set_title("Model comparison")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
