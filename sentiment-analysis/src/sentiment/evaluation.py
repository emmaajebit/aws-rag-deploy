"""Model evaluation (spec step 7)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,  # noqa: E402
                             precision_recall_fscore_support, roc_auc_score)

from .config import LABELS  # noqa: E402


def evaluate(y_true, y_pred, y_proba=None) -> dict:
    """y_* are integer class ids in LABELS order. Macro-F1 is the headline metric (class imbalance)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=range(len(LABELS)), average="macro", zero_division=0)
    res = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p), "recall_macro": float(r), "f1_macro": float(f),
        "f1_weighted": float(precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)[2]),
        "report": classification_report(y_true, y_pred, labels=range(len(LABELS)), target_names=LABELS,
                                        output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=range(len(LABELS))).tolist(),
    }
    if y_proba is not None:
        y_proba = np.asarray(y_proba)
        try:
            res["roc_auc_ovr_macro"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro",
                                                           labels=list(range(len(LABELS)))))
            # Binary view from the spec (positive vs negative only, neutral excluded)
            mask = np.isin(y_true, [0, 2])
            if mask.any() and len(set(y_true[mask])) == 2:
                res["roc_auc_pos_vs_neg"] = float(roc_auc_score(y_true[mask] == 2, y_proba[mask, 2] - y_proba[mask, 0]))
        except ValueError:
            pass
    return res


def plot_confusion_matrix(cm, title: str, out: Path, normalize: bool = True) -> Path:
    cm = np.asarray(cm, dtype=float)
    shown = cm / cm.sum(axis=1, keepdims=True).clip(min=1) if normalize else cm
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set(xticks=range(3), yticks=range(3), xticklabels=LABELS, yticklabels=LABELS,
           xlabel="predicted", ylabel="actual", title=title)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{int(cm[i, j])}\n{shown[i, j]:.0%}" if normalize else f"{int(cm[i, j])}",
                    ha="center", va="center", color="white" if shown[i, j] > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return out


def top_features(pipeline, n: int = 10) -> dict | None:
    """Most indicative terms per class for linear models on sparse features (cheap explainability)."""
    try:
        vec, clf = pipeline.named_steps["vec"], pipeline.named_steps["clf"]
        names = np.array(vec.get_feature_names_out())
        coef = clf.coef_ if hasattr(clf, "coef_") else clf.feature_log_prob_
    except Exception:
        return None
    if coef.shape[0] != len(LABELS):
        return None
    return {lab: names[np.argsort(coef[i])[::-1][:n]].tolist() for i, lab in enumerate(LABELS)}
