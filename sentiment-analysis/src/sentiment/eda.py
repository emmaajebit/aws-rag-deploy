"""Exploratory data analysis (spec step 4): class bars, length distribution, word clouds."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from .config import CLEAN_COL, FIGURES_DIR, LABEL_COL, LABELS, TEXT_COL  # noqa: E402

PALETTE = {"negative": "#c0392b", "neutral": "#7f8c8d", "positive": "#27ae60"}


def plot_class_distribution(df: pd.DataFrame, out: Path) -> Path:
    counts = df[LABEL_COL].value_counts().reindex(LABELS).fillna(0)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette=PALETTE, legend=False, ax=ax)
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{int(v):,}\n({v / counts.sum():.1%})", ha="center", va="bottom", fontsize=9)
    ax.set(title="Reviews by sentiment", xlabel="", ylabel="count")
    ax.set_ylim(0, counts.max() * 1.18)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return out


def plot_length_distribution(df: pd.DataFrame, out: Path) -> Path:
    words = df[TEXT_COL].str.split().str.len()
    cap = words.quantile(0.99)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(data=pd.DataFrame({"words": words.clip(upper=cap), LABEL_COL: df[LABEL_COL]}),
                 x="words", hue=LABEL_COL, hue_order=LABELS, palette=PALETTE, bins=40,
                 element="step", stat="density", common_norm=False, ax=ax)
    ax.set(title=f"Review length (words; clipped at p99={cap:.0f})", xlabel="words")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return out


def plot_wordclouds(df: pd.DataFrame, out: Path, max_words: int = 100) -> Path:
    from wordcloud import WordCloud

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, lab, cmap in zip(axes, ("positive", "negative"), ("Greens", "Reds")):
        text = " ".join(df.loc[df[LABEL_COL] == lab, CLEAN_COL])
        if text.strip():
            wc = WordCloud(width=800, height=400, background_color="white", colormap=cmap,
                           max_words=max_words, random_state=42).generate(text)
            ax.imshow(wc, interpolation="bilinear")
        ax.set_title(f"{lab.title()} reviews"); ax.axis("off")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return out


def run_eda(df: pd.DataFrame, out_dir: Path = FIGURES_DIR) -> dict:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    files = [
        plot_class_distribution(df, out_dir / "class_distribution.png"),
        plot_length_distribution(df, out_dir / "review_length.png"),
        plot_wordclouds(df, out_dir / "wordclouds.png"),
    ]
    words = df[TEXT_COL].str.split().str.len()
    summary = {
        "n_reviews": int(len(df)),
        "class_counts": {k: int(v) for k, v in df[LABEL_COL].value_counts().items()},
        "imbalance_ratio_max_to_min": round(float(df[LABEL_COL].value_counts().max() / df[LABEL_COL].value_counts().min()), 2),
        "words_per_review": {"mean": round(float(words.mean()), 1), "median": float(words.median()),
                             "p95": float(words.quantile(0.95))},
        "figures": [str(f) for f in files],
    }
    (out_dir / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
