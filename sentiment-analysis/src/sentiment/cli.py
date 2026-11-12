"""Command line entry point:  sentiment {data,eda,train,dl,bert,serve,dashboard,predict}"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ARTIFACTS_DIR, BEST_MODEL_DIR, CLEAN_COL, FIGURES_DIR, LABEL2ID, LABEL_COL, PROCESSED_DIR, SEED, TEXT_COL


def _load_processed(path: Path):
    import pandas as pd
    if not path.exists():
        sys.exit(f"{path} not found. Run `sentiment data` first.")
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def cmd_data(a):
    from .data.loaders import load_reviews
    from .preprocessing import prepare_dataset
    raw = load_reviews(a.source, a.path, n=a.n, seed=a.seed, nrows=a.nrows)
    df = prepare_dataset(raw)
    if a.max_per_class:
        df = (df.groupby(LABEL_COL, group_keys=False)
                .apply(lambda g: g.sample(min(len(g), a.max_per_class), random_state=a.seed)).reset_index(drop=True))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{len(raw):,} raw -> {len(df):,} prepared reviews saved to {out}")
    print(df[LABEL_COL].value_counts().to_string())


def cmd_eda(a):
    from .eda import run_eda
    print(json.dumps(run_eda(_load_processed(Path(a.data)), Path(a.fig_dir)), indent=2))


def cmd_train(a):
    from .train_ml import run_experiment
    df = _load_processed(Path(a.data))
    run_experiment(df, models=a.models, vectorizer=a.vectorizer, imbalance=a.imbalance, tune=a.tune, cv=a.cv,
                   fast=a.fast, out_dir=Path(a.out), seed=a.seed)


def _split_for_nn(a):
    from .train_ml import split_data
    df = _load_processed(Path(a.data))
    tr, te, y_tr, y_te = split_data(df, 0.2, a.seed)
    tr, va, y_tr, y_va = __import__("sklearn.model_selection", fromlist=["x"]).train_test_split(
        tr, y_tr, test_size=0.125, stratify=y_tr, random_state=a.seed)
    return tr, va, te, y_tr, y_va, y_te


def _eval_predictor(pred, te, y_te, out_dir: Path):
    from .evaluation import evaluate, plot_confusion_matrix
    proba = pred.predict_proba(te[TEXT_COL].tolist())
    m = evaluate(y_te, proba.argmax(1), proba)
    plot_confusion_matrix(m["confusion_matrix"], out_dir.name, FIGURES_DIR / f"cm_{out_dir.name}.png")
    meta_p = out_dir / "metadata.json"
    meta = json.loads(meta_p.read_text()); meta["test_metrics"] = {k: m[k] for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro")}
    meta["test_roc_auc_ovr"] = m.get("roc_auc_ovr_macro"); meta_p.write_text(json.dumps(meta, indent=2))
    print(f"TEST  acc={m['accuracy']:.4f}  macro-F1={m['f1_macro']:.4f}  ROC-AUC(ovr)={m.get('roc_auc_ovr_macro', float('nan')):.4f}")
    return m


def cmd_dl(a):
    from .models_dl import DLPredictor, train_dl
    tr, va, te, y_tr, y_va, y_te = _split_for_nn(a)
    out = Path(a.out) / f"dl_{a.arch}"
    train_dl(a.arch, tr[TEXT_COL], y_tr, va[TEXT_COL], y_va, out, epochs=a.epochs, seed=a.seed)
    _eval_predictor(DLPredictor(out), te, y_te, out)


def cmd_bert(a):
    from .bert import BertPredictor, fine_tune
    tr, va, te, y_tr, y_va, y_te = _split_for_nn(a)
    if a.max_train:
        tr, y_tr = tr.iloc[:a.max_train], y_tr.iloc[:a.max_train]
    out = Path(a.out) / "bert"
    fine_tune(tr[TEXT_COL], y_tr, va[TEXT_COL], y_va, out, model_name=a.model_name, epochs=a.epochs,
              batch_size=a.batch_size, lr=a.lr, max_length=a.max_length, seed=a.seed)
    _eval_predictor(BertPredictor(out), te, y_te, out)


def cmd_predict(a):
    from .predictor import SentimentPredictor
    for t, r in zip(a.text, SentimentPredictor.load(a.model_dir).predict(a.text)):
        print(f"{r['sentiment']:<9} {r['confidence']:.2%}  {t}")


def cmd_serve(a):
    import os, uvicorn
    os.environ["SENTIMENT_MODEL_DIR"] = a.model_dir
    uvicorn.run("sentiment.api.app:app", host=a.host, port=a.port)


def cmd_dashboard(a):
    import os, subprocess
    os.environ["SENTIMENT_MODEL_DIR"] = a.model_dir
    app = Path(__file__).parent / "dashboard" / "app.py"
    sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(a.port)]))


def main(argv=None):
    p = argparse.ArgumentParser(prog="sentiment", description=__doc__)
    sp = p.add_subparsers(dest="cmd", required=True)
    default_data = str(PROCESSED_DIR / "reviews.csv")

    d = sp.add_parser("data", help="load/generate reviews, dedupe, label, clean")
    d.add_argument("--source", default="synthetic", choices=["synthetic", "amazon", "csv", "jsonl", "hf"])
    d.add_argument("--path"); d.add_argument("--n", type=int, default=20000); d.add_argument("--nrows", type=int)
    d.add_argument("--max-per-class", type=int, help="cap class size (speeds up experiments)")
    d.add_argument("--seed", type=int, default=SEED); d.add_argument("--out", default=default_data); d.set_defaults(fn=cmd_data)

    e = sp.add_parser("eda", help="word clouds, class bars, length distribution")
    e.add_argument("--data", default=default_data); e.add_argument("--fig-dir", default=str(FIGURES_DIR)); e.set_defaults(fn=cmd_eda)

    t = sp.add_parser("train", help="train/compare LR, NB, RF, XGB; save best")
    t.add_argument("--data", default=default_data); t.add_argument("--out", default=str(ARTIFACTS_DIR))
    t.add_argument("--models", nargs="+", default=["lr", "nb", "rf", "xgb"], choices=["lr", "nb", "rf", "xgb"])
    t.add_argument("--vectorizer", default="tfidf", choices=["tfidf", "bow", "w2v"])
    t.add_argument("--imbalance", default="class_weight", choices=["none", "class_weight", "smote"])
    t.add_argument("--tune", action="store_true", help="GridSearchCV (macro-F1)"); t.add_argument("--fast", action="store_true")
    t.add_argument("--cv", type=int, default=5); t.add_argument("--seed", type=int, default=SEED); t.set_defaults(fn=cmd_train)

    dl = sp.add_parser("dl", help="train BiLSTM / TextCNN (needs torch)")
    dl.add_argument("--arch", default="bilstm", choices=["bilstm", "cnn"]); dl.add_argument("--epochs", type=int, default=8)
    dl.add_argument("--data", default=default_data); dl.add_argument("--out", default=str(ARTIFACTS_DIR)); dl.add_argument("--seed", type=int, default=SEED)
    dl.set_defaults(fn=cmd_dl)

    b = sp.add_parser("bert", help="fine-tune DistilBERT/BERT (needs torch+transformers and model weights)")
    b.add_argument("--model-name", default="distilbert-base-uncased"); b.add_argument("--epochs", type=int, default=3)
    b.add_argument("--batch-size", type=int, default=16); b.add_argument("--lr", type=float, default=2e-5)
    b.add_argument("--max-length", type=int, default=128); b.add_argument("--max-train", type=int)
    b.add_argument("--data", default=default_data); b.add_argument("--out", default=str(ARTIFACTS_DIR)); b.add_argument("--seed", type=int, default=SEED)
    b.set_defaults(fn=cmd_bert)

    pr = sp.add_parser("predict", help="score text from the command line")
    pr.add_argument("text", nargs="+"); pr.add_argument("--model-dir", default=str(BEST_MODEL_DIR)); pr.set_defaults(fn=cmd_predict)

    s = sp.add_parser("serve", help="run the FastAPI service")
    s.add_argument("--model-dir", default=str(BEST_MODEL_DIR)); s.add_argument("--host", default="127.0.0.1"); s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)

    da = sp.add_parser("dashboard", help="run the Streamlit dashboard")
    da.add_argument("--model-dir", default=str(BEST_MODEL_DIR)); da.add_argument("--port", type=int, default=8501); da.set_defaults(fn=cmd_dashboard)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
