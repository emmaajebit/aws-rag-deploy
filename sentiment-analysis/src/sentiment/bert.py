"""Fine-tune DistilBERT / BERT for 3-class sentiment (spec step 6, transformer model). Requires torch + transformers.

Uses RAW review text (not the stopword-stripped/lemmatised text): the tokenizer and pretrained model were trained on
natural text and rely on function words, negations and punctuation.
`model_name` may be a Hugging Face id ('distilbert-base-uncased', 'bert-base-uncased') or a local directory.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from .config import ID2LABEL, LABEL2ID, LABELS, SEED


def _batches(texts, labels, tok, batch_size, max_length, shuffle, rng=None):
    idx = np.arange(len(texts))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        b = idx[i:i + batch_size]
        enc = tok([texts[j] for j in b], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        if labels is not None:
            enc["labels"] = torch.tensor([labels[j] for j in b])
        yield enc


@torch.no_grad()
def predict_proba(model, tok, texts, dev, batch_size=64, max_length=128) -> np.ndarray:
    model.eval(); outs = []
    for enc in _batches(list(texts), None, tok, batch_size, max_length, False):
        enc = {k: v.to(dev) for k, v in enc.items()}
        outs.append(torch.softmax(model(**enc).logits.float(), dim=-1).cpu().numpy())
    return np.vstack(outs) if outs else np.zeros((0, len(LABELS)))


def fine_tune(train_texts, y_train, val_texts, y_val, out_dir: Path, model_name: str = "distilbert-base-uncased",
              epochs: int = 3, batch_size: int = 16, lr: float = 2e-5, max_length: int = 128, warmup: float = 0.1,
              seed: int = SEED, verbose: bool = True) -> dict:
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID).to(dev)

    train_texts, val_texts = list(train_texts), list(val_texts)
    y_train, y_val = list(map(int, y_train)), list(map(int, y_val))
    cw = compute_class_weight("balanced", classes=np.arange(len(LABELS)), y=np.asarray(y_train))
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=dev))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * int(np.ceil(len(train_texts) / batch_size))
    sched = get_linear_schedule_with_warmup(opt, int(warmup * steps), steps)
    use_amp = dev == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_f1, hist = -1.0, []
    for ep in range(1, epochs + 1):
        model.train(); total, n = 0.0, 0
        for enc in _batches(train_texts, y_train, tok, batch_size, max_length, True, rng):
            labels = enc.pop("labels").to(dev)
            enc = {k: v.to(dev) for k, v in enc.items()}
            opt.zero_grad()
            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits = model(**enc).logits
            loss = loss_fn(logits.float(), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            total += loss.item() * len(labels); n += len(labels)
        proba = predict_proba(model, tok, val_texts, dev, max_length=max_length)
        f1 = f1_score(y_val, proba.argmax(1), average="macro")
        hist.append({"epoch": ep, "train_loss": total / max(n, 1), "val_f1_macro": float(f1)})
        if verbose:
            print(f"[bert] epoch {ep}/{epochs} loss={total / max(n, 1):.4f} val_macro_f1={f1:.4f}")
        if f1 > best_f1:  # keep best epoch
            best_f1 = f1
            model.save_pretrained(out_dir); tok.save_pretrained(out_dir)
    meta = {"kind": "bert", "base_model": model_name, "max_length": max_length, "labels": LABELS, "epochs": epochs,
            "best_val_f1_macro": float(best_f1), "history": hist}
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    return {"meta": meta, "out_dir": out_dir}


class BertPredictor:
    def __init__(self, model_dir: Path):
        model_dir = Path(model_dir)
        self.meta = json.loads((model_dir / "metadata.json").read_text())
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.dev).eval()

    def predict_proba(self, texts) -> np.ndarray:
        return predict_proba(self.model, self.tok, list(texts), self.dev, max_length=self.meta.get("max_length", 128))
