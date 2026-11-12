"""BiLSTM and TextCNN text classifiers in PyTorch (spec step 6, DL models). Requires torch."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

from .config import LABELS, SEED
from .preprocessing import clean_text

PAD, UNK = 0, 1


def dl_text(text: str) -> str:
    """DL models keep stopwords and skip lemmatisation: sequence models use word order and function words."""
    return clean_text(text, remove_stopwords=False, lemmatize=False)


class Vocab:
    def __init__(self, itos: list[str]):
        self.itos = itos
        self.stoi = {w: i for i, w in enumerate(itos)}

    @classmethod
    def build(cls, texts, max_size: int = 20000, min_freq: int = 2):
        c = Counter(w for t in texts for w in t.split())
        words = [w for w, f in c.most_common(max_size) if f >= min_freq]
        return cls(["<pad>", "<unk>"] + words)

    def encode(self, text: str, max_len: int) -> list[int]:
        ids = [self.stoi.get(w, UNK) for w in text.split()][:max_len]
        return ids + [PAD] * (max_len - len(ids))

    def __len__(self):
        return len(self.itos)


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb=100, hidden=128, n_classes=3, dropout=0.4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=PAD)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, n_classes)

    def forward(self, x):
        mask = (x != PAD).unsqueeze(-1)
        h, _ = self.lstm(self.drop(self.emb(x)))
        h = h.masked_fill(~mask, -1e9).max(dim=1).values  # masked max-pool over time
        return self.fc(self.drop(h))


class TextCNN(nn.Module):
    def __init__(self, vocab_size, emb=100, n_filters=100, kernel_sizes=(2, 3, 4), n_classes=3, dropout=0.4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=PAD)
        self.convs = nn.ModuleList(nn.Conv1d(emb, n_filters, k, padding=k - 1) for k in kernel_sizes)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(kernel_sizes), n_classes)

    def forward(self, x):
        e = self.drop(self.emb(x)).transpose(1, 2)
        feats = [torch.relu(c(e)).max(dim=2).values for c in self.convs]
        return self.fc(self.drop(torch.cat(feats, dim=1)))


ARCHS = {"bilstm": BiLSTMClassifier, "cnn": TextCNN}


def _tensors(vocab, texts, max_len):
    return torch.tensor([vocab.encode(t, max_len) for t in texts], dtype=torch.long)


def train_dl(arch: str, train_texts, y_train, val_texts, y_val, out_dir: Path, max_len: int = 128, epochs: int = 8,
             batch_size: int = 64, lr: float = 2e-3, patience: int = 2, seed: int = SEED, verbose: bool = True) -> dict:
    """Class-weighted CE, AdamW, early stopping on validation macro-F1; saves best weights + vocab."""
    torch.manual_seed(seed); np.random.seed(seed)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr_txt = [dl_text(t) for t in train_texts]; va_txt = [dl_text(t) for t in val_texts]
    vocab = Vocab.build(tr_txt)
    Xtr, Xva = _tensors(vocab, tr_txt, max_len), _tensors(vocab, va_txt, max_len)
    ytr, yva = torch.tensor(np.asarray(y_train)), torch.tensor(np.asarray(y_val))
    model = ARCHS[arch](len(vocab)).to(dev)
    cw = compute_class_weight("balanced", classes=np.arange(len(LABELS)), y=np.asarray(y_train))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=dev))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    best_f1, best_state, bad, hist = -1.0, None, 0, []
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr))
        total = 0.0
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx].to(dev)), ytr[idx].to(dev))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total += loss.item() * len(idx)
        f1 = f1_score(yva.numpy(), _predict(model, Xva, dev).argmax(1), average="macro")
        hist.append({"epoch": ep, "train_loss": total / len(perm), "val_f1_macro": float(f1)})
        if verbose:
            print(f"[{arch}] epoch {ep} loss={total / len(perm):.4f} val_macro_f1={f1:.4f}")
        if f1 > best_f1:
            best_f1, bad = f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), out_dir / "weights.pt")
    (out_dir / "vocab.json").write_text(json.dumps(vocab.itos))
    meta = {"kind": "dl", "arch": arch, "max_len": max_len, "labels": LABELS, "vocab_size": len(vocab),
            "best_val_f1_macro": float(best_f1), "history": hist}
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    return {"model": model, "vocab": vocab, "meta": meta}


def _predict(model, X, dev, batch_size: int = 256) -> np.ndarray:
    model.eval(); outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            outs.append(torch.softmax(model(X[i:i + batch_size].to(dev)), dim=1).cpu().numpy())
    return np.vstack(outs)


class DLPredictor:
    def __init__(self, model_dir: Path):
        model_dir = Path(model_dir)
        self.meta = json.loads((model_dir / "metadata.json").read_text())
        self.vocab = Vocab(json.loads((model_dir / "vocab.json").read_text()))
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = ARCHS[self.meta["arch"]](len(self.vocab))
        self.model.load_state_dict(torch.load(model_dir / "weights.pt", map_location="cpu"))
        self.model.to(self.dev).eval()

    def predict_proba(self, texts) -> np.ndarray:
        X = _tensors(self.vocab, [dl_text(t) for t in texts], self.meta["max_len"])
        return _predict(self.model, X, self.dev)
