"""Text vectorisation (spec step 5): TF-IDF, BoW, Word2Vec (avg), transformer embeddings."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def make_vectorizer(kind: str = "tfidf", max_features: int = 5000, ngram_range=(1, 2)):
    if kind == "tfidf":
        return TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=2, sublinear_tf=True)
    if kind == "bow":
        return CountVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=2)
    if kind == "w2v":
        return Word2VecVectorizer()
    raise ValueError(f"unknown vectorizer '{kind}' (tfidf | bow | w2v)")


class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    """Trains Word2Vec on the *training fold only* and averages word vectors per document.
    Dense output: works with LR / RF / XGBoost, not MultinomialNB (negative values)."""

    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 2, epochs: int = 10, seed: int = 42):
        self.vector_size, self.window, self.min_count, self.epochs, self.seed = vector_size, window, min_count, epochs, seed

    def fit(self, X, y=None):
        from gensim.models import Word2Vec  # lazy

        sents = [t.split() for t in X]
        self.model_ = Word2Vec(sents, vector_size=self.vector_size, window=self.window, min_count=self.min_count,
                               epochs=self.epochs, seed=self.seed, workers=1)
        return self

    def transform(self, X):
        wv = self.model_.wv
        out = np.zeros((len(X), self.vector_size), dtype=np.float32)
        for i, doc in enumerate(X):
            vecs = [wv[w] for w in doc.split() if w in wv]
            if vecs:
                out[i] = np.mean(vecs, axis=0)
        return out


class TransformerEmbedder(BaseEstimator, TransformerMixin):
    """Frozen BERT/DistilBERT mean-pooled sentence embeddings (feed to LR etc.). Needs torch + transformers
    and a model name/path reachable from this machine."""

    def __init__(self, model_name: str = "distilbert-base-uncased", max_length: int = 128, batch_size: int = 32):
        self.model_name, self.max_length, self.batch_size = model_name, max_length, batch_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        import torch
        from transformers import AutoModel, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name).eval()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(dev)
        outs = []
        texts = list(X)
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                enc = tok(texts[i:i + self.batch_size], padding=True, truncation=True,
                          max_length=self.max_length, return_tensors="pt").to(dev)
                h = model(**enc).last_hidden_state
                m = enc["attention_mask"].unsqueeze(-1).float()
                outs.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy())
        return np.vstack(outs)
