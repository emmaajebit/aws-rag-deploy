import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from sentiment.config import LABEL2ID
from sentiment.predictor import SentimentPredictor
from sentiment.train_ml import split_data


@pytest.fixture(scope="module")
def nn_split(small_df):
    sub = small_df.head(400)
    tr, te, ytr, yte = split_data(sub, 0.25)
    return tr, te, ytr, yte


@pytest.mark.parametrize("arch", ["bilstm", "cnn"])
def test_dl_train_and_serve(tmp_path, nn_split, arch):
    from sentiment.models_dl import train_dl
    tr, te, ytr, yte = nn_split
    out = tmp_path / arch
    res = train_dl(arch, tr["review"], ytr, te["review"], yte, out, epochs=2, batch_size=32, verbose=False)
    assert res["meta"]["history"]
    p = SentimentPredictor.load(out)
    proba = p.predict_proba(["great taste", "terrible and stale", ""])
    assert proba.shape == (3, 3) and np.allclose(proba.sum(1), 1, atol=1e-4)


def _tiny_bert(path):
    """Tiny randomly-initialised BERT + locally built WordPiece vocab: exercises the real fine-tuning code
    path offline. (Pretrained weights are NOT used here - see README for the real run.)"""
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
    words = ("great good tasty fresh love loved delicious amazing best terrible awful bad cold soggy stale worst "
             "okay fine average nothing special the was and it not pizza burger very").split()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + words
    path.mkdir(parents=True, exist_ok=True)
    (path / "vocab.txt").write_text("\n".join(vocab))
    BertTokenizerFast(str(path / "vocab.txt"), do_lower_case=True).save_pretrained(path)
    cfg = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=64, max_position_embeddings=64, num_labels=3)
    BertForSequenceClassification(cfg).save_pretrained(path)


def test_bert_finetune_checkpoint_and_serve(tmp_path, nn_split):
    from sentiment.bert import fine_tune
    tr, te, ytr, yte = nn_split
    base = tmp_path / "tiny"; _tiny_bert(base)
    out = tmp_path / "bert"
    res = fine_tune(tr["review"], ytr, te["review"], yte, out, model_name=str(base), epochs=2, batch_size=16,
                    lr=5e-3, max_length=32, verbose=False)
    assert (out / "config.json").exists() and (out / "metadata.json").exists()
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["id2label"] == {"0": "negative", "1": "neutral", "2": "positive"}
    assert res["meta"]["best_val_f1_macro"] > 0
    p = SentimentPredictor.load(out)
    r = p.predict(["great pizza", "terrible burger"])
    assert len(r) == 2 and all(abs(sum(x["probabilities"].values()) - 1) < 1e-3 for x in r)
