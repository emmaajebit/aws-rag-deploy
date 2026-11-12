"""Text cleaning (spec step 2) + sentiment labelling (spec step 3).

Deviation from the spec's snippet: NLTK's stock stopword list contains negations
("not", "no", "nor", "never", ...) and "but"/"very"/"too". Dropping them turns
"not good" into "good" and flips the polarity, so they are kept. See TRADEOFFS.md.
"""
from __future__ import annotations

import html
import re
import warnings
from functools import lru_cache

import pandas as pd

from .config import CLEAN_COL, LABEL_COL, RATING_COL, TEXT_COL

KEEP_WORDS = {
    "no", "nor", "not", "never", "none", "neither", "nothing", "nobody", "cannot",
    "but", "very", "too", "more", "most", "less", "only", "against", "few", "same",
    "without", "than", "off", "down", "up",
}

_CONTRACTIONS = [
    (re.compile(r"\bcan['’]t\b"), "can not"), (re.compile(r"\bwon['’]t\b"), "will not"),
    (re.compile(r"\bain['’]t\b"), "is not"), (re.compile(r"n['’]t\b"), " not"),
    (re.compile(r"['’]re\b"), " are"), (re.compile(r"['’]ve\b"), " have"),
    (re.compile(r"['’]ll\b"), " will"), (re.compile(r"['’]m\b"), " am"),
]
_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+|www\.\S+")
_NON_ALPHA = re.compile(r"[^a-z\s]")
_WS = re.compile(r"\s+")

_FALLBACK_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "am", "and", "or", "of",
    "to", "in", "on", "at", "for", "with", "it", "its", "this", "that", "these", "those", "i", "me",
    "my", "we", "our", "you", "your", "he", "she", "they", "them", "his", "her", "their", "as", "by",
    "from", "had", "has", "have", "do", "does", "did", "so", "if", "then", "there", "here", "will",
    "would", "just", "s", "t",
}


@lru_cache(maxsize=1)
def _resources():
    """Load NLTK stopwords + WordNet lemmatizer. If a corpus is missing, try the NLTK downloader once; if that
    fails too, degrade to a built-in stopword list / no lemmatizer and WARN LOUDLY (see resources_status())."""
    import nltk

    stop, lem = None, None
    for pkg, path in (("stopwords", "corpora/stopwords"), ("wordnet", "corpora/wordnet")):
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass
    try:
        from nltk.corpus import stopwords
        stop = set(stopwords.words("english"))
    except Exception:
        stop = None
    try:
        from nltk.stem import WordNetLemmatizer
        lem = WordNetLemmatizer()
        lem.lemmatize("tests")  # forces corpus load
    except Exception:
        lem = None
    status = {"nltk_stopwords": stop is not None, "nltk_wordnet": lem is not None}
    if not all(status.values()):
        warnings.warn(
            "NLTK data missing (%s): using degraded preprocessing. Run `python -m nltk.downloader stopwords wordnet` "
            "before training AND serving, otherwise train/serve text will differ." %
            ", ".join(k for k, v in status.items() if not v), RuntimeWarning, stacklevel=2)
    _STATUS.update(status)
    return (stop if stop is not None else set(_FALLBACK_STOP)) - KEEP_WORDS, lem


_STATUS: dict = {}


def resources_status() -> dict:
    """{'nltk_stopwords': bool, 'nltk_wordnet': bool} - recorded in model metadata to detect train/serve skew."""
    _resources()
    return dict(_STATUS)


def clean_text(text: str, remove_stopwords: bool = True, lemmatize: bool = True) -> str:
    """lowercase -> strip HTML/URLs/punctuation -> tokenize -> stopwords -> lemmatize."""
    if not isinstance(text, str):
        return ""
    stop, lem = _resources()
    t = html.unescape(text)
    t = _TAG.sub(" ", t)
    t = _URL.sub(" ", t)
    t = t.lower()
    for pat, rep in _CONTRACTIONS:
        t = pat.sub(rep, t)
    t = _NON_ALPHA.sub(" ", t)
    words = _WS.sub(" ", t).strip().split()
    if remove_stopwords:
        words = [w for w in words if w not in stop]
    if lemmatize and lem is not None:
        words = [lem.lemmatize(lem.lemmatize(w, "v"), "n") for w in words]
    return " ".join(words)


def clean_series(s: pd.Series, **kw) -> pd.Series:
    return s.map(lambda x: clean_text(x, **kw))


# ---- labelling -------------------------------------------------------------
def rating_to_sentiment(rating: float) -> str:
    """4-5 -> positive, 3 -> neutral, 1-2 -> negative."""
    if rating >= 4:
        return "positive"
    if rating == 3:
        return "neutral"
    return "negative"


def prepare_dataset(df: pd.DataFrame, dedupe: bool = True, min_tokens: int = 1) -> pd.DataFrame:
    """Dedupe -> label from rating -> clean text. Dedupe happens BEFORE any split to prevent
    identical reviews landing in both train and test (a common source of inflated scores)."""
    out = df.copy()
    if dedupe:
        key = out[TEXT_COL].str.lower().str.strip()
        out = out.loc[~key.duplicated()]
    out[LABEL_COL] = out[RATING_COL].map(rating_to_sentiment)
    out[CLEAN_COL] = clean_series(out[TEXT_COL])
    out = out[out[CLEAN_COL].str.split().str.len() >= min_tokens]
    return out.reset_index(drop=True)
