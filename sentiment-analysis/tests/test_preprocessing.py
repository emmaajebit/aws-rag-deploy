import pandas as pd
import pytest

from sentiment.preprocessing import clean_text, prepare_dataset, rating_to_sentiment, resources_status


def test_nltk_resources_present():
    # If this fails, preprocessing is silently degraded (no lemmatisation / tiny stopword list).
    assert all(resources_status().values()), "run: python -m nltk.downloader stopwords wordnet"


def test_spec_example():
    assert clean_text("The burger was awesome!!!") == "burger awesome"


def test_lemmatises_verbs_and_plurals():
    assert clean_text("I loved the pizzas") == "love pizza"


def test_html_url_punct_removed():
    out = clean_text("Great!<br /> see http://x.com/a &amp; www.y.org  :)")
    assert "<" not in out and "http" not in out and "www" not in out and "amp" not in out


@pytest.mark.parametrize("txt", ["not good", "It wasn't good", "I can't recommend it", "never again", "no flavor"])
def test_negations_survive(txt):
    assert any(w in clean_text(txt).split() for w in ("not", "never", "no"))


def test_non_string_is_empty():
    assert clean_text(None) == "" and clean_text(float("nan")) == ""


@pytest.mark.parametrize("r,lab", [(1, "negative"), (2, "negative"), (3, "neutral"), (4, "positive"), (5, "positive")])
def test_rating_mapping(r, lab):
    assert rating_to_sentiment(r) == lab


def test_prepare_dedupes_labels_and_drops_empty():
    df = pd.DataFrame({"review": ["Great pizza!", "great pizza! ", "!!!", "Awful soup"], "rating": [5, 5, 4, 1]})
    out = prepare_dataset(df)
    assert len(out) == 2                      # duplicate (case/space-insensitive) and empty-after-cleaning removed
    assert set(out["sentiment"]) == {"positive", "negative"}
    assert (out["cleaned_review"].str.len() > 0).all()
