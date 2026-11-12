import json

import pandas as pd
import pytest

from sentiment.data.loaders import load_amazon_fine_food, load_csv, load_reviews
from sentiment.data.scrape import fetch_yelp_reviews, parse_reviews_html
from sentiment.data.synthetic import generate_reviews


def test_synthetic_deterministic_and_skewed():
    a, b = generate_reviews(2000, seed=1), generate_reviews(2000, seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert list(a.columns) == ["review", "rating", "product", "timestamp"]
    assert a["rating"].value_counts(normalize=True)[5] > 0.5     # Amazon-like skew toward 5 stars
    assert a["review"].str.contains("<br />").any()               # HTML noise present


def test_load_amazon_format(tmp_path):
    p = tmp_path / "Reviews.csv"
    pd.DataFrame({"Id": [1, 2, 3], "ProductId": ["B1", "B2", "B3"], "Score": [5, 1, 9],
                  "Time": [1331510400, 1331510400, 1331510400], "Summary": ["x"] * 3,
                  "Text": ["Great stuff", "Awful", "bad rating value"]}).to_csv(p, index=False)
    df = load_amazon_fine_food(p)
    assert len(df) == 2 and df["product"].tolist() == ["B1", "B2"]   # Score=9 dropped
    assert df["timestamp"].iloc[0] == "2012-03-12"


def test_load_csv_mapping_and_missing_cols(tmp_path):
    p = tmp_path / "r.csv"
    pd.DataFrame({"txt": ["good", None, "bad"], "stars": [5, 3, 1]}).to_csv(p, index=False)
    df = load_csv(p, columns={"txt": "review", "stars": "rating"})
    assert len(df) == 2                                               # null review dropped
    with pytest.raises(ValueError):
        load_csv(p)


def test_load_reviews_requires_path():
    with pytest.raises(ValueError):
        load_reviews("amazon")


def test_parse_reviews_html():
    html = ('<div class="r"><span class="s">4.0 out of 5 stars</span><p>Tasty <b>snack</b></p></div>'
            '<div class="r"><span class="s">x</span>Bad</div>')
    df = parse_reviews_html(html, "div.r", "span.s", product="Snack")
    assert df["rating"].tolist()[0] == 4.0 and pd.isna(df["rating"].iloc[1])
    assert "Tasty snack" in df["review"].iloc[0]


def test_yelp_client_uses_env_key_and_parses(monkeypatch):
    class R:
        def raise_for_status(self): pass
        def json(self): return {"reviews": [{"text": "Great", "rating": 5, "time_created": "2024-03-15 10:00:00"}]}
    seen = {}
    class S:
        def get(self, url, headers, params, timeout):
            seen["auth"] = headers["Authorization"]; return R()
    monkeypatch.delenv("YELP_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        fetch_yelp_reviews(["abc"], session=S())
    monkeypatch.setenv("YELP_API_KEY", "k")
    df = fetch_yelp_reviews(["abc"], session=S())
    assert seen["auth"] == "Bearer k" and df["timestamp"].iloc[0] == "2024-03-15"
