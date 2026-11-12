"""Web scraping (BeautifulSoup) and Yelp Fusion API collection.

Only scrape sites whose terms/robots.txt allow it. Both functions are polite by
default (robots.txt check, delay). Network code is isolated here so the rest of the
pipeline never needs the internet.
"""
from __future__ import annotations

import os
import time
import urllib.robotparser
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

UA = "sentiment-analysis-research/1.0"


def parse_reviews_html(html: str, review_sel: str, rating_sel: str | None = None,
                       product: str | None = None) -> pd.DataFrame:
    """Pure parser (testable offline). Ratings are read from text like '4.0 out of 5 stars' or '4'."""
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for node in soup.select(review_sel):
        text = node.get_text(" ", strip=True)
        rating = None
        if rating_sel:
            r = node.select_one(rating_sel)
            if r:
                token = r.get_text(" ", strip=True).split()[0]
                try:
                    rating = float(token)
                except ValueError:
                    rating = None
        rows.append({"review": text, "rating": rating, "product": product,
                     "timestamp": pd.Timestamp.today().strftime("%Y-%m-%d")})
    return pd.DataFrame(rows, columns=["review", "rating", "product", "timestamp"])


def _allowed(url: str) -> bool:
    p = urlparse(url)
    rp = urllib.robotparser.RobotFileParser(f"{p.scheme}://{p.netloc}/robots.txt")
    try:
        rp.read()
    except Exception:
        return False  # fail closed
    return rp.can_fetch(UA, url)


def scrape_reviews(urls: list[str], review_sel: str, rating_sel: str | None = None,
                   delay: float = 2.0, product: str | None = None) -> pd.DataFrame:
    frames = []
    for url in urls:
        if not _allowed(url):
            raise PermissionError(f"robots.txt disallows (or could not be read for) {url}")
        resp = requests.get(url, headers={"User-Agent": UA}, timeout=20)
        resp.raise_for_status()
        frames.append(parse_reviews_html(resp.text, review_sel, rating_sel, product))
        time.sleep(delay)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_yelp_reviews(business_ids: list[str], api_key: str | None = None, session=None) -> pd.DataFrame:
    """Yelp Fusion API (`/businesses/{id}/reviews`; the API returns at most 3 excerpts per business)."""
    api_key = api_key or os.getenv("YELP_API_KEY")
    if not api_key:
        raise RuntimeError("Set YELP_API_KEY (never hard-code it).")
    http = session or requests
    rows = []
    for bid in business_ids:
        r = http.get(f"https://api.yelp.com/v3/businesses/{bid}/reviews",
                     headers={"Authorization": f"Bearer {api_key}"}, params={"limit": 50}, timeout=20)
        r.raise_for_status()
        for rev in r.json().get("reviews", []):
            rows.append({"review": rev["text"], "rating": rev["rating"], "product": bid,
                         "timestamp": rev.get("time_created", "")[:10]})
    return pd.DataFrame(rows, columns=["review", "rating", "product", "timestamp"])
