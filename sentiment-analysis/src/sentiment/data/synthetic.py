"""Synthetic food / e-commerce review generator.

Used ONLY so the pipeline can be run and tested without network access to the real
datasets. It deliberately includes the difficulties the spec calls out: skewed star
distribution, HTML noise, negations ("not bad"), mixed-sentiment 3-star text, and a
small amount of rating/text disagreement (label noise) so metrics are not trivially 100%.
Do NOT quote metrics from this data as real-world performance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PRODUCTS = [
    "Pepperoni Pizza", "Chicken Burger", "Veggie Wrap", "Chocolate Cake", "Instant Noodles",
    "Green Tea", "Protein Bar", "Butter Croissant", "Sushi Platter", "Peanut Butter",
    "Organic Granola", "Olive Oil", "Hot Sauce", "Cold Brew Coffee", "Tomato Soup",
    "Beef Jerky", "Vanilla Ice Cream", "Potato Chips", "Basmati Rice", "Fruit Juice",
]

OPENERS = ["", "", "", "Update: ", "Honestly, ", "To be fair, ", "Ordered this last week. ", "Second time buying. "]
TAILS = ["", "", "", " Just my opinion.", " Ordered it for the family.", " Got it through the app.", " Arrived in a box."]

POS_STRONG = [
    "Absolutely loved the {p}!", "The {p} was delicious and delivered on time.",
    "Best {p} I have ever had, will order again.", "Fantastic quality, the {p} tasted fresh and amazing.",
    "Highly recommend this {p}, worth every penny.", "Wow, the {p} exceeded my expectations. Superb flavor!",
    "Five stars. The {p} was perfect, so tasty.", "Not bad at all, in fact the {p} was excellent.",
]
POS_MILD = [
    "The {p} was good and tasted fresh.", "Pretty happy with the {p}, nice flavor.",
    "Solid {p}, would buy again.", "Good value, the {p} was tasty enough.",
    "The {p} was not bad, quite enjoyable.", "Nice {p}, my kids liked it.",
]
NEUTRAL = [
    "It was okay, nothing special.", "The {p} was fine, about what I expected.",
    "Average {p}. Not great, not terrible.", "Decent but not memorable.",
    "Meh. The {p} is alright I guess.", "The {p} was ok. Would not go out of my way for it.",
    "Neither good nor bad, just an ordinary {p}.",
]
NEG_MILD = [
    "The {p} was disappointing and a bit bland.", "Not worth the price, the {p} was underwhelming.",
    "I would not buy this {p} again.", "The {p} was not good, kind of stale.",
    "Expected more from this {p}, it was mediocre at best.", "Not great. The {p} tasted off.",
]
NEG_STRONG = [
    "Terrible {p}, it was cold and soggy.", "Awful. The {p} arrived late and tasted stale.",
    "Worst {p} ever, complete waste of money.", "The {p} was disgusting, do not buy.",
    "Horrible experience, the {p} was spoiled and support ignored me.",
    "Burger was cold and soggy, absolutely not worth it." ,
]
ASPECT_POS = ["Delivery was quick.", "Packaging was great.", "Price is fair.", "Portion size is generous."]
ASPECT_NEG = ["Delivery was late.", "Packaging was damaged.", "A bit overpriced.", "The portion was tiny."]
SARCASM = ["Oh great, another cold {p}. Just what I wanted.", "Wonderful, the {p} arrived crushed. Thanks a lot."]


def _sentence(rng, pool, p):
    return pool[rng.integers(len(pool))].format(p=p.lower())


def _compose(rng: np.random.Generator, tone: int, product: str) -> str:
    """tone: 1 (very neg) .. 5 (very pos)."""
    s = _sentence
    if tone == 5:
        parts = [s(rng, POS_STRONG, product)]
        if rng.random() < 0.5:
            parts.append(ASPECT_POS[rng.integers(4)])
    elif tone == 4:
        parts = [s(rng, POS_MILD, product)]
        if rng.random() < 0.4:
            parts.append("However, " + ASPECT_NEG[rng.integers(4)].lower())
    elif tone == 3:
        r = rng.random()
        if r < 0.55:
            parts = [s(rng, NEUTRAL, product)]
        else:
            parts = [s(rng, POS_MILD, product), "But " + ASPECT_NEG[rng.integers(4)].lower()]
        if rng.random() < 0.3:
            parts.append(ASPECT_POS[rng.integers(4)])
    elif tone == 2:
        parts = [s(rng, NEG_MILD, product)]
        if rng.random() < 0.4:
            parts.append("The " + ASPECT_POS[rng.integers(4)].lower())
    else:
        parts = [s(rng, SARCASM if rng.random() < 0.08 else NEG_STRONG, product)]
        if rng.random() < 0.5:
            parts.append(ASPECT_NEG[rng.integers(4)])
    text = OPENERS[rng.integers(len(OPENERS))] + " ".join(parts) + TAILS[rng.integers(len(TAILS))]
    return text.strip()


def generate_reviews(n: int = 5000, seed: int = 42, label_noise: float = 0.08) -> pd.DataFrame:
    """Return DataFrame[review, rating, product, timestamp] with Amazon-like star skew."""
    rng = np.random.default_rng(seed)
    ratings = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.09, 0.05, 0.08, 0.14, 0.64])
    products = rng.choice(PRODUCTS, size=n)
    days = rng.integers(0, 3 * 365, size=n)
    ts = (pd.Timestamp("2022-01-01") + pd.to_timedelta(days, unit="D")).strftime("%Y-%m-%d")
    rows = []
    for r, p in zip(ratings, products):
        tone = int(r)
        if rng.random() < label_noise:  # text disagrees with stars (real-data phenomenon)
            tone = int(np.clip(tone + rng.choice([-2, -1, 1, 2]), 1, 5))
        text = _compose(rng, tone, p)
        x = rng.random()
        if x < 0.25:
            text = text.replace(". ", ".<br />", 1) if ". " in text else text + "<br />"
        if rng.random() < 0.2:
            text += "!!!"
        if rng.random() < 0.1:
            text = text.lower()
        rows.append(text)
    return pd.DataFrame({"review": rows, "rating": ratings, "product": products, "timestamp": ts})
