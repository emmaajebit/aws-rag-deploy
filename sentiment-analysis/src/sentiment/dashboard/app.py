"""Streamlit dashboard (spec step 10).

    streamlit run src/sentiment/dashboard/app.py
Env: SENTIMENT_MODEL_DIR (default artifacts/best_model), SENTIMENT_DATA (optional CSV with review[,rating,product,timestamp]).
"""
from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import streamlit as st

from sentiment.config import BEST_MODEL_DIR, LABELS, TEXT_COL
from sentiment.dashboard.helpers import monthly_trend, product_sentiment, score_reviews, top_reviews
from sentiment.data.loaders import load_reviews
from sentiment.predictor import SentimentPredictor

COLORS = {"negative": "#c0392b", "neutral": "#7f8c8d", "positive": "#27ae60"}

st.set_page_config(page_title="Review Sentiment Dashboard", layout="wide")


@st.cache_resource(show_spinner=False)
def get_predictor(path: str):
    return SentimentPredictor.load(path)


@st.cache_data(show_spinner="Scoring reviews…")
def get_scored(df: pd.DataFrame, path: str):
    return score_reviews(df, get_predictor(path))


model_dir = os.getenv("SENTIMENT_MODEL_DIR", str(BEST_MODEL_DIR))
st.title("Customer review sentiment")

try:
    predictor = get_predictor(model_dir)
except FileNotFoundError:
    st.error(f"No trained model found in `{model_dir}`. Run `sentiment train` first.")
    st.stop()

with st.sidebar:
    st.header("Data")
    up = st.file_uploader("Upload reviews CSV (needs a `review` column)", type="csv")
    st.caption(f"Model: **{predictor.description}** ({predictor.kind})")

if up is not None:
    raw = pd.read_csv(up)
    if TEXT_COL not in raw.columns:
        st.error("CSV must contain a `review` column."); st.stop()
elif os.getenv("SENTIMENT_DATA"):
    raw = pd.read_csv(os.environ["SENTIMENT_DATA"])
else:
    raw = load_reviews("synthetic", n=2000)
    st.info("Showing **synthetic demo data**. Upload a CSV or set SENTIMENT_DATA to use real reviews.")

scored = get_scored(raw, model_dir)

c1, c2, c3, c4 = st.columns(4)
share = scored["pred_sentiment"].value_counts(normalize=True)
c1.metric("Reviews", f"{len(scored):,}")
for col, lab in zip((c2, c3, c4), ("positive", "neutral", "negative")):
    col.metric(lab.title(), f"{share.get(lab, 0):.1%}")

left, right = st.columns(2)
with left:
    st.subheader("Sentiment distribution")
    counts = scored["pred_sentiment"].value_counts().reindex(LABELS).fillna(0).reset_index()
    counts.columns = ["sentiment", "count"]
    st.plotly_chart(px.pie(counts, names="sentiment", values="count", color="sentiment",
                           color_discrete_map=COLORS, hole=0.4), use_container_width=True)
with right:
    ps = product_sentiment(scored)
    st.subheader("Net sentiment by product")
    if ps.empty:
        st.caption("No `product` column in data.")
    else:
        st.plotly_chart(px.bar(ps.reset_index(), x="net_sentiment", y="product", orientation="h",
                               color="net_sentiment", color_continuous_scale="RdYlGn", range_color=(-1, 1)),
                        use_container_width=True)

tr = monthly_trend(scored)
if not tr.empty:
    st.subheader("Sentiment over time")
    st.plotly_chart(px.area(tr, x="month", y=LABELS, color_discrete_map=COLORS), use_container_width=True)

t1, t2 = st.columns(2)
with t1:
    st.subheader("Top positive reviews"); st.dataframe(top_reviews(scored, "positive"), use_container_width=True)
with t2:
    st.subheader("Top negative reviews (flag for follow-up)")
    st.dataframe(top_reviews(scored, "negative"), use_container_width=True)

st.divider()
st.subheader("Try a review")
txt = st.text_area("Review text", placeholder="The burger was cold and soggy.")
if st.button("Predict") and txt.strip():
    res = predictor.predict([txt])[0]
    st.markdown(f"**{res['sentiment'].upper()}** — confidence {res['confidence']:.1%}")
    st.bar_chart(pd.Series(res["probabilities"]))
