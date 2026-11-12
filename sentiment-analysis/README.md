# Sentiment Analysis on E-Commerce / Food Product Reviews

End-to-end NLP system that classifies a review as **positive / neutral / negative**: data gathering → cleaning →
labelling → EDA → vectorisation → classical ML, BiLSTM/TextCNN and fine-tuned DistilBERT → evaluation → tuning →
FastAPI service → Streamlit dashboard. Implements the 10-step pipeline from the project brief.

> **Read this first — what has and has not been validated.** This was built in a sandbox that could not reach
> Kaggle, Hugging Face or Yelp. So: (1) every number below comes from a **synthetic** review generator
> (`data/synthetic.py`) and is **not** real-world performance; (2) the **BERT fine-tuning code is tested offline
> with a tiny random model** and has never run on real `distilbert-base-uncased` weights; (3) the Dockerfile was not
> built (no Docker daemon). Section "Running on real data" lists the exact commands. Brief's "BERT >90% F1" claim is
> unverified here.

## Layout
```
src/sentiment/
  data/        loaders.py (Amazon CSV/JSONL/HF/CSV) · scrape.py (BeautifulSoup + Yelp Fusion) · synthetic.py
  preprocessing.py   clean_text + rating→label + dedupe          eda.py          word clouds, class bar, lengths
  features.py        TF-IDF · BoW · Word2Vec(avg) · BERT embeds  evaluation.py   acc/P/R/F1, confusion, ROC-AUC
  models_ml.py       LR · NB · RF · XGBoost (+SMOTE/class wts)   train_ml.py     CV, GridSearchCV, model selection
  models_dl.py       BiLSTM · TextCNN (PyTorch)                  bert.py         DistilBERT/BERT fine-tuning
  predictor.py       one inference interface for all model kinds
  api/app.py         FastAPI                                     dashboard/      Streamlit + Plotly
  cli.py             `sentiment {data,eda,train,dl,bert,predict,serve,dashboard}`
tests/               67 tests · Dockerfile · TRADEOFFS.md · tasks/todo.md
```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt      # core + torch/transformers/gensim + pytest   (core only: requirements.txt)
pip install -e . --no-deps
python -m nltk.downloader stopwords wordnet   # REQUIRED before training and serving
pytest -q
```
`pytest` fails loudly (`test_nltk_resources_present`) if the NLTK data is missing, because preprocessing would
otherwise silently degrade (no lemmatisation, tiny stopword list). The model metadata records this and the
predictor warns if train/serve environments differ.

## Run the pipeline
```bash
sentiment data --source synthetic --n 20000          # or a real source, below
sentiment eda                                        # reports/figures/*.png
sentiment train                                      # LR, NB, RF, XGB — 5-fold CV, saves artifacts/best_model
sentiment train --tune                               # + GridSearchCV on macro-F1
sentiment train --imbalance smote                    # or: none | class_weight (default)
sentiment train --vectorizer w2v --models lr xgb     # tfidf (default) | bow | w2v
sentiment dl --arch bilstm                           # or cnn
sentiment bert --model-name distilbert-base-uncased  # needs HF access + ideally a GPU
sentiment predict "Burger was cold and soggy"
sentiment serve    # http://127.0.0.1:8000/docs         sentiment dashboard   # http://localhost:8501
```
Serve a different model with `--model-dir artifacts/bert` (or env `SENTIMENT_MODEL_DIR`).

## Running on real data
| Source | Command |
|---|---|
| Amazon Fine Food Reviews (`Reviews.csv` from Kaggle/SNAP) | `sentiment data --source amazon --path Reviews.csv --max-per-class 50000` |
| Any CSV (`review`, `rating` columns) | `sentiment data --source csv --path my.csv` |
| Hugging Face dataset | `pip install datasets` then `load_hf(...)` (see `data/loaders.py`) |
| Scrape / Yelp API | `data/scrape.py::scrape_reviews` (checks robots.txt), `fetch_yelp_reviews` (env `YELP_API_KEY`; API returns ≤3 excerpts/business) |

Then `sentiment train --tune` and `sentiment bert --epochs 3` (use a GPU; BERT training time was not measured here).
**Expect real-data numbers to be much lower than below**, and the neutral (3-star) class to be the hardest.

## API
| | |
|---|---|
| `GET /health` | `{"status":"ok","model_loaded":true}` (`degraded` if no model; `/predict` then returns 503) |
| `GET /model-info` | model kind, labels, held-out metrics |
| `POST /predict` | `{"review": "..."}` → `{review, sentiment, confidence, probabilities}` |
| `POST /predict/batch` | `{"reviews": [...]}` (≤100) → list of the above |

Inputs: 1–5000 chars, non-blank; else 422. **No authentication or rate limiting** — put it behind a gateway before
exposing it. Docker: `docker build -t sentiment-api . && docker run -p 8000:8000 sentiment-api` (needs
`artifacts/best_model`; **untested**, see above).

## Results on SYNTHETIC data (20k generated → 16,969 after dedupe; 76/15/9% pos/neg/neu; 80/20 stratified split)
Models are ranked by 5-fold CV macro-F1 on the training split only; the test split is used once for reporting.

| Model (TF-IDF 1-2gram, 5000 feat., class-weighted) | CV macro-F1 | Test acc | Test macro-F1 | ROC-AUC (ovr) |
|---|---|---|---|---|
| Logistic Regression | 0.940 | 0.966 | 0.933 | 0.978 |
| Naive Bayes | 0.838 | 0.896 | 0.840 | 0.974 |
| Random Forest | 0.934 | 0.964 | 0.928 | 0.977 |
| XGBoost | 0.936 | 0.966 | 0.932 | 0.978 |
| BiLSTM (6 ep, early stop) | – | 0.967 | 0.934 | 0.977 |
| TextCNN | – | 0.967 | 0.934 | 0.975 |

LR/BiLSTM/CNN agree on 99.97–100% of test predictions: the synthetic data has a fixed ceiling set by its injected 8%
text-vs-rating noise, and every competent model reaches it. **It therefore cannot rank models or show BERT's
advantage on sarcasm/nuance.** Only the engineering (no leakage, negation handling, serving) is demonstrated, not
modelling quality.

## Design decisions that matter (details in TRADEOFFS.md)
- **Negations are kept** (the brief's stopword snippet turns "not good" into "good").
- **Dedupe before splitting**, vectoriser/SMOTE/Word2Vec fitted **inside CV folds only**, model chosen by CV not test.
- **Macro-F1** is the headline metric (imbalance); accuracy is reported but not used for selection.
- BERT/DL see less-processed text than TF-IDF models, on purpose.
