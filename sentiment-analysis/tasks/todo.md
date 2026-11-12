# Sentiment Analysis on E-Commerce / Food Reviews — implementation plan

Source spec: "Project Explanation For NLP Sentiment Analysis on E-Commerce Food Product Reviews" (10-step pipeline).
Lives in `sentiment-analysis/`, independent of the RAG app at the repo root.

## Environment constraints found (drive design)
- Hugging Face, Kaggle, SNAP, Yelp are NOT reachable from this sandbox; PyPI and raw GitHub are.
- Consequence: real Amazon Fine Food Reviews cannot be downloaded here, and pretrained
  DistilBERT/BERT weights cannot be fetched. Design: pluggable loaders for real data +
  a synthetic generator so everything is runnable/testable offline; BERT script is verified
  offline with a tiny randomly-initialised config and must be run with real weights elsewhere.

## Steps (map to spec sections)
- [x] 1  Data gathering: `data/loaders.py` (Amazon CSV, JSONL, HF), `data/synthetic.py`, `data/scrape.py` (BeautifulSoup + Yelp Fusion client)
- [x] 2  Preprocessing `preprocessing.py` (spec's clean_text, but keep negations — see TRADEOFFS)
- [x] 3  Labeling `rating_to_sentiment` in `preprocessing.py` (4-5 pos, 3 neu, 1-2 neg)
- [x] 4  EDA `eda.py` (word clouds, class bar, length dist)
- [x] 5  Vectorisation `features.py` (TF-IDF, BoW, Word2Vec avg, transformer embeddings)
- [x] 6  Models `models_ml.py` (LR, NB, RF, XGB), `models_dl.py` (BiLSTM, TextCNN), `bert.py` (DistilBERT fine-tune)
- [x] 7  Evaluation `evaluation.py` (acc, P/R/F1, confusion matrix, ROC-AUC OvR)
- [x] 8  Optimisation: GridSearchCV, stratified CV, class_weight, SMOTE (inside pipeline, train folds only)
- [x] 9  FastAPI service + Dockerfile
- [x] 10 Streamlit + Plotly dashboard
- [x] CLI `sentiment {data,eda,train,dl,bert,predict,serve,dashboard}` (evaluation is run inside train/dl/bert)
- [x] pytest suite, README.md, TRADEOFFS.md
- [x] Verify: run full pipeline end to end, run tests, boot API and hit it, import-check dashboard
- [x] Commit + push to `claude/cool-hawking-mg3nfe` (no PR unless asked)

## Review
- Verified: 67 pytest tests pass (4 mutation checks: negation stopwords, SMOTE, selection-on-test, train/serve cleaning
  each fail the suite when broken); real `sentiment serve` + `sentiment dashboard` processes booted and exercised by curl.
- NOT verified: real datasets, BERT on pretrained weights/GPU, Docker build (no daemon).
- Bugs found by running it: NLTK silent fallback (now loud), NLTK 3.10 data perms, pandas-3 Arrow strings breaking CV.
- Lesson: a test that only checks output shape does not prove a component (SMOTE) exists — pin behaviour with a spy.
