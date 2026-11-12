# TRADEOFFS

What was decided, why, and what it costs. Items marked **OPEN** need your confirmation.

## 1. Synthetic data instead of a real corpus (OPEN — biggest caveat)
- **Decision:** ship a generator + pluggable loaders; run everything on generated reviews.
- **Why:** Kaggle, Hugging Face, SNAP and Yelp were unreachable from the build sandbox; the only copies found were in
  third-party GitHub repos outside this session's repository scope, so they were not used.
- **Cost:** reported metrics say nothing about real performance. The generator's templates are also easy: word clouds
  are dominated by its label-independent filler ("last week", "arrive box") and top "positive" features include
  "however" (an artefact of how 4-star text is composed). **Action:** run `sentiment data --source amazon` on the real
  file and treat only those numbers as results.

## 2. BERT is code-complete but not validated on real weights (OPEN)
- Verified: full fine-tuning loop, checkpointing of best epoch, label mapping, and serving, on a tiny random BERT
  built locally in `tests/test_nn.py`. Not verified: convergence/accuracy of `distilbert-base-uncased`, GPU/AMP path,
  the brief's ">90% F1" claim.
- DistilBERT is the default (≈40% smaller, ≈60% faster than BERT-base, ≈97% of its GLUE score per its authors) — fine
  for a CPU-served API; use `--model-name bert-base-uncased` if the accuracy gap matters.
- No hyper-parameter search for BERT (lr 2e-5, 3 epochs, max_len 128 are standard defaults). Reviews longer than 128
  tokens are truncated (the end of long reviews, where verdicts often sit, is lost).

## 3. Departures from the brief's code
| Brief | Here | Why |
|---|---|---|
| Remove all NLTK English stopwords | Keep negations and `but/very/too/…` | Stopword removal flips "not good" → "good". Measured: with them kept, "not good" scores negative, "not bad at all" positive |
| `WordNetLemmatizer().lemmatize(w)` | verb pass then noun pass | default (noun) pass leaves "loved" unchanged, contradicting the brief's own "loved → love" example |
| strip non-letters then tokenise | expand `n't` first | otherwise "wasn't" → "wasnt" |
| TF-IDF `max_features=5000` | same, plus 1–2-grams, `min_df=2`, sublinear tf | bigrams capture "not good" / "not bad"; 5000 cap kept per brief |
| – | dedupe reviews before split | Amazon food data contains many duplicate texts; leaving them in inflates test scores |
| ROC-AUC "for binary versions" | one-vs-rest macro AUC **and** positive-vs-negative AUC | brief's problem is 3-class |

## 4. Imbalance handling
- Default is **class weights** (no data duplication, cheap, no leakage risk). **SMOTE** is available but sits *inside*
  the pipeline so it only ever sees training folds. On 5000-dim sparse TF-IDF, SMOTE interpolates between sparse
  vectors, which are not meaningful sentences; it is slow on large corpora. Prefer class weights at scale.
- Naive Bayes has no `class_weight`; `fit_prior=False` (uniform prior) is used as the equivalent.
- Star-to-label mapping (3★ = neutral) is a **proxy**; 3-star reviews are frequently mixed or mildly positive, so the
  neutral class carries label noise that caps achievable F1 on real data.

## 5. Model selection protocol
- Model chosen by CV macro-F1 on the training split; the test split is reported once. With `--tune`, GridSearchCV's
  `best_score_` is itself optimistic (selected on the same folds) — the held-out test number is the honest one.
- One random 80/20 split, one seed: no confidence intervals. Differences < ~0.5 pt macro-F1 are within noise.
- Time-based splitting (train on past, test on recent) would be more realistic for a deployed system and is not done.

## 6. Serving and operations
- API: no auth, no rate limit, no request logging/metrics, single process. Batch cap 100, text cap 5000 chars.
- Model drift is not monitored; there is no retraining job. Predictions on non-English text are unreliable (the
  models are English-only) — brief lists multilingual support as future work.
- **Dockerfile unbuilt** (no Docker daemon in the sandbox). It runs as non-root and has a healthcheck, but treat it as
  a draft until `docker build` has been run.
- **PII:** reviews can contain personal data. Nothing is stored by the API; the dashboard processes uploads in memory.
  If you ingest scraped/Yelp data, check the source's terms (the scraper honours robots.txt and fails closed).

## 7. Environment gotchas found while building (worth knowing)
- NLTK ≥3.10 refuses data directories that are group/other-writable → `chmod -R go-w ~/nltk_data`.
- pandas ≥3 string columns are Arrow-backed; `.values` breaks scikit-learn CV slicing → use `as_text_array()`.
- Silent preprocessing fallback was the original design; it hid missing NLTK data. It now warns, is recorded in model
  metadata, is checked at load time, and is asserted in the test-suite.

## 8. Not implemented (from the brief's "next version")
Multilingual support, aspect-based sentiment, LIME/SHAP explainability (only top-n n-gram weights for linear models
are stored), and serverless deployment (Lambda / Cloud Run). Heroku/AWS/GCP deploy steps are not scripted.
