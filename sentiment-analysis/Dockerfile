# Inference image: FastAPI + best sklearn model. For a BERT model use requirements-dl.txt instead.
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt pyproject.toml ./
RUN pip install -r requirements.txt
COPY src ./src
ENV NLTK_DATA=/usr/share/nltk_data
RUN pip install --no-deps . && python -c "import nltk; [nltk.download(p, quiet=True, download_dir='/usr/share/nltk_data') for p in ('stopwords','wordnet')]"
# Model artifacts are mounted or baked in at build time (see README "Docker").
COPY artifacts/best_model /app/artifacts/best_model
ENV SENTIMENT_MODEL_DIR=/app/artifacts/best_model
RUN useradd -m app && chown -R app /app
USER app
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "sentiment.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
