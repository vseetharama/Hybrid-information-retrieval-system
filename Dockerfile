FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    NLTK_DATA=/usr/local/share/nltk_data \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python -c "import nltk; [nltk.download(x, download_dir='/usr/local/share/nltk_data') for x in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']]"

COPY . .
RUN mkdir -p artifacts
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]