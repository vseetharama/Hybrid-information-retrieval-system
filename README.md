# Hybrid Information Retrieval System

This project combines NLTK preprocessing, BM25Okapi, TF-IDF cosine similarity, Sentence Transformer embeddings, FAISS inner-product search, and Reciprocal Rank Fusion (RRF). The backend is FastAPI and the UI is plain HTML, CSS, and JavaScript.

## Run locally

Use Python 3.10 or newer (Python 3.11 is recommended for the pinned ecosystem):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn api:app --reload
```

Open `http://127.0.0.1:8000`, click **Rebuild index**, and search. The first indexing run downloads `all-MiniLM-L6-v2` and the NLTK resources. Artifacts are written under `artifacts/` and can be rebuilt whenever the corpus changes.

## API

- `GET /health`
- `POST /index`
- `POST /search` with `{"query": "machine learning", "top_k": 5}`
- `POST /evaluate` with `{"k": 10}`

Search results expose the raw BM25, TF-IDF, and semantic scores, their actual ranks, each document's RRF score, and final rank. RRF uses $k=60$ and is not a percentage.

## Tests and Docker

```powershell
pytest -q
docker build -t hybrid-ir .
docker run --rm -p 8000:8000 hybrid-ir
```

The Docker container includes NLTK data. The Sentence Transformer model is downloaded on the first `/index` request and needs network access unless the Hugging Face cache is supplied separately.