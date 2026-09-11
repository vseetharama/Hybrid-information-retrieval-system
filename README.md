# Hybrid Information Retrieval System

## 1. Overview

The Hybrid Information Retrieval System is an academic project that combines lexical, statistical, and semantic retrieval methods. It preprocesses a JSON document collection, builds three complementary indexes, combines their rankings with Reciprocal Rank Fusion (RRF), and exposes the resulting search service through a FastAPI backend and a plain HTML/CSS/JavaScript web interface.

## 2. Problem Statement

Individual information-retrieval methods have different strengths. Exact term matching can miss documents that express the same idea with different words, while semantic matching can overlook important exact terms. The project addresses this problem by combining lexical matching, TF-IDF similarity, and semantic embedding search into one ranked result set.

## 3. Objectives

- Preprocess documents and queries consistently using NLTK.
- Retrieve documents using BM25Okapi, TF-IDF cosine similarity, and semantic embeddings.
- Use FAISS for efficient inner-product search over normalized embeddings.
- Fuse the three ranked lists using Reciprocal Rank Fusion.
- Provide an HTTP API and browser-based interface for indexing, searching, health checks, and evaluation.
- Measure retrieval quality using Precision@K, Recall@K, and Mean Reciprocal Rank.
- Support repeatable local and Docker-based execution.

## 4. Key Features

- NLTK tokenization, lowercasing, punctuation removal, stop-word filtering, and lemmatization.
- BM25Okapi lexical retrieval.
- TF-IDF vectorization with cosine similarity.
- Sentence Transformer semantic embeddings using `all-MiniLM-L6-v2`.
- FAISS inner-product search over normalized 384-dimensional vectors.
- Reciprocal Rank Fusion with `k = 60`.
- Search results containing document information, a content snippet, method scores, method ranks, RRF score, and final rank.
- JSON document corpus and JSON ground-truth evaluation data.
- FastAPI backend, browser frontend, automated pytest tests, and Docker support.

## 5. System Architecture

```text
User Query
    |
    v
Text Preprocessing
    |
    +----------------+------------------+
    |                |                  |
    v                v                  v
  BM25             TF-IDF        Sentence Transformer
    |                |                  |
    |                |                  v
    |                |                FAISS
    |                |                  |
    +----------------+------------------+
                     |
                     v
          Reciprocal Rank Fusion
                     |
                     v
             Final Ranked Results
                     |
                     v
                API / Web UI
```

The FastAPI application invokes the indexer when `/index` is called. The `SearchEngine.search()` method then runs all three retrieval methods for each query and combines their rankings before returning the final results.

## 6. How the Hybrid Retrieval Works

### Text preprocessing

Documents and queries are normalized with NLTK. The preprocessing pipeline lowercases text, removes punctuation, tokenizes it, removes English stop words, and lemmatizes the remaining alphabetic tokens. Required NLTK resources are downloaded when they are not already available.

### BM25

BM25Okapi provides lexical retrieval based on term frequency, inverse document frequency, and document-length normalization. It is useful when query terms occur explicitly in a document.

### TF-IDF

The system creates a TF-IDF representation of the preprocessed documents and compares a preprocessed query with the document matrix using cosine similarity. This provides a statistical term-similarity ranking that complements BM25.

### Sentence Transformer + FAISS

The `all-MiniLM-L6-v2` Sentence Transformer generates a 384-dimensional embedding for each processed document and query. The vectors are L2-normalized and stored in a FAISS inner-product index. With normalized vectors, the inner product represents semantic similarity between the query and documents.

### Reciprocal Rank Fusion

BM25, TF-IDF, and semantic search each produce a ranked list. RRF combines those ranks using:

```text
RRF score(document) = sum(1 / (k + rank))
```

The configured value is `k = 60`. The RRF score is a ranking score, not a percentage. The fused results are sorted by RRF score and assigned a final rank.

A hybrid approach is useful because BM25 handles lexical matching, TF-IDF provides statistical term similarity, semantic search handles meaning and context, and RRF combines their rankings.

## 7. Technology Stack

| Area | Technology |
| --- | --- |
| Language | Python 3.10 or newer; Python 3.11 recommended |
| Text preprocessing | NLTK |
| Lexical retrieval | `rank-bm25` with BM25Okapi |
| Statistical retrieval | scikit-learn TF-IDF and cosine similarity |
| Semantic retrieval | Sentence Transformers, `all-MiniLM-L6-v2` |
| Vector search | FAISS with normalized inner-product search |
| Backend API | FastAPI and Uvicorn |
| Frontend | Plain HTML, CSS, and JavaScript |
| Evaluation and testing | Python evaluation module and pytest |
| Containerization | Docker |
| Stored data | JSON documents and ground truth |

## 8. Dataset

The project uses a local JSON corpus stored in `data/documents.json`. Each document contains an identifier, title, and content. Retrieval evaluation queries and their relevant document identifiers are stored in `data/ground_truth.json`.

The system is therefore intended for experimentation on this local corpus. Retrieval quality depends on the coverage, correctness, and relevance labels in those files.

## 9. Project Structure

```text
.
├── api.py
├── config.py
├── Dockerfile
├── evaluator.py
├── indexer.py
├── main.py
├── preprocess.py
├── README.md
├── requirements.txt
├── search_engine.py
├── artifacts/                 # Generated index and model artifacts
│   ├── bm25.pkl
│   ├── faiss.index
│   ├── metadata.json
│   └── tfidf.pkl
├── data/
│   ├── documents.json
│   └── ground_truth.json
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── style.css
└── tests/
    ├── test_api.py
    ├── test_evaluator.py
    ├── test_indexer.py
    ├── test_preprocess.py
    └── test_search_engine.py
```

The files under `artifacts/` are generated by indexing and can be rebuilt when the corpus changes. They are not source code.

## 10. Installation

Use Python 3.10 or newer. Python 3.11 is recommended for the current dependency set.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The first indexing operation may download the `all-MiniLM-L6-v2` Sentence Transformer model and the required NLTK resources. Network access is required for those downloads unless the resources and model are already available locally.

## 11. Running the Application

Start the development server with:

```powershell
python -m uvicorn api:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in a browser. Use **Rebuild index** in the web interface before searching if the generated artifacts are missing or the corpus has changed.

## 12. API Endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Returns service health and whether the search engine is loaded. |
| `POST` | `/index` | Builds BM25, TF-IDF, and FAISS artifacts from the JSON corpus. |
| `POST` | `/search` | Runs the three retrieval methods and returns fused results. |
| `POST` | `/evaluate` | Evaluates the search engine against the ground-truth file. |

### `/search` request

```json
{
  "query": "machine learning",
  "top_k": 5
}
```

The `query` must be non-empty and `top_k` must be greater than zero.

### `/evaluate` request

```json
{
  "k": 10
}
```

The value of `k` must be greater than zero.

## 13. Example Search

The implementation has been verified with the query `machine learning algorithms`. One observed top result was:

| Field | Observed value |
| --- | --- |
| Document | Document 2 - Machine Learning Algorithms |
| BM25 rank | 1 |
| TF-IDF rank | 1 |
| Semantic rank | 1 |
| RRF score | `0.04918032786885246` |
| Final rank | 1 |

Search results also include the BM25, TF-IDF, and semantic scores; their ranks; the RRF score; the final rank; the document identifier, title, content, and a short snippet.

## 14. Evaluation

The `/evaluate` endpoint loads `data/ground_truth.json`, runs every ground-truth query through the search engine, and reports per-query and aggregate retrieval measures:

- **Precision@K:** the proportion of the top `K` retrieved documents that are relevant.
- **Recall@K:** the proportion of relevant documents retrieved in the top `K` results.
- **Mean Reciprocal Rank (MRR):** the mean of the reciprocal rank of the first relevant result across evaluation queries.

No aggregate evaluation scores are recorded in the repository documentation or source files, so no performance percentage is claimed here.

## 15. Testing

Run the automated tests with:

```powershell
pytest -q
```

The verified test result is:

```text
12 passed, 2 warnings
```

The two warnings are dependency deprecation warnings; they are not test failures.

## 16. Docker

Build and run the application with:

```powershell
docker build -t hybrid-ir .
docker run --rm -p 8000:8000 hybrid-ir
```

The Docker image installs the Python dependencies and downloads the NLTK resources during the image build. The Sentence Transformer model is downloaded on the first `/index` request, so the container needs network access at that point unless an appropriate Hugging Face cache is supplied separately.

## 17. Limitations

- The system uses a local JSON corpus rather than a production-scale document store.
- Retrieval quality depends on the size and quality of the corpus and ground-truth labels.
- The Sentence Transformer model and FAISS artifacts require local storage and computational resources.
- Generated artifacts must be rebuilt when the corpus changes.
- The first indexing operation can require model and NLTK downloads and therefore network access.
- The project provides a development-oriented API and frontend; authentication, user management, and production deployment concerns are outside the current implementation.

## 18. Future Enhancements

Possible future enhancements, not currently implemented, include:

- Adding larger or domain-specific datasets and richer relevance judgments.
- Supporting incremental indexing and document updates.
- Adding configurable retrieval weights or fusion strategies.
- Improving API authentication, validation, observability, and production deployment.
- Adding richer frontend result filtering, pagination, and analytics.
- Comparing additional embedding models and retrieval configurations through systematic experiments.

## 19. Project Status

The current implementation is functional for local demonstration and academic evaluation. It includes preprocessing, hybrid indexing and search, RRF ranking, evaluation endpoints, automated tests, a browser interface, and Docker support.

## 20. Authors

This is a college academic project. Add the project author or team member names here for the final submission.