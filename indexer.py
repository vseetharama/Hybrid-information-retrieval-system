import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import joblib
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from config import (
    ARTIFACTS_DIR,
    DOCUMENTS_FILE,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
)
from preprocess import preprocess, preprocess_to_text


@dataclass
class Corpus:
    documents: list[dict[str, str]]
    tokens: list[list[str]]
    processed_texts: list[str]


def load_documents(path: Path = DOCUMENTS_FILE) -> Corpus:
    if not path.exists():
        raise FileNotFoundError(f"Document file does not exist: {path}")
    with path.open(encoding="utf-8") as file:
        documents = json.load(file)
    if not isinstance(documents, list) or not documents:
        raise ValueError("documents.json must contain a non-empty list")

    normalized: list[dict[str, str]] = []
    ids: set[str] = set()
    for item in documents:
        if not isinstance(item, dict) or not {"id", "title"}.issubset(item):
            raise ValueError("Each document requires id and title fields")
        content = item.get("content", item.get("text", ""))
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"Document {item.get('id')} has empty content")
        document_id = str(item["id"])
        if document_id in ids:
            raise ValueError(f"Duplicate document id: {document_id}")
        ids.add(document_id)
        normalized.append({"id": document_id, "title": str(item["title"]), "content": content})

    tokens = [preprocess(f"{doc['title']} {doc['content']}") for doc in normalized]
    processed_texts = [" ".join(item) for item in tokens]
    return Corpus(normalized, tokens, processed_texts)


def build_index(
    documents_path: Path = DOCUMENTS_FILE,
    artifacts_dir: Path = ARTIFACTS_DIR,
    model_name: str = EMBEDDING_MODEL,
) -> dict[str, Any]:
    corpus = load_documents(documents_path)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    bm25 = BM25Okapi(corpus.tokens)
    vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus.processed_texts)

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        corpus.processed_texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=False,
    ).astype("float32")
    if embeddings.ndim != 2 or embeddings.shape[1] != EMBEDDING_DIMENSION:
        raise ValueError(f"Expected embeddings with dimension {EMBEDDING_DIMENSION}, got {embeddings.shape}")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
    index.add(embeddings)

    joblib.dump(bm25, artifacts_dir / "bm25.pkl")
    joblib.dump(
        {"vectorizer": vectorizer, "matrix": tfidf_matrix},
        artifacts_dir / "tfidf.pkl",
    )
    faiss.write_index(index, str(artifacts_dir / "faiss.index"))
    metadata = {
        "documents": corpus.documents,
        "processed_texts": corpus.processed_texts,
        "embedding_model": model_name,
        "embedding_dimension": EMBEDDING_DIMENSION,
    }
    (artifacts_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"document_count": len(corpus.documents), "vocabulary_size": len(vectorizer.vocabulary_), "embedding_shape": list(embeddings.shape)}
