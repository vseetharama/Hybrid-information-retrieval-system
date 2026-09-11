import json
from pathlib import Path
from typing import Any

import faiss
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config import ARTIFACTS_DIR, EMBEDDING_DIMENSION, EMBEDDING_MODEL, RRF_K
from preprocess import preprocess, preprocess_to_text


class SearchEngine:
    def __init__(self, artifacts_dir: Path = ARTIFACTS_DIR, model_name: str = EMBEDDING_MODEL) -> None:
        self.artifacts_dir = artifacts_dir
        self.model_name = model_name
        self._loaded = False

    def load(self) -> None:
        metadata_path = self.artifacts_dir / "metadata.json"
        required = [metadata_path, self.artifacts_dir / "bm25.pkl", self.artifacts_dir / "tfidf.pkl", self.artifacts_dir / "faiss.index"]
        if not all(path.exists() for path in required):
            raise FileNotFoundError("Search artifacts are missing; call /index first")
        self.bm25 = joblib.load(self.artifacts_dir / "bm25.pkl")
        tfidf = joblib.load(self.artifacts_dir / "tfidf.pkl")
        self.vectorizer = tfidf["vectorizer"]
        self.tfidf_matrix = tfidf["matrix"]
        self.faiss_index = faiss.read_index(str(self.artifacts_dir / "faiss.index"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.documents = metadata["documents"]
        self.model = SentenceTransformer(metadata.get("embedding_model", self.model_name))
        if self.faiss_index.d != EMBEDDING_DIMENSION:
            raise ValueError(f"Expected FAISS dimension {EMBEDDING_DIMENSION}, got {self.faiss_index.d}")
        self._loaded = True

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    @staticmethod
    def _ranked(scores: np.ndarray, ids: list[str]) -> list[dict[str, Any]]:
        order = sorted(range(len(ids)), key=lambda position: (-float(scores[position]), position))
        return [{"id": ids[position], "score": float(scores[position]), "rank": rank} for rank, position in enumerate(order, start=1)]

    def search(self, query: str, top_k: int = 10) -> dict[str, Any]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        self._ensure_loaded()
        query_tokens = preprocess(query)
        if not query_tokens:
            raise ValueError("query contains no searchable terms")
        ids = [document["id"] for document in self.documents]
        bm25_scores = np.asarray(self.bm25.get_scores(query_tokens), dtype=float)
        bm25 = self._ranked(bm25_scores, ids)
        query_vector = self.vectorizer.transform([" ".join(query_tokens)])
        tfidf_scores = cosine_similarity(query_vector, self.tfidf_matrix).ravel()
        tfidf = self._ranked(tfidf_scores, ids)
        embedding = self.model.encode([preprocess_to_text(query)], convert_to_numpy=True, normalize_embeddings=False).astype("float32")
        faiss.normalize_L2(embedding)
        semantic_scores, semantic_positions = self.faiss_index.search(embedding, len(ids))
        semantic = [{"id": ids[int(position)], "score": float(score), "rank": rank} for rank, (score, position) in enumerate(zip(semantic_scores[0], semantic_positions[0]), start=1) if position >= 0]

        by_method = {name: {item["id"]: item for item in values} for name, values in {"bm25": bm25, "tfidf": tfidf, "semantic": semantic}.items()}
        all_ids = set().union(*[set(values) for values in by_method.values()])
        documents_by_id = {document["id"]: document for document in self.documents}
        fused = []
        for document_id in all_ids:
            ranks = {name: by_method[name].get(document_id, {}).get("rank") for name in by_method}
            score = sum(1.0 / (RRF_K + rank) for rank in ranks.values() if rank is not None)
            fused.append({
                **documents_by_id[document_id],
                "bm25_score": by_method["bm25"].get(document_id, {}).get("score"),
                "tfidf_score": by_method["tfidf"].get(document_id, {}).get("score"),
                "semantic_score": by_method["semantic"].get(document_id, {}).get("score"),
                "bm25_rank": ranks["bm25"],
                "tfidf_rank": ranks["tfidf"],
                "semantic_rank": ranks["semantic"],
                "rrf_score": score,
            })
        fused.sort(key=lambda item: (-item["rrf_score"], item["id"]))
        for rank, item in enumerate(fused, start=1):
            item["final_rank"] = rank
            item["snippet"] = item["content"][:240]
        return {"query": query, "results": fused[:top_k], "total_results": len(fused)}


def reciprocal_rank_fusion(rankings: dict[str, list[str]], k: int = RRF_K) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings.values():
        for rank, document_id in enumerate(ranking, start=1):
            scores[document_id] = scores.get(document_id, 0.0) + 1.0 / (k + rank)
    return scores
