import json
from pathlib import Path
from typing import Any, Callable

from config import GROUND_TRUTH_FILE


def load_ground_truth(path: Path = GROUND_TRUTH_FILE) -> dict[str, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Ground-truth file does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError("ground_truth.json must contain a non-empty object")
    return {str(query): [str(document_id) for document_id in ids] for query, ids in data.items()}


def evaluate(ground_truth: dict[str, list[str]], search: Callable[[str, int], dict[str, Any]], k: int = 10) -> dict[str, Any]:
    if k <= 0:
        raise ValueError("k must be greater than zero")
    per_query = []
    for query, relevant_ids in ground_truth.items():
        results = search(query, k)["results"]
        retrieved_ids = [item["id"] for item in results[:k]]
        relevant = set(relevant_ids)
        hits = [document_id for document_id in retrieved_ids if document_id in relevant]
        first_rank = next((rank for rank, document_id in enumerate(retrieved_ids, start=1) if document_id in relevant), None)
        per_query.append({
            "query": query,
            "num_relevant_documents": len(relevant),
            "retrieved_document_ids": retrieved_ids,
            "precision_at_k": len(hits) / k,
            "recall_at_k": len(hits) / len(relevant) if relevant else 0.0,
            "reciprocal_rank": 1.0 / first_rank if first_rank else 0.0,
        })
    count = len(per_query)
    return {
        "k": k,
        "num_queries": count,
        "per_query": per_query,
        "mean_precision_at_k": sum(item["precision_at_k"] for item in per_query) / count,
        "mean_recall_at_k": sum(item["recall_at_k"] for item in per_query) / count,
        "mrr": sum(item["reciprocal_rank"] for item in per_query) / count,
    }
