import pytest

from evaluator import evaluate


def fake_search(query, top_k):
    return {"results": [{"id": value} for value in {"q1": ["a", "x", "b"], "q2": ["x", "b"]}[query][:top_k]]}


def test_evaluation_calculates_precision_recall_and_mrr():
    result = evaluate({"q1": ["a", "b"], "q2": ["b"]}, fake_search, 2)
    assert result["mean_precision_at_k"] == 0.5
    assert result["mean_recall_at_k"] == 0.75
    assert result["mrr"] == 0.75


def test_evaluation_rejects_invalid_k():
    with pytest.raises(ValueError):
        evaluate({}, fake_search, 0)