from search_engine import reciprocal_rank_fusion


def test_rrf_uses_one_based_actual_ranks():
    scores = reciprocal_rank_fusion({"bm25": ["a", "b"], "tfidf": ["b", "a"], "semantic": ["a"]})
    assert scores["a"] == 1 / 61 + 1 / 62 + 1 / 61
    assert scores["b"] == 1 / 62 + 1 / 61


def test_rrf_missing_rankings_contribute_nothing():
    scores = reciprocal_rank_fusion({"bm25": ["a"], "tfidf": [], "semantic": []})
    assert scores == {"a": 1 / 61}