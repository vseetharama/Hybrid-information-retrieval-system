from preprocess import preprocess


def test_preprocess_normalizes_and_lemmatizes():
    assert preprocess("Cats, RUNNING! in the gardens.") == ["cat", "running", "garden"]


def test_preprocess_handles_empty_and_punctuation():
    assert preprocess("") == []
    assert preprocess("   !!!") == []


def test_stopword_only_query_is_empty():
    assert preprocess("the and is") == []