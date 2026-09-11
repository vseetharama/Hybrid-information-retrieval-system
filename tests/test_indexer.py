import json

import pytest

from indexer import load_documents


def test_dataset_has_valid_unique_documents():
    corpus = load_documents()
    ids = [document["id"] for document in corpus.documents]
    assert len(ids) == len(set(ids))
    assert all(corpus.processed_texts)


def test_invalid_document_is_rejected(tmp_path):
    path = tmp_path / "documents.json"
    path.write_text(json.dumps([{"id": "1", "title": "Missing content"}]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_documents(path)