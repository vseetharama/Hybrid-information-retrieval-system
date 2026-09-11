from fastapi.testclient import TestClient

from api import app


def test_health_endpoint():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_search_rejects_empty_query():
    response = TestClient(app).post("/search", json={"query": "   "})
    assert response.status_code in {400, 422}


def test_evaluate_rejects_invalid_k():
    response = TestClient(app).post("/evaluate", json={"k": 0})
    assert response.status_code == 422