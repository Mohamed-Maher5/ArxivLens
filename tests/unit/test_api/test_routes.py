from fastapi.testclient import TestClient

from app.api.main import create_app


def test_health_route_returns_healthy_status():
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_chat_route_returns_serialized_pipeline_result(monkeypatch):
    class _Result:
        question = "What is attention?"
        answer = "Attention is a weighted aggregation mechanism."
        contextualized_query = "What is attention?"
        sources = []

    monkeypatch.setattr("app.api.routes.chat.get_llm_client", lambda: object())
    monkeypatch.setattr(
        "app.api.routes.chat.Pipeline.run",
        lambda self, message, arxiv_id=None: _Result(),
    )

    client = TestClient(create_app())
    response = client.post("/chat", json={"message": "What is attention?", "arxiv_id": "1706.03762"})

    assert response.status_code == 200
    assert response.json()["answer"] == "Attention is a weighted aggregation mechanism."
