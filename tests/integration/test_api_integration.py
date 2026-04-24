from fastapi.testclient import TestClient

from app.api.main import create_app


def test_ingest_route_uses_public_ingest_and_index_pipeline(monkeypatch):
    class _Paper:
        def model_dump(self):
            return {
                "arxiv_id": "1706.03762",
                "title": "Attention Is All You Need",
                "authors": ["A. Vaswani"],
                "abstract": "",
                "published": "2017-06-12",
                "processed": False,
            }

    monkeypatch.setattr("app.api.routes.ingest.ArxivFetcher.fetch_by_id", lambda self, arxiv_id: _Paper())
    monkeypatch.setattr("app.api.routes.ingest.ingest_paper", lambda paper: {"arxiv_id": "1706.03762"})
    monkeypatch.setattr("app.api.routes.ingest.index_paper", lambda parsed: ["c1", "c2", "c3"])

    client = TestClient(create_app())
    response = client.post("/ingest", json={"arxiv_id": "1706.03762"})

    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert response.json()["chunk_count"] == 3
