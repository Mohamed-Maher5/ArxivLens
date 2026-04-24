from types import SimpleNamespace

from app.indexing.vector_store import VectorStore, collection_name_from_arxiv_id
from app.models.schemas import Chunk


class _FakeQdrantClient:
    def __init__(self):
        self.collections: list[str] = []
        self.created: list[str] = []
        self.upserts: list[tuple[str, list]] = []

    def get_collections(self):
        return SimpleNamespace(
            collections=[SimpleNamespace(name=name) for name in self.collections]
        )

    def create_collection(self, collection_name, **kwargs):
        self.collections.append(collection_name)
        self.created.append(collection_name)

    def upsert(self, collection_name, points):
        self.upserts.append((collection_name, points))

    def get_collection(self, collection_name):
        return SimpleNamespace(vectors_count=1, points_count=1)


def test_collection_name_from_arxiv_id_sanitizes_input():
    assert collection_name_from_arxiv_id("1706.03762v2") == "paper_1706_03762v2"
    assert collection_name_from_arxiv_id("cs/0112017") == "paper_cs_0112017"


def test_vector_store_creates_collection_and_stores_payload():
    client = _FakeQdrantClient()
    store = VectorStore("1706.03762v2", qdrant_client=client)

    chunk = Chunk(
        chunk_id="c1",
        arxiv_id="1706.03762v2",
        paper_title="Attention Is All You Need",
        authors=["A. Vaswani"],
        content="Multi-head attention allows the model to attend to different subspaces.",
        chunk_type="content",
        page_number=4,
    )

    store.store(
        [
            {
                "chunk": chunk,
                "dense_vector": [0.1] * 1024,
                "sparse_vector": {"7": 0.8, "9": 0.2},
            }
        ]
    )

    assert client.created == ["paper_1706_03762v2"]
    assert client.upserts
    collection_name, points = client.upserts[0]
    assert collection_name == "paper_1706_03762v2"
    assert points[0].payload["paper_title"] == "Attention Is All You Need"
    assert points[0].payload["page_number"] == 4
