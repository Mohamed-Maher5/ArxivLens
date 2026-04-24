import re
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    SparseVectorParams,
    SparseIndexParams,
    PointStruct,
    SparseVector
)
from app.core.logger import logger
from app.core.exceptions import QdrantConnectionError
from app.core.settings import settings
from app.interfaces.vector_store import VectorStore as VectorStoreInterface


DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
VECTOR_SIZE = 1024


def collection_name_from_arxiv_id(arxiv_id: str) -> str:
    """
    Derive a Qdrant-safe collection name from an ArXiv paper ID.

    Examples:
        "1706.03762"    -> "paper_1706_03762"
        "1706.03762v2"  -> "paper_1706_03762v2"
        "2301.00001"    -> "paper_2301_00001"

    Rules applied:
        - Prefix with "paper_"
        - Replace dots and any other non-alphanumeric chars (except underscores) with "_"
    """
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", arxiv_id)
    return f"paper_{sanitized}"


class VectorStore(VectorStoreInterface):
    """
    Manages per-paper Qdrant collections.

    Each paper gets its own collection named after its ArXiv ID
    (e.g. paper_1706_03762). Collections are created on first use
    and reused on subsequent calls.
    """

    def __init__(self, arxiv_id: str, qdrant_client: QdrantClient | None = None):
        """
        Args:
            arxiv_id: ArXiv paper ID used to derive the collection name.
        """
        self.collection = collection_name_from_arxiv_id(arxiv_id)
        try:
            if qdrant_client is not None:
                self.client = qdrant_client
                logger.info(f"Using injected Qdrant client (collection: {self.collection})")
            else:
                self.client = QdrantClient(url=settings.qdrant_url)
                logger.info(f"Connected to Qdrant at {settings.qdrant_url} (collection: {self.collection})")

            self._ensure_collection()
            logger.info("VectorStore initialized")
        except Exception as e:
            raise QdrantConnectionError(f"Failed to connect to Qdrant: {e}")

    def _ensure_collection(self):
        """Create the collection if it doesn't already exist."""
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    DENSE_VECTOR_NAME: VectorParams(
                        size=VECTOR_SIZE,
                        distance=Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    SPARSE_VECTOR_NAME: SparseVectorParams(
                        index=SparseIndexParams(on_disk=False)
                    )
                }
            )
            logger.info(f"Created Qdrant collection: {self.collection}")
        else:
            logger.info(f"Using existing collection: {self.collection}")

    def store(self, embedded_chunks: list[dict]):
        logger.info(f"Storing {len(embedded_chunks)} chunks in '{self.collection}'")
        try:
            batch_size = 20
            for i in range(0, len(embedded_chunks), batch_size):
                batch = embedded_chunks[i:i + batch_size]
                points = []
                for item in batch:
                    chunk = item["chunk"]
                    sparse = item["sparse_vector"]
                    points.append(PointStruct(
                        id=str(chunk.chunk_id),
                        vector={
                            DENSE_VECTOR_NAME: item["dense_vector"],
                            SPARSE_VECTOR_NAME: SparseVector(
                                indices=[int(k) for k in sparse.keys()],
                                values=list(sparse.values())
                            )
                        },
                        payload={
                            "chunk_id": chunk.chunk_id,
                            "arxiv_id": chunk.arxiv_id,
                            "paper_title": chunk.paper_title,
                            "authors": chunk.authors,
                            "content": chunk.content,
                            "chunk_type": chunk.chunk_type,
                            "page_number": chunk.page_number,
                            "caption": chunk.caption,
                            "figure_description": chunk.figure_description
                        }
                    ))
                self.client.upsert(
                    collection_name=self.collection,
                    points=points
                )
                logger.info(
                    f"Stored batch {i // batch_size + 1}/"
                    f"{-(-len(embedded_chunks) // batch_size)}"
                )
            logger.info(f"Stored {len(embedded_chunks)} chunks successfully")
        except Exception as e:
            raise QdrantConnectionError(f"Failed to store chunks: {e}")

    def collection_exists(self) -> bool:
        existing = [c.name for c in self.client.get_collections().collections]
        return self.collection in existing

    def get_collection_info(self) -> dict:
        info = self.client.get_collection(self.collection)
        return {
            "vectors_count": info.vectors_count,
            "points_count": info.points_count
        }
