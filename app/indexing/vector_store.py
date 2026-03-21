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


DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
VECTOR_SIZE = 1024


class VectorStore:

    def __init__(self):
        try:
            if settings.qdrant_url:
                self.client = QdrantClient(
                    url=settings.qdrant_url,
                    api_key=settings.qdrant_api_key
                )
                logger.info("Connected to Qdrant Cloud")
            else:
                self.client = QdrantClient(
                    host=settings.qdrant_host,
                    port=settings.qdrant_port
                )
                logger.info("Connected to Qdrant local")
            self.collection = settings.qdrant_collection_name
            self._ensure_collection()
            logger.info("VectorStore initialized")
        except Exception as e:
            raise QdrantConnectionError(f"Failed to connect to Qdrant: {e}")

    def _ensure_collection(self):
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
        logger.info(f"Storing {len(embedded_chunks)} chunks in Qdrant")
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
                            "paper_id": chunk.paper_id,
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
                logger.info(f"Stored batch {i // batch_size + 1}/{-(-len(embedded_chunks) // batch_size)}")
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