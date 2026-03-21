from qdrant_client import QdrantClient
from qdrant_client.models import (
    SearchRequest,
    NamedVector,
    NamedSparseVector,
    SparseVector,
    FusionQuery,
    Fusion,
    Prefetch
)
from app.core.logger import logger
from app.core.exceptions import RetrievalError
from app.core.settings import settings
from app.indexing.embedder import Embedder
from app.indexing.vector_store import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME


class HybridRetriever:

    def __init__(self):
        if settings.qdrant_url:
            self.client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key
            )
        else:
            self.client = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port
            )
        self.collection = settings.qdrant_collection_name
        self.embedder = Embedder()
        self.top_k = settings.top_k_retrieval
        logger.info("HybridRetriever initialized")

    def retrieve(self, query: str) -> list[dict]:
        logger.info(f"Retrieving for query: {query[:50]}...")
        try:
            query_vectors = self.embedder.embed_query(query)
            dense = query_vectors["dense_vector"]
            sparse = query_vectors["sparse_vector"]
            results = self.client.query_points(
                collection_name=self.collection,
                prefetch=[
                    Prefetch(
                        query=dense,
                        using=DENSE_VECTOR_NAME,
                        limit=self.top_k
                    ),
                    Prefetch(
                        query=SparseVector(
                            indices=[int(k) for k in sparse.keys()],
                            values=list(sparse.values())
                        ),
                        using=SPARSE_VECTOR_NAME,
                        limit=self.top_k
                    )
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=self.top_k
            )
            chunks = []
            for point in results.points:
                chunks.append(point.payload)
            logger.info(f"Retrieved {len(chunks)} chunks")
            return chunks
        except Exception as e:
            raise RetrievalError(f"Retrieval failed: {e}")