from qdrant_client import QdrantClient
from qdrant_client.models import (
    FusionQuery,
    Fusion,
    Prefetch,
    SparseVector,
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
        self.embedder = Embedder()
        self.top_k = settings.top_k_retrieval
        logger.info("HybridRetriever initialized")

    def retrieve(self, query: str, collection_name: str) -> list[dict]:
        """
        Hybrid search — dense (BGE-M3) + sparse (BM25) fused via RRF.

        Args:
            query:           The search query string.
            collection_name: The per-paper Qdrant collection to search in.

        Returns:
            List of chunk dicts, each with a 'score' key (0.0–1.0).
        """
        logger.info(
            f"[RETRIEVE] Hybrid search | collection={collection_name} | "
            f"query={query[:60]}..."
        )
        try:
            query_vectors = self.embedder.embed_query(query)
            dense = query_vectors["dense_vector"]
            sparse = query_vectors["sparse_vector"]

            results = self.client.query_points(
                collection_name=collection_name,
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
                limit=self.top_k,
                with_payload=True,
                with_vectors=False,
            )

            chunks = []
            for point in results.points:
                payload = point.payload.copy()
                payload["score"] = round(float(point.score), 4)
                chunks.append(payload)

            logger.info(f"[RETRIEVE] Got {len(chunks)} chunks via hybrid search (RRF)")
            for i, chunk in enumerate(chunks, 1):
                logger.info(
                    f"[RETRIEVE] Chunk {i:02d} | score={chunk.get('score', 0):.4f} | "
                    f"type={chunk.get('chunk_type', '?'):12s} | "
                    f"page={str(chunk.get('page_number', '?')):4s} | "
                    f"preview={chunk.get('content', '')[:60].replace(chr(10), ' ')}..."
                )
                
            return chunks

        except Exception as e:
            raise RetrievalError(f"Retrieval failed for collection '{collection_name}': {e}")