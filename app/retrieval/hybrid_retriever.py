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
        self.collection = settings.qdrant_collection_name
        self.embedder = Embedder()
        self.top_k = settings.top_k_retrieval
        logger.info("HybridRetriever initialized")

    def retrieve(self, query: str) -> list[dict]:
        """
        Hybrid search — dense (BGE-M3) + sparse (BM25) fused via RRF.
        Returns top_k chunks with scores. Each chunk dict has a 'score' key.

        Dense search finds semantically similar chunks.
        Sparse search finds keyword-matching chunks.
        RRF fusion combines both for best recall.

        Full debug logging shows all retrieved chunks and their scores.

        Normal vector search (dense only) is kept below as fallback comment.
        """
        logger.info(f"[RETRIEVE] Hybrid search for: {query[:60]}...")
        try:
            query_vectors = self.embedder.embed_query(query)
            dense = query_vectors["dense_vector"]
            sparse = query_vectors["sparse_vector"]

            # ── Hybrid search (dense + sparse via RRF) ─────────────────────
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
                limit=self.top_k,
                with_payload=True,
                with_vectors=False,
            )

            chunks = []
            for point in results.points:
                payload = point.payload.copy()
                payload["score"] = round(float(point.score), 4)
                chunks.append(payload)

            # Debug: log all retrieved chunks with scores
            logger.info(f"[RETRIEVE] Got {len(chunks)} chunks via hybrid search (RRF)")
            for i, chunk in enumerate(chunks, 1):
                logger.info(
                    f"[RETRIEVE] Chunk {i:02d} | score={chunk.get('score',0):.4f} | "
                    f"type={chunk.get('chunk_type','?'):12s} | "
                    f"page={str(chunk.get('page_number','?')):4s} | "
                    f"preview={chunk.get('content','')[:60].replace(chr(10),' ')}..."
                )

            return chunks

        except Exception as e:
            raise RetrievalError(f"Retrieval failed: {e}")