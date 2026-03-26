from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.cache import Cache
from app.core.logger import logger
from app.core.exceptions import RetrievalError
from app.retrieval.reranker import Reranker

_cache = Cache()
_reranker = Reranker()


def retrieve(query: str) -> list[dict]:
    """
    Retrieve chunks for a query using normal vector search.
    Returns list of chunk dicts, each containing a 'score' key
    (cosine similarity, 0.0 – 1.0) alongside the chunk payload.

    Results are cached by query hash to avoid duplicate API calls
    within the same session.
    """
    cached = _cache.get(query)
    if cached:
        return cached
    try:
        retriever = HybridRetriever()
        chunks = retriever.retrieve(query)
        _cache.set(query, chunks)
        return chunks
    except RetrievalError:
        raise
    except Exception as e:
        raise RetrievalError(f"Retrieval pipeline failed: {e}")
    
def rerank_chunks(query: str, chunks: list[dict]) -> list[dict]:
    """
    Rerank a list of retrieved chunks using Reranker.
    Returns only chunks passing the rerank threshold.
    """
    if not chunks:
        logger.info("No chunks to rerank.")
        return []

    logger.info(f"Reranking {len(chunks)} chunks for query: {query[:60]}...")
    try:
        return _reranker.rerank(query, chunks)
    except Exception as e:
        raise RetrievalError(f"Reranking failed: {e}")