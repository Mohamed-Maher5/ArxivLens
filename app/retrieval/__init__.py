from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import Reranker
from app.retrieval.cache import Cache
from app.core.logger import logger
from app.core.exceptions import RetrievalError


_cache = Cache()


def retrieve(query: str) -> list[dict]:
    cached = _cache.get(query)
    if cached:
        return cached
    try:
        retriever = HybridRetriever()
        reranker = Reranker()
        chunks = retriever.retrieve(query)
        reranked = reranker.rerank(query, chunks)
        _cache.set(query, reranked)
        return reranked
    except RetrievalError:
        raise
    except Exception as e:
        raise RetrievalError(f"Retrieval pipeline failed: {e}")