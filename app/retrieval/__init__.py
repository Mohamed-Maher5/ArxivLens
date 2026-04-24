from app.core.exceptions import RetrievalError
from app.core.logger import logger
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import Reranker


def retrieve(query: str, collection_name: str) -> list[dict]:
    """Retrieve chunks for a query from a specific per-paper collection."""
    try:
        retriever = HybridRetriever()
        return retriever.retrieve(query, collection_name)
    except RetrievalError:
        raise
    except Exception as error:
        raise RetrievalError(f'Retrieval pipeline failed: {error}') from error


def rerank_chunks(query: str, chunks: list[dict]) -> list[dict]:
    """Rerank a list of retrieved chunks using Reranker."""
    if not chunks:
        logger.info('No chunks to rerank.')
        return []

    logger.info(f"Reranking {len(chunks)} chunks for query: {query[:60]}...")
    try:
        return Reranker().rerank(query, chunks)
    except Exception as error:
        raise RetrievalError(f'Reranking failed: {error}') from error
