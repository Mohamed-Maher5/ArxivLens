from concurrent.futures import ThreadPoolExecutor, as_completed
from app.core.logger import logger
from app.core.exceptions import RetrievalError
from app.core.settings import settings
from app.interfaces.llm_provider import LLMProvider


class Reranker:

    def __init__(self, llm_client: LLMProvider | None = None):
        if llm_client is None:
            from app.llm.gemma_client import GemmaClient

            llm_client = GemmaClient()
        self.llm_client = llm_client
        self.top_k = settings.top_k_rerank
        logger.info(f"Reranker initialized (model: {settings.gemma_model})")

    def rerank(self, query: str, chunks: list[dict]) -> list[dict]:
        """
        Rerank chunks using Gemma scoring.
        Returns only chunks that meet the configured rerank threshold.
        """
        logger.info(f"[RERANKER] Scoring {len(chunks)} chunks")
        try:
            if not chunks:
                return []

            scores = self._score_all(query, chunks)

            # Filter by threshold instead of taking top_k
            scored_chunks = []
            for score, chunk in zip(scores, chunks):
                if score >= settings.rerank_score_threshold:
                    chunk_copy = chunk.copy()
                    chunk_copy['rerank_score'] = score  # Store the LLM score
                    scored_chunks.append((score, chunk_copy))
                    logger.info(f"[RERANKER] Kept chunk score={score:.1f}")

            # Sort by score descending
            scored_chunks.sort(key=lambda x: x[0], reverse=True)
            result = [chunk for _, chunk in scored_chunks]
            
            logger.info(
                f"[RERANKER] {len(result)} chunks >= {settings.rerank_score_threshold:.1f} threshold"
            )
            return result

        except Exception as e:
            raise RetrievalError(f"Reranking failed: {e}")

    def _score_all(self, query: str, chunks: list[dict]) -> list[float]:
        """Score all chunks concurrently using ThreadPoolExecutor."""
        scores = [0.0] * len(chunks)
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_idx = {
                executor.submit(
                    self._score_chunk, query, chunk.get("content", "")
                ): i
                for i, chunk in enumerate(chunks)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    scores[idx] = future.result()
                except Exception as e:
                    logger.warning(f"[RERANKER] Chunk {idx} scoring failed: {e}")
                    scores[idx] = 0.0
        return scores

    def _score_chunk(self, query: str, content: str) -> float:
        """Answer-aware reranking: measures if chunk can directly answer the question."""
        return self.llm_client.score_chunk(query, content)
