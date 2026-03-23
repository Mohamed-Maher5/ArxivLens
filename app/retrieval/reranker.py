import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from groq import Groq
from app.core.logger import logger
from app.core.exceptions import RetrievalError
from app.core.settings import settings


class Reranker:

    def __init__(self):
        self.client = Groq(api_key=settings.groq_api_key)
        self.top_k = settings.top_k_rerank
        logger.info(f"Reranker initialized (model: {settings.groq_classifier_model})")

    def rerank(self, query: str, chunks: list[dict]) -> list[dict]:
        """
        Rerank chunks using Groq llama-3.1-8b-instant scoring.
        All chunks scored in parallel to minimise latency.
        Returns exactly top_k chunks sorted by score descending.
        Full debug logging shows every chunk score before and after ranking.
        """
        logger.info(f"[RERANKER] Scoring {len(chunks)} chunks using {settings.groq_classifier_model}")
        try:
            if not chunks:
                return []

            scores = self._score_all(query, chunks)

            # Log all chunk scores for visibility
            for i, (score, chunk) in enumerate(zip(scores, chunks)):
                logger.info(
                    f"[RERANKER] Chunk {i+1:02d} | score={score:5.1f} | "
                    f"type={chunk.get('chunk_type','?'):12s} | "
                    f"page={str(chunk.get('page_number','?')):4s} | "
                    f"preview={chunk.get('content','')[:60].replace(chr(10),' ')}..."
                )

            scored = list(zip(scores, chunks))
            scored.sort(key=lambda x: x[0], reverse=True)

            top = min(self.top_k, len(scored))
            reranked = [chunk for _, chunk in scored[:top]]

            # Log selected top chunks
            logger.info(f"[RERANKER] Selected top {len(reranked)} chunks:")
            for i, (score, chunk) in enumerate(scored[:top], 1):
                logger.info(
                    f"[RERANKER] → #{i} score={score:5.1f} | "
                    f"type={chunk.get('chunk_type','?'):12s} | "
                    f"page={str(chunk.get('page_number','?')):4s} | "
                    f"preview={chunk.get('content','')[:80].replace(chr(10),' ')}..."
                )

            return reranked

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
        """Score a single chunk relevance 0–10 using Groq llama-3.1-8b-instant."""
        try:
            response = self.client.chat.completions.create(
                model=settings.groq_classifier_model,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Score the relevance of this chunk to the query.\n"
                        f"Query: {query}\n"
                        f"Chunk: {content[:400]}\n\n"
                        f"Respond with ONLY a single number between 0 and 10. No explanation."
                    )
                }],
                max_tokens=5,
                temperature=0,
            )
            raw = response.choices[0].message.content.strip()
            match = re.search(r"\d+(\.\d+)?", raw)
            return min(10.0, max(0.0, float(match.group()))) if match else 0.0
        except Exception:
            return 0.0