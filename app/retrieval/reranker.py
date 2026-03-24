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
        Returns chunks with score >= 6.0 only.
        """
        logger.info(f"[RERANKER] Scoring {len(chunks)} chunks")
        try:
            if not chunks:
                return []

            scores = self._score_all(query, chunks)

            # Filter by threshold >= 6.0 instead of taking top_k
            scored_chunks = []
            for score, chunk in zip(scores, chunks):
                if score >= 6.0:
                    chunk_copy = chunk.copy()
                    chunk_copy['rerank_score'] = score  # Store the LLM score
                    scored_chunks.append((score, chunk_copy))
                    logger.info(f"[RERANKER] Kept chunk score={score:.1f}")

            # Sort by score descending
            scored_chunks.sort(key=lambda x: x[0], reverse=True)
            result = [chunk for _, chunk in scored_chunks]
            
            logger.info(f"[RERANKER] {len(result)} chunks >= 6.0 threshold")
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