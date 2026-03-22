from groq import Groq
from app.core.logger import logger
from app.core.exceptions import RetrievalError
from app.core.settings import settings


class Reranker:

    def __init__(self):
        self.client = Groq(api_key=settings.groq_api_key)
        self.top_k = settings.top_k_rerank
        logger.info("Reranker initialized")

    def rerank(self, query: str, chunks: list[dict]) -> list[dict]:
        logger.info(f"Reranking {len(chunks)} chunks to top {self.top_k}")
        try:
            if not chunks:
                return []
            scored = []
            for chunk in chunks:
                score = self._score_chunk(query, chunk["content"])
                scored.append((score, chunk))
            scored.sort(key=lambda x: x[0], reverse=True)
            reranked = [chunk for _, chunk in scored[:self.top_k]]
            logger.info(f"Reranked to top {len(reranked)} chunks")
            return reranked
        except Exception as e:
            raise RetrievalError(f"Reranking failed: {e}")

    def _score_chunk(self, query: str, content: str) -> float:
        try:
            response = self.client.chat.completions.create(
                model=settings.groq_model,
                messages=[
                    {
                        "role": "user",
                        "content": f"""Score relevance of this chunk to the query.
                        Query: {query}
                        Chunk: {content[:300]}
                        Respond with only a number 0-10:"""
                    }
                ],
                max_tokens=5,
                temperature=0
            )
            score_text = response.choices[0].message.content.strip()
            score = float(''.join(c for c in score_text if c.isdigit() or c == '.'))
            return min(10.0, max(0.0, score))
        except Exception:
            return 0.0