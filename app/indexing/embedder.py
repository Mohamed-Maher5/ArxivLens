# app/indexing/embedder.py
import numpy as np
import requests
from app.core.logger import logger
from app.core.exceptions import EmbeddingError
from app.core.settings import settings
from app.models.schemas import Chunk

HF_API_URL = "https://router.huggingface.co/hf-inference/models"

class Embedder:

    def __init__(self):
        self.model = settings.bge_model_name
        self.api_url = f"{HF_API_URL}/{self.model}/pipeline/feature-extraction"
        self.headers = {
            "Authorization": f"Bearer {settings.huggingface_api_key}",
            "Content-Type": "application/json"
        }
        logger.info(f"Embedder initialized with model: {self.model}")

    def embed_chunks(self, chunks: list[Chunk]) -> list[dict]:
        logger.info(f"Embedding {len(chunks)} chunks")
        result = []
        batch_size = 10

        try:
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                texts = [chunk.content for chunk in batch]

                vectors = self._get_embeddings(texts)
                for j, chunk in enumerate(batch):
                    sparse = self._compute_sparse(chunk.content)
                    result.append({
                        "chunk": chunk,
                        "dense_vector": vectors[j],
                        "sparse_vector": sparse
                    })

                logger.info(f"Embedded batch {i // batch_size + 1}/{-(-len(chunks) // batch_size)}")

            logger.info(f"Embedded {len(result)} chunks successfully")
            return result
        except Exception as e:
            raise EmbeddingError(f"Embedding failed: {e}")

    def embed_query(self, query: str) -> dict:
        try:
            vectors = self._get_embeddings([query])
            sparse = self._compute_sparse(query)
            return {
                "dense_vector": vectors[0],
                "sparse_vector": sparse
            }
        except Exception as e:
            raise EmbeddingError(f"Query embedding failed: {e}")

    def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json={"inputs": texts}
        )

        if response.status_code != 200:
            raise EmbeddingError(
                f"HuggingFace API error {response.status_code}: {response.text}"
            )

        embeddings = response.json()

        # Handle nested embeddings (e.g., token-level embeddings)
        if isinstance(embeddings[0], list) and isinstance(embeddings[0][0], list):
            embeddings = [np.mean(np.array(e), axis=0).tolist() for e in embeddings]

        return embeddings

    def _compute_sparse(self, text: str) -> dict:
        words = text.lower().split()
        word_counts = {}
        for word in words:
            word = ''.join(c for c in word if c.isalnum())
            if word and len(word) > 2:
                word_counts[word] = word_counts.get(word, 0) + 1

        total = sum(word_counts.values())
        sparse = {}
        for word, count in word_counts.items():
            token_id = str(abs(hash(word)) % 50000)
            sparse[token_id] = round(count / total, 4)
        return sparse