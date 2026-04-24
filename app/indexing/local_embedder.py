import re

from app.core.exceptions import EmbeddingError
from app.core.logger import logger
from app.core.settings import settings
from app.models.schemas import Chunk


class LocalEmbedder:
    def __init__(self):
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise EmbeddingError(
                "Local embeddings require 'torch' and 'sentence-transformers' to be installed."
            ) from error

        self.model_name = settings.bge_model_name
        self.batch_size = 16
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        try:
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                trust_remote_code=True,
            )
            logger.info(f"LocalEmbedder initialized with model: {self.model_name}")
        except Exception as error:
            raise EmbeddingError(f"Failed to load embedding model '{self.model_name}': {error}")

    def embed_chunks(self, chunks: list[Chunk]) -> list[dict]:
        logger.info(f"Embedding {len(chunks)} chunks locally")
        result = []

        try:
            for i in range(0, len(chunks), self.batch_size):
                batch = chunks[i:i + self.batch_size]
                texts = [chunk.content for chunk in batch]
                vectors = self._get_embeddings(texts)

                for j, chunk in enumerate(batch):
                    result.append(
                        {
                            "chunk": chunk,
                            "dense_vector": vectors[j],
                            "sparse_vector": self._compute_sparse(chunk.content),
                        }
                    )

                logger.info(
                    f"Embedded batch {i // self.batch_size + 1}/{-(-len(chunks) // self.batch_size)}"
                )

            logger.info(f"Embedded {len(result)} chunks successfully")
            return result
        except EmbeddingError:
            raise
        except Exception as error:
            raise EmbeddingError(f"Embedding failed: {error}") from error

    def embed_query(self, query: str) -> dict:
        try:
            vectors = self._get_embeddings([query])
            return {
                "dense_vector": vectors[0],
                "sparse_vector": self._compute_sparse(query),
            }
        except EmbeddingError:
            raise
        except Exception as error:
            raise EmbeddingError(f"Query embedding failed: {error}") from error

    def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return embeddings.tolist()
        except Exception as error:
            raise EmbeddingError(f"Local embedding inference failed: {error}") from error

    def _compute_sparse(self, text: str) -> dict:
        terms = [token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2]
        if not terms:
            return {}

        counts: dict[str, int] = {}
        for term in terms:
            counts[term] = counts.get(term, 0) + 1

        total = sum(counts.values())
        return {
            str(abs(hash(term)) % 50000): round(count / total, 4)
            for term, count in counts.items()
        }
