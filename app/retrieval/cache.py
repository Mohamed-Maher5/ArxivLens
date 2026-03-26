import hashlib
from app.core.logger import logger


class Cache:

    def __init__(self):
        self._cache = {}
        logger.info("Cache initialized")

    def get(self, query: str, collection_name: str = "") -> list[dict] | None:
        key = self._hash(query, collection_name)
        if key in self._cache:
            logger.info(f"Cache hit for query: {query[:50]}... (collection: {collection_name})")
            return self._cache[key]
        return None

    def set(self, query: str, chunks: list[dict], collection_name: str = ""):
        key = self._hash(query, collection_name)
        self._cache[key] = chunks
        logger.info(f"Cached result for query: {query[:50]}... (collection: {collection_name})")

    def clear(self):
        self._cache.clear()
        logger.info("Cache cleared")

    def _hash(self, query: str, collection_name: str = "") -> str:
        """
        Hash is scoped to both query and collection_name so that the same
        query against different paper collections never returns a stale hit.
        """
        combined = f"{collection_name}::{query.lower().strip()}"
        return hashlib.md5(combined.encode()).hexdigest()