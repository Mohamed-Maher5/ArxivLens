import hashlib
import json
from app.core.logger import logger


class Cache:

    def __init__(self):
        self._cache = {}
        logger.info("Cache initialized")

    def get(self, query: str) -> list[dict] | None:
        key = self._hash(query)
        if key in self._cache:
            logger.info(f"Cache hit for query: {query[:50]}...")
            return self._cache[key]
        return None

    def set(self, query: str, chunks: list[dict]):
        key = self._hash(query)
        self._cache[key] = chunks
        logger.info(f"Cached result for query: {query[:50]}...")

    def clear(self):
        self._cache.clear()
        logger.info("Cache cleared")

    def _hash(self, query: str) -> str:
        return hashlib.md5(query.lower().strip().encode()).hexdigest()