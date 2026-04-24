from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def get_llm_client():
    from app.llm.gemma_client import GemmaClient

    return GemmaClient()
