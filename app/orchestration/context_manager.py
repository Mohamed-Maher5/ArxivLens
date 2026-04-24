from app.core.logger import logger
from app.interfaces.llm_provider import LLMProvider


class ContextManager:
    """Owns query contextualization for a single, stateless request."""

    def __init__(self, llm_client: LLMProvider | None = None):
        if llm_client is None:
            from app.llm.gemma_client import GemmaClient

            llm_client = GemmaClient()
        self.llm_client = llm_client

    def contextualize(self, question: str) -> str:
        try:
            result = self.llm_client.contextualize_query(question=question, max_tokens=100)
            return result if result and len(result) > 10 else question
        except Exception as error:
            logger.warning(f'[CONTEXT] Contextualization failed: {error}')
            return question
