from app.interfaces.llm_provider import LLMProvider


class QueryRouter:
    """Owns intent classification and high-level query routing."""

    def __init__(self, llm_client: LLMProvider):
        self.llm_client = llm_client

    def classify_intent(self, question: str) -> str:
        return self.llm_client.classify_intent(question)

    def route(self, question: str) -> str:
        return self.classify_intent(question)
