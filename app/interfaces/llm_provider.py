from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    def classify_intent(self, message: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def contextualize_query(self, question: str, max_tokens: int = 100) -> str:
        raise NotImplementedError

    @abstractmethod
    def score_chunk(self, query: str, content: str) -> float:
        raise NotImplementedError

    @abstractmethod
    def generate_chat_response(self, message: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate_paper_answer(self, question: str, chunks: list[dict]) -> dict:
        raise NotImplementedError

    @abstractmethod
    def generate_general_knowledge(self, question: str, metadata: str) -> dict:
        raise NotImplementedError

    @abstractmethod
    def describe_image(self, image_b64: str, caption: str = '') -> str:
        raise NotImplementedError

    def generate_from_paper_top3(self, question: str, chunks: list[dict]) -> dict:
        return self.generate_paper_answer(question, chunks)
