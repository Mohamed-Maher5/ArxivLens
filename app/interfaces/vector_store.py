from abc import ABC, abstractmethod


class VectorStore(ABC):
    @abstractmethod
    def store(self, embedded_chunks: list[dict]) -> None:
        raise NotImplementedError

    @abstractmethod
    def collection_exists(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_collection_info(self) -> dict:
        raise NotImplementedError
