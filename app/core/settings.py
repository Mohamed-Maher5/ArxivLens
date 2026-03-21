from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    groq_api_key: str
    langsmith_api_key: str
    huggingface_api_key: str = ""
    langsmith_project: str = "arxiv-lens"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection_name: str = "arxiv_papers"
    bge_model_name: str = "BAAI/bge-m3"
    bge_reranker_name: str = "BAAI/bge-reranker-v2-m3"
    groq_model: str = "llama-3.3-70b-versatile"
    vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    ollama_url: str = "http://localhost:11434"
    ollama_summarizer_model: str = "phi3"
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k_retrieval: int = 10
    top_k_rerank: int = 5
    max_history: int = 6

    class Config:
        env_file = ".env"


settings = Settings()