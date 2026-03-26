"""Application settings configuration."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"  # Allow extra env vars without errors
    )
    
    # ── API Keys ───────────────────────────────────────────────
    groq_api_key: str = ""                   # intent + reranker scoring
    langsmith_api_key: str = ""              # LangSmith tracing
    huggingface_api_key: str = ""            # answer generation only

    # ── LangSmith ─────────────────────────────────────────────
    langsmith_project: str = "arxiv-lens"
    langchain_tracing_v2: bool = True
    langchain_endpoint: str = "https://api.smith.langchain.com"
    langchain_api_key: str = ""
    langchain_project: str = "arxiv-lens"

    # ── Qdrant ────────────────────────────────────────────────
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection_name: str = "arxiv_papers"

    # ── Models ────────────────────────────────────────────────
    groq_classifier_model: str = "llama-3.1-8b-instant"  # intent + reranker
    hf_model: str = "Qwen/Qwen3-8B"                      # answer generation
    bge_model_name: str = "BAAI/bge-m3"                  # embeddings
    vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"  # ingestion
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "phi3"  # contextualization & history

    # ── Retrieval ─────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 50
    score_threshold: float = 0.3
    max_history: int = 5
    rerank_score_threshold: float = 6.0
    top_k_retrieval: int = 10
    top_k_rerank: int = 3


settings = Settings()