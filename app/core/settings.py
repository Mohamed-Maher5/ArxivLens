"""Application settings configuration."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # ── API Keys ───────────────────────────────────────────────
    groq_api_key: str = ""
    langsmith_api_key: str = ""
    huggingface_api_key: str = ""

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
    # NOTE: qdrant_collection_name removed — collections are now dynamic per paper.
    # Collection names are derived from paper IDs: paper_1706_03762

    # ── Models ────────────────────────────────────────────────
    groq_classifier_model: str = "llama-3.1-8b-instant"
    hf_model: str = "Qwen/Qwen3-8B"
    bge_model_name: str = "BAAI/bge-m3"
    vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "phi3"

    # ── Retrieval ─────────────────────────────────────────────
    chunk_size: int = 256
    chunk_overlap: int = 30
    score_threshold: float = 0.3
    max_history: int = 6
    rerank_score_threshold: float = 7.0
    top_k_retrieval: int = 10
    top_k_rerank: int = 3


settings = Settings()