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
    # Also support standard LangSmith env vars
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
    # Groq — intent classification, reranker scoring
    groq_classifier_model: str = "llama-3.1-8b-instant"

    # HuggingFace — final answer generation only
    hf_model: str = "Qwen/Qwen3-8B"

    # Embeddings
    bge_model_name: str = "BAAI/bge-m3"

    # Ollama local — query contextualization + history summarization
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "phi3"

    # Ingestion only
    vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    ollama_summarizer_model: str = "phi3"  # Can remove, not used anymore

    # ── Retrieval ─────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k_retrieval: int = 10
    top_k_rerank: int = 3
    score_threshold: float = 0.25
    max_history: int = 5
    rerank_score_threshold: float = 6.0  

    # ── Web Search ────────────────────────────────────────────
    serpapi_api_key: str = "" 


settings = Settings()