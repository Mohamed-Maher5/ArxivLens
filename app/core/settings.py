from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── API Keys ───────────────────────────────────────────────
    groq_api_key: str                   # intent + HyDE + reranker scoring
    langsmith_api_key: str
    huggingface_api_key: str            # answer generation only

    # ── LangSmith ─────────────────────────────────────────────
    langsmith_project: str = "arxiv-lens"

    # ── Qdrant ────────────────────────────────────────────────
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection_name: str = "arxiv_papers"

    # ── Models ────────────────────────────────────────────────
    # Groq — intent classification, HyDE, reranker scoring
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
    ollama_summarizer_model: str = "phi3"

    # ── Retrieval ─────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k_retrieval: int = 10
    top_k_rerank: int = 3
    # RRF hybrid scores range 0.09–0.50 (rank-based, not cosine).
    # Threshold set to 0.10 — filters only truly irrelevant queries.
    # The LLM answer itself handles cases where chunks are off-topic.
    score_threshold: float = 0.10
    max_history: int = 5

    class Config:
        env_file = ".env"


settings = Settings()