"""Centralized application settings with code-owned business defaults."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and code defaults."""

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore',
        case_sensitive=False,
    )

    google_api_key: str = Field(default='', validation_alias='GOOGLE_API_KEY')
    qdrant_url: str = Field(default='http://localhost:6333', validation_alias='QDRANT_URL')

    langsmith_project: str = 'arxiv-lens'
    langchain_tracing_v2: bool = True
    langchain_endpoint: str = 'https://api.smith.langchain.com'

    gemma_model: str = 'gemma-3-27b-it'
    bge_model_name: str = 'BAAI/bge-m3'

    chunk_size: int = 256
    chunk_overlap: int = 30
    score_threshold: float = 0.3
    rerank_score_threshold: float = 7.0
    top_k_retrieval: int = 10
    top_k_rerank: int = 3


settings = Settings()
