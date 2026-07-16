"""
Application configuration loaded from environment variables via pydantic-settings.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration — values come from .env or OS environment variables."""

    # --- Groq LLM ---
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # --- Chunking ---
    CHUNK_SIZE_WORDS: int = 700
    CHUNK_OVERLAP_WORDS: int = 120

    # --- Retrieval ---
    TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.25

    # --- Embedding model ---
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384

    # --- History ---
    MAX_HISTORY_MESSAGES: int = 8

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache()
def get_settings() -> Settings:
    """Singleton accessor — cached so the .env file is only read once."""
    return Settings()
