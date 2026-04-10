"""
app/core/config.py
──────────────────────────────────────────────
All settings for the entire project live here.
Every other file imports from this file.
Settings are read from the .env file automatically.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── OpenAI ───────────────────────────────────
    openai_api_key: str
    openai_llm_model: str = "gpt-4o"
    openai_max_tokens: int = 2048
    openai_temperature: float = 0.0

    # ── PageIndex + BERT ─────────────────────────
    index_dir: str = "./data/index"

    # ── SQLite (chat history) ────────────────────
    sqlite_db_path: str = "./data/legal_chat.db"

    # ── Redis (query cache) ──────────────────────
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_ttl_seconds: int = 3600
    redis_max_connections: int = 10

    # ── FastAPI ──────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = False
    api_log_level: str = "info"

    # ── Gradio ───────────────────────────────────
    gradio_host: str = "0.0.0.0"
    gradio_port: int = 7860
    gradio_api_url: str = "http://localhost:8000"

    # ── RAG settings ─────────────────────────────
    retrieval_top_k: int = 5
    SUPPORTED_ACTS: list[str] = ["BNS", "BNSS", "BSA", "DPDP", "ALL"]

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
