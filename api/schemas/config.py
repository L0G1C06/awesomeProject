"""Configurações centrais da aplicação via pydantic-settings."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # MinIO
    MINIO_ENDPOINT: str
    MINIO_ROOT_USER: str
    MINIO_ROOT_PASSWORD: str

    # PostgreSQL
    DATABASE_URL: str

    # Milvus
    MILVUS_HOST: str
    MILVUS_PORT: int
    MILVUS_COLLECTION: str

    # Ollama
    OLLAMA_HOST: str
    OLLAMA_LLM_MODEL: str
    OLLAMA_EMBED_MODEL: str
    OLLAMA_EMBED_DIMENSION: int

    # HuggingFace
    HUGGINGFACE_API_TOKEN: str
    HF_LLM_MODEL: str
    HF_EMBED_MODEL: str

    # MLflow
    MLFLOW_TRACKING_URI: str
    MLFLOW_EXPERIMENT_NAME: str

    # RAG Pipeline
    CHUNK_SIZE: int
    CHUNK_OVERLAP: int
    TOP_K_RETRIEVAL: int

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()