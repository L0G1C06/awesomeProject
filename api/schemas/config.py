"""Configurações centrais da aplicação via pydantic-settings."""
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # MinIO
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ROOT_USER: str = "minioadmin"
    MINIO_ROOT_PASSWORD: str = "minioadmin"

    # PostgreSQL
    DATABASE_URL: str = "postgresql://raguser:ragpass@localhost:5433/ragdb"

    # Milvus
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_COLLECTION: str = "rag_documents"

    # Ollama
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_LLM_MODEL: str = "mistral"
    OLLAMA_EMBED_MODEL: str = "nomic-embed-text"
    OLLAMA_EMBED_DIMENSION: int = 768

    # HuggingFace ← adicionado
    HUGGINGFACE_API_TOKEN: str = ""
    HF_LLM_MODEL: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    HF_EMBED_MODEL: str = "BAAI/bge-base-en-v1.5" 

    # MLflow
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "rag-enterprise"

    # RAG Pipeline
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    TOP_K_RETRIEVAL: int = 5

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()