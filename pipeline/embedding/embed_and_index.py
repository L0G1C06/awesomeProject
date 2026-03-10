"""
Pipeline — Embedding & Indexação
Lê chunks do Gold, gera embeddings via Ollama e indexa no Milvus.
"""
import json
import os
import uuid
from loguru import logger
from minio import Minio
import ollama
import mlflow

from api.services.milvus_service import MilvusService

MINIO_ENDPOINT  = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_USER      = os.getenv("MINIO_ROOT_USER", "minioadmin")
MINIO_PASS      = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")
BUCKET_GOLD     = "gold"
EMBED_MODEL     = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
MLFLOW_URI      = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
BATCH_SIZE      = 32


def get_minio_client() -> Minio:
    return Minio(MINIO_ENDPOINT, access_key=MINIO_USER, secret_key=MINIO_PASS, secure=False)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Gera embeddings em batch via Ollama."""
    embeddings = []
    for text in texts:
        response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
        embeddings.append(response["embedding"])
    return embeddings


def run():
    logger.info("=== Embedding & Indexação iniciada ===")
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("rag-enterprise-indexing")

    with mlflow.start_run(run_name="embed-and-index"):
        client = get_minio_client()
        milvus = MilvusService()

        objects = list(client.list_objects(BUCKET_GOLD, prefix="", recursive=True))
        total_indexed = 0
        total_files = 0

        for obj in objects:
            if not obj.object_name.endswith(".jsonl"):
                continue

            response = client.get_object(BUCKET_GOLD, obj.object_name)
            lines = response.read().decode("utf-8").strip().split("\n")
            response.close()

            chunks = [json.loads(l) for l in lines if l]
            logger.info(f"Indexando {len(chunks)} chunks de {obj.object_name}")

            # Processa em batches
            for i in range(0, len(chunks), BATCH_SIZE):
                batch = chunks[i:i + BATCH_SIZE]
                texts = [c["content"] for c in batch]
                embeddings = embed_texts(texts)

                for chunk, embedding in zip(batch, embeddings):
                    milvus.insert(
                        doc_uuid=str(uuid.uuid4()),
                        content=chunk["content"],
                        embedding=embedding,
                        metadata=chunk.get("metadata", {}),
                    )
                    total_indexed += 1

            total_files += 1

        mlflow.log_metrics({
            "total_files_indexed": total_files,
            "total_chunks_indexed": total_indexed,
        })
        mlflow.log_params({"embed_model": EMBED_MODEL})

        logger.info(f"=== Indexação concluída: {total_indexed} chunks em {total_files} arquivos ===")


if __name__ == "__main__":
    run()