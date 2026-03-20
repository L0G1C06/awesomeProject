"""
Pipeline — Embedding & Indexação (Otimizado)
Lê chunks do bucket Gold (gold/chunks/) e indexa no Milvus.
Pula arquivos já indexados consultando o object_key na tabela data_files.
"""

import json
import os
import uuid
import sys

from loguru import logger
from minio import Minio
import mlflow

from sentence_transformers import SentenceTransformer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from api.services.milvus_service import MilvusService
from api.services.postgres_service import PostgresService


MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_USER = os.getenv("MINIO_ROOT_USER", "minioadmin")
MINIO_PASS = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")

BUCKET = "gold"
GOLD_PREFIX = os.getenv("GOLD_PREFIX", "")  # "" = todos os datasets

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 256))

DATASET_NAME = os.getenv("DATASET_NAME", "arxiv")
DATASET_DOMAIN = os.getenv("DATASET_DOMAIN", "research")
DATASET_SOURCE_URL = os.getenv("DATASET_SOURCE_URL", "https://arxiv.org/")
DATASET_VERSION = os.getenv("DATASET_VERSION", "1.0.0")


logger.info("Carregando modelo de embeddings {}", EMBED_MODEL)

model = SentenceTransformer(EMBED_MODEL, device="cpu")


# ---------------------------------------------------------------------------
# Infraestrutura
# ---------------------------------------------------------------------------

def get_minio_client() -> Minio:
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_USER,
        secret_key=MINIO_PASS,
        secure=False,
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
        device="cpu",
    )
    return embeddings.tolist()


def stream_jsonl(response):
    buffer = ""
    for chunk in response.stream(32 * 1024):
        buffer += chunk.decode()
        lines = buffer.split("\n")
        buffer = lines[-1]
        for line in lines[:-1]:
            if line.strip():
                yield json.loads(line)
    if buffer.strip():
        yield json.loads(buffer)


def list_gold_files(client: Minio, prefix: str = GOLD_PREFIX) -> list[str]:
    """Lista todos os .jsonl do bucket Gold com o prefixo informado."""
    objects = client.list_objects(BUCKET, prefix=prefix, recursive=True)
    return [
        obj.object_name for obj in objects
        if obj.object_name.endswith(".jsonl") and "/chunks/" in obj.object_name
    ]


def get_indexed_object_keys(db: PostgresService, dataset_id: str) -> set[str]:
    """Retorna object_keys já indexados com status='done' para o dataset."""
    rows = db.list_data_files(dataset_id=dataset_id, status="done")
    return {row["object_key"] for row in rows}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def run() -> None:
    logger.info("=== Embedding & Indexação iniciada ===")

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("rag-enterprise-indexing")

    with mlflow.start_run(run_name="embed-and-index"):

        client = get_minio_client()
        milvus = MilvusService()
        db = PostgresService()

        dataset_id = db.ensure_dataset(
            name=DATASET_NAME,
            domain=DATASET_DOMAIN,
            source_url=DATASET_SOURCE_URL,
            version=DATASET_VERSION,
            description=f"Dataset {DATASET_NAME} indexado no Milvus",
        )

        already_indexed = get_indexed_object_keys(db, dataset_id)
        logger.info("{} arquivo(s) já indexado(s) serão ignorados.", len(already_indexed))

        gold_files = list_gold_files(client, prefix=GOLD_PREFIX)
        logger.info("{} arquivo(s) Gold encontrado(s).", len(gold_files))

        total_files = 0
        total_skipped = 0
        total_indexed = 0

        for object_name in gold_files:

            if object_name in already_indexed:
                logger.info("Pulando (já indexado): {}", object_name)
                total_skipped += 1
                continue

            logger.info("Processando {}", object_name)

            response = client.get_object(BUCKET, object_name)
            chunks = list(stream_jsonl(response))
            response.close()

            if not chunks:
                logger.warning("Nenhum chunk em {}", object_name)
                continue

            data_file_id = db.register_data_file(
                dataset_id=dataset_id,
                layer="gold",
                bucket=BUCKET,
                object_key=object_name,
                row_count=len(chunks),
                size_bytes=None,
                checksum=None,
                status="done",
            )

            logger.info("{} chunks carregados de {}", len(chunks), object_name)

            for i in range(0, len(chunks), BATCH_SIZE):

                batch = chunks[i:i + BATCH_SIZE]

                # Gold já tem content pronto — usa diretamente com o título do metadata
                texts = [
                    f"{c.get('metadata', {}).get('title', '')} {c['content']}".strip()
                    for c in batch
                ]

                embeddings = embed_texts(texts)

                doc_ids = [str(uuid.uuid4()) for _ in batch]
                contents = [c["content"] for c in batch]
                metadatas = [c.get("metadata", {}) for c in batch]

                milvus_ids = milvus.insert_batch(
                    doc_ids=doc_ids,
                    contents=contents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                )

                for chunk, milvus_id in zip(batch, milvus_ids):
                    db.save_document(
                        data_file_id=data_file_id,
                        milvus_id=milvus_id,
                        content=chunk["content"],
                        chunk_index=int(chunk.get("chunk_index", 0)),
                        token_count=len(chunk["content"].split()),
                        metadata=chunk.get("metadata", {}),
                        embed_model=EMBED_MODEL,
                        embed_version=DATASET_VERSION,
                    )

                total_indexed += len(batch)

            total_files += 1

            db.log_audit(
                entity="data_files",
                entity_id=data_file_id,
                action="indexed_milvus",
                details={
                    "object_key": object_name,
                    "chunks_indexed": len(chunks),
                    "embed_model": EMBED_MODEL,
                },
            )

        milvus.flush()

        mlflow.log_metrics({
            "total_files_indexed": total_files,
            "total_files_skipped": total_skipped,
            "total_chunks_indexed": total_indexed,
        })
        mlflow.log_params({"embed_model": EMBED_MODEL})

        logger.info(
            "=== Indexação concluída: {} chunks em {} arquivo(s) novo(s), {} ignorado(s) ===",
            total_indexed, total_files, total_skipped,
        )


if __name__ == "__main__":
    run()