"""
Pipeline — Silver → Gold
Prepara documentos finais para consumo pelo RAG (chunking, formatação).
"""

import json
import os
from io import BytesIO
from loguru import logger
from minio import Minio
from langchain_text_splitters import RecursiveCharacterTextSplitter

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_USER = os.getenv("MINIO_ROOT_USER", "minioadmin")
MINIO_PASS = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")
BUCKET_SILVER = "silver"
BUCKET_GOLD = "gold"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 64))


def get_minio_client() -> Minio:
    return Minio(
        MINIO_ENDPOINT, access_key=MINIO_USER, secret_key=MINIO_PASS, secure=False
    )


def extract_text(record: dict) -> str:
    """
    TODO: Adapte conforme o campo de texto do seu dataset.
    Concatena campos relevantes em um único texto para indexação.
    """
    parts = []
    for field in ["title", "summary", "text", "content", "description"]:
        if val := record.get(field):
            parts.append(str(val))
    return " | ".join(parts) if parts else json.dumps(record)


def extract_metadata(record: dict) -> dict:
    """
    TODO: Adapte os campos de metadados do seu dataset.
    """
    metadata: dict = {}
    ignored_fields = {"text", "content", "description", "summary", "_processed_at"}
    for key, value in record.items():
        if key in ignored_fields or isinstance(value, dict):
            continue
        if isinstance(value, list):
            if value and all(isinstance(item, str) for item in value):
                metadata[key] = ", ".join(value)
            continue
        metadata[key] = value
    return metadata


def chunk_and_prepare(records: list[dict]) -> list[dict]:
    """Divide registros em chunks para indexação."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for record in records:
        text = extract_text(record)
        metadata = extract_metadata(record)
        texts = splitter.split_text(text)
        for i, chunk_text in enumerate(texts):
            chunks.append(
                {
                    "content": chunk_text,
                    "chunk_index": i,
                    "total_chunks": len(texts),
                    "metadata": metadata,
                }
            )
    return chunks


def transform_silver_to_gold(client: Minio, object_key: str) -> str | None:
    try:
        response = client.get_object(BUCKET_SILVER, object_key)
        raw = response.read().decode("utf-8")
        response.close()
    except Exception as e:
        logger.error(f"Erro ao ler Silver {object_key}: {e}")
        return None

    records = [json.loads(line) for line in raw.strip().split("\n") if line]
    chunks = chunk_and_prepare(records)

    if not chunks:
        return None

    gold_key = object_key.replace("cleaned/", "chunks/")
    content_bytes = "\n".join(json.dumps(c, ensure_ascii=False) for c in chunks).encode(
        "utf-8"
    )

    if not client.bucket_exists(BUCKET_GOLD):
        client.make_bucket(BUCKET_GOLD)

    client.put_object(
        bucket_name=BUCKET_GOLD,
        object_name=gold_key,
        data=BytesIO(content_bytes),
        length=len(content_bytes),
        content_type="application/jsonlines",
    )

    logger.info(
        f"✔ Gold: {gold_key} ({len(chunks)} chunks de {len(records)} registros)"
    )
    return gold_key


def run():
    logger.info("=== Silver → Gold iniciado ===")
    client = get_minio_client()
    objects = list(client.list_objects(BUCKET_SILVER, recursive=True))
    logger.info(f"Arquivos Silver encontrados: {len(objects)}")

    for obj in objects:
        transform_silver_to_gold(client, obj.object_name)

    logger.info("=== Silver → Gold concluído ===")


if __name__ == "__main__":
    run()
