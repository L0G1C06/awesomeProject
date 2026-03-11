"""
Pipeline — Silver → Gold
Prepara documentos finais para consumo pelo RAG (chunking, formatação).
"""
import json
import os
import hashlib
from io import BytesIO
from loguru import logger
from minio import Minio
from minio.error import S3Error
from langchain_text_splitters import RecursiveCharacterTextSplitter

from api.services.postgres_service import PostgresService

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_USER     = os.getenv("MINIO_ROOT_USER", "minioadmin")
MINIO_PASS     = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")
BUCKET_SILVER  = "silver"
BUCKET_GOLD    = "gold"
DATASET_NAME   = os.getenv("DATASET_NAME", "arxiv")
SILVER_PREFIX  = os.getenv("SILVER_PREFIX", f"{DATASET_NAME}/cleaned/")
DATASET_DOMAIN = os.getenv("DATASET_DOMAIN", "research")
DATASET_SOURCE_URL = os.getenv("DATASET_SOURCE_URL", "https://arxiv.org/")
DATASET_VERSION = os.getenv("DATASET_VERSION", "1.0.0")
CHUNK_SIZE     = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP  = int(os.getenv("CHUNK_OVERLAP", 64))


def get_minio_client() -> Minio:
    return Minio(MINIO_ENDPOINT, access_key=MINIO_USER, secret_key=MINIO_PASS, secure=False)


def object_exists(client: Minio, bucket: str, object_key: str) -> bool:
    try:
        client.stat_object(bucket, object_key)
        return True
    except S3Error:
        return False


def compute_checksum(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def extract_text(record: dict) -> str:
    """
    Monta o texto final indexável com base nos metadados do arXiv.
    """
    title = str(record.get("title", "")).strip()
    summary = str(record.get("summary", "")).strip()
    authors = ", ".join(record.get("authors", []))
    categories = ", ".join(record.get("categories", []))
    published = str(record.get("published", "")).strip()
    arxiv_id = str(record.get("id", "")).strip()

    sections = [
        f"ArXiv ID: {arxiv_id}" if arxiv_id else "",
        f"Title: {title}" if title else "",
        f"Authors: {authors}" if authors else "",
        f"Categories: {categories}" if categories else "",
        f"Published: {published}" if published else "",
        f"Abstract: {summary}" if summary else "",
    ]
    text = "\n".join([s for s in sections if s]).strip()
    return text if text else json.dumps(record, ensure_ascii=False)


def extract_metadata(record: dict) -> dict:
    """
    Mantém metadados relevantes para filtros e auditoria.
    """
    return {
        "source": record.get("source", "arxiv"),
        "arxiv_id": record.get("id"),
        "title": record.get("title"),
        "authors": record.get("authors", []),
        "categories": record.get("categories", []),
        "primary_category": record.get("primary_category"),
        "published": record.get("published"),
        "updated": record.get("updated"),
        "doi": record.get("doi"),
        "pdf_url": record.get("pdf_url"),
        "query": record.get("query"),
    }


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
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
            chunks.append({
                "content": chunk_text,
                "chunk_index": i,
                "total_chunks": len(texts),
                "source_id": record.get("id"),
                "metadata": metadata,
            })
    return chunks


def transform_silver_to_gold(client: Minio, object_key: str) -> dict | None:
    if "/cleaned/" not in object_key:
        return None

    gold_key = object_key.replace("/cleaned/", "/chunks/", 1)
    if object_exists(client, BUCKET_GOLD, gold_key):
        logger.info(f"Gold já existe, pulando: {gold_key}")
        return {"silver_key": object_key, "gold_key": gold_key, "skipped": True}

    try:
        response = client.get_object(BUCKET_SILVER, object_key)
        raw = response.read().decode("utf-8")
        response.close()
    except Exception as e:
        logger.error(f"Erro ao ler Silver {object_key}: {e}")
        return None

    records = [json.loads(l) for l in raw.strip().split("\n") if l]
    chunks = chunk_and_prepare(records)

    if not chunks:
        return None

    content_bytes = "\n".join(json.dumps(c, ensure_ascii=False) for c in chunks).encode("utf-8")
    checksum = compute_checksum(content_bytes)

    if not client.bucket_exists(BUCKET_GOLD):
        client.make_bucket(BUCKET_GOLD)

    client.put_object(
        bucket_name=BUCKET_GOLD,
        object_name=gold_key,
        data=BytesIO(content_bytes),
        length=len(content_bytes),
        content_type="application/jsonlines",
    )

    logger.info(f"✔ Gold: {gold_key} ({len(chunks)} chunks de {len(records)} registros)")
    return {
        "silver_key": object_key,
        "gold_key": gold_key,
        "row_count": len(chunks),
        "size_bytes": len(content_bytes),
        "checksum": checksum,
        "skipped": False,
    }


def register_gold_metadata(db: PostgresService, info: dict) -> None:
    if info.get("skipped"):
        return

    dataset_id: str | None = None
    silver_meta = db.get_data_file(BUCKET_SILVER, info["silver_key"])
    if silver_meta:
        dataset_id = str(silver_meta.get("dataset_id") or "")
    if not dataset_id:
        dataset_id = db.ensure_dataset(
            name=DATASET_NAME,
            domain=DATASET_DOMAIN,
            source_url=DATASET_SOURCE_URL,
            version=DATASET_VERSION,
            description=f"Dataset {DATASET_NAME} processado para camada gold",
        )

    data_file_id = db.register_data_file(
        dataset_id=dataset_id,
        layer="gold",
        bucket=BUCKET_GOLD,
        object_key=info["gold_key"],
        file_format="jsonl",
        size_bytes=info["size_bytes"],
        row_count=info["row_count"],
        checksum=info["checksum"],
        status="done",
    )
    db.log_audit(
        entity="data_files",
        entity_id=data_file_id,
        action="transform_silver_to_gold",
        details={
            "from": info["silver_key"],
            "to": info["gold_key"],
            "chunks": info["row_count"],
        },
    )


def run():
    logger.info("=== Silver → Gold iniciado ===")
    client = get_minio_client()
    if not client.bucket_exists(BUCKET_SILVER):
        logger.warning("Bucket silver não existe. Nada para processar.")
        return

    db = PostgresService()
    objects = list(client.list_objects(BUCKET_SILVER, prefix=SILVER_PREFIX, recursive=True))
    logger.info(f"Arquivos Silver encontrados: {len(objects)}")

    for obj in objects:
        info = transform_silver_to_gold(client, obj.object_name)
        if info:
            register_gold_metadata(db, info)

    logger.info("=== Silver → Gold concluído ===")


if __name__ == "__main__":
    run()
