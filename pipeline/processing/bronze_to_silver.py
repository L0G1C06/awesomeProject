"""
Pipeline — Bronze → Silver
Limpeza, normalização e enriquecimento dos dados brutos.
"""
import json
import os
import re
import hashlib
from io import BytesIO
from datetime import datetime
from loguru import logger
from minio import Minio
from minio.error import S3Error

from api.services.postgres_service import PostgresService

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_USER     = os.getenv("MINIO_ROOT_USER", "minioadmin")
MINIO_PASS     = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")
BUCKET_BRONZE  = "bronze"
BUCKET_SILVER  = "silver"
DATASET_NAME   = os.getenv("DATASET_NAME", "arxiv")
BRONZE_PREFIX  = os.getenv("BRONZE_PREFIX", f"{DATASET_NAME}/raw/")
DATASET_DOMAIN = os.getenv("DATASET_DOMAIN", "research")
DATASET_SOURCE_URL = os.getenv("DATASET_SOURCE_URL", "https://arxiv.org/")
DATASET_VERSION = os.getenv("DATASET_VERSION", "1.0.0")


def get_minio_client() -> Minio:
    return Minio(MINIO_ENDPOINT, access_key=MINIO_USER, secret_key=MINIO_PASS, secure=False)


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def object_exists(client: Minio, bucket: str, object_key: str) -> bool:
    try:
        client.stat_object(bucket, object_key)
        return True
    except S3Error:
        return False


def compute_checksum(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def list_pending_bronze(client: Minio, prefix: str = BRONZE_PREFIX) -> list[str]:
    """Lista objetos do Bronze para processamento."""
    objects = client.list_objects(BUCKET_BRONZE, prefix=prefix, recursive=True)
    return [obj.object_name for obj in objects if obj.object_name.endswith(".jsonl")]


def clean_record(record: dict) -> dict | None:
    """
    Normaliza metadados do arXiv.
    Retorna None para descartar registros inválidos.
    """
    arxiv_id = normalize_text(str(record.get("id", "")))
    title = normalize_text(str(record.get("title", "")))
    summary = normalize_text(str(record.get("summary", "")))

    if not arxiv_id or not title or len(summary) < 20:
        return None

    authors = [normalize_text(str(a)) for a in record.get("authors", []) if normalize_text(str(a))]
    categories = [normalize_text(str(c)) for c in record.get("categories", []) if normalize_text(str(c))]

    cleaned = {
        "source": "arxiv",
        "id": arxiv_id,
        "title": title,
        "summary": summary,
        "authors": authors,
        "categories": categories,
        "primary_category": normalize_text(str(record.get("primary_category", ""))),
        "published": normalize_text(str(record.get("published", ""))),
        "updated": normalize_text(str(record.get("updated", ""))),
        "comment": normalize_text(str(record.get("comment", ""))),
        "journal_ref": normalize_text(str(record.get("journal_ref", ""))),
        "doi": normalize_text(str(record.get("doi", ""))),
        "pdf_url": normalize_text(str(record.get("pdf_url", ""))),
        "query": normalize_text(str(record.get("query", ""))),
        "abstract_length": len(summary),
        "_processed_at": datetime.now().isoformat(),
    }

    return cleaned


def transform_bronze_to_silver(client: Minio, object_key: str) -> dict | None:
    """Lê um arquivo do Bronze, limpa e salva no Silver."""
    if "/raw/" not in object_key:
        return None

    silver_key = object_key.replace("/raw/", "/cleaned/", 1)
    if object_exists(client, BUCKET_SILVER, silver_key):
        logger.info(f"Silver já existe, pulando: {silver_key}")
        return {"bronze_key": object_key, "silver_key": silver_key, "skipped": True}

    try:
        response = client.get_object(BUCKET_BRONZE, object_key)
        raw_content = response.read().decode("utf-8")
        response.close()
    except Exception as e:
        logger.error(f"Erro ao ler {object_key}: {e}")
        return None

    records = []
    for line in raw_content.strip().split("\n"):
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning(f"Linha inválida ignorada em {object_key}")

    cleaned = [clean_record(r) for r in records]
    cleaned = [r for r in cleaned if r is not None]

    if not cleaned:
        logger.warning(f"Nenhum registro válido em {object_key}")
        return None

    content_bytes = "\n".join(json.dumps(r, ensure_ascii=False) for r in cleaned).encode("utf-8")
    checksum = compute_checksum(content_bytes)

    if not client.bucket_exists(BUCKET_SILVER):
        client.make_bucket(BUCKET_SILVER)

    client.put_object(
        bucket_name=BUCKET_SILVER,
        object_name=silver_key,
        data=BytesIO(content_bytes),
        length=len(content_bytes),
        content_type="application/jsonlines",
    )

    logger.info(f"✔ Silver: {silver_key} ({len(cleaned)}/{len(records)} registros válidos)")
    return {
        "bronze_key": object_key,
        "silver_key": silver_key,
        "row_count": len(cleaned),
        "size_bytes": len(content_bytes),
        "checksum": checksum,
        "skipped": False,
    }


def register_silver_metadata(db: PostgresService, info: dict) -> None:
    if info.get("skipped"):
        return

    dataset_id: str | None = None
    bronze_meta = db.get_data_file(BUCKET_BRONZE, info["bronze_key"])
    if bronze_meta:
        dataset_id = str(bronze_meta.get("dataset_id") or "")
    if not dataset_id:
        dataset_id = db.ensure_dataset(
            name=DATASET_NAME,
            domain=DATASET_DOMAIN,
            source_url=DATASET_SOURCE_URL,
            version=DATASET_VERSION,
            description=f"Dataset {DATASET_NAME} processado para camada silver",
        )

    data_file_id = db.register_data_file(
        dataset_id=dataset_id,
        layer="silver",
        bucket=BUCKET_SILVER,
        object_key=info["silver_key"],
        file_format="jsonl",
        size_bytes=info["size_bytes"],
        row_count=info["row_count"],
        checksum=info["checksum"],
        status="done",
    )
    db.log_audit(
        entity="data_files",
        entity_id=data_file_id,
        action="transform_bronze_to_silver",
        details={
            "from": info["bronze_key"],
            "to": info["silver_key"],
            "rows": info["row_count"],
        },
    )


def run():
    logger.info("=== Bronze → Silver iniciado ===")
    client = get_minio_client()
    if not client.bucket_exists(BUCKET_BRONZE):
        logger.warning("Bucket bronze não existe. Nada para processar.")
        return

    db = PostgresService()
    bronze_files = list_pending_bronze(client)
    logger.info(f"Arquivos Bronze encontrados: {len(bronze_files)}")

    for obj_key in bronze_files:
        info = transform_bronze_to_silver(client, obj_key)
        if info:
            register_silver_metadata(db, info)

    logger.info("=== Bronze → Silver concluído ===")


if __name__ == "__main__":
    run()
