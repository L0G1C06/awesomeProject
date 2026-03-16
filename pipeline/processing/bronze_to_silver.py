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
MINIO_USER = os.getenv("MINIO_ROOT_USER", "minioadmin")
MINIO_PASS = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")
BUCKET_BRONZE = "bronze"
BUCKET_SILVER = "silver"


def get_minio_client() -> Minio:
    return Minio(
        MINIO_ENDPOINT, access_key=MINIO_USER, secret_key=MINIO_PASS, secure=False
    )


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
    # O schema do arXiv usa "summary" como texto principal.
    text_field = (
        record.get("summary")
        or record.get("text")
        or record.get("content")
        or record.get("description")
        or ""
    )
    if not text_field or len(str(text_field).strip()) < 10:
        return None

    if not arxiv_id or not title or len(summary) < 20:
        return None

    # Normalização básica de strings
    for key, value in cleaned.items():
        if isinstance(value, str):
            cleaned[key] = " ".join(value.split())
        elif isinstance(value, list):
            cleaned[key] = [
                " ".join(item.split()) if isinstance(item, str) else item
                for item in value
                if item not in (None, "")
            ]

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

    silver_key = object_key.replace("raw/", "cleaned/")
    content_bytes = "\n".join(
        json.dumps(r, ensure_ascii=False) for r in cleaned
    ).encode("utf-8")

    if not client.bucket_exists(BUCKET_SILVER):
        client.make_bucket(BUCKET_SILVER)

    client.put_object(
        bucket_name=BUCKET_SILVER,
        object_name=silver_key,
        data=BytesIO(content_bytes),
        length=len(content_bytes),
        content_type="application/jsonlines",
    )

    logger.info(
        f"✔ Silver: {silver_key} ({len(cleaned)}/{len(records)} registros válidos)"
    )
    return silver_key


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
