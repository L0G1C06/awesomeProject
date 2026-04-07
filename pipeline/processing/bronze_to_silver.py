"""
Pipeline — Bronze → Silver
Limpeza, normalização e enriquecimento dos dados brutos.
Arquitetura Medallion: bronze/raw/ → silver/cleaned/
"""

import json
import os
import sys
import re
import hashlib
from io import BytesIO
from datetime import datetime, timezone
from loguru import logger
from minio import Minio
from minio.error import S3Error

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from api.services.postgres_service import PostgresService

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_USER = os.getenv("MINIO_ROOT_USER", "minioadmin")
MINIO_PASS = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")

BUCKET_BRONZE = "bronze"
BUCKET_SILVER = "silver"
BRONZE_PREFIX = os.getenv("BRONZE_PREFIX", "")  # "" = todos os datasets

DATASET_NAME = os.getenv("DATASET_NAME", "arxiv")
DATASET_DOMAIN = os.getenv("DATASET_DOMAIN", "research")
DATASET_SOURCE_URL = os.getenv("DATASET_SOURCE_URL", "https://arxiv.org/")
DATASET_VERSION = os.getenv("DATASET_VERSION", "1.0.0")

MIN_SUMMARY_LEN = int(os.getenv("SILVER_MIN_SUMMARY_LEN", "20"))


# ---------------------------------------------------------------------------
# Infraestrutura
# ---------------------------------------------------------------------------

def get_minio_client() -> Minio:
    return Minio(
        MINIO_ENDPOINT, access_key=MINIO_USER, secret_key=MINIO_PASS, secure=False
    )


def compute_checksum(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def object_exists(client: Minio, bucket: str, object_key: str) -> bool:
    try:
        client.stat_object(bucket, object_key)
        return True
    except S3Error:
        return False


def ensure_bucket(client: Minio, bucket: str) -> None:
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
        logger.info("Bucket criado: {}", bucket)


# ---------------------------------------------------------------------------
# Limpeza e normalização (camada Silver)
# ---------------------------------------------------------------------------

def normalize_text(value: str) -> str:
    """Remove espaços extras e normaliza quebras de linha."""
    return re.sub(r"\s+", " ", value).strip()


def clean_record(record: dict) -> dict | None:
    """
    Normaliza e valida um registro bruto do arXiv.
    Retorna None para descartar registros inválidos.
    """
    arxiv_id = record.get("arxiv_id") or record.get("id") or ""
    title = record.get("title") or ""
    summary = (
        record.get("summary")
        or record.get("text")
        or record.get("content")
        or record.get("description")
        or ""
    )

    # Descarta registros sem identificador, título ou texto mínimo
    if not arxiv_id or not title or len(summary.strip()) < MIN_SUMMARY_LEN:
        return None

    authors = record.get("authors") or []
    categories = record.get("categories") or []

    cleaned = {
        # Identidade
        "arxiv_id": normalize_text(str(arxiv_id)),
        "id": record.get("id") or arxiv_id,
        "source": record.get("source", "arXiv"),
        # Conteúdo
        "title": normalize_text(title),
        "summary": normalize_text(summary),
        # Listas normalizadas — remove entradas vazias
        "authors": [
            normalize_text(a) for a in authors
            if isinstance(a, str) and a.strip()
        ],
        "categories": [
            c.strip() for c in categories
            if isinstance(c, str) and c.strip()
        ],
        "primary_category": record.get("primary_category"),
        # URLs
        "html_url": record.get("html_url"),
        "pdf_url": record.get("pdf_url"),
        "doi": record.get("doi"),
        "journal_ref": record.get("journal_ref"),
        # Datas
        "published": record.get("published"),
        "updated": record.get("updated"),
        # Rastreabilidade
        "query": record.get("query"),
        "retrieved_at": record.get("retrieved_at"),
        "silver_processed_at": datetime.now(timezone.utc).isoformat(),
    }

    # Remove chaves com valor None para manter o JSONL limpo
    return {k: v for k, v in cleaned.items() if v is not None}


# ---------------------------------------------------------------------------
# Listagem e transformação
# ---------------------------------------------------------------------------

def list_bronze_files(client: Minio, prefix: str = BRONZE_PREFIX) -> list[str]:
    """Lista todos os .jsonl do bucket Bronze com o prefixo informado."""
    objects = client.list_objects(BUCKET_BRONZE, prefix=prefix, recursive=True)
    return [
        obj.object_name for obj in objects
        if obj.object_name.endswith(".jsonl") and "/raw/" in obj.object_name
    ]


def silver_key_for(bronze_key: str) -> str:
    """Deriva o object_key Silver a partir do Bronze (raw/ → cleaned/)."""
    return bronze_key.replace("/raw/", "/cleaned/", 1)


def transform_bronze_to_silver(
    client: Minio,
    bronze_key: str,
) -> dict | None:
    """
    Lê um arquivo Bronze, aplica limpeza e grava no Silver.

    Retorna um dict com métricas do arquivo processado, ou None em caso de erro.
    Pula silenciosamente se o Silver correspondente já existir.
    """
    silver_key = silver_key_for(bronze_key)

    if object_exists(client, BUCKET_SILVER, silver_key):
        logger.info("Silver já existe, pulando: {}", silver_key)
        return {
            "bronze_key": bronze_key,
            "silver_key": silver_key,
            "skipped": True,
            "total": 0,
            "valid": 0,
            "discarded": 0,
            "checksum": None,
            "size_bytes": None,
        }

    # Leitura do Bronze
    try:
        response = client.get_object(BUCKET_BRONZE, bronze_key)
        raw_content = response.read().decode("utf-8")
        response.close()
    except Exception as exc:
        logger.error("Erro ao ler {}: {}", bronze_key, exc)
        return None

    # Parse linha a linha (JSONL)
    raw_records: list[dict] = []
    for line in raw_content.strip().splitlines():
        if not line.strip():
            continue
        try:
            raw_records.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("Linha JSON inválida ignorada em {}", bronze_key)

    if not raw_records:
        logger.warning("Nenhuma linha válida em {}", bronze_key)
        return None

    # Limpeza
    cleaned_records = [clean_record(r) for r in raw_records]
    cleaned_records = [r for r in cleaned_records if r is not None]
    discarded = len(raw_records) - len(cleaned_records)

    if not cleaned_records:
        logger.warning("Todos os registros descartados em {}", bronze_key)
        return None

    # Serialização e upload para Silver
    content_bytes = "\n".join(
        json.dumps(r, ensure_ascii=False) for r in cleaned_records
    ).encode("utf-8")

    ensure_bucket(client, BUCKET_SILVER)
    client.put_object(
        bucket_name=BUCKET_SILVER,
        object_name=silver_key,
        data=BytesIO(content_bytes),
        length=len(content_bytes),
        content_type="application/jsonlines",
    )

    checksum = compute_checksum(content_bytes)
    logger.info(
        "✔ Silver: {} ({}/{} registros válidos, {} descartados, checksum={}...)",
        silver_key, len(cleaned_records), len(raw_records), discarded, checksum[:12],
    )

    return {
        "bronze_key": bronze_key,
        "silver_key": silver_key,
        "skipped": False,
        "total": len(raw_records),
        "valid": len(cleaned_records),
        "discarded": discarded,
        "checksum": checksum,
        "size_bytes": len(content_bytes),
    }


# ---------------------------------------------------------------------------
# Registro de metadados no Postgres
# ---------------------------------------------------------------------------

def register_silver_metadata(db: PostgresService, dataset_id: str, info: dict) -> None:
    """Registra o arquivo Silver na tabela data_files e loga auditoria."""
    if info.get("skipped"):
        return

    data_file_id = db.register_data_file(
        dataset_id=dataset_id,
        layer="silver",
        bucket=BUCKET_SILVER,
        object_key=info["silver_key"],
        file_format="jsonl",
        size_bytes=info["size_bytes"],
        row_count=info["valid"],
        checksum=info["checksum"],
        status="done",
    )

    db.log_audit(
        entity="data_files",
        entity_id=data_file_id,
        action="silver_transformed",
        details={
            "bronze_key": info["bronze_key"],
            "silver_key": info["silver_key"],
            "total_records": info["total"],
            "valid_records": info["valid"],
            "discarded_records": info["discarded"],
        },
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def run() -> None:
    logger.info("=== Bronze → Silver iniciado ===")

    client = get_minio_client()

    if not client.bucket_exists(BUCKET_BRONZE):
        logger.warning("Bucket '{}' não existe. Nada para processar.", BUCKET_BRONZE)
        return

    db = PostgresService()
    dataset_id = db.ensure_dataset(
        name=DATASET_NAME,
        domain=DATASET_DOMAIN,
        source_url=DATASET_SOURCE_URL,
        version=DATASET_VERSION,
        description=f"Dataset {DATASET_NAME} — camada Silver",
    )

    bronze_files = list_bronze_files(client, prefix=BRONZE_PREFIX)
    logger.info("Arquivos Bronze encontrados: {}", len(bronze_files))

    total_files = 0
    total_skipped = 0
    total_valid = 0
    total_discarded = 0

    for bronze_key in bronze_files:
        info = transform_bronze_to_silver(client, bronze_key)
        if info is None:
            continue

        register_silver_metadata(db, dataset_id, info)

        if info["skipped"]:
            total_skipped += 1
        else:
            total_files += 1
            total_valid += info["valid"]
            total_discarded += info["discarded"]

    logger.info(
        "=== Bronze → Silver concluído: {} arquivo(s) processado(s), "
        "{} ignorado(s), {} registros válidos, {} descartados ===",
        total_files, total_skipped, total_valid, total_discarded,
    )


if __name__ == "__main__":
    run()