"""
Pipeline — Silver → Gold
Prepara documentos finais para consumo pelo RAG (chunking, formatação).
Arquitetura Medallion: silver/cleaned/ → gold/chunks/
"""

import json
import os
import sys
import hashlib
from io import BytesIO
from loguru import logger
from minio import Minio
from minio.error import S3Error
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from api.services.postgres_service import PostgresService

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_USER = os.getenv("MINIO_ROOT_USER", "minioadmin")
MINIO_PASS = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")

BUCKET_SILVER = "silver"
BUCKET_GOLD = "gold"
SILVER_PREFIX = os.getenv("SILVER_PREFIX", "")  # "" = todos os datasets

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 64))

DATASET_NAME = os.getenv("DATASET_NAME", "arxiv")
DATASET_DOMAIN = os.getenv("DATASET_DOMAIN", "research")
DATASET_SOURCE_URL = os.getenv("DATASET_SOURCE_URL", "https://arxiv.org/")
DATASET_VERSION = os.getenv("DATASET_VERSION", "1.0.0")


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
# Extração de texto e metadados
# ---------------------------------------------------------------------------

def extract_text(record: dict) -> str:
    """Monta o texto final indexável com base nos metadados do arXiv."""
    parts = []
    for field in ["title", "summary", "text", "content", "description"]:
        if val := record.get(field):
            parts.append(str(val))
    return " | ".join(parts) if parts else json.dumps(record)


def extract_metadata(record: dict) -> dict:
    """Mantém metadados relevantes para filtros e auditoria."""
    ignored_fields = {"text", "content", "description", "summary", "_processed_at"}
    metadata: dict = {}
    for key, value in record.items():
        if key in ignored_fields or isinstance(value, dict):
            continue
        if isinstance(value, list):
            if value and all(isinstance(item, str) for item in value):
                metadata[key] = ", ".join(value)
            continue
        metadata[key] = value
    return metadata


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Listagem e transformação
# ---------------------------------------------------------------------------

def list_silver_files(client: Minio, prefix: str = SILVER_PREFIX) -> list[str]:
    """Lista todos os .jsonl do bucket Silver com o prefixo informado."""
    objects = client.list_objects(BUCKET_SILVER, prefix=prefix, recursive=True)
    return [
        obj.object_name for obj in objects
        if obj.object_name.endswith(".jsonl") and "/cleaned/" in obj.object_name
    ]


def gold_key_for(silver_key: str) -> str:
    """Deriva o object_key Gold a partir do Silver (cleaned/ → chunks/)."""
    return silver_key.replace("/cleaned/", "/chunks/", 1)


def transform_silver_to_gold(client: Minio, silver_key: str) -> dict | None:
    """
    Lê um arquivo Silver, aplica chunking e grava no Gold.

    Retorna um dict com métricas do arquivo processado, ou None em caso de erro.
    Pula silenciosamente se o Gold correspondente já existir.
    """
    gold_key = gold_key_for(silver_key)

    if object_exists(client, BUCKET_GOLD, gold_key):
        logger.info("Gold já existe, pulando: {}", gold_key)
        return {
            "silver_key": silver_key,
            "gold_key": gold_key,
            "skipped": True,
            "records": 0,
            "chunks": 0,
            "checksum": None,
            "size_bytes": None,
        }

    # Leitura do Silver
    try:
        response = client.get_object(BUCKET_SILVER, silver_key)
        raw = response.read().decode("utf-8")
        response.close()
    except Exception as exc:
        logger.error("Erro ao ler Silver {}: {}", silver_key, exc)
        return None

    # Parse linha a linha (JSONL)
    records: list[dict] = []
    for line in raw.strip().splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("Linha JSON inválida ignorada em {}", silver_key)

    if not records:
        logger.warning("Nenhuma linha válida em {}", silver_key)
        return None

    # Chunking
    chunks = chunk_and_prepare(records)
    if not chunks:
        logger.warning("Nenhum chunk gerado para {}", silver_key)
        return None

    # Serialização e upload para Gold
    content_bytes = "\n".join(
        json.dumps(c, ensure_ascii=False) for c in chunks
    ).encode("utf-8")

    ensure_bucket(client, BUCKET_GOLD)
    client.put_object(
        bucket_name=BUCKET_GOLD,
        object_name=gold_key,
        data=BytesIO(content_bytes),
        length=len(content_bytes),
        content_type="application/jsonlines",
    )

    checksum = compute_checksum(content_bytes)
    logger.info(
        "✔ Gold: {} ({} chunks de {} registros, checksum={}...)",
        gold_key, len(chunks), len(records), checksum[:12],
    )

    return {
        "silver_key": silver_key,
        "gold_key": gold_key,
        "skipped": False,
        "records": len(records),
        "chunks": len(chunks),
        "checksum": checksum,
        "size_bytes": len(content_bytes),
    }


# ---------------------------------------------------------------------------
# Registro de metadados no Postgres
# ---------------------------------------------------------------------------

def register_gold_metadata(db: PostgresService, dataset_id: str, info: dict) -> None:
    """Registra o arquivo Gold na tabela data_files e loga auditoria."""
    if info.get("skipped"):
        return

    data_file_id = db.register_data_file(
        dataset_id=dataset_id,
        layer="gold",
        bucket=BUCKET_GOLD,
        object_key=info["gold_key"],
        file_format="jsonl",
        size_bytes=info["size_bytes"],
        row_count=info["chunks"],
        checksum=info["checksum"],
        status="done",
    )

    db.log_audit(
        entity="data_files",
        entity_id=data_file_id,
        action="gold_chunked",
        details={
            "silver_key": info["silver_key"],
            "gold_key": info["gold_key"],
            "source_records": info["records"],
            "chunks_generated": info["chunks"],
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
        },
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def run() -> None:
    logger.info("=== Silver → Gold iniciado ===")

    client = get_minio_client()

    if not client.bucket_exists(BUCKET_SILVER):
        logger.warning("Bucket '{}' não existe. Nada para processar.", BUCKET_SILVER)
        return

    db = PostgresService()
    dataset_id = db.ensure_dataset(
        name=DATASET_NAME,
        domain=DATASET_DOMAIN,
        source_url=DATASET_SOURCE_URL,
        version=DATASET_VERSION,
        description=f"Dataset {DATASET_NAME} — camada Gold",
    )

    silver_files = list_silver_files(client, prefix=SILVER_PREFIX)
    logger.info("Arquivos Silver encontrados: {}", len(silver_files))

    total_files = 0
    total_skipped = 0
    total_records = 0
    total_chunks = 0

    for silver_key in silver_files:
        info = transform_silver_to_gold(client, silver_key)
        if info is None:
            continue

        register_gold_metadata(db, dataset_id, info)

        if info["skipped"]:
            total_skipped += 1
        else:
            total_files += 1
            total_records += info["records"]
            total_chunks += info["chunks"]

    logger.info(
        "=== Silver → Gold concluído: {} arquivo(s) processado(s), "
        "{} ignorado(s), {} registros → {} chunks ===",
        total_files, total_skipped, total_records, total_chunks,
    )


if __name__ == "__main__":
    run()