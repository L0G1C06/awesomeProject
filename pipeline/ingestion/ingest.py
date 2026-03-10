"""
Pipeline — Ingestão Bronze
Responsável por receber dados brutos e persistir no bucket Bronze (MinIO).
Adapte o método `load_raw_data()` ao seu dataset.
"""
import os
import uuid
import hashlib
from pathlib import Path
from datetime import datetime
from loguru import logger
from minio import Minio

# TODO: Importe/adapte conforme o dataset escolhido
# from datasets import load_dataset
# import pandas as pd

BUCKET_BRONZE = "bronze"
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_USER     = os.getenv("MINIO_ROOT_USER", "minioadmin")
MINIO_PASS     = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")


def get_minio_client() -> Minio:
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_USER,
        secret_key=MINIO_PASS,
        secure=False,
    )


def load_raw_data() -> list[dict]:
    """
    TODO: Substitua pelo carregamento do seu dataset.
    Retorna lista de registros brutos.
    Exemplos:
      - pd.read_csv("path/to/file.csv").to_dict(orient="records")
      - load_dataset("nome_do_dataset")["train"]
      - requests.get(URL_DO_DATASET).json()
    """
    raise NotImplementedError(
        "Implemente load_raw_data() com o dataset escolhido!"
    )


def compute_checksum(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def ingest_to_bronze(records: list[dict], dataset_name: str = "dataset") -> list[str]:
    """
    Salva registros brutos no MinIO bucket Bronze como JSON lines.
    Retorna lista de object_keys gerados.
    """
    client = get_minio_client()
    if not client.bucket_exists(BUCKET_BRONZE):
        client.make_bucket(BUCKET_BRONZE)

    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    object_key = f"{dataset_name}/raw/{batch_id}_{uuid.uuid4().hex[:8]}.jsonl"

    import json
    content = "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
    content_bytes = content.encode("utf-8")

    from io import BytesIO
    client.put_object(
        bucket_name=BUCKET_BRONZE,
        object_name=object_key,
        data=BytesIO(content_bytes),
        length=len(content_bytes),
        content_type="application/jsonlines",
    )

    checksum = compute_checksum(content_bytes)
    logger.info(f"✔ Bronze: {object_key} ({len(records)} registros, checksum={checksum[:12]}...)")
    return object_key


def run():
    logger.info("=== Ingestão Bronze iniciada ===")
    records = load_raw_data()
    logger.info(f"Total de registros carregados: {len(records)}")
    key = ingest_to_bronze(records)
    logger.info(f"=== Ingestão Bronze concluída: {key} ===")


if __name__ == "__main__":
    run()