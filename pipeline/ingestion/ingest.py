"""
Pipeline — Ingestão Bronze
Coleta metadados do arXiv e persiste no bucket Bronze (MinIO).
"""
import json
import os
import uuid
import hashlib
import re
import time
from datetime import datetime, timezone
from io import BytesIO
from typing import Iterator
from xml.etree import ElementTree as ET

import httpx
from loguru import logger
from minio import Minio

from api.services.postgres_service import PostgresService

BUCKET_BRONZE = "bronze"
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_USER     = os.getenv("MINIO_ROOT_USER", "minioadmin")
MINIO_PASS     = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")
DATASET_NAME   = os.getenv("DATASET_NAME", "arxiv")
DATASET_DOMAIN = os.getenv("DATASET_DOMAIN", "research")
DATASET_SOURCE_URL = os.getenv("DATASET_SOURCE_URL", "https://arxiv.org/")
DATASET_VERSION = os.getenv("DATASET_VERSION", datetime.now(timezone.utc).strftime("%Y.%m.%d"))

ARXIV_BASE_URL   = os.getenv("ARXIV_BASE_URL", "https://export.arxiv.org/api/query")
ARXIV_QUERY      = os.getenv("ARXIV_QUERY", "cat:cs.AI")
ARXIV_START      = int(os.getenv("ARXIV_START", "0"))
ARXIV_MAX_RESULTS = int(os.getenv("ARXIV_MAX_RESULTS", "100"))
ARXIV_PAGE_SIZE  = int(os.getenv("ARXIV_PAGE_SIZE", "100"))
ARXIV_MAX_PAGES  = int(os.getenv("ARXIV_MAX_PAGES", "0"))  # 0 = sem limite
ARXIV_REQUEST_DELAY_SECONDS = float(os.getenv("ARXIV_REQUEST_DELAY_SECONDS", "0.25"))
ARXIV_DEDUP_IN_RUN = os.getenv("ARXIV_DEDUP_IN_RUN", "true").strip().lower() in {"1", "true", "yes"}
ARXIV_RETRY_ATTEMPTS = int(os.getenv("ARXIV_RETRY_ATTEMPTS", "5"))
ARXIV_RETRY_BACKOFF_SECONDS = float(os.getenv("ARXIV_RETRY_BACKOFF_SECONDS", "1.0"))
ARXIV_RETRY_BACKOFF_FACTOR = float(os.getenv("ARXIV_RETRY_BACKOFF_FACTOR", "2.0"))
ARXIV_FAIL_FAST = os.getenv("ARXIV_FAIL_FAST", "false").strip().lower() in {"1", "true", "yes"}
ARXIV_SORT_BY    = os.getenv("ARXIV_SORT_BY", "submittedDate")
ARXIV_SORT_ORDER = os.getenv("ARXIV_SORT_ORDER", "descending")

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def get_minio_client() -> Minio:
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_USER,
        secret_key=MINIO_PASS,
        secure=False,
    )


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def get_pdf_url(entry: ET.Element, arxiv_id: str) -> str:
    for link in entry.findall("atom:link", ATOM_NS):
        title = (link.attrib.get("title") or "").lower()
        link_type = (link.attrib.get("type") or "").lower()
        href = link.attrib.get("href")
        if not href:
            continue
        if title == "pdf" or link_type == "application/pdf":
            return href
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"


def parse_feed(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    records: list[dict] = []

    for entry in root.findall("atom:entry", ATOM_NS):
        raw_id = (entry.findtext("atom:id", default="", namespaces=ATOM_NS) or "").strip()
        arxiv_id = raw_id.split("/abs/")[-1] if "/abs/" in raw_id else raw_id

        title = normalize_text(entry.findtext("atom:title", default="", namespaces=ATOM_NS))
        summary = normalize_text(entry.findtext("atom:summary", default="", namespaces=ATOM_NS))

        authors = []
        for author in entry.findall("atom:author", ATOM_NS):
            name = normalize_text(author.findtext("atom:name", default="", namespaces=ATOM_NS))
            if name:
                authors.append(name)

        categories = [
            cat.attrib.get("term", "").strip()
            for cat in entry.findall("atom:category", ATOM_NS)
            if cat.attrib.get("term")
        ]

        primary_category = ""
        primary = entry.find("arxiv:primary_category", ATOM_NS)
        if primary is not None:
            primary_category = (primary.attrib.get("term") or "").strip()

        record = {
            "source": "arxiv",
            "id": arxiv_id,
            "title": title,
            "summary": summary,
            "authors": authors,
            "categories": categories,
            "primary_category": primary_category,
            "published": (entry.findtext("atom:published", default="", namespaces=ATOM_NS) or "").strip(),
            "updated": (entry.findtext("atom:updated", default="", namespaces=ATOM_NS) or "").strip(),
            "comment": normalize_text(entry.findtext("arxiv:comment", default="", namespaces=ATOM_NS)),
            "journal_ref": normalize_text(entry.findtext("arxiv:journal_ref", default="", namespaces=ATOM_NS)),
            "doi": normalize_text(entry.findtext("arxiv:doi", default="", namespaces=ATOM_NS)),
            "pdf_url": get_pdf_url(entry, arxiv_id),
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "query": ARXIV_QUERY,
        }
        records.append(record)

    return records


def load_raw_page(client: httpx.Client, start: int, max_results: int) -> list[dict]:
    """
    Busca uma página de registros no arXiv API (Atom feed).
    """
    params = {
        "search_query": ARXIV_QUERY,
        "start": start,
        "max_results": max_results,
        "sortBy": ARXIV_SORT_BY,
        "sortOrder": ARXIV_SORT_ORDER,
    }
    logger.info("Consultando arXiv: {}", params)

    attempts = max(1, ARXIV_RETRY_ATTEMPTS)
    for attempt in range(1, attempts + 1):
        try:
            response = client.get(
                ARXIV_BASE_URL,
                params=params,
                headers={"User-Agent": "awesomeproject-rag-medallion/1.0"},
            )
            response.raise_for_status()

            records = parse_feed(response.text)
            logger.info("arXiv retornou {} registros (start={})", len(records), start)
            return records
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            retryable = status_code == 429 or (status_code is not None and status_code >= 500)
            if not retryable or attempt >= attempts:
                raise
            backoff = ARXIV_RETRY_BACKOFF_SECONDS * (ARXIV_RETRY_BACKOFF_FACTOR ** (attempt - 1))
            logger.warning(
                "Falha HTTP {} no arXiv (start={}, attempt={}/{}). Tentando novamente em {:.2f}s.",
                status_code,
                start,
                attempt,
                attempts,
                backoff,
            )
            time.sleep(backoff)
        except httpx.RequestError as exc:
            if attempt >= attempts:
                raise
            backoff = ARXIV_RETRY_BACKOFF_SECONDS * (ARXIV_RETRY_BACKOFF_FACTOR ** (attempt - 1))
            logger.warning(
                "Erro de rede no arXiv (start={}, attempt={}/{}): {}. Retry em {:.2f}s.",
                start,
                attempt,
                attempts,
                exc,
                backoff,
            )
            time.sleep(backoff)

    return []


def compute_checksum(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def iter_raw_data_pages() -> Iterator[tuple[int, list[dict]]]:
    """
    Itera sobre o arXiv em páginas para permitir ingestão volumosa sem estourar memória.
    """
    page_size = max(1, ARXIV_PAGE_SIZE)
    total_target = max(1, ARXIV_MAX_RESULTS)
    max_pages = max(0, ARXIV_MAX_PAGES)

    page = 0
    current_start = ARXIV_START
    total_fetched = 0

    with httpx.Client(timeout=60) as client:
        while total_fetched < total_target:
            if max_pages > 0 and page >= max_pages:
                logger.info("Limite de páginas atingido (ARXIV_MAX_PAGES={})", max_pages)
                break

            remaining = total_target - total_fetched
            current_page_size = min(page_size, remaining)
            try:
                records = load_raw_page(client=client, start=current_start, max_results=current_page_size)
            except Exception as exc:
                logger.error(
                    "Falha ao consultar arXiv em start={} (page={}): {}",
                    current_start,
                    page + 1,
                    exc,
                )
                if ARXIV_FAIL_FAST:
                    raise
                logger.warning(
                    "Encerrando paginação com dados parciais. Defina ARXIV_FAIL_FAST=true para abortar em erro."
                )
                break

            if not records:
                logger.info("Sem mais resultados no arXiv a partir de start={}", current_start)
                break

            yield current_start, records
            fetched = len(records)
            total_fetched += fetched
            current_start += fetched
            page += 1

            if fetched < current_page_size:
                logger.info("Página parcial recebida ({}/{}). Encerrando paginação.", fetched, current_page_size)
                break

            if ARXIV_REQUEST_DELAY_SECONDS > 0 and total_fetched < total_target:
                time.sleep(ARXIV_REQUEST_DELAY_SECONDS)


def load_raw_data() -> list[dict]:
    """
    Compatibilidade: carrega todos os registros alvo em memória.
    Para ingestão em volume, prefira `iter_raw_data_pages()`.
    """
    all_records: list[dict] = []
    for _, page_records in iter_raw_data_pages():
        all_records.extend(page_records)
    return all_records


def ingest_to_bronze(
    client: Minio,
    records: list[dict],
    dataset_name: str = DATASET_NAME,
    page_number: int = 1,
    page_start: int = 0,
) -> dict:
    """
    Salva registros brutos no MinIO bucket Bronze como JSON lines.
    Retorna metadados do objeto gerado.
    """
    if not client.bucket_exists(BUCKET_BRONZE):
        client.make_bucket(BUCKET_BRONZE)

    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    object_key = (
        f"{dataset_name}/raw/"
        f"{batch_id}_p{page_number:05d}_s{page_start:07d}_n{len(records):04d}_{uuid.uuid4().hex[:8]}.jsonl"
    )

    content = "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
    content_bytes = content.encode("utf-8")

    client.put_object(
        bucket_name=BUCKET_BRONZE,
        object_name=object_key,
        data=BytesIO(content_bytes),
        length=len(content_bytes),
        content_type="application/jsonlines",
    )

    checksum = compute_checksum(content_bytes)
    logger.info(f"✔ Bronze: {object_key} ({len(records)} registros, checksum={checksum[:12]}...)")
    return {
        "object_key": object_key,
        "row_count": len(records),
        "size_bytes": len(content_bytes),
        "checksum": checksum,
        "page_number": page_number,
        "page_start": page_start,
    }


def register_bronze_metadata(db: PostgresService, dataset_id: str, info: dict) -> None:
    data_file_id = db.register_data_file(
        dataset_id=dataset_id,
        layer="bronze",
        bucket=BUCKET_BRONZE,
        object_key=info["object_key"],
        file_format="jsonl",
        size_bytes=info["size_bytes"],
        row_count=info["row_count"],
        checksum=info["checksum"],
        status="done",
    )
    db.log_audit(
        entity="data_files",
        entity_id=data_file_id,
        action="ingest_bronze",
        details={
            "dataset": DATASET_NAME,
            "query": ARXIV_QUERY,
            "object_key": info["object_key"],
            "rows": info["row_count"],
            "size_bytes": info["size_bytes"],
            "page_number": info.get("page_number"),
            "page_start": info.get("page_start"),
        },
    )


def run():
    logger.info("=== Ingestão Bronze iniciada ===")
    if ARXIV_MAX_RESULTS <= 0:
        logger.warning("ARXIV_MAX_RESULTS <= 0. Encerrando sem ingestão.")
        return

    db = PostgresService()
    client = get_minio_client()
    dataset_id = db.ensure_dataset(
        name=DATASET_NAME,
        domain=DATASET_DOMAIN,
        source_url=DATASET_SOURCE_URL,
        version=DATASET_VERSION,
        description=f"Dataset {DATASET_NAME} ingerido da API do arXiv",
    )

    total_raw = 0
    total_written = 0
    total_files = 0
    total_duplicates = 0
    seen_ids: set[str] = set()
    has_pages = False

    for page_number, (page_start, records) in enumerate(iter_raw_data_pages(), start=1):
        has_pages = True
        total_raw += len(records)
        filtered_records: list[dict] = []

        for record in records:
            record_id = str(record.get("id") or "").strip()
            if ARXIV_DEDUP_IN_RUN and record_id:
                if record_id in seen_ids:
                    total_duplicates += 1
                    continue
                seen_ids.add(record_id)
            filtered_records.append(record)

        if not filtered_records:
            logger.info(
                "Página {} (start={}) sem novos registros após deduplicação.",
                page_number,
                page_start,
            )
            continue

        info = ingest_to_bronze(
            client=client,
            records=filtered_records,
            dataset_name=DATASET_NAME,
            page_number=page_number,
            page_start=page_start,
        )
        register_bronze_metadata(db=db, dataset_id=dataset_id, info=info)
        total_written += len(filtered_records)
        total_files += 1

    if not has_pages:
        logger.warning("Nenhum registro retornado do arXiv. Encerrando sem escrita no Bronze.")
        return

    db.log_audit(
        entity="rag_datasets",
        entity_id=dataset_id,
        action="ingest_bronze_batch_completed",
        details={
            "dataset": DATASET_NAME,
            "query": ARXIV_QUERY,
            "target_records": ARXIV_MAX_RESULTS,
            "page_size": ARXIV_PAGE_SIZE,
            "max_pages": ARXIV_MAX_PAGES,
            "files_created": total_files,
            "raw_records": total_raw,
            "written_records": total_written,
            "duplicates_skipped": total_duplicates,
        },
    )

    logger.info(
        "=== Ingestão Bronze concluída: {} arquivos, {} registros gravados ({} lidos, {} duplicados) ===",
        total_files,
        total_written,
        total_raw,
        total_duplicates,
    )


if __name__ == "__main__":
    run()
