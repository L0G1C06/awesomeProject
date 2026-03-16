"""
Pipeline — Ingestão Bronze
Baixa artigos do arXiv, salva uma cópia local em JSONL e, se disponível,
publica o mesmo lote no bucket Bronze do MinIO.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import httpx
from dotenv import load_dotenv
from loguru import logger
from minio import Minio
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

load_dotenv()

BUCKET_BRONZE = "bronze"
DEFAULT_DATASET_NAME = "arxiv"
MAX_API_SLICE = 2000
MAX_API_RESULTS = 30000
ARXIV_NAMESPACES = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_USER = os.getenv("MINIO_ROOT_USER", "minioadmin")
MINIO_PASS = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")


def parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_categories(raw_categories: str | None) -> list[str]:
    if not raw_categories:
        return ["cs.LG"]
    return [item.strip() for item in raw_categories.split(",") if item.strip()]


def normalize_whitespace(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(value.split())


@dataclass(slots=True)
class IngestionConfig:
    api_url: str = os.getenv("DATASET_SOURCE_URL", "https://export.arxiv.org/api/query")
    categories: list[str] = field(
        default_factory=lambda: parse_categories(os.getenv("ARXIV_CATEGORY", "cs.LG"))
    )
    max_results: int = int(os.getenv("ARXIV_MAX_RESULTS", "10000"))
    batch_size: int = int(os.getenv("ARXIV_BATCH_SIZE", str(MAX_API_SLICE)))
    delay_seconds: float = float(os.getenv("ARXIV_DELAY_SECONDS", "3"))
    sort_by: str = os.getenv("ARXIV_SORT_BY", "submittedDate")
    sort_order: str = os.getenv("ARXIV_SORT_ORDER", "descending")
    output_dir: Path = field(
        default_factory=lambda: Path(os.getenv("ARXIV_OUTPUT_DIR", "data/bronze"))
    )
    dataset_name: str = os.getenv("ARXIV_DATASET_NAME", DEFAULT_DATASET_NAME)
    write_to_minio: bool = parse_bool(os.getenv("ARXIV_WRITE_TO_MINIO"), default=True)
    user_agent: str = os.getenv("ARXIV_USER_AGENT", "awesomeProject/1.0")


def get_minio_client() -> Minio:
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_USER,
        secret_key=MINIO_PASS,
        secure=False,
    )


def compute_checksum(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def build_search_query(categories: list[str]) -> str:
    if not categories:
        raise ValueError("Informe ao menos uma categoria do arXiv em ARXIV_CATEGORY.")
    return " OR ".join(f"cat:{category}" for category in categories)


def extract_arxiv_id(entry_id: str) -> str:
    marker = "/abs/"
    if marker in entry_id:
        return entry_id.split(marker, maxsplit=1)[1]
    return entry_id.rsplit("/", maxsplit=1)[-1]


def extract_link(
    entry: ET.Element, *, title: str | None = None, rel: str | None = None
) -> str | None:
    for link in entry.findall("atom:link", ARXIV_NAMESPACES):
        if title is not None and link.get("title") != title:
            continue
        if rel is not None and link.get("rel") != rel:
            continue
        href = link.get("href")
        if href:
            return href
    return None


def parse_arxiv_feed(feed_xml: str, search_query: str) -> tuple[list[dict], int | None]:
    root = ET.fromstring(feed_xml)
    total_results = root.findtext(
        "opensearch:totalResults", namespaces=ARXIV_NAMESPACES
    )
    total = int(total_results) if total_results else None
    retrieved_at = datetime.now(timezone.utc).isoformat()

    records: list[dict] = []
    for entry in root.findall("atom:entry", ARXIV_NAMESPACES):
        entry_id = normalize_whitespace(
            entry.findtext("atom:id", namespaces=ARXIV_NAMESPACES)
        )
        title = normalize_whitespace(
            entry.findtext("atom:title", namespaces=ARXIV_NAMESPACES)
        )
        summary = normalize_whitespace(
            entry.findtext("atom:summary", namespaces=ARXIV_NAMESPACES)
        )
        authors = [
            normalize_whitespace(
                author.findtext("atom:name", namespaces=ARXIV_NAMESPACES)
            )
            for author in entry.findall("atom:author", ARXIV_NAMESPACES)
            if normalize_whitespace(
                author.findtext("atom:name", namespaces=ARXIV_NAMESPACES)
            )
        ]
        categories = [
            category.get("term")
            for category in entry.findall("atom:category", ARXIV_NAMESPACES)
            if category.get("term")
        ]
        primary_category_node = entry.find("arxiv:primary_category", ARXIV_NAMESPACES)
        primary_category = (
            primary_category_node.get("term")
            if primary_category_node is not None
            else None
        )
        html_url = extract_link(entry, rel="alternate") or entry_id
        pdf_url = extract_link(entry, title="pdf")
        if not pdf_url and entry_id:
            pdf_url = entry_id.replace("/abs/", "/pdf/") + ".pdf"

        records.append(
            {
                "source": "arXiv",
                "query": search_query,
                "id": entry_id,
                "arxiv_id": extract_arxiv_id(entry_id),
                "title": title,
                "summary": summary,
                "authors": authors,
                "categories": categories,
                "primary_category": primary_category,
                "published": normalize_whitespace(
                    entry.findtext("atom:published", namespaces=ARXIV_NAMESPACES)
                ),
                "updated": normalize_whitespace(
                    entry.findtext("atom:updated", namespaces=ARXIV_NAMESPACES)
                ),
                "comment": normalize_whitespace(
                    entry.findtext("arxiv:comment", namespaces=ARXIV_NAMESPACES)
                )
                or None,
                "journal_ref": normalize_whitespace(
                    entry.findtext("arxiv:journal_ref", namespaces=ARXIV_NAMESPACES)
                )
                or None,
                "doi": normalize_whitespace(
                    entry.findtext("arxiv:doi", namespaces=ARXIV_NAMESPACES)
                )
                or None,
                "html_url": html_url,
                "pdf_url": pdf_url,
                "retrieved_at": retrieved_at,
            }
        )

    return records, total


@retry(
    retry=retry_if_exception_type(httpx.HTTPError),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    stop=stop_after_attempt(5),
    reraise=True,
)
def fetch_arxiv_page(
    client: httpx.Client,
    *,
    api_url: str,
    search_query: str,
    start: int,
    batch_size: int,
    sort_by: str,
    sort_order: str,
) -> tuple[list[dict], int | None]:
    response = client.get(
        api_url,
        params={
            "search_query": search_query,
            "start": start,
            "max_results": batch_size,
            "sortBy": sort_by,
            "sortOrder": sort_order,
        },
    )
    response.raise_for_status()
    return parse_arxiv_feed(response.text, search_query)


def load_raw_data(config: IngestionConfig | None = None) -> list[dict]:
    config = config or IngestionConfig()
    requested_total = min(config.max_results, MAX_API_RESULTS)
    if config.max_results > MAX_API_RESULTS:
        logger.warning(
            "ARXIV_MAX_RESULTS acima do limite da API; ajustando para {} registros.",
            MAX_API_RESULTS,
        )

    batch_size = min(max(config.batch_size, 1), MAX_API_SLICE)
    search_query = build_search_query(config.categories)
    records: list[dict] = []
    total_available: int | None = None

    timeout = httpx.Timeout(60.0, connect=10.0)
    with httpx.Client(
        timeout=timeout,
        headers={"User-Agent": config.user_agent},
        follow_redirects=True,
    ) as client:
        for start in range(0, requested_total, batch_size):
            current_batch_size = min(batch_size, requested_total - start)
            logger.info(
                "Baixando arXiv: start={} max_results={} query='{}'",
                start,
                current_batch_size,
                search_query,
            )
            batch_records, batch_total = fetch_arxiv_page(
                client,
                api_url=config.api_url,
                search_query=search_query,
                start=start,
                batch_size=current_batch_size,
                sort_by=config.sort_by,
                sort_order=config.sort_order,
            )
            if total_available is None:
                total_available = batch_total
                if total_available is not None and total_available < requested_total:
                    logger.warning(
                        "A consulta retornou {} registros disponíveis; o download vai parar nesse total.",
                        total_available,
                    )

            records.extend(batch_records)
            if not batch_records or len(batch_records) < current_batch_size:
                break
            if len(records) >= requested_total:
                break
            if config.delay_seconds > 0:
                time.sleep(config.delay_seconds)

    return records[:requested_total]


def serialize_records(records: list[dict]) -> bytes:
    content = "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
    return content.encode("utf-8")


def save_records_locally(records: list[dict], config: IngestionConfig) -> Path:
    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = config.output_dir / config.dataset_name / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{batch_id}_{uuid.uuid4().hex[:8]}.jsonl"

    content_bytes = serialize_records(records)
    file_path.write_bytes(content_bytes)

    checksum = compute_checksum(content_bytes)
    logger.info(
        "✔ Local: {} ({} registros, checksum={}...)",
        file_path,
        len(records),
        checksum[:12],
    )
    return file_path


def ingest_to_bronze(
    records: list[dict], dataset_name: str = DEFAULT_DATASET_NAME
) -> str:
    """
    Salva registros brutos no MinIO bucket Bronze como JSON lines.
    Retorna o object_key gerado.
    """
    if not client.bucket_exists(BUCKET_BRONZE):
        client.make_bucket(BUCKET_BRONZE)

    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    object_key = f"{dataset_name}/raw/{batch_id}_{uuid.uuid4().hex[:8]}.jsonl"
    content_bytes = serialize_records(records)

    client.put_object(
        bucket_name=BUCKET_BRONZE,
        object_name=object_key,
        data=BytesIO(content_bytes),
        length=len(content_bytes),
        content_type="application/jsonlines",
    )

    checksum = compute_checksum(content_bytes)
    logger.info(
        "✔ Bronze: {} ({} registros, checksum={}...)",
        object_key,
        len(records),
        checksum[:12],
    )
    return object_key


def parse_args() -> IngestionConfig:
    parser = argparse.ArgumentParser(
        description="Baixa registros do arXiv e salva localmente."
    )
    parser.add_argument(
        "--categories",
        default=",".join(parse_categories(os.getenv("ARXIV_CATEGORY", "cs.LG"))),
        help="Lista de categorias separadas por vírgula. Ex.: cs.LG,cs.AI",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=int(os.getenv("ARXIV_MAX_RESULTS", "10000")),
        help="Quantidade de registros a baixar.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("ARXIV_BATCH_SIZE", str(MAX_API_SLICE))),
        help="Tamanho de cada página da API (máximo 2000).",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=float(os.getenv("ARXIV_DELAY_SECONDS", "3")),
        help="Atraso entre requisições para respeitar a API.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("ARXIV_OUTPUT_DIR", "data/bronze"),
        help="Diretório raiz para a cópia local.",
    )
    parser.add_argument(
        "--api-url",
        default=os.getenv("DATASET_SOURCE_URL", "https://export.arxiv.org/api/query"),
        help="Endpoint da API do arXiv.",
    )
    parser.add_argument(
        "--sort-by",
        default=os.getenv("ARXIV_SORT_BY", "submittedDate"),
        choices=["relevance", "lastUpdatedDate", "submittedDate"],
        help="Campo de ordenação suportado pela API do arXiv.",
    )
    parser.add_argument(
        "--sort-order",
        default=os.getenv("ARXIV_SORT_ORDER", "descending"),
        choices=["ascending", "descending"],
        help="Direção da ordenação.",
    )
    parser.add_argument(
        "--dataset-name",
        default=os.getenv("ARXIV_DATASET_NAME", DEFAULT_DATASET_NAME),
        help="Nome lógico do dataset para pastas locais e bucket Bronze.",
    )
    parser.add_argument(
        "--write-to-minio",
        action="store_true",
        default=parse_bool(os.getenv("ARXIV_WRITE_TO_MINIO"), default=True),
        help="Envia o lote também para o bucket Bronze.",
    )
    parser.add_argument(
        "--skip-minio",
        action="store_true",
        help="Evita a tentativa de upload no MinIO.",
    )
    parser.add_argument(
        "--user-agent",
        default=os.getenv("ARXIV_USER_AGENT", "awesomeProject/1.0"),
        help="User-Agent usado nas chamadas HTTP.",
    )
    args = parser.parse_args()

    return IngestionConfig(
        api_url=args.api_url,
        categories=parse_categories(args.categories),
        max_results=args.max_results,
        batch_size=args.batch_size,
        delay_seconds=args.delay_seconds,
        sort_by=args.sort_by,
        sort_order=args.sort_order,
        output_dir=Path(args.output_dir),
        dataset_name=args.dataset_name,
        write_to_minio=args.write_to_minio and not args.skip_minio,
        user_agent=args.user_agent,
    )


def run() -> tuple[Path, str | None]:
    logger.info("=== Ingestão Bronze iniciada ===")
    config = parse_args()
    records = load_raw_data(config)
    logger.info("Total de registros carregados: {}", len(records))
    local_path = save_records_locally(records, config)

    object_key: str | None = None
    if config.write_to_minio:
        try:
            object_key = ingest_to_bronze(records, dataset_name=config.dataset_name)
        except Exception as exc:
            logger.warning("MinIO indisponível, mantendo apenas a cópia local: {}", exc)

    logger.info("=== Ingestão Bronze concluída ===")
    return local_path, object_key


if __name__ == "__main__":
    run()
