"""Testes unitários — Pipeline de processamento."""

from pipeline.ingestion.ingest import build_search_query, parse_arxiv_feed
from pipeline.processing.bronze_to_silver import clean_record
from pipeline.processing.silver_to_gold import (
    chunk_and_prepare,
    extract_metadata,
    extract_text,
)


class TestCleanRecord:
    def test_valid_record_passes(self):
        record = {"text": "Este é um texto válido com conteúdo suficiente.", "id": 1}
        result = clean_record(record)
        assert result is not None
        assert "_processed_at" in result

    def test_empty_text_returns_none(self):
        assert clean_record({"text": ""}) is None

    def test_none_values_removed(self):
        record = {"text": "Texto válido com mais de dez caracteres.", "field": None}
        result = clean_record(record)
        assert "field" not in result

    def test_strings_stripped(self):
        record = {"text": "  Texto com espaços   "}
        result = clean_record(record)
        assert result["text"] == "Texto com espaços"

    def test_summary_field_from_arxiv_passes(self):
        record = {
            "summary": "  Resumo do artigo com informacao suficiente para validacao.  ",
            "authors": ["  Alice  ", "Bob"],
        }
        result = clean_record(record)
        assert result is not None
        assert (
            result["summary"]
            == "Resumo do artigo com informacao suficiente para validacao."
        )
        assert result["authors"] == ["Alice", "Bob"]


class TestChunking:
    def test_long_text_creates_multiple_chunks(self):
        long_text = "Esta é uma frase. " * 100
        records = [{"text": long_text}]
        chunks = chunk_and_prepare(records)
        assert len(chunks) > 1

    def test_chunk_has_required_fields(self):
        records = [{"text": "Texto de teste para chunking com tamanho suficiente."}]
        chunks = chunk_and_prepare(records)
        for chunk in chunks:
            assert "content" in chunk
            assert "chunk_index" in chunk
            assert "metadata" in chunk

    def test_extract_text_from_various_fields(self):
        record = {"title": "Título", "description": "Descrição"}
        text = extract_text(record)
        assert "Título" in text
        assert "Descrição" in text

    def test_extract_metadata_flattens_string_lists(self):
        record = {
            "title": "Titulo",
            "summary": "Resumo",
            "authors": ["Alice", "Bob"],
            "categories": ["cs.LG", "cs.AI"],
        }
        metadata = extract_metadata(record)
        assert metadata["authors"] == "Alice, Bob"
        assert metadata["categories"] == "cs.LG, cs.AI"
        assert "summary" not in metadata


class TestArxivIngestion:
    def test_build_search_query_supports_multiple_categories(self):
        query = build_search_query(["cs.LG", "cs.AI"])
        assert query == "cat:cs.LG OR cat:cs.AI"

    def test_parse_arxiv_feed_extracts_expected_fields(self):
        feed = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <opensearch:totalResults>1</opensearch:totalResults>
  <entry>
    <id>http://arxiv.org/abs/2401.12345v1</id>
    <updated>2024-01-31T00:00:00Z</updated>
    <published>2024-01-30T00:00:00Z</published>
    <title>Sample Paper</title>
    <summary>Sample abstract for testing ingestion.</summary>
    <author><name>Alice</name></author>
    <author><name>Bob</name></author>
    <link href="http://arxiv.org/abs/2401.12345v1" rel="alternate" type="text/html" />
    <link title="pdf" href="http://arxiv.org/pdf/2401.12345v1" rel="related" type="application/pdf" />
    <arxiv:primary_category term="cs.LG" scheme="http://arxiv.org/schemas/atom" />
    <category term="cs.LG" scheme="http://arxiv.org/schemas/atom" />
    <category term="cs.AI" scheme="http://arxiv.org/schemas/atom" />
    <arxiv:doi>10.1000/test</arxiv:doi>
  </entry>
</feed>
"""
        records, total = parse_arxiv_feed(feed, "cat:cs.LG")
        assert total == 1
        assert len(records) == 1

        record = records[0]
        assert record["arxiv_id"] == "2401.12345v1"
        assert record["title"] == "Sample Paper"
        assert record["summary"] == "Sample abstract for testing ingestion."
        assert record["authors"] == ["Alice", "Bob"]
        assert record["primary_category"] == "cs.LG"
        assert record["categories"] == ["cs.LG", "cs.AI"]
        assert record["pdf_url"] == "http://arxiv.org/pdf/2401.12345v1"
        assert record["doi"] == "10.1000/test"
