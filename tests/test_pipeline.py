"""Testes unitários — Pipeline de processamento."""
import pytest
from pipeline.processing.bronze_to_silver import clean_record
from pipeline.processing.silver_to_gold import extract_text, chunk_and_prepare


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