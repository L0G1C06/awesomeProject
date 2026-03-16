"""
Serviço Milvus: gerencia coleção, indexação e busca vetorial.
"""
from loguru import logger
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)
from api.schemas.config import settings


class MilvusService:
    def __init__(self):
        connections.connect(host=settings.MILVUS_HOST, port=settings.MILVUS_PORT)
        self._ensure_collection()

    def _ensure_collection(self):
        name = settings.MILVUS_COLLECTION
        if utility.has_collection(name):
            self.collection = Collection(name)
            return

        logger.info(f"Criando coleção Milvus: {name}")
        fields = [
            FieldSchema(name="id",        dtype=DataType.INT64,          is_primary=True, auto_id=True),
            FieldSchema(name="doc_uuid",  dtype=DataType.VARCHAR,        max_length=64),
            FieldSchema(name="content",   dtype=DataType.VARCHAR,        max_length=8192),
            FieldSchema(name="metadata",  dtype=DataType.JSON),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR,   dim=settings.OLLAMA_EMBED_DIMENSION),
        ]
        schema = CollectionSchema(fields, description="RAG Enterprise Documents")
        self.collection = Collection(name=name, schema=schema)

        # Índice HNSW
        self.collection.create_index(
            field_name="embedding",
            index_params={
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {"M": 16, "efConstruction": 200},
            },
        )
        logger.info("Coleção e índice criados.")

    def insert(self, doc_uuid: str, content: str, embedding: list[float], metadata: dict = {}) -> int:
        self.collection.load()
        res = self.collection.insert([
            {"doc_uuid": doc_uuid, "content": content, "embedding": embedding, "metadata": metadata}
        ])
        return res.primary_keys[0]

    def search(self, vector: list[float], top_k: int = 5, collection_name: str = None) -> list[dict]:
        self.collection.load()
        results = self.collection.search(
            data=[vector],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k,
            output_fields=["doc_uuid", "content", "metadata"],
        )
        hits = []
        for hit in results[0]:
            content = hit.entity.get("content")
            metadata = hit.entity.get("metadata")
            hits.append({
                "id": hit.id,
                "content": content,
                "score": hit.score,
                "metadata": metadata or {},
            })
        return hits

    def flush(self) -> None:
        """Força persistência dos segmentos para refletir contagem/estado final."""
        self.collection.flush()

    def delete_collection(self):
        from pymilvus import utility
        if utility.has_collection(settings.MILVUS_COLLECTION):
            utility.drop_collection(settings.MILVUS_COLLECTION)
            logger.warning(f"Coleção {settings.MILVUS_COLLECTION} removida.")
