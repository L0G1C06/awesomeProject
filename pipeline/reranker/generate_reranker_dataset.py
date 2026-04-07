"""
Gera dataset para treino de re-ranker (Cross-Encoder)

Formato de saída (JSONL):
{
  "query": "...",
  "doc": "...",
  "label": 1 ou 0
}

Estratégia:
- Positivo: (query, doc original)
- Negativos fáceis: docs aleatórios
- 🔥 Hard negatives: docs retornados pelo Milvus (sem ser o correto)
"""

import json
import random
from tqdm import tqdm
from pymilvus import Collection, connections
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()

connections.connect(
    alias="default",
    host=os.getenv("MILVUS_HOST", "localhost"),
    port=os.getenv("MILVUS_PORT", "19530")
)

# =========================
# CONFIG
# =========================

COLLECTION_NAME = "rag_documents"
OUTPUT_FILE = "reranker_dataset.jsonl"

NUM_SAMPLES = 3000
NEGATIVES_RANDOM = 2
NEGATIVES_HARD = 2
TOP_K_SEARCH = 5

EMBED_MODEL = "BAAI/bge-base-en-v1.5"

# =========================
# INIT
# =========================

collection = Collection(COLLECTION_NAME)
embedder = SentenceTransformer(EMBED_MODEL)

# =========================
# HELPERS
# =========================

def get_documents(limit=5000):
    results = collection.query(
        expr="id >= 0",
        output_fields=["content"],
        limit=limit
    )
    return [r["content"] for r in results]


def generate_query(doc):
    """
    Gera uma pseudo-query a partir do documento
    """
    sentences = doc.split(".")
    if len(sentences) > 1:
        return sentences[0][:200]
    return doc[:200]


def get_hard_negatives(query, positive_doc):
    """
    Busca no Milvus docs similares (hard negatives)
    """
    vector = embedder.encode(query).tolist()

    results = collection.search(
        data=[vector],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=TOP_K_SEARCH,
        output_fields=["content"]
    )

    negatives = []
    for hit in results[0]:
        doc = hit.entity.get("content")
        if doc != positive_doc:
            negatives.append(doc)

    return negatives[:NEGATIVES_HARD]


# =========================
# MAIN
# =========================

def main():
    print("🔹 Carregando documentos do Milvus...")
    docs = get_documents(NUM_SAMPLES)

    print(f"🔹 Gerando dataset ({len(docs)} docs)...")

    with open(OUTPUT_FILE, "w") as f:

        for doc in tqdm(docs):

            query = generate_query(doc)

            # ----------------------
            # POSITIVE
            # ----------------------
            f.write(json.dumps({
                "query": query,
                "doc": doc,
                "label": 1
            }) + "\n")

            # ----------------------
            # RANDOM NEGATIVES
            # ----------------------
            random_negs = random.sample(docs, NEGATIVES_RANDOM)

            for neg in random_negs:
                if neg == doc:
                    continue

                f.write(json.dumps({
                    "query": query,
                    "doc": neg,
                    "label": 0
                }) + "\n")

            # ----------------------
            # 🔥 HARD NEGATIVES (Milvus)
            # ----------------------
            hard_negs = get_hard_negatives(query, doc)

            for neg in hard_negs:
                f.write(json.dumps({
                    "query": query,
                    "doc": neg,
                    "label": 0
                }) + "\n")

    print(f"\n✅ Dataset salvo em: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()