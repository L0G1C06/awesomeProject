"""
Serviço RAG: orquestra embedding, retrieval e geração.
"""
import time
import uuid
from loguru import logger

# ── Must set BEFORE importing mlflow ──────────────────────────
import os
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"

import mlflow  # ← now mlflow sees the env vars on first import

from api.schemas.query import QueryResponse, RetrievedDoc
from api.services.milvus_service import MilvusService
from api.services.postgres_service import PostgresService
from api.services.rag_huggingface_service import HuggingFaceService
from api.schemas.config import settings

SYSTEM_PROMPT = (
    "You are a scientific article assistant. "
    "Answer the user's question in a single cohesive paragraph or short paragraphs. "
    "Do NOT invent sub-questions. Do NOT create Q&A format. "
    "Answer ONLY using the provided documents. "
    "Do NOT use external knowledge. "
    "Cite [Document N] inline for each claim. "
    "If the documents lack sufficient information, reply only with: "
    "'The retrieved documents do not contain enough information to answer this question.'"
)


class RAGService:
    def __init__(self):
        self.milvus = MilvusService()
        self.db = PostgresService()
        self.hf = HuggingFaceService()

    async def query(self, query: str, top_k: int = 5, llm_model: str = None) -> QueryResponse:
        llm_model = settings.HF_LLM_MODEL
        run_id = str(uuid.uuid4())
        start = time.time()

        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

        with mlflow.start_run(run_name=f"rag-query-{run_id[:8]}") as run:
            mlflow.log_params({
                "query": query[:200],
                "top_k": top_k,
                "llm_model": llm_model,
                "embed_model": settings.HF_EMBED_MODEL,  # ← atualizado
            })

            # ── 1. Embed da query ──────────────────────────────
            logger.info("Gerando embedding da query...")
            query_vector = self.hf.embed(query)  # ← era ollama.embeddings

            # ── 2. Retrieval no Milvus ─────────────────────────
            logger.info(f"Buscando top-{top_k} documentos...")
            hits = self.milvus.search(
                vector=query_vector,
                top_k=top_k,
                collection_name=settings.MILVUS_COLLECTION,
            )
            retrieved_docs = [
                RetrievedDoc(
                    id=str(h["id"]),
                    content=h["content"],
                    score=float(h["score"]),
                    metadata=h.get("metadata", {}),
                )
                for h in hits
            ]

            # ── 3. Construção do prompt ────────────────────────
            context = "\n\n".join(
                f"[Documento {i+1}] (score: {doc.score:.2f})\n{doc.content}"
                for i, doc in enumerate(retrieved_docs)
            )
            prompt = self._build_prompt(query=query, context=context)

            # ── 4. Geração via HuggingFace ─────────────────────
            logger.info(f"Gerando resposta com {llm_model}...")
            answer = self.hf.generate(
                prompt=self._build_prompt(query=query, context=context),
                system=SYSTEM_PROMPT,
                max_tokens=768,
            )

            # HuggingFace Inference API não retorna contagem de tokens,
            # então estimamos pelo tamanho do texto
            prompt_tokens   = len(prompt.split())
            response_tokens = len(answer.split())
            total_tokens    = prompt_tokens + response_tokens

            latency_ms = int((time.time() - start) * 1000)

            # ── 5. Log no MLflow ───────────────────────────────
            mlflow.log_metrics({
                "latency_ms":      latency_ms,
                "docs_retrieved":  len(retrieved_docs),
                "prompt_tokens":   prompt_tokens,
                "response_tokens": response_tokens,
                "total_tokens":    total_tokens,
            })
            mlflow.log_param("prompt_preview", prompt[:490])
            mlflow.log_param("response_preview", answer[:490])

            # ── 6. Persiste no PostgreSQL ──────────────────────
            await self.db.save_rag_run(
                run_id=run_id,
                mlflow_run_id=run.info.run_id,
                query=query,
                retrieved_docs=[d.model_dump() for d in retrieved_docs],
                prompt_used=prompt,
                response=answer,
                llm_model=llm_model,
                latency_ms=latency_ms,
                top_k=top_k,
                embed_model=settings.HF_EMBED_MODEL,  # ← atualizado
            )

        return QueryResponse(
            run_id=run_id,
            query=query,
            answer=answer,
            retrieved_docs=retrieved_docs,
            llm_model=llm_model,
            latency_ms=latency_ms,
            mlflow_run_id=run.info.run_id,
        )

    def _build_prompt(self, query: str, context: str) -> str:
        return f"""Retrieved documents:

    {context}

    The user submitted the following query: "{query}"

    If this is a broad topic rather than a specific question, summarize what the retrieved documents say about it in a concise, integrated paragraph. If it is a specific question, answer it directly.

    Cite [Document N] for each claim. Base your answer exclusively on the documents above."""

    async def register_feedback(self, run_id: str, feedback: int):
        await self.db.update_feedback(run_id=run_id, feedback=feedback)