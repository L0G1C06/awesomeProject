"""
Serviço RAG: orquestra embedding, retrieval e geração.
"""
import time
import uuid
from coverage import context
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
from api.services.reranker_service import RerankerService
from api.schemas.config import settings

SYSTEM_PROMPT = """You are a scientific article assistant. Your ONLY knowledge source is the documents provided in each query. You have no external knowledge and must never use it.

STRICT RULES:
- Answer exclusively from the provided documents. If information is not there, say so.
- Never fabricate, infer beyond the text, or fill gaps with general knowledge.
- Cite [Document N] inline for every factual claim.
- Write in cohesive prose (one or a few short paragraphs). No Q&A format, no invented sub-questions, no bullet lists.
- For broad topics: synthesize what the documents collectively say into an integrated summary.
- For specific questions: answer directly and precisely, citing sources per claim.
- For complex or multi-part queries: identify the central theme or intent, then address it holistically — do not decompose into sub-questions or answer only isolated keywords.

If the documents lack sufficient information to address the query, respond only with:
"The retrieved documents do not contain enough information to answer this question."
"""


class RAGService:
    def __init__(self):
        self.milvus = MilvusService()
        self.db = PostgresService()
        self.hf = HuggingFaceService()
        self.reranker = RerankerService()

    async def query(self, query: str, top_k: int = 8) -> QueryResponse:
        llm_model = settings.HF_LLM_MODEL
        run_id = str(uuid.uuid4())
        start = time.time()

        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

        with mlflow.start_run(run_name=f"rag-query-{run_id[:8]}") as run:

            mlflow.log_params({
                "query": query[:200],
                "top_k_before_rerank": top_k,
                "top_k_after_rerank": 3,
                "llm_model": llm_model,
                "embed_model": settings.HF_EMBED_MODEL,
                "reranker_used": True
            })

            # ── 1. EMBEDDING ───────────────────────────────
            logger.info("Gerando embedding da query...")
            query_vector = self.hf.embed(query)

            # ── 2. RETRIEVAL ───────────────────────────────
            logger.info(f"Buscando top-{top_k} documentos...")
            hits = self.milvus.search(
                vector=query_vector,
                top_k=top_k,
                collection_name=settings.MILVUS_COLLECTION,
            )

            retrieved_docs = [
                RetrievedDoc(
                    score=float(h["score"]),
                    title=h.get("metadata", {}).get("title"),
                    url=h.get("metadata", {}).get("id"),
                    arxiv_id=h.get("metadata", {}).get("arxiv_id"),
                    authors=h.get("metadata", {}).get("authors"),
                    categories=h.get("metadata", {}).get("categories"),
                    primary_category=h.get("metadata", {}).get("primary_category"),
                    content=h["content"],
                    published=h.get("metadata", {}).get("published"),
                    updated=h.get("metadata", {}).get("updated")
                )
                for h in hits
            ]

            # ── 3. RE-RANKER 🔥 ─────────────────────────────
            logger.info("Aplicando re-ranking...")
            retrieved_docs = self.reranker.rerank(
                query=query,
                docs=retrieved_docs,
                top_k=3
            )

            mlflow.log_metric("docs_after_rerank", len(retrieved_docs))

            # ── 4. PROMPT ──────────────────────────────────
            context = "\n\n".join(
                f"[Documento {i+1}] (score: {doc.score:.2f})\n{doc.content}"
                for i, doc in enumerate(retrieved_docs)
            )

            prompt = self._build_prompt(query=query, context=context)

            # ── 5. LLM ─────────────────────────────────────
            logger.info(f"Gerando resposta com {llm_model}...")
            answer = self.hf.generate(
                prompt=prompt,
                system=SYSTEM_PROMPT,
                max_tokens=768,
            )

            # ── 6. MÉTRICAS ────────────────────────────────
            latency_ms = int((time.time() - start) * 1000)

            prompt_tokens = len(prompt.split())
            response_tokens = len(answer.split())

            mlflow.log_metrics({
                "latency_ms": latency_ms,
                "prompt_tokens": prompt_tokens,
                "response_tokens": response_tokens,
                "total_tokens": prompt_tokens + response_tokens,
            })

            mlflow.log_param("prompt_preview", prompt[:300])
            mlflow.log_param("response_preview", answer[:300])

            # ── 7. DATABASE ────────────────────────────────
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
                embed_model=settings.HF_EMBED_MODEL,
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

            ---

            The user submitted the following query:
            \"{query}\"

            Instructions:
            1. Identify the core intent of the query. If it is long or multi-part, determine the central theme rather than focusing on isolated keywords.
            2. Synthesize what the documents above say that is relevant to that core intent.
            3. If the query is broad, write a concise integrated paragraph summarizing the documents' collective stance.
            4. If the query is specific, answer it directly and precisely.
            5. Cite [Document N] for every factual claim.
            6. Base your answer exclusively on the documents above. Do not use external knowledge under any circumstances.
            7. If the documents do not contain enough information to answer the question, respond only with:
            "The retrieved documents do not contain enough information to answer this question."
            """

    async def register_feedback(self, run_id: str, feedback: int):
        await self.db.update_feedback(run_id=run_id, feedback=feedback)