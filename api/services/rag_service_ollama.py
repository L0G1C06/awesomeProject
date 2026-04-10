"""
Serviço RAG: orquestra embedding, retrieval e geração.
"""
import time
import uuid
from loguru import logger
import mlflow
from ollama import Client

from api.schemas.query import QueryResponse, RetrievedDoc
from api.services.milvus_service import MilvusService
from api.services.postgres_service import PostgresService
from api.schemas.config import settings


class RAGService:
    def __init__(self):
        self.milvus = MilvusService()
        self.db = PostgresService()

    async def query(self, query: str, top_k: int = 5, llm_model: str = None) -> QueryResponse:
        llm_model = llm_model or settings.OLLAMA_LLM_MODEL
        # Cria cliente com timeout maior (padrão é 30s)
        client = Client(host=settings.OLLAMA_HOST, timeout=120.0)
        run_id = str(uuid.uuid4())
        start = time.time()

        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

        with mlflow.start_run(run_name=f"rag-query-{run_id[:8]}") as run:
            mlflow.log_params({
                "query": query[:200],
                "top_k": top_k,
                "llm_model": llm_model,
                "embed_model": settings.OLLAMA_EMBED_MODEL,
            })

            # ── 1. Embed da query ──────────────────────────────
            logger.info("Gerando embedding da query...")
            logger.info(f"Conectando ao Ollama em {settings.OLLAMA_HOST} com modelo {settings.OLLAMA_EMBED_MODEL}")
            try:
                embed_response = client.embeddings(
                    model=settings.OLLAMA_EMBED_MODEL,
                    prompt=query,
                )
                logger.info("Embedding gerado com sucesso")
                query_vector = embed_response["embedding"]
            except Exception as e:
                logger.error(f"Erro ao gerar embedding: {str(e)}")
                raise

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
                f"[Documento {i+1}]\n{doc.content}"
                for i, doc in enumerate(retrieved_docs)
            )
            prompt = self._build_prompt(query=query, context=context)

            # ── 4. Geração via Ollama ──────────────────────────
            logger.info(f"Gerando resposta com {llm_model}...")
            logger.info(f"Conectando ao Ollama em {settings.OLLAMA_HOST} com modelo {llm_model}")
            try:
                response = client.chat(
                    model=llm_model,
                    messages=[{"role": "user", "content": prompt}],
                )
                logger.info("Resposta gerada com sucesso")
                answer = response["message"]["content"]
            except Exception as e:
                logger.error(f"Erro ao gerar resposta: {str(e)}")
                raise

            # Captura tokens da resposta do Ollama
            prompt_tokens    = response.get("prompt_eval_count", 0)
            response_tokens  = response.get("eval_count", 0)
            total_tokens     = prompt_tokens + response_tokens

            latency_ms = int((time.time() - start) * 1000)

            # ── 5. Log no MLflow ───────────────────────────────
            mlflow.log_metrics({
                "latency_ms":      latency_ms,
                "docs_retrieved":  len(retrieved_docs),
                "prompt_tokens":   prompt_tokens,
                "response_tokens": response_tokens,
                "total_tokens":    total_tokens,
            })
            mlflow.log_text(prompt, "prompt.txt")
            mlflow.log_text(answer, "response.txt")

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
                embed_model=settings.OLLAMA_EMBED_MODEL,
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
        """
        TODO: Adapte o system prompt ao domínio do seu dataset.
        """
        return f"""Você é um assistente especialista. Use os documentos abaixo para responder à pergunta.
Responda de forma clara e objetiva. Se não souber, diga que não encontrou informações suficientes.

=== DOCUMENTOS RECUPERADOS ===
{context}

=== PERGUNTA ===
{query}

=== RESPOSTA ==="""

    async def register_feedback(self, run_id: str, feedback: int):
        await self.db.update_feedback(run_id=run_id, feedback=feedback)
