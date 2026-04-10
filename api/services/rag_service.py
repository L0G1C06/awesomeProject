"""
Serviço RAG: orquestra embedding, retrieval e geração.
Suporta roteamento automático entre Ollama, HuggingFace e OpenAI.
"""
import json
import tempfile
import time
import uuid

from loguru import logger

# ── Must set BEFORE importing mlflow ──────────────────────────
import os
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"]      = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"]  = "minioadmin"
os.environ["MLFLOW_TRACKING_URI"]    = "http://localhost:5000"

import mlflow

from api.schemas.query import QueryResponse, RetrievedDoc
from api.services.milvus_service import MilvusService
from api.services.openai_service import OpenAIService
from api.services.postgres_service import PostgresService
from api.services.rag_huggingface_service import HuggingFaceService
from api.services.reranker_service import RerankerService
from api.schemas.config import settings


def _detect_provider(model: str) -> str:
    """
    Detecta o provedor baseado no nome do modelo.
    - HuggingFace: 'org/model' (contém /)
    - OpenAI: Começa com 'gpt-' ou 'gpt4'
    - Ollama: Outros nomes simples (llama2, neural-chat, etc)
    """
    if not model:
        return "huggingface"
    
    model_lower = model.lower()
    
    # OpenAI
    if model_lower.startswith("gpt-") or model_lower.startswith("gpt4"):
        return "openai"
    
    # HuggingFace (format org/model)
    if "/" in model:
        return "huggingface"
    
    # Ollama (nomes locais simples)
    return "ollama"

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


def _normalize_text_metadata(value):
    """Converte listas/tuplas em texto estável para resposta e logs."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value if item is not None) or None
    return str(value)


def _log_docs_table(docs: list[RetrievedDoc], tag: str) -> None:
    """Loga documentos como tabela e JSON completo no MLflow."""

    # Tabela resumida (aparece em Artifacts como JSON navegável)
    rows = [
        {
            "rank": i + 1,
            "score": round(doc.score, 4),
            "title": doc.title or "",
            "arxiv_id": doc.arxiv_id or "",
            "authors": doc.authors or "",
            "primary_category": doc.primary_category or "",
            "published": str(doc.published or ""),
            "content_preview": (doc.content or "")[:200],
        }
        for i, doc in enumerate(docs)
    ]

    mlflow.log_table(
        data={
            "columns": list(rows[0].keys()),
            "data": [list(r.values()) for r in rows],
        },
        artifact_file=f"docs_{tag}.json",
    )

    # JSON completo com conteúdo inteiro
    full_rows = [
        {
            "rank": i + 1,
            "score": round(doc.score, 4),
            "title": doc.title,
            "arxiv_id": doc.arxiv_id,
            "authors": doc.authors,
            "categories": doc.categories,
            "primary_category": doc.primary_category,
            "published": str(doc.published),
            "updated": str(doc.updated),
            "url": doc.url,
            "content": doc.content,
        }
        for i, doc in enumerate(docs)
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(full_rows, f, ensure_ascii=False, indent=2)
        tmp_path = f.name

    mlflow.log_artifact(tmp_path, artifact_path=f"docs_{tag}_full")


class RAGService:
    def __init__(self):
        self.milvus = MilvusService()
        self.db = PostgresService()
        self.hf = HuggingFaceService()
        self.openai = OpenAIService()
        self.reranker = RerankerService()

    async def query(
        self,
        query: str,
        top_k: int = 5,
        llm_model: str = None,
        llm_provider: str = None,
    ) -> QueryResponse:
        """
        Realiza uma query RAG com suporte a Ollama, HuggingFace e OpenAI.
        
        Args:
            query: Texto da pergunta
            top_k: Número de documentos a recuperar
            llm_model: Nome do modelo. Se None, usa padrão do provider.
            llm_provider: 'ollama', 'huggingface' ou 'openai'. 
                         Se None, detecta pelo llm_model ou usa .env
        """
        # Determina o provider
        if llm_provider and llm_provider.lower() in ("ollama", "huggingface", "openai"):
            provider = llm_provider.lower()
        elif llm_model:
            provider = _detect_provider(llm_model)
        else:
            # Usa padrão do .env
            provider = settings.LLM_PROVIDER.strip().lower()
        
        # Determina o modelo
        if not llm_model:
            if provider == "openai":
                llm_model = settings.OPENAI_MODEL
            elif provider == "ollama":
                llm_model = settings.OLLAMA_LLM_MODEL
            else:
                llm_model = settings.HF_LLM_MODEL
        
        # Roteia para o provider correto
        if provider == "openai":
            return await self._query_with_openai(
                query=query,
                top_k=top_k,
                llm_model=llm_model,
            )
        elif provider == "ollama":
            return await self._query_with_ollama(
                query=query,
                top_k=top_k,
                llm_model=llm_model,
            )
        else:
            return await self._query_with_huggingface(
                query=query,
                top_k=top_k,
                llm_model=llm_model,
            )

    async def _query_with_openai(self, query: str, top_k: int, llm_model: str) -> QueryResponse:
        """Implementação RAG com OpenAI."""
        run_id = str(uuid.uuid4())
        start = time.time()

        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

        with mlflow.start_run(run_name=f"rag-query-{run_id[:8]}") as run:

            mlflow.log_params({
                "query": query[:200],
                "top_k": top_k,
                "llm_model": llm_model,
                "embed_model": settings.HF_EMBED_MODEL,
                "llm_provider": "openai",
            })

            # ── 1. EMBEDDING ───────────────────────────────
            logger.info("Gerando embedding da query com HuggingFace...")
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
                    authors=_normalize_text_metadata(
                        h.get("metadata", {}).get("authors")
                    ),
                    categories=_normalize_text_metadata(
                        h.get("metadata", {}).get("categories")
                    ),
                    primary_category=_normalize_text_metadata(
                        h.get("metadata", {}).get("primary_category")
                    ),
                    content=h["content"],
                    published=_normalize_text_metadata(
                        h.get("metadata", {}).get("published")
                    ),
                    updated=_normalize_text_metadata(
                        h.get("metadata", {}).get("updated")
                    ),
                )
                for h in hits
            ]

            mlflow.log_metric("docs_retrieved", len(retrieved_docs))
            _log_docs_table(retrieved_docs, tag="retrieved")

            # ── 3. PROMPT ──────────────────────────────────
            context = "\n\n".join(
                f"[Documento {i+1}] (score: {doc.score:.2f})\n{doc.content}"
                for i, doc in enumerate(retrieved_docs)
            )

            prompt = self._build_prompt(query=query, context=context)

            # ── 4. LLM ─────────────────────────────────────
            logger.info(f"Gerando resposta com OpenAI:{llm_model}...")
            answer = self.openai.generate(
                prompt=prompt,
                system=SYSTEM_PROMPT,
                max_tokens=768,
            )

            # ── 5. MÉTRICAS ────────────────────────────────
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

            # ── 6. DATABASE ────────────────────────────────
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
            llm_provider="openai",
            llm_model=llm_model,
            latency_ms=latency_ms,
            mlflow_run_id=run.info.run_id,
        )

    async def _query_with_huggingface(self, query: str, top_k: int, llm_model: str) -> QueryResponse:
        """Implementação RAG com HuggingFace."""
        run_id = str(uuid.uuid4())
        start = time.time()

        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

        with mlflow.start_run(run_name=f"rag-query-{run_id[:8]}") as run:

            mlflow.log_params({
                "query": query[:200],
                "top_k_before_rerank": top_k,
                "top_k_after_rerank": 3,
                "llm_provider": "huggingface",
                "llm_model": llm_model,
                "embed_model": settings.HF_EMBED_MODEL,
                "reranker_used": True,
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
                    authors=_normalize_text_metadata(
                        h.get("metadata", {}).get("authors")
                    ),
                    categories=_normalize_text_metadata(
                        h.get("metadata", {}).get("categories")
                    ),
                    primary_category=_normalize_text_metadata(
                        h.get("metadata", {}).get("primary_category")
                    ),
                    content=h["content"],
                    published=_normalize_text_metadata(
                        h.get("metadata", {}).get("published")
                    ),
                    updated=_normalize_text_metadata(
                        h.get("metadata", {}).get("updated")
                    ),
                )
                for h in hits
            ]

            # Loga documentos ANTES do rerank
            mlflow.log_metric("docs_before_rerank", len(retrieved_docs))
            _log_docs_table(retrieved_docs, tag="before_rerank")

            # ── 3. RE-RANKER 🔥 ─────────────────────────────
            logger.info("Aplicando re-ranking...")
            retrieved_docs = self.reranker.rerank(
                query=query,
                docs=retrieved_docs,
                top_k=3,
            )

            # Loga documentos DEPOIS do rerank
            mlflow.log_metric("docs_after_rerank", len(retrieved_docs))
            _log_docs_table(retrieved_docs, tag="after_rerank")

            # ── 4. PROMPT ──────────────────────────────────
            context = "\n\n".join(
                f"[Documento {i+1}] (score: {doc.score:.2f})\n{doc.content}"
                for i, doc in enumerate(retrieved_docs)
            )

            prompt = self._build_prompt(query=query, context=context)

            # ── 5. LLM ─────────────────────────────────────
            logger.info(f"Gerando resposta com HuggingFace:{llm_model}...")
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
            llm_provider="huggingface",
            llm_model=llm_model,
            latency_ms=latency_ms,
            mlflow_run_id=run.info.run_id,
        )

    async def _query_with_ollama(self, query: str, top_k: int, llm_model: str) -> QueryResponse:
        """Implementação RAG com Ollama (local)."""
        from ollama import Client
        
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
                "llm_provider": "ollama",
            })

            # ── 1. EMBEDDING ───────────────────────────────
            logger.info("Gerando embedding da query com Ollama...")
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
                    authors=_normalize_text_metadata(
                        h.get("metadata", {}).get("authors")
                    ),
                    categories=_normalize_text_metadata(
                        h.get("metadata", {}).get("categories")
                    ),
                    primary_category=_normalize_text_metadata(
                        h.get("metadata", {}).get("primary_category")
                    ),
                    content=h["content"],
                    published=_normalize_text_metadata(
                        h.get("metadata", {}).get("published")
                    ),
                    updated=_normalize_text_metadata(
                        h.get("metadata", {}).get("updated")
                    ),
                )
                for h in hits
            ]

            mlflow.log_metric("docs_retrieved", len(retrieved_docs))

            # ── 3. PROMPT ──────────────────────────────────
            context = "\n\n".join(
                f"[Documento {i+1}] (score: {doc.score:.2f})\n{doc.content}"
                for i, doc in enumerate(retrieved_docs)
            )

            prompt = self._build_prompt(query=query, context=context)

            # ── 4. LLM ─────────────────────────────────────
            logger.info(f"Gerando resposta com {llm_model} (Ollama)...")
            logger.info(f"Conectando ao Ollama em {settings.OLLAMA_HOST} com modelo {llm_model}")
            try:
                response = client.chat(
                    model=llm_model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                )
                logger.info("Resposta gerada com sucesso")
                answer = response["message"]["content"]
            except Exception as e:
                logger.error(f"Erro ao gerar resposta: {str(e)}")
                raise

            prompt_tokens = response.get("prompt_eval_count", 0)
            response_tokens = response.get("eval_count", 0)

            # ── 5. MÉTRICAS ────────────────────────────────
            latency_ms = int((time.time() - start) * 1000)

            mlflow.log_metrics({
                "latency_ms": latency_ms,
                "prompt_tokens": prompt_tokens,
                "response_tokens": response_tokens,
                "total_tokens": prompt_tokens + response_tokens,
            })

            mlflow.log_param("prompt_preview", prompt[:300])
            mlflow.log_param("response_preview", answer[:300])

            # ── 6. DATABASE ────────────────────────────────
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
            llm_provider="ollama",
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
