"""
Serviço RAG: orquestra embedding, retrieval e geração.
Suporta roteamento automático entre Ollama, HuggingFace e OpenAI.
"""
import time
import uuid

from loguru import logger

from api.schemas.query import QueryResponse, RetrievedDoc
from api.services.milvus_service import MilvusService
from api.services.openai_service import OpenAIService
from api.services.postgres_service import PostgresService
from api.services.rag_huggingface_service import HuggingFaceService
from api.services.reranker_service import RerankerService
from api.schemas.config import settings


def _detect_provider(model: str) -> str:
    if not model:
        return "huggingface"
    model_lower = model.lower()
    if model_lower.startswith("gpt-") or model_lower.startswith("gpt4"):
        return "openai"
    if "/" in model:
        return "huggingface"
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
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value if item is not None) or None
    return str(value)


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
        if llm_provider and llm_provider.lower() in ("ollama", "huggingface", "openai"):
            provider = llm_provider.lower()
        elif llm_model:
            provider = _detect_provider(llm_model)
        else:
            provider = settings.LLM_PROVIDER.strip().lower()

        if not llm_model:
            if provider == "openai":
                llm_model = settings.OPENAI_MODEL
            elif provider == "ollama":
                llm_model = settings.OLLAMA_LLM_MODEL
            else:
                llm_model = settings.HF_LLM_MODEL

        if provider == "openai":
            return await self._query_with_openai(query=query, top_k=top_k, llm_model=llm_model)
        elif provider == "ollama":
            return await self._query_with_ollama(query=query, top_k=top_k, llm_model=llm_model)
        else:
            return await self._query_with_huggingface(query=query, top_k=top_k, llm_model=llm_model)

    async def _query_with_openai(self, query: str, top_k: int, llm_model: str) -> QueryResponse:
        run_id = str(uuid.uuid4())
        start = time.time()

        logger.info("Gerando embedding da query com HuggingFace...")
        query_vector = self.hf.embed(query)

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
                authors=_normalize_text_metadata(h.get("metadata", {}).get("authors")),
                categories=_normalize_text_metadata(h.get("metadata", {}).get("categories")),
                primary_category=_normalize_text_metadata(h.get("metadata", {}).get("primary_category")),
                content=h["content"],
                published=_normalize_text_metadata(h.get("metadata", {}).get("published")),
                updated=_normalize_text_metadata(h.get("metadata", {}).get("updated")),
            )
            for h in hits
        ]

        context = "\n\n".join(
            f"[Documento {i+1}] (score: {doc.score:.2f})\n{doc.content}"
            for i, doc in enumerate(retrieved_docs)
        )
        prompt = self._build_prompt(query=query, context=context)

        logger.info(f"Gerando resposta com OpenAI:{llm_model}...")
        answer = self.openai.generate(prompt=prompt, system=SYSTEM_PROMPT, max_tokens=768)

        latency_ms = int((time.time() - start) * 1000)

        await self.db.save_rag_run(
            run_id=run_id,
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
        )

    async def _query_with_huggingface(self, query: str, top_k: int, llm_model: str) -> QueryResponse:
        run_id = str(uuid.uuid4())
        start = time.time()

        logger.info("Gerando embedding da query...")
        query_vector = self.hf.embed(query)

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
                authors=_normalize_text_metadata(h.get("metadata", {}).get("authors")),
                categories=_normalize_text_metadata(h.get("metadata", {}).get("categories")),
                primary_category=_normalize_text_metadata(h.get("metadata", {}).get("primary_category")),
                content=h["content"],
                published=_normalize_text_metadata(h.get("metadata", {}).get("published")),
                updated=_normalize_text_metadata(h.get("metadata", {}).get("updated")),
            )
            for h in hits
        ]

        logger.info("Aplicando re-ranking...")
        retrieved_docs = self.reranker.rerank(query=query, docs=retrieved_docs, top_k=3)

        context = "\n\n".join(
            f"[Documento {i+1}] (score: {doc.score:.2f})\n{doc.content}"
            for i, doc in enumerate(retrieved_docs)
        )
        prompt = self._build_prompt(query=query, context=context)

        logger.info(f"Gerando resposta com HuggingFace:{llm_model}...")
        answer = self.hf.generate(prompt=prompt, system=SYSTEM_PROMPT, max_tokens=768)

        latency_ms = int((time.time() - start) * 1000)

        await self.db.save_rag_run(
            run_id=run_id,
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
        )

    async def _query_with_ollama(self, query: str, top_k: int, llm_model: str) -> QueryResponse:
        from ollama import Client
        client = Client(host=settings.OLLAMA_HOST, timeout=120.0)

        run_id = str(uuid.uuid4())
        start = time.time()

        logger.info("Gerando embedding da query com Ollama...")
        try:
            embed_response = client.embeddings(model=settings.OLLAMA_EMBED_MODEL, prompt=query)
            query_vector = embed_response["embedding"]
        except Exception as e:
            logger.error(f"Erro ao gerar embedding: {str(e)}")
            raise

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
                authors=_normalize_text_metadata(h.get("metadata", {}).get("authors")),
                categories=_normalize_text_metadata(h.get("metadata", {}).get("categories")),
                primary_category=_normalize_text_metadata(h.get("metadata", {}).get("primary_category")),
                content=h["content"],
                published=_normalize_text_metadata(h.get("metadata", {}).get("published")),
                updated=_normalize_text_metadata(h.get("metadata", {}).get("updated")),
            )
            for h in hits
        ]

        context = "\n\n".join(
            f"[Documento {i+1}] (score: {doc.score:.2f})\n{doc.content}"
            for i, doc in enumerate(retrieved_docs)
        )
        prompt = self._build_prompt(query=query, context=context)

        logger.info(f"Gerando resposta com {llm_model} (Ollama)...")
        try:
            response = client.chat(
                model=llm_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            answer = response["message"]["content"]
        except Exception as e:
            logger.error(f"Erro ao gerar resposta: {str(e)}")
            raise

        prompt_tokens = response.get("prompt_eval_count", 0)
        response_tokens = response.get("eval_count", 0)
        latency_ms = int((time.time() - start) * 1000)

        await self.db.save_rag_run(
            run_id=run_id,
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
