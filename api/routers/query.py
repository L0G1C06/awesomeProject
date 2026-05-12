"""
Router: RAG Query
Endpoint principal de pergunta e resposta com retrieval.
"""
from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from api.schemas.query import QueryRequest, QueryResponse
from api.services.rag_service import RAGService

router = APIRouter()


def get_rag_service() -> RAGService:
    return RAGService()


@router.post("/", response_model=QueryResponse, summary="Realiza query RAG")
async def rag_query(
    request: QueryRequest,
    service: RAGService = Depends(get_rag_service),
):
    """
    Recebe uma pergunta, recupera documentos relevantes do Milvus
    e gera resposta via LLM (Ollama, HuggingFace ou OpenAI).
    """
    try:
        logger.info(f"Query recebida: {request.query[:80]}...")
        result = await service.query(
            query=request.query,
            top_k=request.top_k,
            llm_model=request.llm_model,
            llm_provider=request.llm_provider,
            filter_category=request.filter_category,
            filter_author=request.filter_author,
            filter_date_from=request.filter_date_from,
            filter_date_to=request.filter_date_to,
        )
        return result
    except Exception as e:
        logger.error(f"Erro na query RAG: {e}")
        message = str(e)

        if "insufficient_quota" in message or "exceeded your current quota" in message.lower():
            raise HTTPException(
                status_code=429,
                detail=(
                    "A chave da OpenAI foi aceita, mas a conta está sem cota/crédito de API "
                    "ou atingiu o limite mensal de gasto. Verifique o billing em "
                    "https://platform.openai.com/account/billing/overview."
                ),
            )

        raise HTTPException(status_code=500, detail=str(e))
