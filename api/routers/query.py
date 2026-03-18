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
    e gera resposta via LLM (Ollama).
    """
    try:
        logger.info(f"Query recebida: {request.query[:80]}...")
        result = await service.query(
            query=request.query,
            top_k=request.top_k,
        )
        return result
    except Exception as e:
        logger.error(f"Erro na query RAG: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback", summary="Registra feedback do usuário")
async def submit_feedback(
    run_id: str,
    feedback: int,  # -1, 0, 1
    service: RAGService = Depends(get_rag_service),
):
    """Registra avaliação humana do resultado RAG."""
    await service.register_feedback(run_id=run_id, feedback=feedback)
    return {"status": "ok", "run_id": run_id}