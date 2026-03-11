"""Persistência de runs RAG no PostgreSQL.

Implementação mínima para manter a API operacional.
"""
from loguru import logger


class PostgresService:
    async def save_rag_run(
        self,
        run_id: str,
        mlflow_run_id: str,
        query: str,
        retrieved_docs: list[dict],
        prompt_used: str,
        response: str,
        llm_model: str,
        latency_ms: int,
    ) -> None:
        logger.info(
            "Persistência PostgreSQL pendente de implementação. run_id={}, mlflow_run_id={}",
            run_id,
            mlflow_run_id,
        )

    async def update_feedback(self, run_id: str, feedback: int) -> None:
        logger.info(
            "Feedback pendente de persistência. run_id={}, feedback={}",
            run_id,
            feedback,
        )
