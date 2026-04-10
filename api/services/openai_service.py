"""
Serviço OpenAI: geração de texto via SDK oficial e Responses API.
"""
from loguru import logger

from api.schemas.config import settings


class OpenAIService:
    def __init__(self):
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Dependência 'openai' não instalada. "
                "Instale os requisitos do projeto antes de usar LLM_PROVIDER=openai."
            ) from exc

        if not settings.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY não configurada. "
                "Defina a chave no ambiente para usar o provider OpenAI."
            )

        self.model_name = settings.OPENAI_MODEL
        self.provider_name = "openai"
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def generate(self, prompt: str, system: str | None = None, max_tokens: int = 768) -> str:
        logger.info(f"Geração com modelo OpenAI: {self.model_name}")

        request_args = {
            "model": self.model_name,
            "input": prompt,
            "max_output_tokens": max_tokens,
        }

        if system:
            request_args["instructions"] = system

        if settings.OPENAI_REASONING_EFFORT:
            request_args["reasoning"] = {"effort": settings.OPENAI_REASONING_EFFORT}

        response = self.client.with_options(
            timeout=settings.OPENAI_TIMEOUT_SECONDS,
        ).responses.create(**request_args)

        return (response.output_text or "").strip()
