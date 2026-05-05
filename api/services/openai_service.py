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

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_completion_tokens=max_tokens,
            temperature=0.1,
            timeout=settings.OPENAI_TIMEOUT_SECONDS,
        )

        return response.choices[0].message.content.strip()
