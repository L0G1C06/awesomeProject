"""
Serviço HuggingFace: geração e embedding via Inference API (SDK oficial).
"""
from loguru import logger
from huggingface_hub import InferenceClient
from api.schemas.config import settings


class HuggingFaceService:
    def __init__(self):
        if not settings.HUGGINGFACE_API_TOKEN:
            raise ValueError(
                "HUGGINGFACE_API_TOKEN não configurado. "
                "Acesse https://huggingface.co/settings/tokens"
            )
        # hf-inference: suporta feature_extraction (embeddings)
        self.embed_client = InferenceClient(
            provider="hf-inference",
            api_key=settings.HUGGINGFACE_API_TOKEN,
        )
        # together: suporta chat_completion com LLMs modernos
        self.llm_client = InferenceClient(
        provider="auto",  # HF escolhe o melhor provider disponível
        api_key=settings.HUGGINGFACE_API_TOKEN,
        )

    def embed(self, text: str) -> list[float]:
        logger.info(f"Embedding com modelo: {settings.HF_EMBED_MODEL}")
        result = self.embed_client.feature_extraction(
            text,
            model=settings.HF_EMBED_MODEL,
        )
        if hasattr(result, "tolist"):
            result = result.tolist()
        if isinstance(result[0], list):
            dim = len(result[0])
            return [sum(v[i] for v in result) / len(result) for i in range(dim)]
        return result

    def generate(self, prompt: str, system: str = None, max_tokens: int = 768) -> str:
        logger.info(f"Geração com modelo: {settings.HF_LLM_MODEL}")
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        result = self.llm_client.chat_completion(
            messages=messages,
            model=settings.HF_LLM_MODEL,
            max_tokens=max_tokens,
            temperature=0.1,
        )
        return result.choices[0].message.content