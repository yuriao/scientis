"""LLM client abstraction with tiered model selection.

Tiers:
  - cheap:   GPT-4o mini → figure interpretation, chart transcription, small extractions
  - local:   vLLM (open-weight models) → bulk claim extraction, embeddings
  - heavy:   Gemini 2.0 Flash / 2.5 Flash → long-context multimodal reasoning
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class ModelTier(str, Enum):
    cheap = "cheap"      # GPT-4o mini
    local = "local"      # vLLM (open-weight)
    heavy = "heavy"      # Gemini Flash


@dataclass
class LLMResponse:
    content: str
    model: str
    tier: ModelTier
    usage: dict = field(default_factory=dict)


class LLMClient:
    """Unified interface across OpenAI, vLLM, and Gemini providers."""

    def __init__(
        self,
        openai_api_key: str = "",
        openai_base_url: str = "",
        gemini_api_key: str = "",
        vllm_base_url: str = "http://localhost:8000/v1",
        cheap_model: str = "gpt-4o-mini",
        local_model: str = "meta-llama/Llama-3.1-8B-Instruct",
        heavy_model: str = "gemini-2.0-flash",
    ):
        self._openai_key = openai_api_key
        self._openai_base = openai_base_url
        self._gemini_key = gemini_api_key
        self._vllm_base = vllm_base_url

        self._cheap_model = cheap_model
        self._local_model = local_model
        self._heavy_model = heavy_model

        self._openai_client = None
        self._vllm_client = None
        self._gemini_client = None

    # ── Lazy init ──────────────────────────────────────

    @property
    def openai(self):
        if self._openai_client is None:
            from openai import AsyncOpenAI
            self._openai_client = AsyncOpenAI(
                api_key=self._openai_key,
                base_url=self._openai_base or None,
            )
        return self._openai_client

    @property
    def vllm(self):
        if self._vllm_client is None:
            from openai import AsyncOpenAI
            self._vllm_client = AsyncOpenAI(
                api_key="not-needed",
                base_url=self._vllm_base,
            )
        return self._vllm_client

    @property
    def gemini(self):
        if self._gemini_client is None:
            from google import genai
            self._gemini_client = genai.Client(api_key=self._gemini_key)
        return self._gemini_client

    # ── Generation ─────────────────────────────────────

    async def generate(
        self,
        messages: list[dict],
        tier: ModelTier = ModelTier.cheap,
        response_format: Optional[dict] = None,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> LLMResponse:
        """Generate a response from the appropriate tier."""
        if tier == ModelTier.heavy and self._gemini_key:
            return await self._generate_gemini(messages, max_tokens, temperature)
        elif tier == ModelTier.local:
            return await self._generate_vllm(messages, response_format, max_tokens, temperature)
        else:
            return await self._generate_openai(
                messages, self._cheap_model, response_format, max_tokens, temperature
            )

    async def _generate_openai(
        self, messages, model, response_format, max_tokens, temperature
    ) -> LLMResponse:
        kwargs = dict(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature)
        if response_format:
            kwargs["response_format"] = response_format

        resp = await self.openai.chat.completions.create(**kwargs)
        return LLMResponse(
            content=resp.choices[0].message.content or "",
            model=resp.model,
            tier=ModelTier.cheap,
            usage=resp.usage.model_dump() if resp.usage else {},
        )

    async def _generate_vllm(
        self, messages, response_format, max_tokens, temperature
    ) -> LLMResponse:
        kwargs = dict(model=self._local_model, messages=messages, max_tokens=max_tokens, temperature=temperature)
        if response_format:
            kwargs["extra_body"] = {"guided_json": response_format.get("json_schema", {})}

        resp = await self.vllm.chat.completions.create(**kwargs)
        return LLMResponse(
            content=resp.choices[0].message.content or "",
            model=resp.model,
            tier=ModelTier.local,
            usage=resp.usage.model_dump() if resp.usage else {},
        )

    async def _generate_gemini(
        self, messages, max_tokens, temperature
    ) -> LLMResponse:
        # Convert OpenAI-format messages to Gemini format
        system_prompt = ""
        contents = []
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            elif msg["role"] == "user":
                contents.append({"role": "user", "parts": [{"text": msg["content"]}]})
            elif msg["role"] == "assistant":
                contents.append({"role": "model", "parts": [{"text": msg["content"]}]})

        resp = self.gemini.models.generate_content(
            model=f"models/{self._heavy_model}",
            contents=contents[-1]["parts"] if contents else [{"text": ""}],
            config={
                "system_instruction": system_prompt,
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        return LLMResponse(
            content=resp.text or "",
            model=self._heavy_model,
            tier=ModelTier.heavy,
        )

    # ── Embeddings ─────────────────────────────────────

    async def embed(self, texts: list[str], tier: ModelTier = ModelTier.local) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        if tier == ModelTier.local:
            resp = await self.openai.embeddings.create(
                model="text-embedding-3-small",
                input=texts,
            )
            return [d.embedding for d in resp.data]
        else:
            resp = await self.openai.embeddings.create(
                model="text-embedding-3-small",
                input=texts,
            )
            return [d.embedding for d in resp.data]


# ── Global pool ────────────────────────────────────

_llm_pool: Optional[LLMClient] = None


def get_llm() -> LLMClient:
    global _llm_pool
    if _llm_pool is None:
        from scientis.config import get_settings
        s = get_settings()
        _llm_pool = LLMClient(
            openai_api_key=s.openai_api_key,
            gemini_api_key=s.gemini_api_key,
            vllm_base_url=s.vllm_base_url,
        )
    return _llm_pool
