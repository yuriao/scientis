"""LLM client abstraction with tiered model selection.

Text tiers:
  - cheap:  GPT-4o mini  — query expansion, evidence compilation, small extractions
  - local:  vLLM         — bulk claim extraction, embeddings (open-weight models)
  - heavy:  Gemini Flash  — long-context multimodal reasoning

Vision tiers (for figure/panel understanding):
  - vision_cheap:   qwen3-vl-8b  — figure detection, panel description (~$0.0004/paper)
  - vision_default: qwen3-vl-32b — fallback for complex figures (~$0.0002/paper)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ModelTier(str, Enum):  # noqa: UP042
    cheap = "cheap"  # GPT-4o mini
    local = "local"  # vLLM (open-weight)
    heavy = "heavy"  # Gemini Flash
    vision_cheap = "vision_cheap"  # qwen3-vl-8b  (figure detection, panel description)
    vision_default = "vision_default"  # qwen3-vl-32b (fallback)


@dataclass
class LLMResponse:
    content: str
    model: str
    tier: ModelTier
    usage: dict = field(default_factory=dict)


class LLMClient:
    """Unified async interface across OpenAI, vLLM, Gemini, and OpenRouter providers."""

    def __init__(
        self,
        openai_api_key: str = "",
        openai_base_url: str = "",
        gemini_api_key: str = "",
        vllm_base_url: str = "http://localhost:8000/v1",
        cheap_model: str = "gpt-4o-mini",
        local_model: str = "meta-llama/Llama-3.1-8B-Instruct",
        heavy_model: str = "gemini-2.0-flash",
        vision_cheap_model: str = "qwen/qwen3-vl-8b-instruct",
        vision_default_model: str = "qwen/qwen3-vl-32b-instruct",
        openrouter_api_key: str = "",
        openrouter_base_url: str = "https://openrouter.ai/api/v1",
    ):
        self._openai_key = openai_api_key
        self._openai_base = openai_base_url
        self._gemini_key = gemini_api_key
        self._vllm_base = vllm_base_url
        self._cheap_model = cheap_model
        self._local_model = local_model
        self._heavy_model = heavy_model
        self._vision_cheap_model = vision_cheap_model
        self._vision_default_model = vision_default_model
        self._openrouter_key = openrouter_api_key
        self._openrouter_base = openrouter_base_url
        self._openai_client = None
        self._vllm_client = None
        self._gemini_client = None
        self._openrouter_client = None

    # ── Lazy-initialised provider clients ─────────────────────────────────

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

    @property
    def openrouter(self):
        if self._openrouter_client is None:
            from openai import AsyncOpenAI

            self._openrouter_client = AsyncOpenAI(
                api_key=self._openrouter_key,
                base_url=self._openrouter_base,
            )
        return self._openrouter_client

    # ── Text generation ────────────────────────────────────────────────────

    async def generate(
        self,
        messages: list[dict],
        tier: ModelTier = ModelTier.cheap,
        response_format: dict | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> LLMResponse:
        """Generate a response using the appropriate tier."""
        if tier == ModelTier.heavy and self._gemini_key:
            return await self._generate_gemini(messages, max_tokens, temperature)
        if tier == ModelTier.local:
            return await self._generate_vllm(messages, response_format, max_tokens, temperature)
        return await self._generate_openai(
            messages, self._cheap_model, response_format, max_tokens, temperature
        )

    async def _generate_openai(
        self,
        messages: list[dict],
        model: str,
        response_format: dict | None,
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        kwargs: dict = dict(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
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
        self,
        messages: list[dict],
        response_format: dict | None,
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        kwargs: dict = dict(
            model=self._local_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if response_format:
            # vLLM uses guided_json for structured output
            kwargs["extra_body"] = {"guided_json": response_format.get("json_schema", {})}
        resp = await self.vllm.chat.completions.create(**kwargs)
        return LLMResponse(
            content=resp.choices[0].message.content or "",
            model=resp.model,
            tier=ModelTier.local,
            usage=resp.usage.model_dump() if resp.usage else {},
        )

    async def _generate_gemini(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        from google.genai import types as genai_types

        system_prompt = ""
        user_parts: list[dict] = []
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            elif msg["role"] == "user":
                user_parts.append({"text": msg["content"]})
            elif msg["role"] == "assistant":
                user_parts.append({"text": msg["content"]})

        # Use the async client (aio) to avoid blocking the event loop
        resp = await self.gemini.aio.models.generate_content(
            model=self._heavy_model,
            contents=user_parts or [{"text": ""}],
            config=genai_types.GenerateContentConfig(
                system_instruction=system_prompt or None,
                max_output_tokens=max_tokens,
                temperature=temperature,
            ),
        )
        return LLMResponse(
            content=resp.text or "",
            model=self._heavy_model,
            tier=ModelTier.heavy,
        )

    # ── Vision generation ──────────────────────────────────────────────────

    async def generate_vision(
        self,
        messages: list[dict],
        tier: ModelTier = ModelTier.vision_cheap,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> LLMResponse:
        """Generate a response from a vision-language model via OpenRouter.

        Tiered fallback:
          - vision_cheap:   qwen3-vl-8b (primary, fast)
          - vision_default: qwen3-vl-32b (fallback, more capable)

        Content in messages should be structured for multimodal:
          - {"role": "user", "content": [{"type": "text", "text": "..."},
                                         {"type": "image_url", "image_url": {"url": "data:..."}}]}
        """
        model = (
            self._vision_default_model
            if tier == ModelTier.vision_default
            else self._vision_cheap_model
        )
        try:
            resp = await self.openrouter.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                extra_headers={
                    "HTTP-Referer": "https://github.com/yuriao/scientis",
                    "X-Title": "Scientis",
                },
            )
            return LLMResponse(
                content=resp.choices[0].message.content or "",
                model=resp.model or model,
                tier=tier,
                usage=resp.usage.model_dump() if resp.usage else {},
            )
        except Exception:
            logger.exception("Vision generation failed with tier %s on %s", tier, model)
            # Fallback: try vision_default if we were on vision_cheap
            if tier == ModelTier.vision_cheap:
                logger.info("Retrying vision generation with vision_default tier")
                return await self.generate_vision(
                    messages=messages,
                    tier=ModelTier.vision_default,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            raise

    # ── Embeddings ─────────────────────────────────────────────────────────

    async def embed(self, texts: list[str], tier: ModelTier = ModelTier.local) -> list[list[float]]:
        """Generate embeddings. Uses vLLM for local tier, OpenAI otherwise."""
        if tier == ModelTier.local and self._vllm_base:
            # Use the vLLM embeddings endpoint with the local model
            resp = await self.vllm.embeddings.create(
                model=self._local_model,
                input=texts,
            )
        else:
            resp = await self.openai.embeddings.create(
                model="text-embedding-3-small",
                input=texts,
            )
        return [d.embedding for d in resp.data]


# ── Module-level singleton ─────────────────────────────────────────────────

_llm_pool: LLMClient | None = None


def get_llm() -> LLMClient:
    global _llm_pool
    if _llm_pool is None:
        from scientis.config import get_settings

        s = get_settings()
        _llm_pool = LLMClient(
            openai_api_key=s.openai_api_key,
            gemini_api_key=s.gemini_api_key,
            vllm_base_url=s.vllm_base_url,
            openrouter_api_key=s.openrouter_api_key,
        )
    return _llm_pool
