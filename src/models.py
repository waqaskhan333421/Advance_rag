"""Shared model clients (Gemini, NVIDIA NIM, CrossEncoder)."""

import logging
import time
from functools import lru_cache
from typing import List, Optional

from google import genai
from google.genai import types
from groq import Groq
from openai import OpenAI
from sentence_transformers import CrossEncoder

from src.config import CONFIG

logger = logging.getLogger(__name__)


class GeminiClient:
    """Unified Gemini client for embeddings and generation."""

    def __init__(self):
        self.api_key = CONFIG.get_gemini_api_key()
        self.client = genai.Client(api_key=self.api_key) if self.api_key else None
        self.embedding_model = CONFIG.models.gemini.embedding_model or CONFIG.models.embedding.model_name
        self.llm_model = CONFIG.models.gemini.llm_model or CONFIG.models.llm.model_name
        self.output_dim = CONFIG.models.embedding.output_dimensionality

    def embed(self, texts: list[str], model: Optional[str] = None) -> list[list[float]]:
        """Batch embed texts with Gemini Embedding 2 Preview with rate-limit handling."""
        if not self.client:
            raise ValueError("Gemini API Key is not set in environment or .env file.")
        if not texts:
            return []
        
        all_embeddings = []
        batch_size = 64
        emb_model = model or self.embedding_model
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            for attempt in range(5):
                try:
                    result = self.client.models.embed_content(
                        model=emb_model,
                        contents=batch,
                        config=types.EmbedContentConfig(
                            output_dimensionality=self.output_dim
                        ),
                    )
                    all_embeddings.extend([e.values for e in result.embeddings])
                    time.sleep(0.3)
                    break
                except Exception as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        wait_time = (2 ** attempt) * 2
                        logger.warning(f"Gemini embedding rate limit hit at batch {i}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Gemini embedding batch failed at {i}: {e}")
                        raise
        return all_embeddings

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> str:
        """Generate text with Gemini with rate-limit retry handling."""
        if not self.client:
            raise ValueError("Gemini API Key is not set in environment or .env file.")
        
        temp = temperature if temperature is not None else CONFIG.models.llm.temperature
        max_out = max_tokens if max_tokens is not None else CONFIG.models.llm.max_output_tokens
        llm_model = model or self.llm_model

        contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
        config = types.GenerateContentConfig(
            temperature=temp,
            max_output_tokens=max_out,
            system_instruction=system_instruction,
        )

        for attempt in range(5):
            try:
                response = self.client.models.generate_content(
                    model=llm_model,
                    contents=contents,
                    config=config,
                )
                return response.text or ""
            except Exception as e:
                if ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)) and attempt < 4:
                    wait_time = (2 ** attempt) * 6
                    logger.warning(f"Gemini LLM rate limit hit. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Gemini generation failed: {e}")
                    raise


class NvidiaClient:
    """NVIDIA NIM client (OpenAI-compatible) for embeddings and generation."""

    def __init__(self):
        self.api_key = CONFIG.get_nvidia_api_key()
        self.base_url = CONFIG.get_nvidia_base_url()
        self.llm_model = CONFIG.models.nvidia.llm_model
        self.embedding_model = CONFIG.models.nvidia.embedding_model
        self.client = (
            OpenAI(base_url=self.base_url, api_key=self.api_key)
            if self.api_key
            else None
        )

    def embed(self, texts: list[str], model: Optional[str] = None) -> list[list[float]]:
        """Batch embed texts using NVIDIA Embedding model via NIM."""
        if not self.client:
            raise ValueError("NVIDIA API Key is not set in environment or .env file.")
        if not texts:
            return []

        all_embeddings = []
        batch_size = 32
        emb_model = model or self.embedding_model

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            for attempt in range(5):
                try:
                    response = self.client.embeddings.create(
                        input=batch,
                        model=emb_model,
                    )
                    all_embeddings.extend([d.embedding for d in response.data])
                    break
                except Exception as e:
                    if "429" in str(e) or "rate" in str(e).lower():
                        wait_time = (2 ** attempt) * 2
                        logger.warning(f"NVIDIA embedding rate limit hit. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"NVIDIA embedding batch failed at {i}: {e}")
                        raise
        return all_embeddings

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> str:
        """Generate text using NVIDIA NIM LLM models."""
        if not self.client:
            raise ValueError("NVIDIA API Key is not set in environment or .env file.")

        temp = temperature if temperature is not None else CONFIG.models.llm.temperature
        max_out = max_tokens if max_tokens is not None else CONFIG.models.llm.max_output_tokens
        llm_model = model or self.llm_model

        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(5):
            try:
                response = self.client.chat.completions.create(
                    model=llm_model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=max_out,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                if ("429" in str(e) or "rate" in str(e).lower()) and attempt < 4:
                    wait_time = (2 ** attempt) * 6
                    logger.warning(f"NVIDIA LLM rate limit hit. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"NVIDIA generation failed: {e}")
                    raise


class GroqClient:
    """Groq client for ultra-fast generation via LLaMA 3.3, LLaMA 3.1, Mixtral, etc."""

    def __init__(self):
        self.api_key = CONFIG.get_groq_api_key()
        self.llm_model = (
            CONFIG.models.groq.llm_model
            if hasattr(CONFIG.models, "groq") and CONFIG.models.groq
            else "llama-3.3-70b-versatile"
        )
        self.client = Groq(api_key=self.api_key) if self.api_key else None

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> str:
        """Generate text using Groq API."""
        if not self.client:
            raise ValueError("Groq API Key is not set in environment or .env file.")

        temp = temperature if temperature is not None else CONFIG.models.llm.temperature
        max_out = max_tokens if max_tokens is not None else CONFIG.models.llm.max_output_tokens
        llm_model = model or self.llm_model

        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(5):
            try:
                response = self.client.chat.completions.create(
                    model=llm_model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=max_out,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                if ("429" in str(e) or "rate" in str(e).lower()) and attempt < 4:
                    wait_time = (2 ** attempt) * 4
                    logger.warning(f"Groq LLM rate limit hit. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Groq generation failed: {e}")
                    raise


class UnifiedLLMClient:
    """Unified client that supports Google Gemini, Groq, NVIDIA NIM, and auto-failover."""

    def __init__(self):
        self.gemini = GeminiClient()
        self.groq = GroqClient()
        self.nvidia = NvidiaClient()

    def _route(self, provider: Optional[str], model: Optional[str]) -> str:
        """Decide the primary provider. An explicit user selection is authoritative;
        only when no provider is given do we infer from the model name / config default."""
        if provider:
            return provider.lower()
        if model:
            m = model.lower()
            if "llama-3.3" in m or "llama-3.1" in m or "mixtral" in m or "deepseek-r1" in m or "groq" in m:
                return "groq"
            if "/" in model or "nemotron" in m:
                return "nvidia"
        return (CONFIG.models.provider or "groq").lower()

    def _client_for(self, provider: str):
        prov = provider.lower()
        if prov == "groq":
            return self.groq
        elif prov == "nvidia":
            return self.nvidia
        else:
            return self.gemini

    def generate(
        self,
        prompt: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Backward-compatible wrapper that returns only the generated text."""
        text, _ = self.generate_with_meta(
            prompt=prompt,
            provider=provider,
            model=model,
            system_instruction=system_instruction,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return text

    def generate_with_meta(
        self,
        prompt: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> "tuple[str, dict]":
        """Generate text using the selected provider/model. If a provider is explicitly
        supplied, use only that provider (no automatic fallback). If no provider is
        given, fall back to the default routing logic.
        """
        if provider:
            # Explicit provider selection – use only this client.
            prov = provider.lower()
            client = self._client_for(prov)
            if client.client is None:
                raise ValueError(f"API key for provider '{prov}' is not configured")
            try:
                text = client.generate(
                    prompt=prompt,
                    system_instruction=system_instruction,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    model=model,
                )
                meta = {
                    "provider": prov,
                    "model": model or client.llm_model,
                    "failed_over": False,
                    "requested_provider": prov,
                    "requested_model": model or client.llm_model,
                    "primary_error": None,
                }
                return text, meta
            except Exception as e:
                raise RuntimeError(f"Generation failed for provider '{prov}': {e}")
        # No explicit provider – retain existing routing/fallback behavior.
        primary = self._route(provider, model)
        order = ["groq", "gemini", "nvidia"] if primary == "groq" else (["gemini", "groq", "nvidia"] if primary == "gemini" else ["nvidia", "groq", "gemini"])
        requested_model = model or self._client_for(primary).llm_model
        errors: dict[str, str] = {}
        for prov in order:
            client = self._client_for(prov)
            if client.client is None:
                errors[prov] = "API key not configured"
                continue
            is_primary = prov == primary
            use_model = model if is_primary else None
            try:
                text = client.generate(
                    prompt=prompt,
                    system_instruction=system_instruction,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    model=use_model,
                )
                if not is_primary:
                    logger.warning(
                        f"Primary provider '{primary}' unavailable; answered via fallback '{prov}'."
                    )
                meta = {
                    "provider": prov,
                    "model": use_model or client.llm_model,
                    "failed_over": not is_primary,
                    "requested_provider": primary,
                    "requested_model": requested_model,
                    "primary_error": None if is_primary else errors.get(primary),
                }
                return text, meta
            except Exception as e:
                errors[prov] = str(e)
                logger.warning(
                    f"{prov} generation failed ({str(e)[:150]})"
                    + ("; attempting failover..." if is_primary else "")
                )

        raise RuntimeError(
            f"All providers failed (requested '{primary}'). Errors: {errors}"
        )

    def embed(
        self,
        texts: list[str],
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> list[list[float]]:
        """Embed texts. If provider is nvidia, use nvidia; otherwise use gemini."""
        chosen_provider = (provider or CONFIG.models.provider).lower()
        if chosen_provider == "nvidia":
            return self.nvidia.embed(texts, model=model)
        else:
            # Gemini embedding client (used for gemini and groq)
            return self.gemini.embed(texts, model=model)


@lru_cache(maxsize=1)
def get_groq_client() -> GroqClient:
    return GroqClient()


@lru_cache(maxsize=1)
def get_gemini_client() -> GeminiClient:
    return GeminiClient()


@lru_cache(maxsize=1)
def get_nvidia_client() -> NvidiaClient:
    return NvidiaClient()


@lru_cache(maxsize=1)
def get_unified_client() -> UnifiedLLMClient:
    return UnifiedLLMClient()


@lru_cache(maxsize=1)
def get_cross_encoder() -> CrossEncoder:
    logger.info(f"Loading cross-encoder: {CONFIG.models.cross_encoder.model_name}")
    return CrossEncoder(CONFIG.models.cross_encoder.model_name)
