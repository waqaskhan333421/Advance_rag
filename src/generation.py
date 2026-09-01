"""LLM generation with inline citations."""

import logging
from typing import List, Optional

from src.config import CONFIG
from src.models import UnifiedLLMClient, get_unified_client
from src.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a precise research assistant. Answer the user's question using ONLY the provided context.
Cite your sources inline using [Page X, Section: Y] format.
If the context does not contain enough information, say "I don't know based on the provided documents."
Do not make up facts."""


class Generator:
    """Multi-provider answer generation with citations (Gemini & NVIDIA NIM)."""

    def __init__(self, client: Optional[UnifiedLLMClient] = None):
        self.client = client or get_unified_client()

    def build_prompt(self, query: str, chunks: List[RetrievedChunk]) -> str:
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            page = chunk.metadata.get("page_number", "?")
            section = chunk.metadata.get("section_title", "Unknown")
            context_parts.append(
                f"[Source {i}] Page {page}, Section: {section}\n{chunk.text}\n"
            )
        context_block = "\n".join(context_parts)

        return f"""{SYSTEM_PROMPT}

--- CONTEXT ---
{context_block}
--- END CONTEXT ---

Question: {query}

Answer (with inline citations):"""

    def generate(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> dict:
        """Generate answer with citations."""
        prompt = self.build_prompt(query, chunks)
        logger.info(f"Generating answer with {len(chunks)} compressed chunks via provider={provider or 'default'}, model={model or 'default'}")

        try:
            answer, meta = self.client.generate_with_meta(
                prompt=prompt,
                provider=provider,
                model=model,
                system_instruction=SYSTEM_PROMPT,
                temperature=temperature if temperature is not None else CONFIG.models.llm.temperature,
                max_tokens=max_tokens if max_tokens is not None else CONFIG.models.llm.max_output_tokens,
            )
            return {
                "answer": answer,
                "citations": [
                    {
                        "chunk_id": c.chunk_id,
                        "doc_id": c.metadata.get("doc_id"),
                        "file_name": c.metadata.get("file_name"),
                        "page": c.metadata.get("page_number"),
                        "section": c.metadata.get("section_title"),
                        "score": c.score,
                        "text": c.text,
                        "visual_asset_path": c.metadata.get("visual_asset_path"),
                    }
                    for c in chunks
                ],
                "provider_used": meta["provider"],
                "model_used": meta["model"],
                "failed_over": meta["failed_over"],
                "requested_provider": meta["requested_provider"],
                "requested_model": meta["requested_model"],
                "primary_error": meta.get("primary_error"),
            }
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
