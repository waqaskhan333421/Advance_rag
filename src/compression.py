"""LLM-based context compression: keep only query-relevant sentences."""

import logging
import re
from typing import List, Optional

from src.config import CONFIG
from src.models import UnifiedLLMClient, get_unified_client
from src.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

BATCH_COMPRESSION_PROMPT = """You are a context compression engine.
Given a user query and numbered document chunks, extract ONLY the sentences from each chunk that are relevant to answering the query.
Preserve any factual claims, numbers, and specific references. If the entire chunk is relevant, keep it unchanged. If nothing in a chunk is relevant, output [IRRELEVANT] for that chunk.

Format your output strictly as:
[CHUNK 1]
<extract or [IRRELEVANT]>
[CHUNK 2]
<extract or [IRRELEVANT]>

Query: {query}

--- CHUNKS ---
{chunks_block}
--- END CHUNKS ---

Extracts:"""


class ContextCompressor:
    """Compress top-k chunks to reduce noise before generation in a single batch pass."""

    def __init__(self, client: Optional[UnifiedLLMClient] = None):
        self.client = client or get_unified_client()

    def compress(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """Compress all candidate chunks in a single LLM request."""
        if not chunks:
            return []

        chunks_formatted = []
        for i, chunk in enumerate(chunks, 1):
            page = chunk.metadata.get("page_number", "?")
            sec = chunk.metadata.get("section_title", "Unknown")
            chunks_formatted.append(f"[CHUNK {i}] (Page {page}, Section: {sec})\n{chunk.text}")

        prompt = BATCH_COMPRESSION_PROMPT.format(
            query=query,
            chunks_block="\n\n".join(chunks_formatted),
        )

        try:
            raw_response = self.client.generate(
                prompt=prompt,
                provider=provider,
                model=model,
                temperature=0.0,
                max_tokens=1500,
            ).strip()

            # Parse [CHUNK i] sections
            parts = re.split(r"\[CHUNK\s+(\d+)\]", raw_response)
            extracted_by_idx = {}
            if len(parts) > 1:
                for j in range(1, len(parts), 2):
                    idx = int(parts[j])
                    content = parts[j + 1].strip()
                    extracted_by_idx[idx] = content

            compressed = []
            for i, chunk in enumerate(chunks, 1):
                extract = extracted_by_idx.get(i)
                if extract:
                    if "[IRRELEVANT]" not in extract.upper():
                        chunk.text = extract
                        compressed.append(chunk)
                    else:
                        logger.debug(f"Chunk {chunk.chunk_id} filtered out by compression")
                else:
                    compressed.append(chunk)

            return compressed if compressed else chunks

        except Exception as e:
            logger.warning(f"Batch compression failed ({e}); falling back to uncompressed chunks.")
            return chunks

