"""Cross-encoder reranking."""

import logging
from typing import List

from src.config import CONFIG
from src.models import get_cross_encoder
from src.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder reranker for high-precision final ranking."""

    def __init__(self):
        self.model = get_cross_encoder()
        self.top_k = CONFIG.retrieval.rerank_top_k

    def rerank(self, query: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """Rerank chunks by (query, chunk) relevance."""
        if not chunks:
            return []

        pairs = [(query, c.text) for c in chunks]
        scores = self.model.predict(pairs, batch_size=16, show_progress_bar=False)

        scored = [(chunk, float(score)) for chunk, score in zip(chunks, scores)]
        scored.sort(key=lambda x: x[1], reverse=True)

        top = scored[: self.top_k]
        for chunk, score in top:
            chunk.score = score
            chunk.source = "reranked"
        return [c for c, _ in top]
