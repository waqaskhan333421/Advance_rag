"""Hybrid retrieval: Pinecone (dense) + BM25 (sparse) + RRF fusion."""

import logging
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional

from pinecone import Pinecone, ServerlessSpec

from src.config import CONFIG
from src.ingestion import Chunk
from src.models import get_gemini_client

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    metadata: dict
    source: str  # 'dense' | 'sparse' | 'fusion'


class PineconeIndex:
    """Dense vector store via Pinecone with metadata filtering support."""

    def __init__(self):
        self.pc = Pinecone(api_key=CONFIG.get_pinecone_api_key())
        self.index_name = CONFIG.pinecone.index_name
        self.namespace = CONFIG.pinecone.namespace
        self.dimension = CONFIG.models.embedding.output_dimensionality
        self._ensure_index()
        self.index = self.pc.Index(self.index_name)

    def _ensure_index(self):
        existing = [i["name"] for i in self.pc.list_indexes()]
        if self.index_name not in existing:
            logger.info(f"Creating Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=CONFIG.pinecone.cloud,
                    region=CONFIG.pinecone.region,
                ),
            )

    def upsert(self, chunks: List[Chunk], embeddings: List[List[float]]):
        """Upsert chunks with metadata into Pinecone."""
        vectors = []
        for chunk, emb in zip(chunks, embeddings):
            vectors.append({
                "id": chunk.chunk_id,
                "values": emb,
                "metadata": {
                    **chunk.metadata,
                    "text": chunk.text[:1000],  # store truncated text in metadata
                },
            })

        # Batch upsert in groups of 100
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch, namespace=self.namespace)
        logger.info(f"Upserted {len(vectors)} vectors to Pinecone")

    def query(
        self,
        embedding: List[float],
        top_k: int = 30,
        filter_dict: Optional[dict] = None,
    ) -> List[RetrievedChunk]:
        """Query dense index."""
        response = self.index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=self.namespace,
            filter=filter_dict,
        )
        results = []
        for match in response.matches:
            meta = match.metadata or {}
            results.append(RetrievedChunk(
                chunk_id=match.id,
                text=meta.get("text", ""),
                score=match.score,
                metadata={k: v for k, v in meta.items() if k != "text"},
                source="dense",
            ))
        return results


class BM25Index:
    """Local sparse lexical index using rank_bm25."""

    def __init__(self):
        self.path = CONFIG.paths.bm25_index
        self.bm25 = None
        self.corpus: List[Chunk] = []
        self.tokenized_corpus: List[List[str]] = []

    def build(self, chunks: List[Chunk]):
        """Build BM25 index from chunks."""
        from rank_bm25 import BM25Okapi
        import nltk

        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)

        self.corpus = chunks
        self.tokenized_corpus = [
            nltk.word_tokenize(c.text.lower()) for c in chunks
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.save()
        logger.info(f"Built BM25 index with {len(chunks)} documents")

    def query(self, query_text: str, top_k: int = 30, filter_dict: Optional[dict] = None) -> List[RetrievedChunk]:
        """Query BM25 index with optional metadata/doc_id filtering."""
        if self.bm25 is None:
            self.load()

        if not self.corpus:
            return []

        import nltk
        tokenized_query = nltk.word_tokenize(query_text.lower())
        scores = self.bm25.get_scores(tokenized_query)

        # Determine target doc_id if filtering is active
        target_doc_id = None
        if filter_dict and "doc_id" in filter_dict:
            val = filter_dict["doc_id"]
            if isinstance(val, dict) and "$eq" in val:
                target_doc_id = val["$eq"]
            elif isinstance(val, str):
                target_doc_id = val

        # Filter candidate indices if a specific document is targeted
        if target_doc_id:
            candidate_indices = [
                i for i, c in enumerate(self.corpus)
                if getattr(c, "doc_id", None) == target_doc_id
                or (c.metadata and c.metadata.get("doc_id") == target_doc_id)
            ]
        else:
            candidate_indices = list(range(len(scores)))

        top_indices = sorted(candidate_indices, key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            chunk = self.corpus[idx]
            results.append(RetrievedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                score=float(scores[idx]),
                metadata=chunk.metadata,
                source="sparse",
            ))
        return results

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "corpus": self.corpus}, f)

    def load(self):
        with open(self.path, "rb") as f:
            data = pickle.load(f)
            self.bm25 = data["bm25"]
            self.corpus = data["corpus"]


def reciprocal_rank_fusion(
    dense_results: List[RetrievedChunk],
    sparse_results: List[RetrievedChunk],
    k: int = 60,
    top_k: int = 50,
) -> List[RetrievedChunk]:
    """Merge dense and sparse results via Reciprocal Rank Fusion."""
    scores: Dict[str, float] = {}
    metadata_map: Dict[str, dict] = {}
    text_map: Dict[str, str] = {}

    def _process(results: List[RetrievedChunk], source: str):
        for rank, item in enumerate(results, start=1):
            scores[item.chunk_id] = scores.get(item.chunk_id, 0.0) + 1.0 / (k + rank)
            metadata_map[item.chunk_id] = item.metadata
            text_map[item.chunk_id] = item.text

    _process(dense_results, "dense")
    _process(sparse_results, "sparse")

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_k]

    fused = []
    for cid in sorted_ids:
        fused.append(RetrievedChunk(
            chunk_id=cid,
            text=text_map[cid],
            score=scores[cid],
            metadata=metadata_map[cid],
            source="fusion",
        ))
    return fused
