"""End-to-end RAG pipeline orchestration with Multi-Provider Support (Gemini & NVIDIA NIM)."""

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from src.compression import ContextCompressor
from src.config import CONFIG
from src.generation import Generator
from src.ingestion import Chunk, IngestionPipeline
from src.models import UnifiedLLMClient, get_unified_client
from src.retrieval import BM25Index, PineconeIndex, reciprocal_rank_fusion
from src.rerank import Reranker

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    query: str
    rewritten_query: str
    hyde_doc: str
    dense_results: int
    sparse_results: int
    fused_results: int
    reranked_results: int
    final_chunks: List[dict]
    answer: str
    latency_ms: dict
    provider_used: str = "gemini"
    model_used: str = ""
    # Transparent-failover reporting: when the selected provider fails, we fall back
    # to the other one but record what was requested vs. what actually answered so
    # the UI can notify the user instead of silently mislabeling the result.
    failed_over: bool = False
    requested_provider: str = ""
    requested_model: str = ""
    primary_error: Optional[str] = None


class QueryTransformer:
    """Rewrite + HyDE + multi-query generation with unified single-call transformation."""

    def __init__(self, client: Optional[UnifiedLLMClient] = None):
        self.client = client or get_unified_client()

    def transform(
        self,
        query: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        use_rewriting: bool = True,
        use_hyde: bool = True,
    ) -> tuple[str, str]:
        """Perform query rewriting and/or HyDE in a single unified prompt to minimize latency and API quota usage."""
        if not use_rewriting and not use_hyde:
            return query, ""
        if use_rewriting and not use_hyde:
            prompt = f"Rewrite the following question into a clear, decontextualized search query:\n\n{query}\n\nSearch query:"
            res = self.client.generate(prompt=prompt, provider=provider, model=model, temperature=0.2, max_tokens=256).strip()
            return res or query, ""
        if not use_rewriting and use_hyde:
            prompt = f"Write a short hypothetical document (2-3 sentences) that would answer this question in an authoritative tone:\n\nQuestion: {query}\n\nHypothetical document:"
            res = self.client.generate(prompt=prompt, provider=provider, model=model, temperature=0.3, max_tokens=256).strip()
            return query, res

        # Both active: combine into 1 API call
        prompt = (
            f"Analyze the following question.\n"
            f"1. Provide a clear, decontextualized search query.\n"
            f"2. Provide a short 2-sentence authoritative hypothetical passage answering it.\n\n"
            f"Question: {query}\n\n"
            f"Format strictly as:\n"
            f"REWRITTEN: <search query>\n"
            f"HYDE: <passage>"
        )
        try:
            resp = self.client.generate(prompt=prompt, provider=provider, model=model, temperature=0.2, max_tokens=300).strip()
            rewritten = query
            hyde_doc = ""
            for line in resp.split("\n"):
                line_s = line.strip()
                if line_s.upper().startswith("REWRITTEN:"):
                    cand = line_s.split(":", 1)[1].strip()
                    if cand:
                        rewritten = cand
                elif line_s.upper().startswith("HYDE:"):
                    cand = line_s.split(":", 1)[1].strip()
                    if cand:
                        hyde_doc = cand
            return rewritten, hyde_doc
        except Exception as e:
            logger.warning(f"Query transform error ({e}); using original query.")
            return query, ""


class AdvancedRAGPipeline:
    """Full pipeline: Ingestion → Indexing → Query → Retrieve → Rerank → Compress → Generate."""

    def __init__(self):
        self.client = get_unified_client()
        self.ingestion = IngestionPipeline()
        self.dense_index = PineconeIndex()
        self.sparse_index = BM25Index()
        self.transformer = QueryTransformer(self.client)
        self.reranker = Reranker()
        self.compressor = ContextCompressor(self.client)
        self.generator = Generator(self.client)

    def index_documents(
        self,
        pdf_dir: Optional[str] = None,
        force_reindex: bool = False,
        extract_images: bool = False,
        skip_captioning: bool = False,
        embedding_provider: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        """Parse PDFs, chunk, embed, and index documents. Option B runs Gemini Vision image extraction first."""
        pdf_dir = pdf_dir or CONFIG.paths.documents_dir

        if extract_images:
            logger.info("Option B enabled: Running Gemini Vision image extraction prior to indexing...")
            from src.image_extractor import run as run_image_extractor
            run_image_extractor(pdf_dir=pdf_dir, skip_captioning=skip_captioning)

        existing_doc_ids = set()
        existing_doc_names = set()
        if not force_reindex:
            existing_docs = self.list_documents()
            for d in existing_docs:
                if d.get("doc_id"):
                    existing_doc_ids.add(d["doc_id"])
                if d.get("file_name") and d["file_name"] != "Unknown":
                    existing_doc_names.add(d["file_name"].lower())

        chunks = self.ingestion.ingest_directory(
            pdf_dir,
            skip_doc_ids=None if force_reindex else existing_doc_ids,
            skip_doc_names=None if force_reindex else existing_doc_names,
        )

        if not chunks:
            logger.info("No new documents to index. All files in directory are already ingested.")
            return

        texts = [c.text for c in chunks]
        logger.info(f"Embedding {len(texts)} new chunks via {embedding_provider or 'default'}...")
        embeddings = self.client.embed(texts, provider=embedding_provider, model=embedding_model)

        # Index dense (Pinecone) and sparse (BM25)
        self.dense_index.upsert(chunks, embeddings)

        combined_chunks = (self.sparse_index.corpus if not force_reindex else []) + chunks
        self.sparse_index.build(combined_chunks)
        logger.info("Indexing complete")

    def list_documents(self) -> List[dict]:
        """List summary of all ingested documents."""
        if not self.sparse_index.corpus:
            self.sparse_index.load()

        pdf_map = {}
        pdf_dir = Path(CONFIG.paths.documents_dir)
        if pdf_dir.exists():
            for p in pdf_dir.glob("*.pdf"):
                try:
                    doc_id = hashlib.md5(p.read_bytes()).hexdigest()[:16]
                    pdf_map[doc_id] = p.name
                except Exception:
                    pass

        doc_stats = {}
        for chunk in self.sparse_index.corpus:
            doc_id = chunk.doc_id
            if doc_id not in doc_stats:
                file_name = chunk.metadata.get("file_name") or pdf_map.get(doc_id, "Unknown")
                doc_stats[doc_id] = {
                    "doc_id": doc_id,
                    "file_name": file_name,
                    "chunks": 0,
                    "max_page": 0,
                }
            doc_stats[doc_id]["chunks"] += 1
        # Ensure all existing PDF files in documents_dir are present in the list
        for doc_id, p_name in pdf_map.items():
            if doc_id not in doc_stats:
                doc_stats[doc_id] = {
                    "doc_id": doc_id,
                    "file_name": p_name,
                    "chunks": 0,
                    "max_page": 0,
                }

        return list(doc_stats.values())

    def query(
        self,
        question: str,
        metadata_filter: Optional[dict] = None,
        provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        # Technique flags — each can be individually toggled from the UI
        use_query_rewriting: bool = True,
        use_hyde: bool = True,
        use_dense: bool = True,
        use_sparse: bool = True,
        use_rrf: bool = True,
        use_reranking: bool = True,
        use_compression: bool = True,
    ) -> PipelineResult:
        """Run the full retrieval + generation pipeline with provider/model selection.

        Technique flags let individual RAG stages be enabled/disabled dynamically.
        """
        latencies = {}
        t0 = time.perf_counter()

        chosen_provider = provider or CONFIG.models.provider
        if llm_model:
            chosen_llm = llm_model
        elif chosen_provider == "groq":
            chosen_llm = getattr(CONFIG.models.groq, "llm_model", "llama-3.3-70b-versatile")
        elif chosen_provider == "nvidia":
            chosen_llm = getattr(CONFIG.models.nvidia, "llm_model", "nvidia/llama-3.1-nemotron-70b-instruct")
        else:
            chosen_llm = getattr(CONFIG.models.gemini, "llm_model", "gemini-2.5-flash")

        # 1. Query Transformation (Rewriting + HyDE in a single step)
        t1 = time.perf_counter()
        rewritten, hyde_doc = self.transformer.transform(
            question,
            provider=chosen_provider,
            model=chosen_llm,
            use_rewriting=use_query_rewriting,
            use_hyde=use_hyde,
        )
        latencies["query_transform"] = round((time.perf_counter() - t1) * 1000, 2)

        # 2. Hybrid Retrieval
        t2 = time.perf_counter()
        dense: list = []
        sparse: list = []

        if use_dense or use_hyde:
            # Build embedding input: rewritten query + optional HyDE doc
            emb_inputs = [rewritten] + ([hyde_doc] if hyde_doc else [])
            embs = self.client.embed(emb_inputs, provider=chosen_provider, model=embedding_model)
            query_emb = [sum(x) / len(x) for x in zip(*embs)]

        if use_dense:
            dense = self.dense_index.query(
                query_emb,
                top_k=CONFIG.retrieval.dense_top_k,
                filter_dict=metadata_filter,
            )

        if use_sparse:
            sparse = self.sparse_index.query(
                rewritten,
                top_k=CONFIG.retrieval.sparse_top_k,
                filter_dict=metadata_filter,
            )

        latencies["retrieval"] = round((time.perf_counter() - t2) * 1000, 2)

        # 3. RRF Fusion (only when both dense and sparse available, else use whichever exists)
        t3 = time.perf_counter()
        if use_rrf and dense and sparse:
            fused = reciprocal_rank_fusion(
                dense, sparse,
                k=CONFIG.retrieval.rrf_k,
                top_k=CONFIG.retrieval.fusion_top_k,
            )
        elif dense:
            fused = dense[:CONFIG.retrieval.fusion_top_k]
        else:
            fused = sparse[:CONFIG.retrieval.fusion_top_k]
        latencies["rrf"] = round((time.perf_counter() - t3) * 1000, 2)

        # 4. Cross-Encoder Rerank
        t4 = time.perf_counter()
        if use_reranking and fused:
            reranked = self.reranker.rerank(rewritten, fused)
        else:
            reranked = fused[:CONFIG.retrieval.rerank_top_k]
        latencies["rerank"] = round((time.perf_counter() - t4) * 1000, 2)

        # 5. Context Compression
        t5 = time.perf_counter()
        if use_compression and reranked:
            compressed = self.compressor.compress(rewritten, reranked, provider=chosen_provider, model=chosen_llm)
        else:
            compressed = reranked
        latencies["compression"] = round((time.perf_counter() - t5) * 1000, 2)

        # 6. Generation
        t6 = time.perf_counter()
        result = self.generator.generate(
            question,
            compressed,
            provider=chosen_provider,
            model=chosen_llm,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        latencies["generation"] = round((time.perf_counter() - t6) * 1000, 2)

        latencies["total"] = round((time.perf_counter() - t0) * 1000, 2)

        actual_provider = result.get("provider_used", chosen_provider)
        actual_model = result.get("model_used", chosen_llm)
        did_failover = result.get("failed_over", False)

        if did_failover:
            logger.warning(
                f"Pipeline complete in {latencies['total']}ms — FAILOVER: requested "
                f"{chosen_provider}/{chosen_llm}, answered via {actual_provider}/{actual_model}"
            )
        else:
            logger.info(f"Pipeline complete in {latencies['total']}ms via provider={actual_provider}, model={actual_model}")
        for stage, ms in latencies.items():
            if stage != "total":
                logger.info(f"  {stage}: {ms}ms")

        return PipelineResult(
            query=question,
            rewritten_query=rewritten,
            hyde_doc=hyde_doc,
            dense_results=len(dense),
            sparse_results=len(sparse),
            fused_results=len(fused),
            reranked_results=len(reranked),
            final_chunks=result["citations"],
            answer=result["answer"],
            latency_ms=latencies,
            provider_used=actual_provider,
            model_used=actual_model,
            failed_over=did_failover,
            requested_provider=chosen_provider,
            requested_model=chosen_llm,
            primary_error=result.get("primary_error"),
        )


