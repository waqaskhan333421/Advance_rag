# Advanced RAG Pipeline for PDF Q&A

Production-grade Retrieval-Augmented Generation with **Pinecone**, **Gemini Embedding 2 Preview**, **Gemini 1.5 Pro**, hybrid search (dense + BM25), RRF fusion, cross-encoder reranking, and LLM-based context compression.

## Architecture

```
User Query
  → Query Transformation (Rewrite + HyDE)
  → Hybrid Search (Pinecone Dense + BM25 Sparse)
  → Reciprocal Rank Fusion (RRF)
  → Top-50 Candidates
  → Cross-Encoder Reranker
  → Top-5 Compressed Context
  → Gemini 1.5 Pro Generation
  → Answer with Citations
```

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set environment variables
export GEMINI_API_KEY="your-gemini-api-key"
export PINECONE_API_KEY="your-pinecone-api-key"
export NVIDIA_API_KEY="your-nvidia-api-key" # Base URL: https://integrate.api.nvidia.com/v1

# 3. (Optional) Download NLTK data
python -c "import nltk; nltk.download('punkt')"

# 4. Place PDFs in ./data/pdfs/
mkdir -p data/pdfs
cp your-docs/*.pdf data/pdfs/
```

## Usage

### CLI

```bash
# Index documents
python main.py index --pdf-dir ./data/pdfs

# Ask a question
python main.py ask "What are the key findings?"

# Filter by document metadata
python main.py ask "What are the key findings?" --doc-id abc123
```

### Streamlit Web UI

Launch the interactive web interface:

```bash
streamlit run app.py
```

Features included in the Streamlit UI:
- **Interactive Chat Interface**: Chat with your PDFs with streaming citations and collapsible context viewers.
- **Document Management**: Drag-and-drop PDF uploader with automatic OCR, semantic chunking, and vector indexing.
- **Retrieval Scope Filtering**: Choose between searching across all documents or filtering specifically by document ID.
- **Pipeline & Latency Inspector**: Stage-by-stage latency breakdowns (Query Rewriting, Hybrid Retrieval, RRF Fusion, Cross-Encoder Reranking, Context Compression, Generation).
- **Query Transformation Transparency**: Inspect the rewritten queries and generated HyDE documents.

### FastAPI Server

```bash
python main.py serve --host 0.0.0.0 --port 8000
```

Then POST to `http://localhost:8000/query`:

```json
{
  "question": "What is the methodology?",
  "metadata_filter": {"doc_id": {"$eq": "abc123"}}
}
```

### Evaluation

```bash
python evaluation.py
```

## Configuration

Edit `config.yaml` to customize:

| Key | Description |
|-----|-------------|
| `models.embedding.model_name` | `gemini-embedding-2-preview` |
| `models.llm.model_name` | `gemini-1.5-pro` |
| `pinecone.index_name` | Pinecone index name |
| `chunking.chunk_size` | Tokens per chunk (~words) |
| `retrieval.rerank_top_k` | Final chunks sent to LLM |

## Metadata Support

Every chunk carries rich metadata stored in Pinecone and BM25:

- `doc_id` — MD5 hash of source PDF
- `page_number` — 1-based page index
- `section_title` — Detected header from font analysis
- `chunk_index` — Global ordering

Use `metadata_filter` in `/query` to restrict search by document, page, or section.

## Testing

```bash
pytest tests/
```

## Swapping Models

- **Embedding**: Change `models.embedding.model_name` to `gemini-embedding-001` or adjust `output_dimensionality` (768, 1536, 3072).
- **LLM**: Change `models.llm.model_name` to any Gemini model (e.g., `gemini-1.5-flash`).
- **Cross-Encoder**: Swap `cross-encoder/ms-marco-MiniLM-L-6-v2` for `BAAI/bge-reranker-large` (requires more VRAM).

---

## Full Project Workflow

This section walks through the complete lifecycle of the Advanced RAG system — from raw PDFs on disk to a cited, grounded answer delivered to the user.

### Overview Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        INGESTION  PIPELINE                              │
│                                                                         │
│  PDF Files  ──►  PDF Parser  ──►  Semantic Chunker  ──►  Embedder      │
│  (./data/pdfs)   (PyMuPDF +       (Recursive split,     (Gemini /       │
│                   OCR fallback)    400-word chunks,      NVIDIA)         │
│                                    60-word overlap)          │           │
│                                                              ▼           │
│                                              ┌─────────────────────┐    │
│                                              │  Pinecone (Dense)   │    │
│                                              │  BM25 Index (Sparse)│    │
│                                              └─────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                         QUERY  PIPELINE                                 │
│                                                                         │
│  User Query                                                             │
│      │                                                                  │
│      ▼                                                                  │
│  [1] Query Transformation                                               │
│      ├─ Query Rewriting  (decontextualized search query)                │
│      └─ HyDE             (hypothetical document generation)             │
│      │                                                                  │
│      ▼                                                                  │
│  [2] Hybrid Retrieval                                                   │
│      ├─ Dense Search  → Pinecone (top-30 by cosine similarity)          │
│      └─ Sparse Search → BM25 index (top-30 by BM25 score)              │
│      │                                                                  │
│      ▼                                                                  │
│  [3] RRF Fusion                                                         │
│      └─ Reciprocal Rank Fusion merges both lists → top-50 candidates   │
│      │                                                                  │
│      ▼                                                                  │
│  [4] Cross-Encoder Reranking                                            │
│      └─ ms-marco-MiniLM scores each candidate → top-5 kept             │
│      │                                                                  │
│      ▼                                                                  │
│  [5] Context Compression                                                │
│      └─ LLM strips irrelevant sentences, retains answer-bearing text   │
│      │                                                                  │
│      ▼                                                                  │
│  [6] Generation                                                         │
│      └─ Gemini / NVIDIA LLM generates a cited, grounded answer         │
│      │                                                                  │
│      ▼                                                                  │
│  Answer + Source Citations + Latency Breakdown                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### Stage 1 — Ingestion Pipeline

**Trigger**: `python main.py index --pdf-dir ./data/pdfs`  
**Code**: `src/ingestion.py` → `src/pipeline.py::AdvancedRAGPipeline.index_documents()`

| Step | Component | What happens |
|------|-----------|--------------|
| 1a | `PDFParser` | Opens each PDF with **PyMuPDF**. Extracts raw text page by page. If a page yields fewer than 50 characters (scanned page), falls back to **RapidOCR** (preferred) or **pytesseract**. |
| 1b | Header detection | Font-size heuristics identify bold text > 12 pt as section headers. The first header on a page becomes `section_title` metadata. |
| 1c | `SemanticChunker` | Recursively splits text using separator hierarchy (`\n## `, `\n### `, `\n\n`, `\n`, `. `, ` `). Chunks target **400 words** with a **60-word overlap** for cross-boundary continuity. |
| 1d | Metadata tagging | Each `Chunk` receives: `doc_id` (MD5 of file bytes), `page_number`, `section_title`, `chunk_index`, and `file_name`. |
| 1e | Embedding | All chunk texts are batch-embedded using **Gemini Embedding 2 Preview** (768-dim) via `UnifiedLLMClient`. |
| 1f | Dense indexing | Embeddings + metadata upserted into **Pinecone** (cosine, serverless, AWS us-east-1). |
| 1g | Sparse indexing | BM25 index built from all chunk texts and persisted to `./data/bm25_index.pkl`. |

> **Incremental indexing**: Already-ingested documents (matched by MD5 doc_id or filename) are skipped automatically unless `--force-reindex` is passed.

> **Image extraction (Option B)**: Pass `--extract-images` to run `src/image_extractor.py` first. It uses **Gemini Vision** to caption embedded images/figures before text chunking.

---

### Stage 2 — Query Transformation

**Code**: `src/pipeline.py::QueryTransformer.transform()`

Two transformations run in a **single LLM call** to minimise API quota usage and latency:

| Technique | Purpose | Config flag |
|-----------|---------|-------------|
| **Query Rewriting** | Rewrites the user's conversational question into a precise, decontextualized search query | `use_query_rewriting=True` |
| **HyDE** (Hypothetical Document Embeddings) | Generates a short 2-sentence authoritative passage that *would* answer the question. Its embedding is averaged with the rewritten query embedding to shift the retrieval vector toward the answer space. | `use_hyde=True` |

Both flags are independently toggleable from the Streamlit UI or the `/query` API.

---

### Stage 3 — Hybrid Retrieval

**Code**: `src/retrieval.py`

Two complementary retrieval signals are gathered in parallel:

| Index | Technology | Signal | Top-K |
|-------|-----------|--------|-------|
| Dense | **Pinecone** (cosine similarity) | Semantic/conceptual proximity | 30 |
| Sparse | **BM25** (rank-bm25) | Exact keyword overlap | 30 |

The query embedding is the **average** of the rewritten query embedding and the HyDE document embedding (when HyDE is active), giving a vector that is semantically anchored to both the question form and the expected answer form.

Metadata filters (e.g., `{"doc_id": {"$eq": "abc123"}}`) are propagated to both indexes to support document-scoped queries.

---

### Stage 4 — RRF Fusion

**Code**: `src/retrieval.py::reciprocal_rank_fusion()`

Reciprocal Rank Fusion merges the dense and sparse ranked lists without requiring score calibration:

```
RRF_score(chunk) = Σ  1 / (k + rank_i)   where k = 60 (config: retrieval.rrf_k)
```

The top **50** fused candidates (config: `retrieval.fusion_top_k`) are forwarded to the reranker. If only one retrieval mode is active, that list is used directly.

---

### Stage 5 — Cross-Encoder Reranking

**Code**: `src/rerank.py::Reranker`

A **cross-encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) scores each of the 50 candidate (query, chunk) pairs jointly. Unlike bi-encoders, the cross-encoder sees both texts simultaneously, yielding significantly more accurate relevance scores.

The top **5** chunks (config: `retrieval.rerank_top_k`) are kept and passed downstream.

---

### Stage 6 — Context Compression

**Code**: `src/compression.py::ContextCompressor`

The LLM (same provider/model selected for generation) reads each of the 5 reranked chunks and extracts only the sentences directly relevant to the rewritten query. This:

- Reduces prompt token count (lower latency, lower cost).
- Removes noise that could distract the generator.
- Keeps passage-level attribution intact for citations.

---

### Stage 7 — Generation

**Code**: `src/generation.py::Generator`

The compressed context chunks are assembled into a structured prompt and sent to the configured LLM:

| Provider | Default Model | Override |
|----------|--------------|---------|
| Gemini | `gemini-3.5-flash-lite` | `models.gemini.llm_model` in `config.yaml` |
| NVIDIA NIM | `nvidia/llama-3.1-nemotron-70b-instruct` | `models.nvidia.llm_model` in `config.yaml` |

The response includes:
- A **grounded answer** with inline citations (`[Source: filename, p. N]`).
- A **`PipelineResult`** object containing the rewritten query, HyDE doc, per-stage latencies, chunk count at each stage, and failover metadata.

---

### Provider Failover

**Code**: `src/models.py::UnifiedLLMClient`

If the primary provider (Gemini or NVIDIA) throws an API error, the client automatically retries with the fallback provider. The `PipelineResult` records:

| Field | Meaning |
|-------|---------|
| `provider_used` | Provider that actually answered |
| `requested_provider` | Provider originally requested |
| `failed_over` | `True` when a silent failover occurred |
| `primary_error` | Error message from the primary provider |

The Streamlit UI surfaces a warning banner when `failed_over` is `True`.

---

### Toggleable RAG Techniques

All stages can be individually enabled or disabled at query time (CLI flags, API body, or Streamlit toggles):

| Flag | Default | Disabling effect |
|------|---------|-----------------|
| `use_query_rewriting` | `True` | Original query used as-is |
| `use_hyde` | `True` | No hypothetical document; only rewritten query embedding |
| `use_dense` | `True` | Skip Pinecone retrieval |
| `use_sparse` | `True` | Skip BM25 retrieval |
| `use_rrf` | `True` | Whichever active list is used without fusion |
| `use_reranking` | `True` | Top-N from fusion passed directly to compression |
| `use_compression` | `True` | Full reranked chunks sent to LLM |

This is useful for **ablation studies** and the built-in **Pipeline & Latency Inspector** in the Streamlit UI.

---

### End-to-End Latency Breakdown

A `latency_ms` dictionary is returned with every query result:

| Key | Stage measured |
|-----|---------------|
| `query_transform` | Query rewriting + HyDE (single LLM call) |
| `retrieval` | Dense + sparse search (parallel) |
| `rrf` | RRF fusion computation |
| `rerank` | Cross-encoder scoring of all candidates |
| `compression` | LLM context compression |
| `generation` | Final answer generation |
| `total` | Wall-clock time from query receipt to answer |

These latencies are displayed stage-by-stage in the **Pipeline & Latency Inspector** panel of the Streamlit UI.

---

### Data Flow Summary

```
./data/pdfs/*.pdf
        │
        ▼ (index command)
  Parse + OCR + Chunk + Embed
        │
        ├──► Pinecone index  (dense vectors)
        └──► BM25 index      (./data/bm25_index.pkl)

User Question
        │
        ▼ (query command / API POST / Streamlit chat)
  Query Rewrite + HyDE  ──► averaged embedding
        │
        ├──► Pinecone top-30  ──┐
        └──► BM25 top-30       ──┤ RRF → top-50
                                 │
                         Cross-Encoder → top-5
                                 │
                     LLM Compression → compressed snippets
                                 │
                      LLM Generation → Answer + Citations
```
