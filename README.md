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
