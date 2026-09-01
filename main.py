"""CLI and FastAPI entry point."""

import logging
import os
from typing import Optional

import typer
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from src.config import CONFIG
from src.pipeline import AdvancedRAGPipeline, PipelineResult

# Setup logging
logging.basicConfig(
    level=getattr(logging, CONFIG.app.log_level),
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title=CONFIG.app.name)
pipeline = AdvancedRAGPipeline()


class QueryRequest(BaseModel):
    question: str
    metadata_filter: Optional[dict] = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[dict]
    latency_ms: dict
    rewritten_query: str
    hyde_doc: str


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    result = pipeline.query(req.question, metadata_filter=req.metadata_filter)
    return QueryResponse(
        answer=result.answer,
        citations=result.final_chunks,
        latency_ms=result.latency_ms,
        rewritten_query=result.rewritten_query,
        hyde_doc=result.hyde_doc,
    )


@app.post("/index")
async def index_endpoint(
    pdf_dir: Optional[str] = None,
    extract_images: bool = False,
    skip_captioning: bool = False,
):
    pipeline.index_documents(pdf_dir, extract_images=extract_images, skip_captioning=skip_captioning)
    return {"status": "indexed", "option_b_extracted_images": extract_images}


@app.post("/extract-images")
async def extract_images_endpoint(
    pdf_dir: Optional[str] = None,
    skip_captioning: bool = False,
):
    from src.image_extractor import run as run_image_extractor
    run_image_extractor(pdf_dir=pdf_dir, skip_captioning=skip_captioning)
    return {"status": "success", "message": "Extracted images and built corpus PDF."}


@app.get("/documents")
async def list_documents_endpoint():
    return {"documents": pipeline.list_documents()}


# CLI
cli = typer.Typer()


@cli.command()
def index(
    pdf_dir: str = CONFIG.paths.documents_dir,
    force: bool = typer.Option(False, "--force", "-f", help="Force re-indexing of all documents"),
    extract_images: bool = typer.Option(False, "--extract-images", "-e", help="Option B: Run Gemini Vision image extraction before indexing"),
    skip_captioning: bool = typer.Option(False, "--skip-captioning", help="Skip Gemini Vision captioning step (extraction test only)"),
):
    """Index PDFs. Use -e / --extract-images for Option B (Gemini Vision first, then ingest)."""
    pipeline.index_documents(
        pdf_dir,
        force_reindex=force,
        extract_images=extract_images,
        skip_captioning=skip_captioning,
    )
    typer.echo("Indexing complete.")


@cli.command("extract-images")
def extract_images_cmd(
    pdf_dir: str = CONFIG.paths.documents_dir,
    skip_captioning: bool = typer.Option(False, "--skip-captioning", help="Skip Gemini Vision captioning step"),
    delay: float = typer.Option(1.0, "--delay", help="Seconds delay between Gemini Vision API calls"),
):
    """Option B: Extract images from all PDF books, caption with Gemini Vision, and create searchable image corpus PDF."""
    from src.image_extractor import run as run_image_extractor
    typer.echo("Starting Option B — Gemini Vision image extraction...")
    run_image_extractor(pdf_dir=pdf_dir, skip_captioning=skip_captioning, delay=delay)
    typer.echo("Option B image extraction complete.")


@cli.command("list-docs")
def list_docs():
    """List all ingested documents and chunk stats."""
    docs = pipeline.list_documents()
    if not docs:
        typer.echo("No documents ingested yet.")
        return
    typer.echo(f"\nIngested Documents ({len(docs)} total):")
    typer.echo("=" * 60)
    for d in docs:
        typer.echo(f"ID: {d['doc_id']} | Chunks: {d['chunks']} | Max Page: {d['max_page']} | File: {d['file_name']}")


@cli.command()
def ask(
    question: str,
    filter_doc_id: Optional[str] = typer.Option(None, "--doc-id"),
):
    """Ask a question via CLI."""
    metadata_filter = {"doc_id": {"$eq": filter_doc_id}} if filter_doc_id else None
    result = pipeline.query(question, metadata_filter=metadata_filter)

    typer.echo(f"\n{'='*60}")
    typer.echo(f"Question: {result.query}")
    typer.echo(f"Rewritten: {result.rewritten_query}")
    typer.echo(f"HyDE: {result.hyde_doc}")
    typer.echo(f"{'='*60}")
    typer.echo(f"\nAnswer:\n{result.answer}")
    typer.echo(f"\nCitations:")
    for c in result.final_chunks:
        typer.echo(f"  - Page {c['page']}, Section: {c['section']} (score: {c['score']:.3f})")
    typer.echo(f"\nLatency: {result.latency_ms['total']}ms")


@cli.command()
def serve(host: str = "0.0.0.0", port: int = 8000):
    """Run FastAPI server."""
    uvicorn.run("main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    cli()
