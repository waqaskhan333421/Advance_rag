"""PDF parsing, semantic chunking, and metadata extraction."""

import hashlib
import logging
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import fitz  # pymupdf
from PIL import Image

from src.config import CONFIG

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    page_number: int
    section_title: Optional[str]
    metadata: dict


class PDFParser:
    """Parse PDFs with PyMuPDF; fallback to RapidOCR/pytesseract for scanned pages."""

    def __init__(self, ocr_enabled: bool = True):
        self.ocr_enabled = ocr_enabled
        self.ocr_engine = None
        self.ocr_type = None

        try:
            from rapidocr_onnxruntime import RapidOCR
            self.ocr_engine = RapidOCR()
            self.ocr_type = "rapidocr"
            logger.info("Initialized RapidOCR engine for scanned PDF fallback")
        except ImportError:
            try:
                import pytesseract
                self.pytesseract = pytesseract
                self.ocr_type = "pytesseract"
            except ImportError:
                logger.warning("No OCR engine available; OCR fallback disabled")

    def _ocr_page(self, page) -> str:
        """Perform OCR on a PyMuPDF page."""
        try:
            pix = page.get_pixmap(dpi=200)
            if self.ocr_type == "rapidocr" and self.ocr_engine:
                result, _ = self.ocr_engine(pix.tobytes("png"))
                if result:
                    return "\n".join([line[1] for line in result])
            elif self.ocr_type == "pytesseract" and getattr(self, "pytesseract", None):
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                return self.pytesseract.image_to_string(img)
        except Exception as e:
            logger.warning(f"OCR failed for page: {e}")
        return ""

    def parse(self, pdf_path: str) -> List[dict]:
        """Extract pages with text, tables, and layout info."""
        doc = fitz.open(pdf_path)
        doc_id = self._doc_id(pdf_path)
        file_name = Path(pdf_path).name
        pages = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()

            # Fallback OCR if page looks scanned (very little extractable text)
            if self.ocr_enabled and self.ocr_type and len(text.strip()) < 50:
                ocr_text = self._ocr_page(page)
                if ocr_text.strip():
                    text = ocr_text

            # Detect section headers via font size heuristics
            blocks = page.get_text("dict")["blocks"]
            headers = []
            for b in blocks:
                if "lines" not in b:
                    continue
                for line in b["lines"]:
                    for span in line["spans"]:
                        if span["size"] > 12 and span["flags"] & 2 ** 4:  # bold-ish
                            headers.append(span["text"].strip())

            section_title = headers[0] if headers else None

            pages.append({
                "doc_id": doc_id,
                "file_name": file_name,
                "page_number": page_num + 1,
                "text": text,
                "section_title": section_title,
            })

        doc.close()
        logger.info(f"Parsed {pdf_path}: {len(pages)} pages")
        return pages

    @staticmethod
    def _doc_id(pdf_path: str) -> str:
        return hashlib.md5(Path(pdf_path).read_bytes()).hexdigest()[:16]


class SemanticChunker:
    """Recursive character chunker that preserves section headers & page numbers."""

    def __init__(self):
        self.chunk_size = CONFIG.chunking.chunk_size
        self.chunk_overlap = CONFIG.chunking.chunk_overlap
        self.separators = CONFIG.chunking.separators

    def chunk(self, pages: List[dict]) -> List[Chunk]:
        """Split pages into overlapping chunks with rich metadata."""
        chunks: List[Chunk] = []
        global_idx = 0

        for page in pages:
            text = page["text"]
            if not text.strip():
                continue

            page_chunks = self._split_text(text)
            for i, chunk_text in enumerate(page_chunks):
                chunk_id = f"{page['doc_id']}_p{page['page_number']}_c{i}"
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    doc_id=page["doc_id"],
                    text=chunk_text,
                    page_number=page["page_number"],
                    section_title=page.get("section_title"),
                    metadata={
                        "doc_id": page["doc_id"],
                        "file_name": page.get("file_name", ""),
                        "page_number": page["page_number"],
                        "section_title": page.get("section_title") or "",
                        "chunk_index": global_idx,
                    },
                ))
                global_idx += 1

        logger.info(f"Produced {len(chunks)} chunks")
        return chunks

    def _split_text(self, text: str) -> List[str]:
        """Recursive splitting by separators."""
        chunks = []
        self._recursive_split(text, 0, chunks)
        return chunks

    def _recursive_split(self, text: str, sep_idx: int, chunks: List[str]):
        sep = self.separators[sep_idx] if sep_idx < len(self.separators) else ""
        parts = text.split(sep) if sep else list(text)

        current = ""
        for part in parts:
            candidate = (current + sep + part).strip() if current else part.strip()
            if len(candidate.split()) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # Handle oversized single piece
                if len(part.split()) > self.chunk_size and sep_idx + 1 < len(self.separators):
                    self._recursive_split(part, sep_idx + 1, chunks)
                else:
                    current = part.strip()

        if current:
            # Apply overlap for continuity
            if chunks and self.chunk_overlap > 0:
                words = current.split()
                overlap_words = words[:self.chunk_overlap]
                overlap_text = " ".join(overlap_words)
                chunks.append(overlap_text + " " + current if overlap_text else current)
            else:
                chunks.append(current)


class IngestionPipeline:
    """End-to-end ingestion: parse → chunk → embed → store."""

    def __init__(self):
        self.parser = PDFParser()
        self.chunker = SemanticChunker()
        self.gemini = None  # set externally to avoid circular import

    def ingest_pdf(self, pdf_path: str):
        """Parse and chunk a single PDF."""
        pages = self.parser.parse(pdf_path)
        return self.chunker.chunk(pages)

    def ingest_directory(
        self,
        directory: str,
        skip_doc_ids: Optional[set] = None,
        skip_doc_names: Optional[set] = None,
    ) -> List[Chunk]:
        """Ingest PDFs in a directory, skipping already ingested files."""
        all_chunks = []
        skip_ids = skip_doc_ids or set()
        skip_names = skip_doc_names or set()

        for fname in os.listdir(directory):
            if fname.lower().endswith(".pdf"):
                path = os.path.join(directory, fname)
                try:
                    file_bytes = Path(path).read_bytes()
                    doc_id = hashlib.md5(file_bytes).hexdigest()[:16]
                except Exception:
                    doc_id = ""

                if fname.lower() in skip_names or doc_id in skip_ids:
                    logger.info(f"Skipping '{fname}' (ID: {doc_id}) - already ingested")
                    continue

                logger.info(f"Ingesting new document: '{fname}' (ID: {doc_id})")
                all_chunks.extend(self.ingest_pdf(path))
        return all_chunks
