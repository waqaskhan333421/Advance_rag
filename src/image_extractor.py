"""
Image Extraction Pipeline for Avance RAG
=========================================
Extracts images from all PDF books → captions them with Gemini Vision
→ writes a new searchable PDF corpus (extracted_images_corpus.pdf)
so that the existing IngestionPipeline can embed and index image content.

Usage:
    python -m src.image_extractor
    python -m src.image_extractor --pdf-dir ./data/pdfs --min-size 5000 --output ./data/pdfs/extracted_images_corpus.pdf
"""

import argparse
import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import fitz  # PyMuPDF — already in requirements
from PIL import Image

from src.config import CONFIG

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ---------------------------------------------------------------------------
# Output PDF filename — saved into data/pdfs/ so ingestion picks it up
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_PDF = "./data/pdfs/extracted_images_corpus.pdf"
DEFAULT_IMAGES_DIR = "./data/images"

# ---------------------------------------------------------------------------
# Minimum image byte size to skip tiny icons / decorative bullets
# ---------------------------------------------------------------------------
DEFAULT_MIN_BYTES = 5_000  # ~5 KB


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ExtractedImage:
    source_file: str          # original PDF filename
    doc_id: str               # MD5 hash of source PDF
    page_number: int          # 1-indexed
    image_index: int          # index on that page
    width: int
    height: int
    ext: str                  # png / jpeg / etc.
    image_path: str           # saved path on disk
    caption: str = ""         # filled by Gemini Vision


# ---------------------------------------------------------------------------
# Step 1 — Extract images from all PDFs
# ---------------------------------------------------------------------------

class ImageExtractor:
    """Extract embedded images from PDF files using PyMuPDF."""

    def __init__(self, images_dir: str = DEFAULT_IMAGES_DIR, min_bytes: int = DEFAULT_MIN_BYTES):
        self.images_dir = Path(images_dir)
        self.min_bytes = min_bytes

    def _doc_id(self, pdf_path: Path) -> str:
        return hashlib.md5(pdf_path.read_bytes()).hexdigest()[:16]

    def extract_from_pdf(self, pdf_path: Path) -> List[ExtractedImage]:
        """Extract all qualifying images from a single PDF."""
        doc_id = self._doc_id(pdf_path)
        out_dir = self.images_dir / doc_id
        out_dir.mkdir(parents=True, exist_ok=True)

        doc = fitz.open(str(pdf_path))
        extracted: List[ExtractedImage] = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            image_list = page.get_images(full=True)

            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                except Exception as e:
                    logger.warning(f"  Could not extract image xref={xref} on page {page_num+1}: {e}")
                    continue

                img_bytes = base_image["image"]
                img_ext = base_image.get("ext", "png")
                img_width = base_image.get("width", 0)
                img_height = base_image.get("height", 0)

                # Skip tiny/trivial images
                if len(img_bytes) < self.min_bytes:
                    logger.debug(f"  Skipping small image ({len(img_bytes)} bytes) on page {page_num+1}")
                    continue

                # Save image to disk
                img_filename = f"page{page_num+1}_img{img_index}.{img_ext}"
                img_path = out_dir / img_filename
                img_path.write_bytes(img_bytes)

                extracted.append(ExtractedImage(
                    source_file=pdf_path.name,
                    doc_id=doc_id,
                    page_number=page_num + 1,
                    image_index=img_index,
                    width=img_width,
                    height=img_height,
                    ext=img_ext,
                    image_path=str(img_path),
                ))
                logger.info(f"  Extracted: {img_filename} ({img_width}x{img_height}, {len(img_bytes)//1024}KB)")

        doc.close()
        return extracted

    def extract_all(self, pdf_dir: str) -> List[ExtractedImage]:
        """Extract images from all PDFs in a directory."""
        pdf_dir_path = Path(pdf_dir)
        all_images: List[ExtractedImage] = []

        pdf_files = sorted(pdf_dir_path.glob("*.pdf"))
        # Skip the output corpus itself to avoid recursion on re-runs
        pdf_files = [p for p in pdf_files if p.name != Path(DEFAULT_OUTPUT_PDF).name]

        logger.info(f"Found {len(pdf_files)} PDF(s) to scan for images.")

        for pdf_path in pdf_files:
            logger.info(f"Scanning: {pdf_path.name}")
            images = self.extract_from_pdf(pdf_path)
            logger.info(f"  -> {len(images)} image(s) extracted from {pdf_path.name}")
            all_images.extend(images)

        logger.info(f"Total images extracted: {len(all_images)}")
        return all_images


# ---------------------------------------------------------------------------
# Step 2 — Caption images with Gemini Vision
# ---------------------------------------------------------------------------

class GeminiVisionCaptioner:
    """Send images to Gemini Vision API and get rich text descriptions."""

    VISION_PROMPT = (
        "You are analyzing an image extracted from an Islamic book. "
        "Provide a detailed, accurate description of this image covering:\n"
        "1. What type of image it is (calligraphy, diagram, chart, illustration, table, photograph, etc.)\n"
        "2. The main content and subject matter\n"
        "3. Any visible Arabic text -- transcribe and translate if possible\n"
        "4. Colors, style, and visual composition\n"
        "5. How this image relates to Islamic topics if applicable\n\n"
        "Be specific and thorough. This description will be used for semantic search."
    )

    def __init__(self):
        api_key = CONFIG.get_gemini_api_key()
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set. Cannot use Gemini Vision for captioning.")
        from google import genai
        self.client = genai.Client(api_key=api_key)
        self.vision_model = CONFIG.models.gemini.llm_model  # use model from config.yaml

    def caption(self, image_path: str, retries: int = 3) -> str:
        """Generate a caption for an image using Gemini Vision."""
        img_path = Path(image_path)
        if not img_path.exists():
            logger.warning(f"Image file not found: {image_path}")
            return "Image file not available for captioning."

        try:
            pil_image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Could not open image {image_path}: {e}")
            return "Could not open image for captioning."

        for attempt in range(retries):
            try:
                response = self.client.models.generate_content(
                    model=self.vision_model,
                    contents=[self.VISION_PROMPT, pil_image],
                )
                caption = response.text.strip() if response.text else ""
                if caption:
                    return caption
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    wait = (2 ** attempt) * 5
                    logger.warning(f"Rate limit hit, retrying in {wait}s... (attempt {attempt+1}/{retries})")
                    time.sleep(wait)
                else:
                    logger.error(f"Gemini Vision failed for {image_path}: {e}")
                    return f"[Captioning failed: {e}]"

        return "[Caption unavailable after retries]"

    def caption_all(self, images: List[ExtractedImage], delay_between: float = 1.0) -> List[ExtractedImage]:
        """Caption all images, adding captions in-place."""
        total = len(images)
        logger.info(f"Captioning {total} image(s) with Gemini Vision...")

        for i, img in enumerate(images, 1):
            logger.info(f"  [{i}/{total}] Captioning: {Path(img.image_path).name}")
            img.caption = self.caption(img.image_path)
            logger.info(f"  Caption preview: {img.caption[:120]}...")
            time.sleep(delay_between)

        return images


# ---------------------------------------------------------------------------
# Step 3 — Write structured PDF corpus
# ---------------------------------------------------------------------------

class ImageCorpusPDFWriter:
    """Write extracted image metadata + captions into a structured PDF corpus."""

    def __init__(self, output_path: str = DEFAULT_OUTPUT_PDF):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, images: List[ExtractedImage]) -> str:
        """Generate the image corpus PDF. Returns the output path."""
        from fpdf import FPDF

        if not images:
            logger.warning("No images to write into corpus PDF.")
            return ""

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_margins(left=15, top=15, right=15)

        for img in images:
            pdf.add_page()

            # Header block
            pdf.set_font("Helvetica", style="B", size=13)
            pdf.set_text_color(20, 60, 120)
            pdf.cell(0, 8, "[IMAGE RECORD]", new_x="LMARGIN", new_y="NEXT")

            meta_lines = [
                ("Source File",  img.source_file),
                ("Page Number",  str(img.page_number)),
                ("Image Index",  str(img.image_index)),
                ("Dimensions",   f"{img.width} x {img.height} px"),
                ("Format",       img.ext.upper()),
                ("Doc ID",       img.doc_id),
                ("Image Path",   img.image_path),
            ]

            for label, value in meta_lines:
                pdf.set_font("Helvetica", style="B", size=9)
                pdf.set_text_color(50, 50, 50)
                pdf.cell(32, 6, f"{label}:", new_x="RIGHT", new_y="TOP")
                pdf.set_font("Helvetica", size=9)
                safe_value = value[:90] if len(value) > 90 else value
                safe_value = safe_value.encode("latin-1", errors="replace").decode("latin-1")
                pdf.cell(0, 6, safe_value, new_x="LMARGIN", new_y="NEXT")

            pdf.ln(4)

            # Caption block
            pdf.set_font("Helvetica", style="B", size=11)
            pdf.set_text_color(20, 100, 60)
            pdf.cell(0, 7, "[DESCRIPTION]", new_x="LMARGIN", new_y="NEXT")

            pdf.set_font("Helvetica", size=10)
            pdf.set_text_color(30, 30, 30)
            safe_caption = img.caption.encode("latin-1", errors="replace").decode("latin-1")
            pdf.multi_cell(0, 6, safe_caption)

            pdf.ln(4)

            # Searchable metadata tag line (used by chunker for metadata pickup)
            pdf.set_font("Helvetica", style="I", size=8)
            pdf.set_text_color(120, 120, 120)
            tags = (
                f"[META] source_pdf={img.source_file} | page={img.page_number} | "
                f"image_index={img.image_index} | doc_id={img.doc_id} | "
                f"image_path={img.image_path} | width={img.width} | height={img.height}"
            )
            safe_tags = tags.encode("latin-1", errors="replace").decode("latin-1")
            pdf.multi_cell(0, 5, safe_tags)

        output_str = str(self.output_path)
        pdf.output(output_str)
        logger.info(f"Image corpus PDF written: {output_str} ({len(images)} records)")
        return output_str


# ---------------------------------------------------------------------------
# Step 4 — Summary report
# ---------------------------------------------------------------------------

def print_summary(images: List[ExtractedImage], output_pdf: str):
    from collections import Counter
    source_counts = Counter(img.source_file for img in images)

    print("\n" + "=" * 65)
    print("  IMAGE EXTRACTION COMPLETE")
    print("=" * 65)
    print(f"  Total images extracted : {len(images)}")
    print(f"  Output corpus PDF      : {output_pdf}")
    print(f"\n  Breakdown by source PDF:")
    for fname, count in source_counts.most_common():
        print(f"    * {fname:<50} {count:>4} image(s)")
    print("=" * 65)
    print("\n  Next step: Run your ingestion pipeline to embed the")
    print("  new corpus PDF. It will be picked up automatically.")
    print("  Example:  python main.py ingest\n")


# ---------------------------------------------------------------------------
# Main pipeline runner
# ---------------------------------------------------------------------------

def run(
    pdf_dir: str = None,
    output_pdf: str = DEFAULT_OUTPUT_PDF,
    images_dir: str = DEFAULT_IMAGES_DIR,
    min_bytes: int = DEFAULT_MIN_BYTES,
    skip_captioning: bool = False,
    delay: float = 1.0,
):
    """
    Full pipeline:
      1. Extract images from all PDFs
      2. Caption with Gemini Vision
      3. Write corpus PDF
    """
    pdf_dir = pdf_dir or CONFIG.paths.documents_dir

    # Step 1: Extract
    extractor = ImageExtractor(images_dir=images_dir, min_bytes=min_bytes)
    images = extractor.extract_all(pdf_dir)

    if not images:
        logger.warning("No images found across all PDFs. Nothing to write.")
        return

    # Step 2: Caption
    if not skip_captioning:
        captioner = GeminiVisionCaptioner()
        images = captioner.caption_all(images, delay_between=delay)
    else:
        logger.info("Captioning skipped. Using placeholder captions.")
        for img in images:
            img.caption = (
                f"Image extracted from '{img.source_file}', page {img.page_number}, "
                f"index {img.image_index}. Dimensions: {img.width}x{img.height}px."
            )

    # Step 3: Write corpus PDF
    writer = ImageCorpusPDFWriter(output_path=output_pdf)
    result_path = writer.write(images)

    # Step 4: Summary
    if result_path:
        print_summary(images, result_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract images from PDF books, caption with Gemini Vision, write searchable corpus PDF."
    )
    parser.add_argument("--pdf-dir", default=None,
                        help="Directory of source PDF books (default: from config.yaml)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PDF,
                        help=f"Output corpus PDF path (default: {DEFAULT_OUTPUT_PDF})")
    parser.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR,
                        help=f"Directory to save raw extracted images (default: {DEFAULT_IMAGES_DIR})")
    parser.add_argument("--min-size", type=int, default=DEFAULT_MIN_BYTES,
                        help=f"Min image byte size to include (default: {DEFAULT_MIN_BYTES})")
    parser.add_argument("--skip-captioning", action="store_true",
                        help="Skip Gemini Vision captioning (for testing extraction only)")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Seconds between Gemini Vision API calls (default: 1.0)")

    args = parser.parse_args()

    run(
        pdf_dir=args.pdf_dir,
        output_pdf=args.output,
        images_dir=args.images_dir,
        min_bytes=args.min_size,
        skip_captioning=args.skip_captioning,
        delay=args.delay,
    )
