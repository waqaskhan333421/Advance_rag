"""image_to_pdf.py

Smart image extraction pipeline for RAG system.

For each image under data/images/<book_name>/:
  - Parses book name, page number, image index from folder/filename
  - Runs Tesseract OCR
  - Classifies image as 'text' (>=3 usable lines, >=15 words) or 'visual' (chart/map/diagram)
  - Text images  -> written to data/pdfs/extracted_data.pdf with rich metadata header
  - Visual images -> copied to data/visual_assets/ as PNG for RAG display
  - All records   -> written to data/extracted_metadata.json

Dependencies:
  pip install pytesseract Pillow fpdf2
  Tesseract-OCR: https://github.com/UB-Mannheim/tesseract/wiki
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# PIL
try:
    from PIL import Image
except ImportError:
    print("Pillow not installed.  Run: pip install Pillow")
    sys.exit(1)

# pytesseract
try:
    import pytesseract
except ImportError:
    print("pytesseract not installed.  Run: pip install pytesseract")
    sys.exit(1)

# Locate Tesseract executable
_tess = os.getenv("TESSERACT_PATH") or shutil.which("tesseract")
if not _tess:
    _default = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.isfile(_default):
        _tess = _default
    else:
        print("Tesseract not found. Install from https://github.com/UB-Mannheim/tesseract/wiki")
        sys.exit(1)
pytesseract.pytesseract.tesseract_cmd = _tess

# fpdf2
try:
    from fpdf import FPDF
except ImportError:
    print("fpdf2 not installed.  Run: pip install fpdf2")
    sys.exit(1)

# matplotlib font (DejaVu Sans - Unicode)
try:
    import matplotlib as _mpl
    _font_dir = Path(_mpl.__file__).parent / "mpl-data" / "fonts" / "ttf"
    FONT_REGULAR = str(_font_dir / "DejaVuSans.ttf")
    FONT_BOLD    = str(_font_dir / "DejaVuSans-Bold.ttf")
    if not os.path.isfile(FONT_REGULAR):
        raise FileNotFoundError
except Exception:
    FONT_REGULAR = FONT_BOLD = None  # fall back to Helvetica (ASCII only)


# ============================================================
# Configuration
# ============================================================

MIN_USABLE_LINES  = 3    # minimum non-trivial lines to keep as text image
MIN_LINE_LENGTH   = 10   # chars per line to count as "usable"
MIN_WORD_COUNT    = 15   # total words required

# Visual asset: fails text filter AND is large enough to be a real chart/diagram
VISUAL_MIN_BYTES  = 80_000   # 80 KB

IMAGE_EXTENSIONS  = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif"}


# ============================================================
# Helpers
# ============================================================

def parse_page_info(stem: str) -> tuple[int, int]:
    """Return (page_number, image_index) from e.g. 'page22_img0'."""
    m = re.match(r"page(\d+)_img(\d+)", stem, re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2))
    return 0, 0


def ocr_image(image_path: Path) -> str:
    """Run Tesseract OCR and return raw text (empty string on failure)."""
    try:
        return pytesseract.image_to_string(Image.open(image_path))
    except Exception as exc:
        print(f"  [OCR error] {image_path.name}: {exc}")
        return ""


def usable_lines(text: str) -> list[str]:
    """Return lines long enough to be considered real content."""
    return [ln.strip() for ln in text.splitlines()
            if len(ln.strip()) >= MIN_LINE_LENGTH]


def is_text_rich(text: str) -> bool:
    lines = usable_lines(text)
    words = text.split()
    return len(lines) >= MIN_USABLE_LINES and len(words) >= MIN_WORD_COUNT


def is_visual_asset(text: str, file_size: int) -> bool:
    return not is_text_rich(text) and file_size >= VISUAL_MIN_BYTES


def extract_title(text: str) -> Optional[str]:
    lines = usable_lines(text)
    return lines[0] if lines else None


def get_image_dimensions(image_path: Path) -> tuple[int, int]:
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception:
        return 0, 0


def clean_text(text: str) -> str:
    """Collapse excessive blank lines from OCR output."""
    lines = text.splitlines()
    cleaned, prev_blank = [], False
    for ln in lines:
        blank = ln.strip() == ""
        if blank and prev_blank:
            continue
        cleaned.append(ln)
        prev_blank = blank
    return "\n".join(cleaned).strip()


# ============================================================
# PDF builder
# ============================================================

def make_pdf() -> FPDF:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    if FONT_REGULAR:
        pdf.add_font("DejaVu", "",  FONT_REGULAR)
        pdf.add_font("DejaVu", "B", FONT_BOLD)
    return pdf


def _set_font(pdf: FPDF, bold: bool = False, size: int = 11):
    name  = "DejaVu" if FONT_REGULAR else "Helvetica"
    style = "B" if bold else ""
    pdf.set_font(name, style=style, size=size)


def add_text_page(pdf: FPDF, meta: dict, text: str):
    """Add one page to the PDF for a text-rich image."""
    pdf.add_page()

    # Book title (bold, large)
    _set_font(pdf, bold=True, size=13)
    pdf.cell(0, 8, f"Book:   {meta['book_name']}",
             new_x="LMARGIN", new_y="NEXT")

    # Page / image index
    _set_font(pdf, bold=False, size=11)
    pdf.cell(0, 6,
             f"Page:   {meta['page_number']}   |   Image index: {meta['image_index']}",
             new_x="LMARGIN", new_y="NEXT")

    # Title (first meaningful OCR line)
    if meta.get("title"):
        pdf.cell(0, 6, f"Title:  {meta['title']}",
                 new_x="LMARGIN", new_y="NEXT")

    # File info
    pdf.cell(0, 6,
             f"Size:   {meta['file_size_bytes']:,} bytes   |   "
             f"Dimensions: {meta['dimensions']}",
             new_x="LMARGIN", new_y="NEXT")

    # Horizontal rule
    pdf.ln(2)
    pdf.set_draw_color(120, 120, 120)
    pdf.set_line_width(0.4)
    pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + pdf.epw, pdf.get_y())
    pdf.ln(4)

    # OCR text body
    _set_font(pdf, bold=False, size=10)
    pdf.multi_cell(0, 5, text or "[No text extracted]")


# ============================================================
# Main pipeline
# ============================================================

def main():
    base_dir         = Path(__file__).parent
    images_dir       = base_dir / "data" / "images"
    pdfs_dir         = base_dir / "data" / "pdfs"
    visual_dir       = base_dir / "data" / "visual_assets"
    output_pdf_path  = pdfs_dir / "extracted_data.pdf"
    output_json_path = base_dir / "data" / "extracted_metadata.json"

    pdfs_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)

    # Collect all image files
    image_files = sorted(
        p for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_files:
        print("No image files found under", images_dir)
        sys.exit(0)

    pdf      = make_pdf()
    records  = []
    counts   = {"text": 0, "visual": 0, "skipped": 0}
    total    = len(image_files)

    for idx, image_path in enumerate(image_files, 1):
        book_name        = image_path.parent.name
        page_no, img_idx = parse_page_info(image_path.stem)
        file_size        = image_path.stat().st_size
        width, height    = get_image_dimensions(image_path)
        dimensions       = f"{width}x{height}"

        print(f"[{idx}/{total}] {book_name} / {image_path.name}", end=" ... ", flush=True)

        raw_text = ocr_image(image_path)
        text     = clean_text(raw_text)
        title    = extract_title(text)

        record: dict = {
            "book_name":         book_name,
            "page_number":       page_no,
            "image_index":       img_idx,
            "filename":          image_path.name,
            "dimensions":        dimensions,
            "file_size_bytes":   file_size,
            "title":             title,
            "ocr_text":          text,
            "image_type":        None,
            "visual_asset_path": None,
        }

        if is_text_rich(text):
            # Text-rich image -> add to PDF
            record["image_type"] = "text"
            add_text_page(pdf, record, text)
            counts["text"] += 1
            print("text OK")

        elif is_visual_asset(text, file_size):
            # Visual asset (chart/map/diagram) -> save as PNG
            safe_book  = re.sub(r"[^\w\-]", "_", book_name)
            asset_name = f"{safe_book}__{image_path.stem}.png"
            asset_path = visual_dir / asset_name
            try:
                with Image.open(image_path) as img:
                    img.save(str(asset_path), "PNG")
                rel = str(asset_path.relative_to(base_dir)).replace("\\", "/")
                record["image_type"]        = "visual"
                record["visual_asset_path"] = rel
                counts["visual"] += 1
                print(f"visual -> {asset_name}")
            except Exception as e:
                record["image_type"] = "visual"
                counts["visual"] += 1
                print(f"visual (save error: {e})")

        else:
            # Blank / cover / too little content
            record["image_type"] = "skipped"
            counts["skipped"] += 1
            print("skipped (blank/cover)")

        records.append(record)

    # Save PDF
    pdf.output(str(output_pdf_path))
    print(f"\nPDF written  -> {output_pdf_path}")

    # Save JSON
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"JSON written -> {output_json_path}")

    # Summary
    print("\n========= Summary =========")
    print(f"  Total images processed : {total}")
    print(f"  Text-rich  (-> PDF)    : {counts['text']}")
    print(f"  Visual assets (-> PNG) : {counts['visual']}")
    print(f"  Skipped (blank/cover)  : {counts['skipped']}")
    print(f"  Visual assets folder   : {visual_dir}")
    print(f"  Metadata JSON          : {output_json_path}")
    print("============================")


if __name__ == "__main__":
    main()


