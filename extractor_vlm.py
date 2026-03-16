"""
PDF Extraction Pipeline — extractor_vlm.py
(Hybrid: Digital + Mistral VLM for OCR)

Page routing:
  1. Digital (clean text)     → pdfplumber + PyMuPDF
  2. Font-corrupt (cid/ctrl)  → strip-split render → Mistral OCR
  3. Image/scanned page       → OSD rotation → adaptive OpenCV detection
                                → per-crop Mistral OCR
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import cv2
import fitz
import numpy as np
import pdfplumber
import pytesseract
from dotenv import load_dotenv
from mistralai.client import Mistral
from PIL import Image
from pytesseract import Output

# ── Setup ─────────────────────────────────────────────────────────────────────
load_dotenv()
client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

# ── Tunables ──────────────────────────────────────────────────────────────────
DIGITAL_TEXT_THRESHOLD  = 50
TABLE_MIN_ROWS          = 2
CID_RATIO_THRESHOLD     = 0.05
OCR_DPI_SCALE           = 2
ROW_STRIP_SIZE          = 15     # rows per strip for corrupt-font pages
HEADER_ROWS             = 3      # merged header rows to include in strip 1
TABLE_MIN_W             = 100    # min pixel width for floating table detection
TABLE_MIN_H             = 100
TABLE_MAX_W             = 600    # max pixel width (rejects merged blobs)
TABLE_MAX_H             = 900
MISTRAL_MODEL           = "mistral-ocr-latest"
# ─────────────────────────────────────────────────────────────────────────────

_CID_RE = re.compile(r"\(cid:\d+\)")


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class PageResult:
    page_num: int
    mode: str
    markdown: str
    tables: list[dict]
    warnings: list[str] = field(default_factory=list)


@dataclass
class ExtractionResult:
    source_path: str
    total_pages: int
    pages: list[PageResult]
    metadata: dict

    def to_markdown(self) -> str:
        parts = [f"# Document: {Path(self.source_path).name}\n"]
        for k, v in self.metadata.items():
            if v:
                parts.append(f"**{k}:** {v}  ")
        parts.append("\n---\n")
        for page in self.pages:
            parts.append(f"\n## Page {page.page_num} _{page.mode}_\n")
            parts.append(page.markdown.strip())
            parts.append(f"\n\n---\n")
        return "\n".join(parts)

    def to_json(self) -> str:
        return json.dumps(
            {"source_path": self.source_path, "total_pages": self.total_pages,
             "metadata": self.metadata, "pages": [asdict(p) for p in self.pages]},
            indent=2,
        )


# ── Shared helpers ────────────────────────────────────────────────────────────

def _has_cid_garbage(text: str) -> bool:
    tokens = text.split()
    if not tokens:
        return False
    if (len(_CID_RE.findall(text)) / len(tokens)) > CID_RATIO_THRESHOLD:
        return True
    control_chars = sum(1 for c in text if ord(c) < 32 and c not in '\t\n\r')
    if len(text) > 0 and (control_chars / len(text)) > 0.01:
        return True
    return False


def _is_digital_page(page: fitz.Page) -> bool:
    return len(page.get_text("text").strip()) >= DIGITAL_TEXT_THRESHOLD


def _rects_overlap(r1: tuple, r2: tuple) -> bool:
    return not (r1[2] <= r2[0] or r1[0] >= r2[2] or r1[3] <= r2[1] or r1[1] >= r2[3])


def _pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _mistral_ocr_image(img: Image.Image) -> tuple[str, list]:
    """Send a PIL image to Mistral OCR. Returns (markdown, tables)."""
    b64 = _pil_to_base64(img)
    response = client.ocr.process(
        model=MISTRAL_MODEL,
        document={"type": "image_url", "image_url": f"data:image/png;base64,{b64}"},
        table_format="markdown",
        include_image_base64=True,
    )
    page_data = response.pages[0] if response.pages else None
    if not page_data:
        return "", []
    if page_data.tables:
        md = "\n\n".join(t.content for t in page_data.tables)
    else:
        md = page_data.markdown or ""
    return md, page_data.tables or []


def _render_page(fitz_page: fitz.Page, scale: int = OCR_DPI_SCALE) -> Image.Image:
    """Render page to PIL image with OSD-based rotation correction."""
    mat = fitz.Matrix(scale, scale).prerotate(fitz_page.rotation)
    pix = fitz_page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    pil_raw = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # OSD rotation correction
    try:
        osd = pytesseract.image_to_osd(pil_raw, output_type=Output.DICT)
        angle = osd.get("rotate", 0)
        ANGLE_FIX = {0: 0, 90: 90, 180: 180, 270: -90}
        fix = ANGLE_FIX.get(angle, 0)
        if fix:
            print(f"    [OSD] Correcting {angle}° rotation")
            return pil_raw.rotate(fix, expand=True)
    except Exception:
        pass
    return pil_raw


# ── Route 2: Font-corrupt page → strip-split Mistral OCR ─────────────────────

def _get_table_row_bboxes(plumber_page) -> list[tuple]:
    """Get (y0, y1) per row using pdfplumber graphics — immune to font corruption."""
    try:
        tables = plumber_page.find_tables()
        if not tables:
            return []
        biggest = max(tables, key=lambda t: len(t.extract() or []))
        cells = biggest.cells
        rows_dict: dict[int, list] = {}
        for cell in cells:
            row_key = round(cell[1])
            rows_dict.setdefault(row_key, []).append(cell)
        sorted_rows = sorted(rows_dict.items())
        return [(min(c[1] for c in row), max(c[3] for c in row))
                for _, row in sorted_rows]
    except Exception:
        return []


def _get_strip_image(
    fitz_page: fitz.Page,
    plumber_page,
    data_indices: list[int],
    scale: int = OCR_DPI_SCALE,
    include_header: bool = False,
) -> Image.Image:
    """Render a horizontal strip of table rows as a PIL image."""
    ROTATION_FIX = {0: 0, 90: 90, 180: 180, 270: -90}
    mat = fitz.Matrix(scale, scale).prerotate(fitz_page.rotation)
    pil_fix = ROTATION_FIX.get(fitz_page.rotation, 0)

    tables = plumber_page.find_tables()
    if not tables:
        return _render_page(fitz_page, scale)
    target = max(tables, key=lambda t: len(t.extract() or []))

    x0 = target.bbox[0]
    x1 = target.bbox[2]
    strip_y0 = target.rows[data_indices[0]].bbox[1]
    strip_y1 = target.rows[data_indices[-1]].bbox[3]

    s_pix = fitz_page.get_pixmap(matrix=mat, clip=fitz.Rect(x0, strip_y0, x1, strip_y1))
    s_img = Image.frombytes("RGB", [s_pix.width, s_pix.height], s_pix.samples)
    if pil_fix:
        s_img = s_img.rotate(pil_fix, expand=True)

    if not include_header:
        return s_img

    header_y0 = target.rows[0].bbox[1]
    header_y1 = target.rows[HEADER_ROWS - 1].bbox[3]
    h_pix = fitz_page.get_pixmap(matrix=mat, clip=fitz.Rect(x0, header_y0, x1, header_y1))
    h_img = Image.frombytes("RGB", [h_pix.width, h_pix.height], h_pix.samples)
    if pil_fix:
        h_img = h_img.rotate(pil_fix, expand=True)

    combined = Image.new("RGB", (max(h_img.width, s_img.width),
                                  h_img.height + s_img.height), (255, 255, 255))
    combined.paste(h_img, (0, 0))
    combined.paste(s_img, (0, h_img.height))
    return combined


def _merge_strip_markdowns(strips: list[str]) -> str:
    """Merge multiple markdown tables — keep header from strip 0, data rows from rest."""
    if not strips:
        return ""
    if len(strips) == 1:
        return strips[0]

    def split_header_data(md: str):
        lines = [l for l in md.splitlines() if l.strip()]
        sep_idx = next((i for i, l in enumerate(lines)
                        if re.match(r"^\|[\s\-|]+\|$", l)), None)
        if sep_idx is None:
            return lines, []
        return lines[:sep_idx + 1], lines[sep_idx + 1:]

    header_lines, first_data = split_header_data(strips[0])
    all_data = list(first_data)
    for md in strips[1:]:
        _, data = split_header_data(md)
        all_data.extend(data)
    return "\n".join(header_lines + all_data)


def _ocr_corrupt_page(fitz_page: fitz.Page, plumber_page) -> str:
    """Strip-split a font-corrupt page and OCR via Mistral."""
    row_bboxes = _get_table_row_bboxes(plumber_page)
    total_rows = len(row_bboxes)
    print(f"    [Strip OCR] {total_rows} rows detected")

    if total_rows <= ROW_STRIP_SIZE or not row_bboxes:
        print(f"    [Strip OCR] Single shot (≤{ROW_STRIP_SIZE} rows)")
        img = _render_page(fitz_page)
        md, _ = _mistral_ocr_image(img)
        return md

    tables = plumber_page.find_tables()
    if not tables:
        img = _render_page(fitz_page)
        md, _ = _mistral_ocr_image(img)
        return md

    data_indices = list(range(HEADER_ROWS, total_rows))
    strip_groups = [data_indices[i:i + ROW_STRIP_SIZE]
                    for i in range(0, len(data_indices), ROW_STRIP_SIZE)]
    total_strips = len(strip_groups)
    print(f"    [Strip OCR] {total_strips} strips of {ROW_STRIP_SIZE} rows each")

    strip_markdowns = []
    for n, group in enumerate(strip_groups, 1):
        print(f"    [Strip OCR] Strip {n}/{total_strips} (rows {group[0]}–{group[-1]})...")
        img = _get_strip_image(fitz_page, plumber_page, group,
                               include_header= True)
        md, _ = _mistral_ocr_image(img)
        strip_markdowns.append(md.strip())

    return _merge_strip_markdowns(strip_markdowns)


# ── Route 3: Image/scanned page → OpenCV detection → per-crop Mistral OCR ────

def _detect_floating_tables(pil_img: Image.Image) -> list[tuple]:
    """
    Detect floating table bounding boxes via adaptive threshold + contour detection.
    Returns list of (x, y, w, h) in image pixel coordinates.
    """
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold — handles varying contrast across image
    img_bin = cv2.adaptiveThreshold(
        img_gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15, C=4
    )

    # Detect horizontal and vertical lines
    h_len = img_gray.shape[1] // 30
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    h_lines = cv2.dilate(cv2.erode(img_bin, h_kernel, iterations=2), h_kernel, iterations=2)

    v_len = img_gray.shape[0] // 30
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    v_lines = cv2.dilate(cv2.erode(img_bin, v_kernel, iterations=2), v_kernel, iterations=2)

    grid = cv2.addWeighted(h_lines, 0.5, v_lines, 0.5, 0.0)
    _, grid = cv2.threshold(grid, 0, 255, cv2.THRESH_BINARY)

    merge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    merged = cv2.dilate(grid, merge_kernel, iterations=1)

    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    table_boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / h if h > 0 else 0
        if (TABLE_MIN_W < w < TABLE_MAX_W and
                TABLE_MIN_H < h < TABLE_MAX_H and
                0.2 < aspect < 6.0):
            table_boxes.append((x, y, w, h))

    # Remove overlapping boxes
    def overlap_ratio(b1, b2):
        ix0 = max(b1[0], b2[0]); iy0 = max(b1[1], b2[1])
        ix1 = min(b1[0]+b1[2], b2[0]+b2[2]); iy1 = min(b1[1]+b1[3], b2[1]+b2[3])
        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0
        return ((ix1-ix0)*(iy1-iy0)) / min(b1[2]*b1[3], b2[2]*b2[3])

    filtered = []
    for box in sorted(table_boxes, key=lambda b: b[2]*b[3], reverse=True):
        if not any(overlap_ratio(box, kept) > 0.3 for kept in filtered):
            filtered.append(box)

    return sorted(filtered, key=lambda b: (b[1] // 100, b[0]))


def _ocr_image_page(fitz_page: fitz.Page) -> str:
    """Detect and OCR floating tables on an image-based page."""
    pil_img = _render_page(fitz_page)
    table_boxes = _detect_floating_tables(pil_img)
    print(f"    [Image OCR] Detected {len(table_boxes)} floating tables")

    if not table_boxes:
        # Fallback: send full page
        print(f"    [Image OCR] No tables detected — sending full page")
        md, _ = _mistral_ocr_image(pil_img)
        return md

    PADDING = 10
    table_markdowns = []
    for i, (x, y, w, h) in enumerate(table_boxes, 1):
        print(f"    [Image OCR] Table {i}/{len(table_boxes)}...")
        crop = pil_img.crop((
            max(0, x - PADDING), max(0, y - PADDING),
            min(pil_img.width, x + w + PADDING),
            min(pil_img.height, y + h + PADDING),
        ))
        md, _ = _mistral_ocr_image(crop)
        if md.strip():
            table_markdowns.append(f"### Detected Table {i}\n\n{md}")

    return "\n\n---\n\n".join(table_markdowns)


# ── Route 1: Digital page ─────────────────────────────────────────────────────

def _extract_digital_text(fitz_page: fitz.Page, table_bboxes: list[tuple]) -> str:
    blocks = fitz_page.get_text("blocks")
    valid = []
    for b in blocks:
        if b[6] != 0 or not b[4].strip():
            continue
        if not any(_rects_overlap((b[0], b[1], b[2], b[3]), tb) for tb in table_bboxes):
            valid.append(b)
    return "\n\n".join(b[4].strip() for b in
                       sorted(valid, key=lambda b: (round(b[1] / 20), b[0])))


def _extract_tables(plumber_page) -> list[dict]:
    results = []
    try:
        found_tables = plumber_page.find_tables()
    except Exception:
        return results
    for t in found_tables:
        table = t.extract()
        if not table or len(table) < TABLE_MIN_ROWS:
            continue
        clean = lambda c: str(c).replace("\n", " ").strip() if c is not None else ""
        header = table[0]
        md = ["| " + " | ".join(clean(c) for c in header) + " |",
              "| " + " | ".join("---" for _ in header) + " |"]
        for row in table[1:]:
            padded = list(row) + [""] * (len(header) - len(row))
            md.append("| " + " | ".join(clean(c) for c in padded) + " |")
        results.append({"markdown": "\n".join(md), "bbox": t.bbox})
    return results


def _merge_tables(text: str, tables: list[dict]) -> str:
    if not tables:
        return text
    parts = [text]
    for i, t in enumerate(tables, 1):
        parts.append(f"\n\n### Table {i}\n\n{t['markdown']}")
    return "\n".join(parts)


# ── Main extraction loop ──────────────────────────────────────────────────────

def extract_pdf(pdf_path: str | Path, page_filter: Optional[set[int]] = None) -> ExtractionResult:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    doc = fitz.open(str(pdf_path))
    metadata = {k: doc.metadata.get(k, "") for k in ("title", "author", "subject", "creator")}
    metadata["pages"] = doc.page_count
    pages: list[PageResult] = []

    with pdfplumber.open(str(pdf_path)) as plumber_doc:
        for i, fitz_page in enumerate(doc):
            page_num = i + 1
            if page_filter and page_num not in page_filter:
                continue

            warnings: list[str] = []
            plumber_page = plumber_doc.pages[i]
            raw_text = fitz_page.get_text("text")

            # ── Classify page ─────────────────────────────────────────────
            is_digital  = _is_digital_page(fitz_page)
            is_corrupt  = is_digital and _has_cid_garbage(raw_text)
            is_image    = not is_digital

            # ── Route ─────────────────────────────────────────────────────
            if is_corrupt:
                # Route 2: font-corrupt → strip-split Mistral
                warnings.append("cid/control-char encoding — strip OCR via Mistral")
                mode = "vlm_ocr_strips"
                try:
                    markdown = _ocr_corrupt_page(fitz_page, plumber_page)
                except Exception as e:
                    warnings.append(f"Strip OCR failed: {e}")
                    markdown = ""

            elif is_image:
                # Route 3: image page → floating table detection → Mistral
                mode = "vlm_ocr_image"
                try:
                    markdown = _ocr_image_page(fitz_page)
                except Exception as e:
                    warnings.append(f"Image OCR failed: {e}")
                    markdown = ""

            else:
                # Route 1: clean digital → pdfplumber + PyMuPDF
                mode = "digital"
                try:
                    tables = _extract_tables(plumber_page)
                except Exception as e:
                    tables = []
                    warnings.append(f"Table extraction failed: {e}")
                table_bboxes = [t["bbox"] for t in tables]
                text = _extract_digital_text(fitz_page, table_bboxes)
                markdown = _merge_tables(text, tables)

            pages.append(PageResult(page_num, mode, markdown, [], warnings))
            print(f"  Page {page_num}/{doc.page_count} [{mode}]"
                  + (f" ⚠ {warnings}" if warnings else ""))

    doc.close()
    return ExtractionResult(str(pdf_path), len(pages), pages, metadata)


# """
# PDF Extraction Pipeline — extractor_vlm.py
# (Hybrid: Digital + Mistral VLM for OCR)
# """

# from __future__ import annotations

# import json
# import os
# import re
# import sys
# from dataclasses import dataclass, field, asdict
# from pathlib import Path
# from typing import Optional

# import fitz
# import pdfplumber
# from dotenv import load_dotenv

# # ── FIX: Updated Import based on Mistral Documentation ──
# from mistralai.client import Mistral

# # ── Setup Mistral API ─────────────────────────────────────────────────────────
# load_dotenv()
# client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

# # ── Tunables ──────────────────────────────────────────────────────────────────
# DIGITAL_TEXT_THRESHOLD = 50
# TABLE_MIN_ROWS = 2
# CID_RATIO_THRESHOLD = 0.05
# # ─────────────────────────────────────────────────────────────────────────────

# _CID_RE = re.compile(r"\(cid:\d+\)")

# @dataclass
# class PageResult:
#     page_num: int
#     mode: str
#     markdown: str
#     tables: list[dict]
#     warnings: list[str] = field(default_factory=list)

# @dataclass
# class ExtractionResult:
#     source_path: str
#     total_pages: int
#     pages: list[PageResult]
#     metadata: dict

#     def to_markdown(self) -> str:
#         parts = [f"# Document: {Path(self.source_path).name}\n"]
#         for k, v in self.metadata.items():
#             if v:
#                 parts.append(f"**{k}:** {v}  ")
#         parts.append("\n---\n")
#         for page in self.pages:
#             parts.append(f"\n## Page {page.page_num} _{page.mode}_\n")
#             parts.append(page.markdown.strip())
#             parts.append(f"\n\n---\n")
#         return "\n".join(parts)

#     def to_json(self) -> str:
#         return json.dumps(
#             {"source_path": self.source_path, "total_pages": self.total_pages,
#              "metadata": self.metadata, "pages": [asdict(p) for p in self.pages]},
#             indent=2,
#         )

# def _has_cid_garbage(text: str) -> bool:
#     """Detect font encoding corruption — (cid:XX) or raw control chars."""
#     tokens = text.split()
#     if not tokens:
#         return False
#     if (len(_CID_RE.findall(text)) / len(tokens)) > CID_RATIO_THRESHOLD:
#         return True
#     control_chars = sum(1 for c in text if ord(c) < 32 and c not in '\t\n\r')
#     if (control_chars / len(text)) > 0.01:
#         return True
#     return False

# def _is_digital_page(page: fitz.Page) -> bool:
#     return len(page.get_text("text").strip()) >= DIGITAL_TEXT_THRESHOLD

# def _rects_overlap(r1: tuple, r2: tuple) -> bool:
#     return not (r1[2] <= r2[0] or r1[0] >= r2[2] or r1[3] <= r2[1] or r1[1] >= r2[3])

# def _extract_digital_text(fitz_page: fitz.Page, table_bboxes: list[tuple]) -> str:
#     blocks = fitz_page.get_text("blocks")
#     valid_blocks = []
#     for b in blocks:
#         if b[6] != 0 or not b[4].strip():
#             continue
#         block_rect = (b[0], b[1], b[2], b[3])
#         if not any(_rects_overlap(block_rect, tb) for tb in table_bboxes):
#             valid_blocks.append(b)
#     sorted_blocks = sorted(valid_blocks, key=lambda b: (round(b[1] / 20), b[0]))
#     return "\n\n".join(b[4].strip() for b in sorted_blocks)

# def _extract_tables(plumber_page) -> list[dict]:
#     results = []
#     try:
#         found_tables = plumber_page.find_tables()
#     except Exception:
#         return results
#     for t in found_tables:
#         table = t.extract()
#         if not table or len(table) < TABLE_MIN_ROWS:
#             continue
#         clean = lambda c: str(c).replace("\n", " ").strip() if c is not None else ""
#         header = table[0]
#         md = ["| " + " | ".join(clean(c) for c in header) + " |",
#               "| " + " | ".join("---" for _ in header) + " |"]
#         for row in table[1:]:
#             padded = list(row) + [""] * (len(header) - len(row))
#             md.append("| " + " | ".join(clean(c) for c in padded) + " |")
#         results.append({"markdown": "\n".join(md), "bbox": t.bbox})
#     return results

# def _merge_tables(text: str, tables: list[dict]) -> str:
#     if not tables:
#         return text
#     parts = [text]
#     for i, t in enumerate(tables, 1):
#         parts.append(f"\n\n### Table {i}\n\n{t['markdown']}")
#     return "\n".join(parts)

# # ── Mistral Integration ───────────────────────────────────────────────────────
# def _ocr_page_mistral(pdf_path: Path, page_num: int) -> str:
#     """Send only the corrupted page to Mistral OCR API."""
#     print(f"    [Mistral API] Uploading and processing Page {page_num}...")
    
#     # 1. Upload the file
#     with open(pdf_path, "rb") as f:
#         uploaded_file = client.files.upload(
#             file={
#                 "file_name": pdf_path.name,
#                 "content": f,
#             },
#             purpose="ocr",
#         )

#     # 2. Get signed URL
#     signed_url = client.files.get_signed_url(file_id=uploaded_file.id)

#     # 3. Process the document (No Base64 images, default inline tables)
#     pdf_response = client.ocr.process(
#         model="mistral-ocr-latest",
#         document={
#             "type": "document_url",
#             "document_url": signed_url.url,
#         },
#         pages=[page_num - 1] 
#         # Notice we removed table_format and include_image_base64 entirely!
#         # This forces Mistral to just give us pure, inline Markdown text and tables.
#     )

#     # 4. Cleanup
#     client.files.delete(file_id=uploaded_file.id)

#     if pdf_response.pages:
#         # Just return the pure markdown!
#         return pdf_response.pages[0].markdown

#     return ""

# # ─────────────────────────────────────────────────────────────────────────────

# def extract_pdf(pdf_path: str | Path, page_filter: Optional[set[int]] = None) -> ExtractionResult:
#     pdf_path = Path(pdf_path)
#     if not pdf_path.exists():
#         raise FileNotFoundError(pdf_path)

#     doc = fitz.open(str(pdf_path))
#     metadata = {k: doc.metadata.get(k, "") for k in ("title", "author", "subject", "creator")}
#     metadata["pages"] = doc.page_count
#     pages: list[PageResult] = []

#     with pdfplumber.open(str(pdf_path)) as plumber_doc:
#         for i, fitz_page in enumerate(doc):
#             page_num = i + 1
#             if page_filter and page_num not in page_filter:
#                 continue

#             warnings: list[str] = []
#             plumber_page = plumber_doc.pages[i]

#             try:
#                 tables = _extract_tables(plumber_page)
#             except Exception as e:
#                 tables = []
#                 warnings.append(f"Table extraction failed: {e}")

#             table_bboxes = [t["bbox"] for t in tables]
            
#             raw_full_text = fitz_page.get_text("text")
#             force_ocr = not _is_digital_page(fitz_page)

#             if not force_ocr and _has_cid_garbage(raw_full_text):
#                 warnings.append("cid font encoding on full page — routing to Mistral OCR")
#                 force_ocr = True
#                 tables = [] # Mistral handles tables natively

#             mode = "digital"
#             markdown = ""

#             if not force_ocr:
#                 text = _extract_digital_text(fitz_page, table_bboxes)
#                 markdown = _merge_tables(text, tables)
#             else:
#                 mode = "vlm_ocr"
#                 try:
#                     markdown = _ocr_page_mistral(pdf_path, page_num)
#                 except Exception as e:
#                     warnings.append(f"Mistral OCR failed: {e}")
#                     markdown = ""

#             pages.append(PageResult(page_num, mode, markdown, tables, warnings))
#             print(f"  Page {page_num}/{doc.page_count} [{mode}]"
#                   + (f" ⚠ {warnings}" if warnings else ""))

#     doc.close()
#     return ExtractionResult(str(pdf_path), len(pages), pages, metadata)