"""
post_processor.py
=================
Cleans raw extraction output and enriches it with structural signals
useful for downstream LLM question-answering.

Runs AFTER extractor.py takes an ExtractionResult and returns a
cleaned, annotated version.
"""

from __future__ import annotations

import re

from dataclasses import dataclass


# ── Regex patterns ────────────────────────────────────────────────────────────
_HEADING_PATTERNS = [
    # All-caps short line (common in env reports)
    re.compile(r"^([A-Z][A-Z\s\d\-:]{3,60})$"),
    # Numbered section  e.g. "2.1 Site Description"
    re.compile(r"^(\d+(?:\.\d+)*)\s+([A-Z].{3,80})$"),
    # "Table X" / "Figure X" captions
    re.compile(r"^(Table\s+\d+[\.\-:]?.*)$", re.I),
    re.compile(r"^(Figure\s+\d+[\.\-:]?.*)$", re.I),
]

_JUNK_LINES = re.compile(
    r"^\s*("
    r"page\s+\d+\s*(of\s*\d+)?"      # page footers
    r"|confidential"
    r"|draft"
    r"|\d{1,2}/\d{1,2}/\d{2,4}"      # bare date lines
    r")\s*$",
    re.I,
)


def _clean_ocr_text(text: str) -> str:
    """Fix common Tesseract artefacts."""
    # Remove form-feed chars
    text = text.replace("\x0c", "\n")
    # Collapse 3+ newlines → 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove lines that are just noise
    lines = [l for l in text.splitlines() if not _JUNK_LINES.match(l)]
    return "\n".join(lines)


def _promote_headings(text: str) -> str:
    """
    Heuristically promote likely headings to Markdown ## / ###.
    Only applies to lines not already inside a Markdown table.
    """
    out_lines = []
    in_table = False
    for line in text.splitlines():
        if line.startswith("|"):
            in_table = True
            out_lines.append(line)
            continue
        if in_table and not line.startswith("|"):
            in_table = False

        promoted = False
        if not in_table:
            for pat in _HEADING_PATTERNS:
                m = pat.match(line.strip())
                if m:
                    # Use ## for all-caps, ### for numbered subsections
                    if re.match(r"^\d+\.\d+", line.strip()):
                        out_lines.append(f"### {line.strip()}")
                    else:
                        out_lines.append(f"## {line.strip()}")
                    promoted = True
                    break
        if not promoted:
            out_lines.append(line)
    return "\n".join(out_lines)


def _extract_key_values(text: str) -> list[dict]:
    """
    Pull out KEY: Value pairs common in env/regulatory docs.
    e.g. "Sample ID: MW-114S", "Analysis Date: 02/25/2020"
    Returns list of {"key": ..., "value": ...}
    """
    pattern = re.compile(r"^([A-Za-z][A-Za-z\s\-/]{2,40}):\s+(.+)$")
    kv_pairs = []
    for line in text.splitlines():
        m = pattern.match(line.strip())
        if m:
            kv_pairs.append({"key": m.group(1).strip(), "value": m.group(2).strip()})
    return kv_pairs


@dataclass
class EnrichedPage:
    page_num: int
    mode: str
    raw_markdown: str
    clean_markdown: str
    key_values: list[dict]
    has_tables: bool
    warnings: list[str]


def post_process(result: ExtractionResult) -> list[EnrichedPage]:
    """Clean and enrich every page in an ExtractionResult."""
    enriched = []
    for page in result.pages:
        
        if page.mode in ("ocr", "ocr-tesseract"):
            clean = _clean_ocr_text(page.markdown)
            clean = _promote_headings(clean)
        else:
            # digital, vlm_ocr_strips, vlm_ocr_image — already clean
            clean = page.markdown

        # 2. Always extract key-values (useful regardless of the engine)
        kv = _extract_key_values(clean)

        enriched.append(EnrichedPage(
            page_num=page.page_num,
            mode=page.mode,
            raw_markdown=page.markdown,
            clean_markdown=clean,
            key_values=kv,
            has_tables=bool(page.tables),
            warnings=page.warnings,
        ))
    return enriched


def enriched_to_markdown(pages: list[EnrichedPage], doc_title: str = "") -> str:
    """Render enriched pages back to a single clean Markdown document."""
    parts = []
    if doc_title:
        parts.append(f"# {doc_title}\n")

    for page in pages:
        parts.append(f"\n## Page {page.page_num} _{page.mode}_\n")
        parts.append(page.clean_markdown.strip())
        if page.key_values:
            parts.append("\n\n**Extracted key–value pairs:**\n")
            for kv in page.key_values:
                parts.append(f"- **{kv['key']}**: {kv['value']}")
        parts.append(f"\n\n---  <!-- end page {page.page_num} -->")

    return "\n".join(parts)