#!/usr/bin/env python3
"""
run_pipeline.py
===============
    python run_pipeline.py report.pdf
    python run_pipeline.py report.pdf --output my_output.md
"""

import argparse
import sys
from pathlib import Path

from post_processor import post_process, enriched_to_markdown


def main():
    parser = argparse.ArgumentParser(description="PDF Extraction Pipeline")
    parser.add_argument("pdf", help="Path to input PDF")
    parser.add_argument("--output", "-o", help="Output file path (default: <pdf_name>.md)")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found", file=sys.stderr)
        sys.exit(1)

    from extractor_vlm import extract_pdf

    print(f"\n{'='*60}")
    print(f"  PDF Extraction Pipeline")
    print(f"  Input: {pdf_path.name}")
    print(f"{'='*60}\n")

    print("Step 1/2: Extracting pages...")
    result = extract_pdf(pdf_path)

    print("\nStep 2/2: Post-processing...")
    enriched = post_process(result)
    doc_title = result.metadata.get("title") or pdf_path.stem
    final_md = enriched_to_markdown(enriched, doc_title=doc_title)

    out_path = Path(args.output) if args.output else pdf_path.with_suffix(".md")
    out_path.write_text(final_md, encoding="utf-8")

    digital = sum(1 for p in result.pages if p.mode == "digital")
    ocr_strips = sum(1 for p in result.pages if p.mode == "vlm_ocr_strips")
    ocr_image = sum(1 for p in result.pages if p.mode == "vlm_ocr_image")

    print(f"\n{'='*60}")
    print(f"  ✓ Done!")
    print(f"  Pages total     : {result.total_pages}")
    print(f"  Digital         : {digital}")
    print(f"  OCR (strips)    : {ocr_strips}")
    print(f"  OCR (image)     : {ocr_image}")
    print(f"  Output chars    : {len(final_md):,}")
    print(f"  Output path     : {out_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

# #!/usr/bin/env python3
# """
# run_pipeline.py
# ===============
# End-to-end entrypoint:
#     python run_pipeline.py report.pdf
#     python run_pipeline.py report.pdf --engine mistral
#     python run_pipeline.py report.pdf --json
# """

# import argparse
# import sys
# from pathlib import Path

# from post_processor import post_process, enriched_to_markdown

# def main():
#     parser = argparse.ArgumentParser(description="PDF Extraction Pipeline")
#     parser.add_argument("pdf", help="Path to input PDF")
#     parser.add_argument("--engine", choices=["traditional", "mistral"], default="traditional", 
#                         help="OCR engine for corrupted/complex pages (default: traditional)")
#     parser.add_argument("--output", "-o", help="Output file path (default: auto-generates based on engine)")
#     parser.add_argument("--json", action="store_true", help="Also write raw JSON extraction")
#     parser.add_argument("--no-post", action="store_true", help="Skip post-processing")
#     args = parser.parse_args()

#     pdf_path = Path(args.pdf)
#     if not pdf_path.exists():
#         print(f"Error: {pdf_path} not found", file=sys.stderr)
#         sys.exit(1)

#     # ── Dynamic Module Loading ──
#     try:
#         if args.engine == "mistral":
#             from extractor_vlm import extract_pdf
#         else:
#             from extractor_traditional import extract_pdf
#     except ImportError as e:
#         print(f"Error loading '{args.engine}' extractor: {e}", file=sys.stderr)
#         print("Ensure 'extractor_traditional.py' and 'extractor_vlm.py' exist in your directory.", file=sys.stderr)
#         sys.exit(1)

#     print(f"\n{'='*60}")
#     print(f"  PDF Extraction Pipeline [Engine: {args.engine.upper()}]")
#     print(f"  Input: {pdf_path.name}")
#     print(f"{'='*60}\n")

#     # ── Step 1: Raw extraction ──
#     print("Step 1/2: Extracting pages...")
#     result = extract_pdf(pdf_path)

#     if args.json:
#         json_path = pdf_path.with_name(f"{pdf_path.stem}_{args.engine}.raw.json")
#         json_path.write_text(result.to_json(), encoding="utf-8")
#         print(f"  ↳ Raw JSON: {json_path}")

#     # ── Step 2: Post-processing ──
#     if args.no_post:
#         final_md = result.to_markdown()
#         doc_title = pdf_path.stem
#     else:
#         print("\nStep 2/2: Post-processing (cleaning + heading detection)...")
#         enriched = post_process(result)
#         doc_title = result.metadata.get("title") or pdf_path.stem
#         final_md = enriched_to_markdown(enriched, doc_title=doc_title)

#     # ── Write output ──
#     if args.output:
#         out_path = Path(args.output)
#     else:
#         # Auto-name the file so traditional and mistral outputs don't overwrite each other
#         out_path = pdf_path.with_name(f"{pdf_path.stem}_{args.engine}.md")
        
#     out_path.write_text(final_md, encoding="utf-8")

#     # ── Calculate Stats ──
#     total_chars = len(final_md)
#     digital = sum(1 for p in result.pages if p.mode == "digital")
#     ocr_pages = sum(1 for p in result.pages if p.mode == "ocr")
#     vlm_pages = sum(1 for p in result.pages if p.mode == "vlm_ocr")

#     print(f"\n{'='*60}")
#     print(f"  ✓ Done!")
#     print(f"  Pages total  : {result.total_pages}")
#     print(f"  Digital      : {digital}")
#     if args.engine == "traditional":
#         print(f"  OCR (Tess)   : {ocr_pages}")
#     else:
#         print(f"  OCR (VLM)    : {vlm_pages}")
#     print(f"  Output chars : {total_chars:,}")
#     print(f"  Output path  : {out_path}")
#     print(f"{'='*60}\n")

# if __name__ == "__main__":
#     main()