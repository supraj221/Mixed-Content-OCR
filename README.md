# PDF Extraction Pipeline

A generalizable PDF extraction pipeline that converts mixed-content PDFs into LLM-friendly Markdown. Handles digitally-embedded text, font-corrupt pages, scanned images, and complex floating-table layouts — all in a single pipeline with no hardcoded document-specific rules.

Built as part of an environmental data engineering assessment. Validated by correctly answering domain-specific questions from the extracted output alone, with no reference to the original PDF.

---

## How It Works

Every page is independently classified and routed to the appropriate extraction strategy:

| Route | Trigger | Handler |
|---|---|---|
| **Digital** | Extractable text ≥ 50 chars, no font corruption | pdfplumber + PyMuPDF |
| **Font-Corrupt** | Control characters > 1% of page text | Strip-split render → Mistral OCR |
| **Image / Scanned** | Extractable text < 50 chars | OSD rotation → OpenCV detection → Mistral OCR |

### Page Type Handling

**Font-Corrupt Pages**
Some PDFs embed fonts without Unicode mapping tables. PyMuPDF and pdfplumber both fail silently, producing raw control characters (`\x14`, `\x16`) instead of text. The pipeline detects this via control character ratio analysis, renders the page as horizontal image strips, and sends each strip to Mistral OCR — successfully extracting dense tables (tested: 59 rows × 27 columns).

**Image / Scanned Pages**
For pages like site maps with floating tables, the pipeline:
1. Auto-corrects rotation via Tesseract OSD (no hardcoded angles)
2. Applies adaptive Gaussian thresholding to detect table borders regardless of contrast
3. Uses OpenCV contour detection to identify individual floating table bounding boxes
4. Sends each crop separately to Mistral OCR

**Digital Pages**
pdfplumber detects tables using PDF vector primitives (line segments, rectangles). PyMuPDF extracts non-table text in reading order via bounding box sorting. Text and tables are merged into clean Markdown — text blocks that overlap table regions are excluded to avoid duplication.

---

## Output Format

```
# Document: report.pdf

## Page 1 _vlm_ocr_strips_

| Sample ID | MW-100S | MW-101S | ... |
| --- | --- | --- | ... |
| Arsenic, Dissolved | <0.002 | <0.002 | ... |

---  <!-- end page 1 -->

## Page 2 _digital_

...
```

- `## Page N _mode_` headers with mode labels (`digital`, `vlm_ocr_strips`, `vlm_ocr_image`) for provenance
- GitHub-Flavored Markdown pipe tables
- `<!-- end page N -->` boundary markers for LLM citation grounding


---

## Setup

**Requirements:** Python 3.12+, Tesseract OCR installed on system

```bash
# Install Tesseract (Windows)
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
# Add to PATH after install

1. `python -m venv env`
 
2. Activate the environment:
   - Windows: `env\Scripts\activate`
   - Mac/Linux: `source env/bin/activate`
 
3. `pip install -r requirements.txt`
 
4. Install Tesseract binary and add to PATH:
   - Windows: Download installer from https://github.com/UB-Mannheim/tesseract/wiki — check "Add to PATH" during install
   - Mac: `brew install tesseract`
   - Ubuntu: `sudo apt install tesseract-ocr`
   - Full installation guide: https://tesseract-ocr.github.io/tessdoc/Installation.html
 
5. Add your Mistral API key to a `.env` file:
   ```
   MISTRAL_API_KEY=your_key_here
   ```
   Get a free Mistral API key at [console.mistral.ai](https://console.mistral.ai)
 
6. Run:
   ```bash
   python run_pipeline.py report.pdf
   ```

---

## File Structure

```
pdf-extraction-pipeline/
├── extractor_vlm.py          # Mistral OCR engine (recommended)
├── run_pipeline.py           # CLI entrypoint
├── post_processor.py         # Heading detection, KV extraction
├── requirements.txt
├── .env                      # API key (not committed)
└── README.md
```

---

## Dependencies

| Library | Role |
|---|---|
| PyMuPDF | Digital text extraction, page rendering |
| pdfplumber | Table detection via PDF vector primitives |
| mistralai | Mistral OCR API client |
| pytesseract | OSD orientation detection |
| opencv-python | Adaptive thresholding, contour detection |
| Pillow | Image rendering, rotation correction |
| python-dotenv | API key management |
| numpy | Image array operations |

---

## Known Limitations

- **Cell highlight colors** — Regulatory exceedance flags (colored cell backgrounds) are not preserved in Markdown output
- **Floating table detection** — OpenCV detected 12/13 floating tables on the test site map; the 13th had partially rendered borders
- **Merged header cells** — Multi-row merged headers are flattened; column labels may appear blank in some rows
- **Multi-strip formatting** — Date formats (e.g. `02/25/2020` vs `02-25-2020`) can vary across strips since each is sent as an independent OCR call
- **Mistral rate limits** — Free tier applies; retry logic is implemented but production use requires a paid tier

---

## Tech Stack

PyMuPDF · pdfplumber · Mistral OCR · Tesseract OSD · OpenCV · Pillow · Python 3.12
