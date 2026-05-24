"""PDF parsing service.

Extracts text by section, figures/tables, captions, and layout structure.
Uses pymupdf for text extraction, figure cropping, and page rendering.

For the figure understanding pipeline, this module:
  - Renders pages at 300 DPI for VLM consumption (captures vector graphics)
  - Extracts text blocks with exact positions via get_text("dict")
  - Provides deterministic caption-to-figure matching
"""

import json
import logging
import re

from scientis.config import Settings
from scientis.storage.object_store import ObjectStore

logger = logging.getLogger(__name__)


# ── Page rendering for VLM consumption ────────────────────


def render_page(doc, page_num: int, dpi: int = 300) -> bytes:
    """Render a page at high DPI as PNG bytes.

    Uses get_pixmap to capture vector graphics (plots, charts, diagrams)
    that get_images() misses. At 300 DPI, multi-panel figures resolve well
    enough for VLMs to distinguish panels.
    """
    page = doc[page_num]
    pix = page.get_pixmap(dpi=dpi)
    return pix.tobytes("png")


def render_all_pages(doc, dpi: int = 300) -> list[bytes]:
    """Render all pages as PNG bytes for VLM processing."""
    return [render_page(doc, i, dpi) for i in range(len(doc))]


# ── Text block extraction for caption matching ────────────


def extract_text_blocks(doc) -> list[dict]:
    """Extract text blocks with exact positions from all pages.

    Returns a list of dicts with: page_num, text, x1, y1, x2, y2, block_type.
    Uses pymupdf's get_text("dict") for precise layout coordinates.
    """
    from fitz import Rect

    blocks: list[dict] = []
    for page_num, page in enumerate(doc):
        page_rect = page.rect
        pw = page_rect.width
        ph = page_rect.height
        td = page.get_text("dict")
        for block in td.get("blocks", []):
            if block.get("type") != 0:  # skip images
                continue
            bbox = Rect(block["bbox"])
            text = " ".join(
                line["spans"][0]["text"]
                for line in block.get("lines", [])
                if line.get("spans")
            ).strip()
            if text:
                blocks.append({
                    "page_num": page_num + 1,
                    "text": text,
                    "x1": bbox.x0 / pw,
                    "y1": bbox.y0 / ph,
                    "x2": bbox.x1 / pw,
                    "y2": bbox.y1 / ph,
                    "block_type": "text",
                })
    return blocks


# ── Caption-to-figure matching (deterministic, no VLM) ───


def match_captions_to_figures(
    detected_figures: list,
    text_blocks: list[dict],
) -> dict[str, str]:
    """Match figure bounding boxes to their captions using positional heuristics.

    Strategy:
    1. Parse figure numbers from detected labels ("Figure 1" → 1)
    2. Search text_blocks for caption patterns: r'(?:Figure|Fig\\.?)\\s*{num}[\\.:]'
    3. Match by: same page + caption bbox is BELOW figure bbox (y1_caption > y2_figure)
    4. Fallback: match by page + figure number only

    This is deterministic — no VLM needed for this step.
    pymupdf get_text("dict") gives us exact text positions.

    Args:
        detected_figures: list of FigureBBox (or dicts with figure_label, figure_number,
                          page_num, y2 fields)
        text_blocks: list of dicts from extract_text_blocks()

    Returns:
        dict mapping figure_id → caption_text
    """
    from scientis.models.figure import FigureBBox

    matches: dict[str, str] = {}

    for fig in detected_figures:
        # Unwrap pydantic model if needed
        if isinstance(fig, FigureBBox):
            fnum = fig.figure_number
            fpage = fig.page_num
            fy2 = fig.y2
            fid = fig.figure_id
        else:
            fnum = fig.get("figure_number", 0)
            fpage = fig.get("page_num", 0)
            fy2 = fig.get("y2", 0.0)
            fid = fig.get("figure_id", "")

        if not fnum or not fid:
            continue

        # Build caption pattern: r'(?:Figure|Fig\.?)\s*{fnum}[\s:.]'
        pattern = re.compile(
            rf"(?:Figure|Fig\.?)\s*{re.escape(str(fnum))}[\s:.]",
            re.IGNORECASE,
        )

        # Find text blocks on the same page matching the caption pattern
        same_page_blocks = [
            b for b in text_blocks if b["page_num"] == fpage
        ]

        # Priority 1: caption below figure (y1_caption > y2_figure)
        best_block = None
        for block in same_page_blocks:
            if pattern.search(block["text"]) and block["y1"] > fy2:
                best_block = block
                break

        # Priority 2: any caption on same page (fallback)
        if best_block is None:
            for block in same_page_blocks:
                if pattern.search(block["text"]):
                    best_block = block
                    break

        if best_block:
            matches[fid] = best_block["text"]

    return matches


# ── Main parsing entry point ──────────────────────────────


async def parse_paper(paper_id: str, store: ObjectStore, settings: Settings) -> dict:
    """Parse a paper into structured artifacts.

    Returns dict with paths to:
        paper_text.json, figures/, tables.json, layout.json, citations.json
    """
    import fitz  # pymupdf

    # Download raw PDF from object store
    raw_key = f"papers/{paper_id}/raw/"
    raw_files = store.list(raw_key)
    pdf_key = next((k for k in raw_files if k.endswith(".pdf")), None)
    if not pdf_key:
        raise FileNotFoundError(f"No PDF found for paper {paper_id}")

    pdf_bytes = store.get(pdf_key)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    # ── Extract text by page ────────────────────────
    pages = []
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        pages.append(
            {
                "page_num": page_num + 1,
                "text": text,
                "char_count": len(text),
            }
        )

    # ── Extract figures ──────────────────────────────
    figures = []
    for page_num, page in enumerate(doc):
        image_list = page.get_images(full=True)
        for img_idx, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]

            fig_key = f"papers/{paper_id}/figures/page{page_num + 1}_img{img_idx}.{ext}"
            store.put(fig_key, image_bytes)

            figures.append(
                {
                    "figure_id": f"fig-{paper_id}-{page_num + 1}-{img_idx}",
                    "page_num": page_num + 1,
                    "storage_key": fig_key,
                    "width": base_image["width"],
                    "height": base_image["height"],
                    "ext": ext,
                }
            )

    doc.close()

    # ── Render pages for VLM figure understanding ───
    # Re-open doc for rendering (fitz doesn't support rendering on closed docs)
    doc2 = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page_num in range(len(doc2)):
        png_bytes = render_page(doc2, page_num, dpi=300)
        page_key = f"papers/{paper_id}/renders/page_{page_num + 1:03d}.png"
        store.put(page_key, png_bytes, content_type="image/png")
    doc2.close()
    logger.info("Rendered %d pages for %s", len(doc2), paper_id)

    # ── Save artifacts to object store ───────────────
    text_json = json.dumps({"pages": pages, "total_pages": len(pages)}, indent=2)
    store.put(
        f"papers/{paper_id}/artifacts/paper_text.json",
        text_json.encode(),
        content_type="application/json",
    )

    figures_json = json.dumps({"figures": figures, "total_figures": len(figures)}, indent=2)
    store.put(
        f"papers/{paper_id}/artifacts/figures.json",
        figures_json.encode(),
        content_type="application/json",
    )

    # Placeholder artifacts (filled by understanding layer later)
    for artifact in ["tables.json", "layout.json", "citations.json"]:
        store.put(
            f"papers/{paper_id}/artifacts/{artifact}",
            b"[]",
            content_type="application/json",
        )

    logger.info(
        "Paper parsed: %s — %d pages, %d figures",
        paper_id,
        len(pages),
        len(figures),
    )
    return {"pages": len(pages), "figures": len(figures)}
