"""PDF parsing service.

Extracts text by section, figures/tables, captions, and layout structure.
Uses pymupdf for text extraction and figure cropping.
"""

import json
import logging

from scientis.config import Settings
from scientis.storage.object_store import ObjectStore

logger = logging.getLogger(__name__)


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
