"""Tests for the figure understanding pipeline.

Covers:
  - Deterministic caption matching (match_captions_to_figures)
  - Model validation (FigureBBox, PanelDescription, FigureDetection)
  - Helpers (encode_image, page_key_sort, enrich_with_page_text)
  - Edge cases (empty pages, multi-part figures, low confidence, VLM fallback)
"""

import json

import pytest

from scientis.models.figure import (
    FigureBBox,
    FigureDetection,
    FigureUnderstandingResult,
    PanelDescription,
)
from scientis.services.figure_understanding import (
    _encode_image,
    _enrich_with_page_text,
    _page_key_sort,
)
from scientis.services.parsing import match_captions_to_figures


# ── Deterministic caption matching ────────────────────────


def test_match_captions_below_figure():
    """Caption directly below figure bbox should match first."""
    detected = [
        FigureBBox(
            figure_label="Figure 1",
            figure_number=1,
            page_num=3,
            x1=0.1,
            y1=0.1,
            x2=0.9,
            y2=0.5,
            confidence=0.95,
            figure_id="p1-fig001",
        )
    ]
    text_blocks = [
        {
            "page_num": 3,
            "text": "Figure 1: Experimental results for treatment A.",
            "x1": 0.1,
            "y1": 0.55,  # just below figure
            "x2": 0.9,
            "y2": 0.65,
            "block_type": "text",
        },
        {
            "page_num": 3,
            "text": "Figure 1. Methods overview.",
            "x1": 0.1,
            "y1": 0.05,  # above figure (should not match)
            "x2": 0.9,
            "y2": 0.08,
            "block_type": "text",
        },
    ]
    result = match_captions_to_figures(detected, text_blocks)
    assert len(result) == 1
    assert result["p1-fig001"] == "Figure 1: Experimental results for treatment A."


def test_match_captions_fallback_same_page():
    """When no caption is below, fall back to any caption on same page."""
    detected = [
        FigureBBox(
            figure_label="Fig. 2",
            figure_number=2,
            page_num=5,
            x1=0.0,
            y1=0.3,
            x2=1.0,
            y2=0.9,
            confidence=0.80,
            figure_id="p1-fig002",
        )
    ]
    text_blocks = [
        {
            "page_num": 5,
            "text": "Fig. 2: Comparison of three methods.",
            "x1": 0.1,
            "y1": 0.1,  # above the figure
            "x2": 0.9,
            "y2": 0.25,
            "block_type": "text",
        }
    ]
    result = match_captions_to_figures(detected, text_blocks)
    assert len(result) == 1
    assert "Comparison of three methods" in result["p1-fig002"]


def test_match_captions_wrong_page():
    """Caption on wrong page should not match."""
    detected = [
        FigureBBox(
            figure_label="Figure 3",
            figure_number=3,
            page_num=4,
            x1=0.1,
            y1=0.1,
            x2=0.9,
            y2=0.5,
            confidence=0.90,
            figure_id="p1-fig003",
        )
    ]
    text_blocks = [
        {
            "page_num": 3,  # different page
            "text": "Figure 3: Discussion of results.",
            "x1": 0.1,
            "y1": 0.6,
            "x2": 0.9,
            "y2": 0.7,
            "block_type": "text",
        }
    ]
    result = match_captions_to_figures(detected, text_blocks)
    assert len(result) == 0


def test_match_captions_no_figures():
    """Empty detection should return empty dict."""
    result = match_captions_to_figures([], [])
    assert result == {}


def test_match_captions_multiple_figures_same_page():
    """Multiple figures on same page should each match their own caption."""
    detected = [
        FigureBBox(
            figure_label="Figure 4",
            figure_number=4,
            page_num=2,
            x1=0.1,
            y1=0.1,
            x2=0.45,
            y2=0.5,
            confidence=0.92,
            figure_id="p1-fig004",
        ),
        FigureBBox(
            figure_label="Figure 5",
            figure_number=5,
            page_num=2,
            x1=0.55,
            y1=0.1,
            x2=0.9,
            y2=0.5,
            confidence=0.88,
            figure_id="p1-fig005",
        ),
    ]
    text_blocks = [
        {
            "page_num": 2,
            "text": "Figure 4: Control group results.",
            "x1": 0.1,
            "y1": 0.55,
            "x2": 0.45,
            "y2": 0.65,
            "block_type": "text",
        },
        {
            "page_num": 2,
            "text": "Figure 5: Treatment group results.",
            "x1": 0.55,
            "y1": 0.55,
            "x2": 0.9,
            "y2": 0.65,
            "block_type": "text",
        },
    ]
    result = match_captions_to_figures(detected, text_blocks)
    assert len(result) == 2
    assert "Control" in result["p1-fig004"]
    assert "Treatment" in result["p1-fig005"]


def test_match_captions_fig_prefix():
    """'Fig.' abbreviation should match same as 'Figure'."""
    detected = [
        FigureBBox(
            figure_label="Fig. 6",
            figure_number=6,
            page_num=1,
            x1=0.1,
            y1=0.1,
            x2=0.9,
            y2=0.4,
            confidence=0.95,
            figure_id="p1-fig006",
        )
    ]
    text_blocks = [
        {
            "page_num": 1,
            "text": "Fig. 6: Dose-response curve.",
            "x1": 0.1,
            "y1": 0.6,
            "x2": 0.9,
            "y2": 0.7,
            "block_type": "text",
        }
    ]
    result = match_captions_to_figures(detected, text_blocks)
    assert len(result) == 1
    assert "Dose-response" in result["p1-fig006"]


def test_match_captions_dict_input():
    """Should also accept plain dicts (not just pydantic models)."""
    detected = [{
        "figure_label": "Figure 7",
        "figure_number": 7,
        "page_num": 3,
        "y2": 0.6,
        "figure_id": "p1-fig007",
    }]
    text_blocks = [{
        "page_num": 3,
        "text": "Figure 7: Summary of findings.",
        "x1": 0.1,
        "y1": 0.7,
        "x2": 0.9,
        "y2": 0.8,
        "block_type": "text",
    }]
    result = match_captions_to_figures(detected, text_blocks)
    assert len(result) == 1
    assert "Summary" in result["p1-fig007"]


# ── Model validation ─────────────────────────────────────


def test_figure_bbox_defaults():
    bbox = FigureBBox()
    assert bbox.figure_label == ""
    assert bbox.figure_number == 0
    assert bbox.confidence == 0.0


def test_figure_bbox_validation():
    bbox = FigureBBox(
        figure_label="Figure 1",
        figure_number=1,
        page_num=3,
        x1=0.1,
        y1=0.2,
        x2=0.9,
        y2=0.7,
        confidence=0.95,
        figure_id="paper-abc-fig001",
    )
    assert bbox.figure_id == "paper-abc-fig001"
    assert bbox.figure_number == 1
    assert abs(bbox.confidence - 0.95) < 0.001


def test_panel_description_validation():
    panel = PanelDescription(
        panel_id="paper-abc-fig001a",
        figure_id="paper-abc-fig001",
        figure_label="Figure 1",
        panel_label="A",
        page_num=3,
        description="Western blot showing protein expression levels.",
        chart_type="western_blot",
        key_observations=["Band intensity increases with dose", "GAPDH loading control present"],
        confidence=0.88,
        model_tier="vision_cheap",
    )
    assert panel.panel_id == "paper-abc-fig001a"
    assert panel.chart_type == "western_blot"
    assert len(panel.key_observations) == 2


def test_figure_detection_no_figures():
    """A page with no figures should have an empty list."""
    detection = FigureDetection(page_num=4, figures=[])
    assert detection.page_num == 4
    assert detection.figures == []


def test_figure_understanding_result():
    result = FigureUnderstandingResult(
        paper_id="test-paper",
        unprocessed_figures=["test-paper-fig003"],
    )
    assert result.paper_id == "test-paper"
    assert result.figures == []
    assert result.panels == []
    assert "test-paper-fig003" in result.unprocessed_figures


def test_figure_understanding_result_serialization():
    result = FigureUnderstandingResult(
        paper_id="test-paper",
        figures=[FigureDetection(page_num=1, figures=[])],
        panels=[
            PanelDescription(
                panel_id="fig1a",
                figure_id="test-paper-fig001",
                figure_label="Figure 1",
                panel_label="A",
                page_num=1,
                description="A bar chart.",
                chart_type="bar_chart",
                key_observations=["Group A higher than B"],
                confidence=0.90,
                model_tier="vision_cheap",
            )
        ],
        unprocessed_figures=[],
    )
    js = result.model_dump_json()
    data = json.loads(js)
    assert data["paper_id"] == "test-paper"
    assert len(data["panels"]) == 1
    assert data["panels"][0]["panel_label"] == "A"


# ── Helpers ──────────────────────────────────────────────


def test_encode_image():
    """Should return a valid base64 data URI."""
    small_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    uri = _encode_image(small_png)
    assert uri.startswith("data:image/png;base64,")
    # Should be decodable
    import base64
    decoded = base64.b64decode(uri.split(",", 1)[1])
    assert decoded == small_png


def test_page_key_sort():
    """Should sort page keys by embedded page number."""
    keys = [
        "papers/p1/renders/page_010.png",
        "papers/p1/renders/page_002.png",
        "papers/p1/renders/page_100.png",
    ]
    sorted_keys = sorted(keys, key=_page_key_sort)
    assert sorted_keys[0].endswith("page_002.png")
    assert sorted_keys[1].endswith("page_010.png")
    assert sorted_keys[2].endswith("page_100.png")


def test_page_key_sort_no_match():
    """Keys without page_### should sort to 0."""
    assert _page_key_sort("papers/p1/renders/something_else.png") == 0


def test_enrich_with_page_text_found():
    """Figure number found in page text should be enriched."""
    figures = [
        FigureBBox(
            figure_label="Figure 1",
            figure_number=1,
            page_num=2,
            figure_id="p1-fig001",
        )
    ]
    pages = [
        {"page_num": 1, "text": "Introduction text."},
        {"page_num": 2, "text": "Figure 1 shows the experimental setup."},
    ]
    result = _enrich_with_page_text(figures, "p1", pages)
    assert len(result) == 1
    assert result[0].figure_id == "p1-fig001"


def test_enrich_with_page_text_not_found():
    """Figure number not in any page text should still be returned but logged."""
    figures = [
        FigureBBox(
            figure_label="Figure 99",
            figure_number=99,
            page_num=10,
            figure_id="p1-fig099",
        )
    ]
    pages = [{"page_num": 10, "text": "No figure 99 mentioned here."}]
    result = _enrich_with_page_text(figures, "p1", pages)
    assert len(result) == 1  # Still returned, just not pruned


def test_enrich_with_page_text_no_number():
    """Figure with no figure_number should pass through unchanged."""
    figures = [FigureBBox(figure_id="p1-fig000")]
    pages = [{"page_num": 1, "text": "Some text."}]
    result = _enrich_with_page_text(figures, "p1", pages)
    assert len(result) == 1
    assert result[0].figure_id == "p1-fig000"


# ── Edge cases ───────────────────────────────────────────


def test_match_captions_supplemental_figure():
    """Supplemental figures: when VLM extracts just the number (e.g. 1 from 'S1'),
    the regex won't match 'Figure S1' because the 'S' is between 'Figure ' and '1'.
    This is a known limitation — supplemental figures need separate handling
    (higher page numbers, separate pipeline pass)."""
    detected = [
        FigureBBox(
            figure_label="Figure S1",
            figure_number=1,  # VLM may extract just the number
            page_num=20,
            x1=0.1,
            y1=0.1,
            x2=0.9,
            y2=0.5,
            confidence=0.85,
            figure_id="p1-figs001",
        )
    ]
    text_blocks = [
        {
            "page_num": 20,
            "text": "Figure S1: Supplemental validation data.",
            "x1": 0.1,
            "y1": 0.6,
            "x2": 0.9,
            "y2": 0.7,
            "block_type": "text",
        }
    ]
    result = match_captions_to_figures(detected, text_blocks)
    # Known limitation: 'Figure 1' pattern does not match 'Figure S1'
    # Supplemental figures should be processed separately
    assert len(result) == 0
    # Workaround: if figure_label contains 'S', prepend 'S' to the number pattern
    # This would require modifying the regex to r'(?:Figure|Fig\.?)\s*S?{num}'


def test_no_figures_on_page():
    """Detection with empty figures list is valid."""
    detection = FigureDetection(page_num=1, figures=[])
    assert detection.figures == []


def test_panel_description_missing_optional_fields():
    """PanelDescription with minimal fields should work."""
    panel = PanelDescription(
        panel_id="minimal",
        figure_id="fig1",
        figure_label="Figure 1",
        panel_label="",
        page_num=1,
        description="Minimal description.",
        chart_type="unknown",
        key_observations=[],
        confidence=0.5,
        model_tier="vision_cheap",
    )
    assert panel.key_observations == []
    assert panel.chart_type == "unknown"
