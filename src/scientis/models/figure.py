"""Figure and panel models for the figure understanding pipeline.

Detached from the database — panel descriptions live in S3 as JSON.
The existing EvidenceSpan model in claim.py already has figure_id + panel fields.
"""

from pydantic import BaseModel, Field


class FigureBBox(BaseModel):
    """A figure bounding box detected on a page by the VLM."""

    figure_label: str = ""  # e.g. "Figure 1", "Fig. 2"
    figure_number: int = 0  # parsed integer: 1, 2, ...
    page_num: int = 0
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0  # bottom of the figure (in pymupdf coords, y increases downward)
    confidence: float = 0.0  # VLM detection confidence [0, 1]
    figure_id: str = ""  # composite: "{paper_id}-fig{num}"


class FigureDetection(BaseModel):
    """Full page detection result from VLM."""

    page_num: int
    figures: list[FigureBBox] = Field(default_factory=list)


class PanelDescription(BaseModel):
    """A VLM-generated description of a single figure panel."""

    panel_id: str = ""  # e.g. "fig1a", "fig2b"
    figure_id: str = ""  # composite: "{paper_id}-fig{N}"
    figure_label: str = ""  # e.g. "Figure 1"
    panel_label: str = ""  # e.g. "A", "B", "C", or "" for single-panel figures
    page_num: int = 0
    description: str = ""  # 2-4 sentence visual description
    chart_type: str = ""  # "bar_chart", "line_plot", "heatmap", "microscopy", "diagram", "table", "unknown"
    key_observations: list[str] = Field(default_factory=list)
    confidence: float = 0.0  # VLM confidence
    model_tier: str = ""  # which tier generated this ("vision_cheap" or "vision_default")


class FigureUnderstandingResult(BaseModel):
    """Aggregate result for a paper's figure understanding pass."""

    paper_id: str
    figures: list[FigureDetection] = Field(default_factory=list)
    panels: list[PanelDescription] = Field(default_factory=list)
    unprocessed_figures: list[str] = Field(default_factory=list)  # figure_ids that failed
