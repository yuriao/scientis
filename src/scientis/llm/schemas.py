"""Structured output schemas for LLM extraction passes.

These are JSON Schema objects used with OpenAI's response_format or
vLLM's guided_json for constrained generation.
"""

# ── Claim extraction schema ──────────────────────────────
CLAIM_EXTRACTION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "claim_extraction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "claims": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "claim": {
                                "type": "string",
                                "description": "A single scientific claim from the paper. Must be a declarative, falsifiable statement.",
                            },
                            "section": {
                                "type": "string",
                                "enum": [
                                    "introduction",
                                    "methods",
                                    "results",
                                    "discussion",
                                    "abstract",
                                ],
                                "description": "Paper section where this claim appears.",
                            },
                            "claim_type": {
                                "type": "string",
                                "enum": [
                                    "hypothesis",
                                    "finding",
                                    "method",
                                    "limitation",
                                    "comparison",
                                ],
                                "description": "Type of the claim.",
                            },
                            "evidence_spans": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "type": {
                                            "type": "string",
                                            "enum": ["text", "figure", "table"],
                                        },
                                        "quote": {
                                            "type": "string",
                                            "description": "Verbatim text quote supporting the claim.",
                                        },
                                        "figure_id": {
                                            "type": "string",
                                            "description": "Reference to figure (e.g. 'fig2', 'Fig. 3B').",
                                        },
                                        "panel": {
                                            "type": "string",
                                            "description": "Panel within a figure (e.g. 'B', 'C').",
                                        },
                                    },
                                    "required": ["type", "quote"],
                                    "additionalProperties": False,
                                },
                            },
                            "entities": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Genes, proteins, diseases, pathways, cell types, methods mentioned.",
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                                "description": "How well-supported is this claim by the evidence provided?",
                            },
                            "contradicting_text": {
                                "type": "string",
                                "description": "Any text in the paper that weakens or contradicts this claim. Empty if none.",
                            },
                        },
                        "required": ["claim", "section", "confidence"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["claims"],
            "additionalProperties": False,
        },
    },
}


# ── Entity canonicalization schema ────────────────────────
ENTITY_CANONICALIZATION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "entity_canonicalization",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "canonical": {"type": "string", "description": "Canonical name for this entity."},
                "entity_type": {
                    "type": "string",
                    "enum": [
                        "Disease",
                        "Gene",
                        "Protein",
                        "Pathway",
                        "Mechanism",
                        "Method",
                        "Biomarker",
                        "CellType",
                        "Drug",
                        "Assay",
                    ],
                },
                "aliases": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Known synonyms or alternate names.",
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["canonical", "entity_type", "confidence"],
            "additionalProperties": False,
        },
    },
}


# ── Mechanism induction schema ────────────────────────────
MECHANISM_INDUCTION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "mechanism_induction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "hypotheses": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "mechanism": {
                                "type": "string",
                                "description": "Name of the shared mechanism.",
                            },
                            "description": {
                                "type": "string",
                                "description": "2-3 sentence description of how this mechanism operates.",
                            },
                            "supporting_claims": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Claim IDs that support this mechanism.",
                            },
                            "contradicting_claims": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Claim IDs that contradict or weaken this mechanism.",
                            },
                            "diseases": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Diseases linked to this mechanism.",
                            },
                            "genes": {"type": "array", "items": {"type": "string"}},
                            "pathways": {"type": "array", "items": {"type": "string"}},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            "next_experiments": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Suggested follow-up experiments.",
                            },
                            "gaps": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Knowledge gaps and open questions.",
                            },
                        },
                        "required": ["mechanism", "description", "confidence"],
                        "additionalProperties": False,
                    },
                }
            },
        },
    },
}


# ── VLM Figure Detection schema ─────────────────────────
FIGURE_DETECTION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "figure_detection",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "page_num": {"type": "integer"},
                "figures": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "figure_label": {
                                "type": "string",
                                "description": "The figure number as written, e.g. 'Figure 1', 'Fig. 2', 'Fig. S1'",
                            },
                            "figure_number": {
                                "type": "integer",
                                "description": "Parsed integer from the label, e.g. 1 from 'Figure 1'",
                            },
                            "x1": {
                                "type": "number",
                                "description": "Left edge of figure bounding box (0-1 fraction of page width)",
                            },
                            "y1": {
                                "type": "number",
                                "description": "Top edge of figure bounding box (0-1 fraction of page height)",
                            },
                            "x2": {
                                "type": "number",
                                "description": "Right edge of figure bounding box (0-1 fraction of page width)",
                            },
                            "y2": {
                                "type": "number",
                                "description": "Bottom edge of figure bounding box (0-1 fraction of page height)",
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                                "description": "Detection confidence. < 0.6 will be logged and flagged.",
                            },
                        },
                        "required": ["figure_label", "figure_number", "x1", "y1", "x2", "y2", "confidence"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["page_num", "figures"],
            "additionalProperties": False,
        },
    },
}

# ── VLM Panel Description schema ────────────────────────
PANEL_DESCRIPTION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "panel_description",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "figure_label": {
                    "type": "string",
                    "description": "The figure number as written, e.g. 'Figure 1'",
                },
                "panels": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "panel_label": {
                                "type": "string",
                                "description": "Panel letter (e.g. 'A', 'B', 'C') or empty for single-panel figures",
                            },
                            "description": {
                                "type": "string",
                                "description": "2-4 sentence visual description of what this panel shows",
                            },
                            "chart_type": {
                                "type": "string",
                                "enum": [
                                    "bar_chart",
                                    "line_plot",
                                    "scatter_plot",
                                    "heatmap",
                                    "microscopy",
                                    "western_blot",
                                    "diagram",
                                    "table",
                                    "photograph",
                                    "unknown",
                                ],
                                "description": "Type of visual in this panel",
                            },
                            "key_observations": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "2-5 bullet-point observations from this panel",
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                                "description": "Description confidence. < 0.6 flagged for review.",
                            },
                        },
                        "required": ["panel_label", "description", "chart_type", "key_observations", "confidence"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["figure_label", "panels"],
            "additionalProperties": False,
        },
    },
}
