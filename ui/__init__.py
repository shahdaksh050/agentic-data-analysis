"""Frontend components for the Streamlit console (see app.py)."""

from ui.pipeline_3d import (
    PALETTE,
    PALETTES,
    Stage,
    StageStatus,
    build_cinematic_document,
    build_document,
    export_cinematic_html,
    extract_cinematic_state,
    render,
    render_cinematic,
)

__all__ = [
    "PALETTE",
    "PALETTES",
    "Stage",
    "StageStatus",
    "build_cinematic_document",
    "build_document",
    "export_cinematic_html",
    "extract_cinematic_state",
    "render",
    "render_cinematic",
]
