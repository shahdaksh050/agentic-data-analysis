"""
The plate — the seven-stage RLM workflow drawn as a live technical drawing.

The scene itself lives in ``assets/pipeline_3d.js`` (Three.js + GSAP); this
module is the seam between it and Streamlit. It owns the state contract —
what a stage is, which inks the scene may use — and inlines both assets
into a single sandboxed iframe, because ``components.html`` has no way to
serve sibling files.

Streamlit re-runs the whole script on every interaction, so the component
mounts fresh each time and renders one snapshot of ``st.session_state``.
The scene replays the run on mount rather than streaming frames live.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import streamlit as st
import streamlit.components.v1 as components

__all__ = ["PALETTE", "Stage", "StageStatus", "build_document", "render"]

_ASSETS = Path(__file__).parent / "assets"

#: Matches the status vocabulary ``app.py`` already writes into ``stage_log``.
StageStatus = Literal["pending", "active", "done", "skipped", "error"]

#: DESIGN.md tokens the scene is allowed to use. The ``:root`` block in
#: ``app.py`` mirrors these values — change both together.
#:
#: The drawing is inked in two pens: ``pen`` for what the run measured,
#: ``risk`` for where it failed. Everything not yet reached stays in pencil.
PALETTE: dict[str, str] = {
    "stock": "#dcdbd3",
    "sheet": "#efeee8",
    "ink": "#171c1f",
    "graphite": "#54585b",
    "pen": "#12467e",
    "risk": "#b5271a",
}


@dataclass(frozen=True, slots=True)
class Stage:
    """One workflow stage, in the shape the scene needs to draw it.

    Args:
        num: Stage number as displayed, e.g. ``"3"``.
        name: Human-readable stage name.
        status: Where the run got to for this stage.
        detail: Short free-text note shown when the stage is hovered.
    """

    num: str
    name: str
    status: StageStatus = "pending"
    detail: str = ""


@lru_cache(maxsize=2)
def _asset(name: str) -> str:
    """Read a bundled asset once per process."""
    return (_ASSETS / name).read_text(encoding="utf-8")


def _embed(document: str, height: int) -> None:
    """Mount ``document`` in a sandboxed iframe.

    ``st.iframe`` is the current API and ``components.html`` is deprecated, but
    requirements.txt still allows streamlit>=1.35, which predates ``st.iframe``.
    """
    if hasattr(st, "iframe"):
        st.iframe(document, height=height)
    else:  # pragma: no cover - only reached on older Streamlit
        components.html(document, height=height, scrolling=False)


def build_document(stages: Sequence[Stage]) -> str:
    """Assemble the standalone HTML document for ``stages``.

    Split out from :func:`render` so the scene can be previewed without a
    Streamlit server — see ``scripts/preview_pipeline_3d.py``.

    Args:
        stages: The workflow stages, in execution order.

    Returns:
        A self-contained HTML document, bar the Three.js and GSAP CDN tags.
    """
    state = {
        "stages": [
            {"num": s.num, "name": s.name, "status": s.status, "detail": s.detail}
            for s in stages
        ],
        "palette": PALETTE,
    }
    # Stage details come from tool output, so escape anything that could close
    # the inline <script> early. The scene renders them with textContent.
    state_json = json.dumps(state, ensure_ascii=False).replace("<", "\\u003c")

    return (
        _asset("pipeline_3d.html")
        .replace("__STATE_JSON__", state_json)
        .replace("__SCENE_SCRIPT__", _asset("pipeline_3d.js"))
    )


def render(stages: Sequence[Stage], *, height: int = 400) -> None:
    """Draw the plate for ``stages``.

    Args:
        stages: The workflow stages, in execution order.
        height: Iframe height in pixels. The scene reframes itself to fit.
    """
    _embed(build_document(stages), height)
