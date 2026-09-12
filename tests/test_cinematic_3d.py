"""Tests for the 6-Section 3D Cinematic Fullpage Experience (ui/cinematic_3d.py)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]

from ui.cinematic_3d import (  # noqa: E402
    CINEMATIC_PALETTES,
    _read_asset,
    build_cinematic_document,
    export_cinematic_html,
    extract_cinematic_state,
    render_cinematic,
)


@pytest.fixture
def captured_iframe(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_iframe(src: str, **kwargs: Any) -> None:
        calls.append({"document": src, **kwargs})

    import ui.cinematic_3d as cmod
    monkeypatch.setattr(cmod.st, "iframe", fake_iframe)
    return calls


def _state_of(document: str) -> dict[str, Any]:
    match = re.search(r"window\.__CINEMATIC_STATE__ = (\{.*?\});", document, re.S)
    assert match, "state payload not found in document"
    return json.loads(match.group(1))


def test_cinematic_assets_exist() -> None:
    html_content = _read_asset("cinematic_3d.html")
    assert html_content.lstrip().startswith("<!doctype html>")
    assert "fullpage" in html_content
    assert "anime" in html_content
    js_content = _read_asset("cinematic_3d.js")
    assert "THREE" in js_content
    assert "setAnimationLoop" in js_content


def test_no_placeholders_left_in_cinematic_doc() -> None:
    doc = build_cinematic_document()
    assert "__CINEMATIC_STATE_JSON__" not in doc
    assert "__CINEMATIC_SCENE_SCRIPT__" not in doc


def test_cinematic_state_structure() -> None:
    doc = build_cinematic_document(theme="night")
    state = _state_of(doc)
    assert state["theme"] == "night"
    assert "dataset" in state
    assert "stages" in state
    assert len(state["stages"]) == 7
    assert "statistics" in state
    assert "ml" in state
    assert "synthesis" in state
    assert state["palette"] == CINEMATIC_PALETTES["night"]


def test_render_cinematic_height_and_iframe(captured_iframe: list[dict[str, Any]]) -> None:
    render_cinematic(height=900, theme="day")
    assert len(captured_iframe) == 1
    assert captured_iframe[0]["height"] == 900
    doc = captured_iframe[0]["document"]
    assert "window.__CINEMATIC_STATE__" in doc
    assert "day" in doc


def test_export_cinematic_html(tmp_path: Path) -> None:
    target = tmp_path / "cinematic_presentation.html"
    exported = export_cinematic_html(target, theme="night")
    assert exported.exists()
    content = exported.read_text(encoding="utf-8")
    assert "<!doctype html>" in content
    assert "fullpage" in content


def test_injection_safety_in_cinematic_doc() -> None:
    hostile = "</script><script>window.pwned=1</script>"
    state = extract_cinematic_state({
        "preview_name": hostile,
        "user_objective": hostile,
        "theme": "night",
    })
    doc = build_cinematic_document(state)
    assert "</script><script>" not in doc
    extracted = _state_of(doc)
    assert extracted["dataset"]["name"] == hostile


def test_extract_cinematic_state_with_nones() -> None:
    """Session state can contain None values before analysis runs."""
    state = extract_cinematic_state({
        "final_report": None,
        "profile": None,
        "tool_results": None,
        "preview_df": None,
        "metadata": None,
        "stage_log": None,
        "theme": "day",
    })
    assert state["theme"] == "day"
    assert state["dataset"]["name"] == "sample_dataset.csv"
    assert len(state["stages"]) == 7
    assert state["ml"]["best_model"] == "GradientBoosting"
    assert "findings" in state["synthesis"]


def test_extract_cinematic_state_with_real_profile_columns() -> None:
    """profile["columns"] is a LIST of column dicts, not a mapping.

    Both DatasetProfile.to_dict() (`[c.to_dict() for c in self.columns]`) and
    the sample-report demo state build it as a list. Every other test here
    passes profile=None, which lands on the `{}` fallback and hides the
    difference — so the list shape went unexercised.
    """
    state = extract_cinematic_state({
        "profile": {
            "quality_score": 92,
            "column_count": 4,
            "columns": [
                {"name": "tenure", "kind": "numeric", "dtype": "int64"},
                {"name": "charges", "kind": "numeric", "dtype": "float64"},
                {"name": "contract", "kind": "categorical", "dtype": "str"},
                {"name": "signed_at", "kind": "datetime", "dtype": "datetime64[ns]"},
            ],
        },
        "theme": "day",
    })
    assert state["dataset"]["quality_score"] == 92
    assert state["dataset"]["column_types"]["numeric"] == 2
    assert state["dataset"]["column_types"]["categorical"] == 1
    assert state["dataset"]["column_types"]["datetime"] == 1


def test_extract_cinematic_state_with_mapping_profile_columns() -> None:
    """A name->column mapping must keep working alongside the list shape."""
    state = extract_cinematic_state({
        "profile": {
            "quality_score": 88,
            "columns": {
                "tenure": {"kind": "numeric"},
                "contract": {"kind": "categorical"},
            },
        },
        "theme": "day",
    })
    assert state["dataset"]["quality_score"] == 88
    assert state["dataset"]["column_types"]["numeric"] == 1
    assert state["dataset"]["column_types"]["categorical"] == 1
