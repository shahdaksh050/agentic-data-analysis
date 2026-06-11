"""Unit tests for src/core/html_report.py — the shareable HTML report."""
from __future__ import annotations

from typing import Any

from src.core.html_report import build_html_report

INSIGHTS: dict[str, Any] = {
    "reasoning": "Support calls drive churn.",
    "insights": ["High call volume predicts churn."],
    "recommendations": ["Add proactive support outreach."],
    "best_model": "random_forest",
    "key_metrics": {"cv_mean": 0.86},
}

CHARTS: list[dict[str, Any]] = [{
    "chart_id": "hist_x",
    "title": "Distribution — x",
    "description": "Histogram of x.",
    "spec": {"data": {"values": [{"x": 1}]}, "mark": "bar", "encoding": {}},
}]

TOOL_RESULTS: list[dict[str, Any]] = [
    {"tool_name": "train_model", "status": "success",
     "output": {"summary": "trained", "treatments_applied": ["Applied log1p to 'amount'."]}},
    {"tool_name": "evaluate_model", "status": "success",
     "output": {"summary": "evaluated",
                "driver_narrative": ["#1 driver: 'support_calls' — higher values push toward '1'."]}},
]


class TestBuildHtmlReport:
    def test_contains_all_sections(self) -> None:
        doc = build_html_report(
            "churn", INSIGHTS, TOOL_RESULTS, CHARTS,
            objective="what drives churn?",
            profile={"quality_score": 91, "row_count": 300, "column_count": 8},
        )
        assert doc.startswith("<!DOCTYPE html>")
        assert "what drives churn?" in doc
        assert "Support calls drive churn." in doc
        assert "High call volume predicts churn." in doc
        assert "Add proactive support outreach." in doc
        assert "support_calls" in doc          # drivers section
        assert "log1p" in doc                  # treatments section
        assert "quality 91/100" in doc
        assert "vegaEmbed" in doc and "Distribution — x" in doc

    def test_html_escaping_of_malicious_content(self) -> None:
        evil = {"reasoning": "<script>alert('xss')</script>", "insights": [], "recommendations": []}
        doc = build_html_report("ds", evil, [], [])
        assert "<script>alert" not in doc
        assert "&lt;script&gt;" in doc

    def test_script_close_tag_in_chart_data_neutralised(self) -> None:
        charts = [{
            "chart_id": "c", "title": "t", "description": "d",
            "spec": {"data": {"values": [{"v": "</script><script>alert(1)</script>"}]},
                     "mark": "bar"},
        }]
        doc = build_html_report("ds", {}, [], charts)
        # The raw close tag must never appear inside the embedded JSON
        assert "</script><script>alert(1)" not in doc

    def test_minimal_inputs_produce_valid_shell(self) -> None:
        doc = build_html_report("empty", {}, [], [])
        assert "<h1>Analysis Report — empty</h1>" in doc
        assert "Interactive Dashboard" not in doc  # no charts, no section
