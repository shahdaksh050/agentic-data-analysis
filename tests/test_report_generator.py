"""Unit tests for the Report Generator Tool — Stage 7."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.tools.report_generator import GenerateReportTool

LLM_INSIGHTS = {
    "status": "complete",
    "reasoning": "All analyses completed successfully.",
    "insights": ["Feature A predicts the target.", "No overfitting detected."],
    "recommendations": ["Deploy the best model."],
    "best_model": "random_forest",
    "key_metrics": {"cv_mean": "0.87", "train_test_gap": "0.03"},
}

TOOL_RESULTS = [
    {
        "tool_name": "clean_data",
        "status": "success",
        "output": {"summary": "Cleaned 5 missing values."},
        "error": None,
    }
]


@pytest.fixture
def report_dir(tmp_path: pytest.TempPathFactory) -> str:
    return str(tmp_path / "reports")


class TestGenerateReportTool:
    def test_markdown_and_json_files_created(self, report_dir: str) -> None:
        result = GenerateReportTool().run(
            dataset_name="unit_test",
            tool_results_json=json.dumps(TOOL_RESULTS),
            llm_insights=LLM_INSIGHTS,
            output_dir=report_dir,
        )
        assert result.status == "success"
        assert Path(result.output["markdown_path"]).exists()
        assert Path(result.output["json_path"]).exists()

    def test_markdown_contains_expected_sections(self, report_dir: str) -> None:
        result = GenerateReportTool().run(
            dataset_name="unit_test",
            tool_results_json=json.dumps(TOOL_RESULTS),
            llm_insights=LLM_INSIGHTS,
            output_dir=report_dir,
        )
        content = Path(result.output["markdown_path"]).read_text(encoding="utf-8")
        assert "## Key Insights" in content
        assert "## Recommendations" in content
        assert "## Model Performance" in content
        assert "## Tool Execution Log" in content
        assert "clean_data" in content

    def test_tolerates_missing_llm_insights(self, report_dir: str) -> None:
        result = GenerateReportTool().run(
            dataset_name="no_insights",
            tool_results_json="[]",
            llm_insights=None,
            output_dir=report_dir,
        )
        assert result.status == "success"

    def test_tolerates_double_serialised_insights(self, report_dir: str) -> None:
        """The LLM sometimes passes llm_insights as a JSON string."""
        result = GenerateReportTool().run(
            dataset_name="stringy",
            tool_results_json="[]",
            llm_insights=json.dumps(LLM_INSIGHTS),
            output_dir=report_dir,
        )
        assert result.status == "success"
        assert result.output["n_insights"] == 2

    def test_tolerates_malformed_tool_results_json(self, report_dir: str) -> None:
        result = GenerateReportTool().run(
            dataset_name="broken",
            tool_results_json="{not json at all",
            llm_insights=LLM_INSIGHTS,
            output_dir=report_dir,
        )
        assert result.status == "success"

    def test_summary_key_present(self, report_dir: str) -> None:
        result = GenerateReportTool().run(
            dataset_name="x", tool_results_json="[]", llm_insights={}, output_dir=report_dir
        )
        assert "summary" in result.output
