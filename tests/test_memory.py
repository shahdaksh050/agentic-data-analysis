"""Unit tests for the Memory System."""
from __future__ import annotations

import pytest

from src.core.memory import AnalysisStep, DatasetMetadata, MemorySystem, ToolResult


@pytest.fixture
def sample_metadata() -> DatasetMetadata:
    return DatasetMetadata(
        file_path="test_data.csv",
        row_count=1000,
        column_count=5,
        columns={"age": "int64", "income": "float64", "churn": "int64"},
        missing_values={"income": 12},
        numerical_cols=["age", "income"],
        categorical_cols=[],
        target_column="churn",
        task_type="classification",
    )


@pytest.fixture
def memory() -> MemorySystem:
    return MemorySystem()


class TestMemorySystem:
    def test_store_and_retrieve_metadata(self, memory: MemorySystem, sample_metadata: DatasetMetadata) -> None:
        memory.store_dataset_metadata(sample_metadata)
        assert memory.dataset_metadata is not None
        assert memory.dataset_metadata.row_count == 1000

    def test_get_metadata_prompt_is_string(self, memory: MemorySystem, sample_metadata: DatasetMetadata) -> None:
        memory.store_dataset_metadata(sample_metadata)
        prompt = memory.get_metadata_prompt()
        assert isinstance(prompt, str)
        assert "1000" in prompt

    def test_get_metadata_prompt_raises_if_no_dataset(self, memory: MemorySystem) -> None:
        with pytest.raises(ValueError, match="No dataset metadata"):
            memory.get_metadata_prompt()

    def test_append_tool_result(self, memory: MemorySystem) -> None:
        result = ToolResult(tool_name="clean_data", status="success", output={"summary": "done"})
        memory.append_tool_result(result)
        assert len(memory.tool_results) == 1

    def test_results_summary_shows_all_tools(self, memory: MemorySystem) -> None:
        for name in ["clean_data", "detect_outliers"]:
            memory.append_tool_result(
                ToolResult(tool_name=name, status="success", output={"summary": "ok"})
            )
        summary = memory.get_results_summary()
        assert "clean_data" in summary
        assert "detect_outliers" in summary

    def test_pending_steps_filters_incomplete(self, memory: MemorySystem) -> None:
        steps = [
            AnalysisStep(1, "clean_data", {}, "rationale"),
            AnalysisStep(2, "detect_outliers", {}, "rationale"),
        ]
        memory.store_analysis_plan(steps)
        # Mark step 1 complete
        result = ToolResult(tool_name="clean_data", status="success", output={})
        memory.mark_step_complete(1, result)
        pending = memory.get_pending_steps()
        assert len(pending) == 1
        assert pending[0].step_number == 2

    def test_generic_context_store(self, memory: MemorySystem) -> None:
        memory.set_context("sub_result_A", {"score": 0.92})
        assert memory.get_context("sub_result_A") == {"score": 0.92}
        assert memory.get_context("missing_key", "default") == "default"
