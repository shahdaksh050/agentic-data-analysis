"""Unit tests for Data Processing Tools."""
from __future__ import annotations

import pandas as pd
import pytest

from src.tools.data_processing import (
    CleanDataTool,
    CorrelationAnalysisTool,
    DetectOutliersTool,
    IngestDatasetTool,
)


@pytest.fixture
def sample_csv(tmp_path: pytest.TempPathFactory) -> str:
    """Create a small sample CSV file for testing."""
    df = pd.DataFrame({
        "age": [25, 30, 35, None, 28, 45, 200, 22],  # 200 is an outlier
        "income": [50000.0, 60000.0, None, 70000.0, 55000.0, 80000.0, 90000.0, 40000.0],
        "score": [0.8, 0.6, 0.9, 0.7, 0.5, 0.85, 0.95, 0.4],
        "churn": [0, 0, 1, 1, 0, 0, 1, 1],
    })
    path = tmp_path / "test_data.csv"
    df.to_csv(path, index=False)
    return str(path)


class TestIngestDatasetTool:
    def test_ingests_csv_successfully(self, sample_csv: str) -> None:
        tool = IngestDatasetTool()
        result = tool.run(file_path=sample_csv)
        assert result.status == "success"
        assert result.output["metadata"]["row_count"] == 8
        assert result.output["metadata"]["column_count"] == 4

    def test_returns_correct_column_types(self, sample_csv: str) -> None:
        result = IngestDatasetTool().run(file_path=sample_csv)
        metadata = result.output["metadata"]
        assert "age" in metadata["numerical_cols"]
        assert "churn" in metadata["numerical_cols"]

    def test_detects_missing_values(self, sample_csv: str) -> None:
        result = IngestDatasetTool().run(file_path=sample_csv)
        missing = result.output["metadata"]["missing_values"]
        assert missing.get("age", 0) == 1
        assert missing.get("income", 0) == 1

    def test_file_not_found_returns_error(self) -> None:
        result = IngestDatasetTool().run(file_path="/nonexistent/path/data.csv")
        assert result.status == "error"
        assert result.error_message is not None


class TestCleanDataTool:
    def test_median_imputation_removes_missing(self, sample_csv: str) -> None:
        result = CleanDataTool().run(file_path=sample_csv, strategy="median")
        assert result.status == "success"
        assert result.output["missing_after"] == 0

    def test_drop_rows_reduces_row_count(self, sample_csv: str) -> None:
        result = CleanDataTool().run(file_path=sample_csv, strategy="drop_rows")
        assert result.status == "success"
        assert result.output["rows_after"] < result.output["rows_before"]

    def test_invalid_strategy_returns_error(self, sample_csv: str) -> None:
        result = CleanDataTool().run(file_path=sample_csv, strategy="invalid_strategy")
        assert result.status == "error"


class TestDetectOutliersTool:
    def test_iqr_detects_outlier(self, sample_csv: str) -> None:
        result = DetectOutliersTool().run(file_path=sample_csv, method="iqr")
        assert result.status == "success"
        assert result.output["total_outliers"] >= 1  # age=200 is an outlier

    def test_zscore_method_works(self, sample_csv: str) -> None:
        result = DetectOutliersTool().run(file_path=sample_csv, method="zscore", threshold=2.0)
        assert result.status == "success"


class TestCorrelationAnalysisTool:
    def test_returns_top_correlations(self, sample_csv: str) -> None:
        result = CorrelationAnalysisTool().run(file_path=sample_csv, target_column="churn")
        assert result.status == "success"
        assert len(result.output["top_correlations"]) > 0

    def test_returns_target_correlations_when_specified(self, sample_csv: str) -> None:
        result = CorrelationAnalysisTool().run(file_path=sample_csv, target_column="churn")
        assert "churn" not in result.output["target_correlations"]  # self-corr excluded
        assert len(result.output["target_correlations"]) > 0
