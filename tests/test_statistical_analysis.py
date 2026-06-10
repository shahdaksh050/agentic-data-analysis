"""Unit tests for Statistical Analysis Tools — Stage 3 hypothesis testing."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.tools.statistical_analysis import SelectStatisticalTestTool


@pytest.fixture
def grouped_csv(tmp_path: pytest.TempPathFactory) -> str:
    """Numeric feature with a binary group column — t-test / Mann-Whitney territory."""
    rng = np.random.default_rng(42)
    n = 120
    group = rng.integers(0, 2, n)
    df = pd.DataFrame(
        {
            "row_id": range(1, n + 1),
            "value": rng.normal(0, 1, n) + group * 0.8,
            "category": rng.choice(["a", "b", "c"], n),
            "group": group,
        }
    )
    p = tmp_path / "grouped.csv"
    df.to_csv(p, index=False)
    return str(p)


class TestSelectStatisticalTestTool:
    def test_numeric_feature_two_groups_succeeds(self, grouped_csv: str) -> None:
        result = SelectStatisticalTestTool().run(
            file_path=grouped_csv, feature_column="value", group_column="group"
        )
        assert result.status == "success"
        assert result.output["n_groups"] == 2
        assert result.output["test_name"] in (
            "Independent T-Test", "Welch's T-Test", "Mann-Whitney U"
        )

    def test_p_value_in_valid_range(self, grouped_csv: str) -> None:
        result = SelectStatisticalTestTool().run(
            file_path=grouped_csv, feature_column="value", group_column="group"
        )
        assert 0.0 <= result.output["p_value"] <= 1.0

    def test_categorical_feature_uses_chi_square(self, grouped_csv: str) -> None:
        result = SelectStatisticalTestTool().run(
            file_path=grouped_csv, feature_column="category", group_column="group"
        )
        assert result.status == "success"
        assert result.output["test_name"] == "Chi-Square Test of Independence"

    def test_missing_feature_column_errors(self, grouped_csv: str) -> None:
        result = SelectStatisticalTestTool().run(
            file_path=grouped_csv, feature_column="nope", group_column="group"
        )
        assert result.status == "error"

    def test_id_like_column_rejected(self, grouped_csv: str) -> None:
        """Monotonic, ~100% unique columns are row IDs, not features."""
        result = SelectStatisticalTestTool().run(
            file_path=grouped_csv, feature_column="row_id", group_column="group"
        )
        assert result.status == "error"
        assert "row ID" in (result.error_message or "")

    def test_interpretation_present(self, grouped_csv: str) -> None:
        result = SelectStatisticalTestTool().run(
            file_path=grouped_csv, feature_column="value", group_column="group"
        )
        assert result.output["interpretation"]
        assert "summary" in result.output

    def test_summary_key_present(self, grouped_csv: str) -> None:
        result = SelectStatisticalTestTool().run(
            file_path=grouped_csv, feature_column="value", group_column="group"
        )
        assert "summary" in result.output
