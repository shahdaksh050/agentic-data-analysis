"""Unit tests for src/core/dashboard.py — the dynamic dashboard agent."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.core.dashboard import MAX_POINTS, ChartSpec, build_dashboard, dashboard_to_json
from src.core.profiler import profile_dataframe

RNG = np.random.default_rng(7)


@pytest.fixture
def churn_df() -> pd.DataFrame:
    n = 300
    support_calls = RNG.integers(0, 10, n)
    churn = (support_calls + RNG.normal(0, 2, n) > 6).astype(int)
    return pd.DataFrame({
        "customer_id": range(1, n + 1),
        "age": RNG.integers(18, 80, n),
        "monthly_charge": RNG.normal(60, 15, n),
        "support_calls": support_calls,
        "plan": RNG.choice(["basic", "pro"], n),
        "churn": churn,
    })


@pytest.fixture
def train_result() -> dict:  # type: ignore[type-arg]
    return {
        "tool_name": "train_model",
        "status": "success",
        "output": {
            "task_type": "classification",
            "best_model": "random_forest",
            "models_trained": {
                "logistic_regression": {
                    "train_metrics": {"accuracy": 0.82}, "test_metrics": {"accuracy": 0.80},
                    "cv_mean": 0.79,
                },
                "random_forest": {
                    "train_metrics": {"accuracy": 0.95}, "test_metrics": {"accuracy": 0.88},
                    "cv_mean": 0.86,
                },
            },
        },
    }


def _ids(charts: list[ChartSpec]) -> list[str]:
    return [c.chart_id for c in charts]


class TestChartSelection:
    def test_classification_gets_class_balance_and_boxplot(
        self, churn_df: pd.DataFrame
    ) -> None:
        profile = profile_dataframe(churn_df, target_column="churn")
        charts = build_dashboard(
            churn_df, profile, target_column="churn", task_type="classification"
        )
        ids = _ids(charts)
        assert "class_balance" in ids
        assert any(i.startswith("box_") for i in ids)
        assert any(i.startswith("hist_") for i in ids)

    def test_eda_mode_has_no_target_charts(self, churn_df: pd.DataFrame) -> None:
        df = churn_df.drop(columns=["churn"])
        profile = profile_dataframe(df)
        ids = _ids(build_dashboard(df, profile, task_type="eda"))
        assert "class_balance" not in ids
        assert not any(i.startswith("box_") for i in ids)
        assert any(i.startswith("hist_") for i in ids)

    def test_identifiers_never_charted(self, churn_df: pd.DataFrame) -> None:
        profile = profile_dataframe(churn_df, target_column="churn")
        charts = build_dashboard(
            churn_df, profile, target_column="churn", task_type="classification"
        )
        assert "hist_customer_id" not in _ids(charts)

    def test_model_comparison_built_from_tool_results(
        self, churn_df: pd.DataFrame, train_result: dict  # type: ignore[type-arg]
    ) -> None:
        profile = profile_dataframe(churn_df, target_column="churn")
        charts = build_dashboard(
            churn_df, profile, target_column="churn",
            task_type="classification", tool_results=[train_result],
        )
        ids = _ids(charts)
        assert ids[0] == "model_comparison"  # results charts lead the dashboard
        chart = charts[0]
        scores = {v["model"] for v in chart.spec["data"]["values"]}
        assert scores == {"logistic_regression", "random_forest"}

    def test_correlation_chart_from_tool_results(self, churn_df: pd.DataFrame) -> None:
        corr_result = {
            "tool_name": "correlation_analysis",
            "status": "success",
            "output": {"top_correlations": [
                {"col_a": "support_calls", "col_b": "churn", "correlation": 0.49},
            ]},
        }
        profile = profile_dataframe(churn_df, target_column="churn")
        charts = build_dashboard(
            churn_df, profile, target_column="churn",
            task_type="classification", tool_results=[corr_result],
        )
        assert "top_correlations" in _ids(charts)

    def test_failed_tool_results_ignored(self, churn_df: pd.DataFrame) -> None:
        failed = {"tool_name": "train_model", "status": "error", "output": {}}
        profile = profile_dataframe(churn_df, target_column="churn")
        charts = build_dashboard(
            churn_df, profile, target_column="churn",
            task_type="classification", tool_results=[failed],
        )
        assert "model_comparison" not in _ids(charts)

    def test_cluster_chart_from_tool_results(self, churn_df: pd.DataFrame) -> None:
        cluster_result = {
            "tool_name": "cluster_data",
            "status": "success",
            "output": {
                "n_clusters": 3,
                "silhouette_score": 0.61,
                "pca_points": [
                    {"x": float(i), "y": float(-i), "cluster": f"cluster_{i % 3}"}
                    for i in range(30)
                ],
            },
        }
        df = churn_df.drop(columns=["churn"])
        profile = profile_dataframe(df)
        charts = build_dashboard(df, profile, task_type="eda",
                                 tool_results=[cluster_result])
        ids = _ids(charts)
        assert "cluster_scatter" in ids
        spec = next(c for c in charts if c.chart_id == "cluster_scatter").spec
        assert len(spec["data"]["values"]) == 30

    def test_cluster_chart_skipped_without_points(self, churn_df: pd.DataFrame) -> None:
        cluster_result = {
            "tool_name": "cluster_data", "status": "success",
            "output": {"n_clusters": 2, "pca_points": []},
        }
        profile = profile_dataframe(churn_df)
        charts = build_dashboard(churn_df, profile, tool_results=[cluster_result])
        assert "cluster_scatter" not in _ids(charts)

    def test_time_series_chart_for_datetime_data(self) -> None:
        n = 400
        df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=n, freq="D"),
            "sales": RNG.normal(1000, 150, n),
        })
        profile = profile_dataframe(df)
        assert "time_series" in _ids(build_dashboard(df, profile))


class TestSpecQuality:
    def test_all_specs_json_serialisable(
        self, churn_df: pd.DataFrame, train_result: dict  # type: ignore[type-arg]
    ) -> None:
        profile = profile_dataframe(churn_df, target_column="churn")
        charts = build_dashboard(
            churn_df, profile, target_column="churn",
            task_type="classification", tool_results=[train_result],
        )
        parsed = json.loads(dashboard_to_json(charts))
        assert len(parsed) == len(charts)
        assert all("spec" in c and "title" in c for c in parsed)

    def test_inline_data_respects_point_cap(self) -> None:
        n = 5000
        df = pd.DataFrame({
            "x": RNG.normal(0, 1, n),
            "y": RNG.normal(0, 1, n),
        })
        profile = profile_dataframe(df)
        charts = build_dashboard(df, profile)
        for chart in charts:
            values = chart.spec.get("data", {}).get("values", [])
            assert len(values) <= MAX_POINTS

    def test_deterministic_output(self, churn_df: pd.DataFrame) -> None:
        profile = profile_dataframe(churn_df, target_column="churn")
        a = dashboard_to_json(build_dashboard(
            churn_df, profile, target_column="churn", task_type="classification"))
        b = dashboard_to_json(build_dashboard(
            churn_df, profile, target_column="churn", task_type="classification"))
        assert a == b

    def test_empty_dataframe_yields_no_charts(self) -> None:
        df = pd.DataFrame({"a": pd.Series(dtype="float64")})
        profile = profile_dataframe(df)
        assert build_dashboard(df, profile) == []
