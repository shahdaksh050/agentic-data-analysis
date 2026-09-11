"""Tests for ML pipeline enhancements: auto-treatments, tuning, explainability."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.tools.ml_pipeline import (
    EvaluateModelTool,
    TrainModelTool,
    _prepare_features,
)

RNG = np.random.default_rng(42)


def _make_df(n: int = 200) -> pd.DataFrame:
    signal = RNG.normal(0, 1, n)
    return pd.DataFrame({
        "customer_id": range(1, n + 1),
        "signal": signal,
        "amount": np.exp(RNG.normal(3, 1.2, n)),          # heavy right skew
        "plan": RNG.choice(["basic", "pro"], n),
        "label": (signal + RNG.normal(0, 0.5, n) > 0).astype(int),
    })


class TestAutoTreatments:
    def test_identifier_dropped(self) -> None:
        X, _y, treatments = _prepare_features(_make_df(), "label")
        assert "customer_id" not in X.columns
        assert any("identifier" in t for t in treatments)

    def test_skewed_feature_log_transformed(self) -> None:
        df = _make_df()
        raw_max = float(df["amount"].max())
        X, _y, treatments = _prepare_features(df, "label")
        assert any("log1p" in t and "amount" in t for t in treatments)
        assert float(X["amount"].max()) < raw_max  # compressed

    def test_treatments_deterministic_across_calls(self) -> None:
        df = _make_df()
        _, _, t1 = _prepare_features(df, "label")
        _, _, t2 = _prepare_features(df.copy(), "label")
        assert t1 == t2

    def test_clean_data_gets_no_treatments(self) -> None:
        df = pd.DataFrame({
            "a": RNG.normal(0, 1, 100),
            "b": RNG.normal(5, 2, 100),
            "label": RNG.choice([0, 1], 100),
        })
        _, _, treatments = _prepare_features(df, "label")
        assert treatments == []

    def test_imbalance_triggers_class_weighting(self, tmp_path: Path) -> None:
        n = 400
        f = RNG.normal(0, 1, n)
        label = np.zeros(n, dtype=int)
        label[np.argsort(f)[-20:]] = 1  # 5% minority
        df = pd.DataFrame({"f": f, "g": RNG.normal(0, 1, n), "label": label})
        p = tmp_path / "imb.csv"
        df.to_csv(p, index=False)

        result = TrainModelTool().run(
            file_path=str(p), target_column="label", task_type="classification",
            models=["logistic_regression"], tune_hyperparameters=False,
            output_dir=str(tmp_path / "m"),
        )
        assert result.status == "success"
        assert any("Class weighting" in t for t in result.output["treatments_applied"])

    def test_balanced_data_gets_no_class_weighting(self, tmp_path: Path) -> None:
        df = _make_df()
        p = tmp_path / "bal.csv"
        df.to_csv(p, index=False)
        result = TrainModelTool().run(
            file_path=str(p), target_column="label", task_type="classification",
            models=["logistic_regression"], tune_hyperparameters=False,
            output_dir=str(tmp_path / "m"),
        )
        assert not any("Class weighting" in t for t in result.output["treatments_applied"])

    def test_datetime_column_expanded_not_dropped(self) -> None:
        """P0.4: a time-series dataset's best-detected signal must survive
        into the feature matrix instead of being thrown away."""
        df = _make_df()
        df["signup_date"] = pd.date_range("2023-01-01", periods=len(df), freq="D")
        X, _y, treatments = _prepare_features(df, "label")
        assert "signup_date" not in X.columns
        assert any("Expanded datetime column" in t for t in treatments)
        for suffix in ("year", "month", "day", "dayofweek", "hour", "is_weekend", "days_since_min"):
            assert f"signup_date_{suffix}" in X.columns
        assert not X["signup_date_days_since_min"].isna().any()


class TestSplitStrategy:
    """P0.3: the profiler detects time-series/panel structure — the
    splitter must actually hear about it instead of shuffling regardless."""

    def test_prepare_params_wires_time_series_from_profile(self) -> None:
        from src.core.memory import MemorySystem

        memory = MemorySystem()
        memory.set_context(
            "data_profile",
            {"is_time_series": True, "datetime_cols": ["signup_date"], "panel_group_cols": []},
        )
        params = TrainModelTool().prepare_params(
            {"file_path": "x.csv", "target_column": "label"}, memory, "output"
        )
        assert params["split_strategy"] == "time_series"
        assert params["time_column"] == "signup_date"

    def test_prepare_params_wires_panel_from_profile(self) -> None:
        from src.core.memory import MemorySystem

        memory = MemorySystem()
        memory.set_context(
            "data_profile",
            {"is_time_series": False, "datetime_cols": [], "panel_group_cols": ["customer_id"]},
        )
        params = TrainModelTool().prepare_params(
            {"file_path": "x.csv", "target_column": "label"}, memory, "output"
        )
        assert params["split_strategy"] == "panel"
        assert params["group_column"] == "customer_id"

    def test_prepare_params_leaves_random_when_planner_chose_it(self) -> None:
        from src.core.memory import MemorySystem

        memory = MemorySystem()
        memory.set_context(
            "data_profile",
            {"is_time_series": True, "datetime_cols": ["signup_date"], "panel_group_cols": []},
        )
        params = TrainModelTool().prepare_params(
            {"file_path": "x.csv", "target_column": "label", "split_strategy": "random"},
            memory, "output",
        )
        assert params["split_strategy"] == "random"

    def test_time_series_split_reported_and_chronological(self, tmp_path: Path) -> None:
        n = 200
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        # Oscillating signal (both classes present throughout) so every
        # TimeSeriesSplit fold sees both classes — isolates the invariant
        # under test (chronological split, reported strategy) from noise.
        wave = np.sin(np.linspace(0, 6 * np.pi, n))
        df = pd.DataFrame({
            "date": dates,
            "f1": wave + RNG.normal(0, 0.1, n),
            "label": (wave + RNG.normal(0, 0.1, n) > 0).astype(int),
        })
        p = tmp_path / "ts.csv"
        df.to_csv(p, index=False)

        result = TrainModelTool().run(
            file_path=str(p), target_column="label", task_type="classification",
            models=["logistic_regression"], tune_hyperparameters=False,
            split_strategy="time_series", time_column="date",
            output_dir=str(tmp_path / "m"),
        )
        assert result.status == "success"
        assert result.output["split_strategy"] == "time_series"
        assert result.output["train_samples"] + result.output["test_samples"] == n
        # Chronological: the held-out fraction is exactly the tail, so
        # test_samples matches test_size applied to the full ordered series.
        assert result.output["test_samples"] == int(n * 0.2)

    def test_panel_split_keeps_entities_out_of_both_folds(self, tmp_path: Path) -> None:
        n_entities = 40
        rows_per_entity = 5
        rng = np.random.default_rng(3)
        entity = np.repeat(np.arange(n_entities), rows_per_entity)
        f1 = rng.normal(0, 1, n_entities * rows_per_entity)
        label = (f1 + rng.normal(0, 0.3, n_entities * rows_per_entity) > 0).astype(int)
        df = pd.DataFrame({"entity_id": entity, "f1": f1, "label": label})
        p = tmp_path / "panel.csv"
        df.to_csv(p, index=False)

        result = TrainModelTool().run(
            file_path=str(p), target_column="label", task_type="classification",
            models=["logistic_regression"], tune_hyperparameters=False,
            split_strategy="panel", group_column="entity_id",
            output_dir=str(tmp_path / "m"),
        )
        assert result.status == "success"
        assert result.output["split_strategy"] == "panel"

    def test_evaluate_reuses_trains_split_strategy(self, tmp_path: Path) -> None:
        """EvaluateModelTool must recreate train_model's actual partition —
        not always a random one — or its 'held-out' metrics are inflated
        on exactly the datasets P0.3 targets."""
        n = 200
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        wave = np.sin(np.linspace(0, 6 * np.pi, n))
        df = pd.DataFrame({
            "date": dates,
            "f1": wave + RNG.normal(0, 0.1, n),
            "label": (wave + RNG.normal(0, 0.1, n) > 0).astype(int),
        })
        p = tmp_path / "ts_eval.csv"
        df.to_csv(p, index=False)
        models_dir = str(tmp_path / "m")

        train = TrainModelTool().run(
            file_path=str(p), target_column="label", task_type="classification",
            models=["logistic_regression"], tune_hyperparameters=False,
            split_strategy="time_series", time_column="date",
            output_dir=models_dir,
        )
        assert train.status == "success"
        model_path = train.output["models_trained"]["logistic_regression"]["model_path"]

        eval_result = EvaluateModelTool().run(
            model_path=model_path, file_path=str(p), target_column="label",
            task_type="classification",
            split_strategy="time_series", time_column="date",
        )
        assert eval_result.status == "success"
        # Same chronological partition train_model used: the trained model's
        # own test-fold accuracy must match evaluate_model's, since both are
        # scoring the identical held-out rows.
        assert eval_result.output["accuracy"] == train.output["models_trained"]["logistic_regression"]["test_metrics"]["accuracy"]

    def test_evaluate_prepare_params_pulls_persisted_split_strategy(self) -> None:
        from src.core.memory import MemorySystem

        memory = MemorySystem()
        memory.set_context("split_strategy", "panel")
        memory.set_context("split_time_column", None)
        memory.set_context("split_group_column", "entity_id")
        params = EvaluateModelTool().prepare_params(
            {"model_path": "m.pkl", "file_path": "x.csv", "target_column": "label"},
            memory, "output",
        )
        assert params["split_strategy"] == "panel"
        assert params["group_column"] == "entity_id"


class TestHyperparameterTuning:
    def test_tuning_reports_best_params(self, tmp_path: Path) -> None:
        df = _make_df()
        p = tmp_path / "tune.csv"
        df.to_csv(p, index=False)
        result = TrainModelTool().run(
            file_path=str(p), target_column="label", task_type="classification",
            models=["logistic_regression"], tune_hyperparameters=True,
            output_dir=str(tmp_path / "m"),
        )
        assert result.status == "success"
        assert result.output["hyperparameter_tuning"] is True
        params = result.output["models_trained"]["logistic_regression"]["best_params"]
        assert "C" in params

    def test_tuning_can_be_disabled(self, tmp_path: Path) -> None:
        df = _make_df()
        p = tmp_path / "notune.csv"
        df.to_csv(p, index=False)
        result = TrainModelTool().run(
            file_path=str(p), target_column="label", task_type="classification",
            models=["logistic_regression"], tune_hyperparameters=False,
            output_dir=str(tmp_path / "m"),
        )
        assert result.output["hyperparameter_tuning"] is False
        assert result.output["models_trained"]["logistic_regression"]["best_params"] == {}


class TestExplainability:
    @pytest.fixture
    def trained(self, tmp_path: Path) -> tuple[str, str]:
        df = _make_df()
        p = tmp_path / "data.csv"
        df.to_csv(p, index=False)
        train = TrainModelTool().run(
            file_path=str(p), target_column="label", task_type="classification",
            models=["random_forest"], tune_hyperparameters=False,
            output_dir=str(tmp_path / "m"),
        )
        model_path = train.output["models_trained"]["random_forest"]["model_path"]
        return str(p), model_path

    def test_drivers_identify_the_signal(self, trained: tuple[str, str]) -> None:
        data_path, model_path = trained
        result = EvaluateModelTool().run(
            model_path=model_path, file_path=data_path,
            target_column="label", task_type="classification",
        )
        assert result.status == "success"
        drivers = result.output["top_drivers"]
        assert drivers, "permutation importance must yield at least one driver"
        assert drivers[0]["feature"] == "signal"
        assert drivers[0]["direction"] == "increases"

    def test_narrative_is_plain_language(self, trained: tuple[str, str]) -> None:
        data_path, model_path = trained
        result = EvaluateModelTool().run(
            model_path=model_path, file_path=data_path,
            target_column="label", task_type="classification",
        )
        narrative = result.output["driver_narrative"]
        assert narrative
        assert "driver" in narrative[0]
        assert "signal" in narrative[0]
