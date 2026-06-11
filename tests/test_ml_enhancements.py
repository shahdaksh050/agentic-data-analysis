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
