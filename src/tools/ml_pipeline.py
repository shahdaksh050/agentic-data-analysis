"""
ML Pipeline Tools — Execution Layer.

Handles automatic model training and evaluation.
Supports classification, regression, and clustering task types.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.tools.base import BaseTool, ToolExecutionError


class TrainModelTool(BaseTool):
    """
    Trains one or more ML models on a cleaned dataset.

    Supports:
      - Classification: RandomForest, XGBoost, LogisticRegression
      - Regression: RandomForest, XGBoost, LinearRegression, Ridge
      - Clustering: KMeans, DBSCAN
    """

    name = "train_model"
    description = (
        "Train one or more machine learning models. Auto-detects task type "
        "from the target column. Returns model paths and training metrics."
    )

    CLASSIFICATION_MODELS = ["random_forest", "xgboost", "logistic_regression"]
    REGRESSION_MODELS = ["random_forest", "xgboost", "linear_regression", "ridge"]
    CLUSTERING_MODELS = ["kmeans", "dbscan"]

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        target_column: str,
        task_type: str = "auto",
        models: list[str] | None = None,
        test_size: float = 0.2,
        output_dir: str = "output/models",
        **_: Any,
    ) -> dict[str, Any]:
        import pickle  # noqa: S403

        from sklearn.model_selection import train_test_split  # type: ignore
        from sklearn.preprocessing import LabelEncoder  # type: ignore

        path = Path(file_path)
        df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_excel(path)

        if target_column not in df.columns:
            raise ToolExecutionError(f"Target column '{target_column}' not in dataset.")

        df = df.dropna(subset=[target_column])
        y = df[target_column]
        X = df.drop(columns=[target_column])

        # Encode categorical features
        for col in X.select_dtypes(include="object").columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

        # Auto-detect task type
        if task_type == "auto":
            if y.dtype == object or y.nunique() <= 20:
                task_type = "classification"
            else:
                task_type = "regression"

        # Select models to train
        if models is None:
            if task_type == "classification":
                models = self.CLASSIFICATION_MODELS
            elif task_type == "regression":
                models = self.REGRESSION_MODELS
            else:
                models = self.CLUSTERING_MODELS

        if task_type in {"classification", "regression"}:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        else:
            X_train, X_test, y_train, y_test = X, X, y, y  # clustering

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        results: dict[str, Any] = {}

        for model_name in models:
            model = self._build_model(model_name, task_type)
            if model is None:
                continue
            model.fit(X_train, y_train)  # type: ignore[arg-type]
            metrics = self._evaluate(model, X_test, y_test, task_type)
            model_path = Path(output_dir) / f"{model_name}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            results[model_name] = {**metrics, "model_path": str(model_path)}

        best_model = self._pick_best(results, task_type)

        return {
            "summary": (
                f"Trained {len(results)} models ({task_type}). "
                f"Best: {best_model} with metrics: {results.get(best_model, {})}."
            ),
            "task_type": task_type,
            "models_trained": results,
            "best_model": best_model,
            "test_size": test_size,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }

    def _build_model(self, name: str, task_type: str) -> Any:
        """Instantiate named model for the given task type."""
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # type: ignore
        from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge  # type: ignore
        from sklearn.cluster import KMeans, DBSCAN  # type: ignore

        model_map: dict[str, Any] = {
            "random_forest_classification": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "random_forest_regression": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "logistic_regression": LogisticRegression(max_iter=500, random_state=42),
            "linear_regression": LinearRegression(),
            "ridge": Ridge(alpha=1.0),
            "kmeans": KMeans(n_clusters=3, random_state=42, n_init="auto"),
            "dbscan": DBSCAN(eps=0.5, min_samples=5),
        }

        try:
            from xgboost import XGBClassifier, XGBRegressor  # type: ignore
            model_map["xgboost_classification"] = XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss")
            model_map["xgboost_regression"] = XGBRegressor(n_estimators=100, random_state=42)
        except ImportError:
            pass

        key = f"{name}_{task_type}" if f"{name}_{task_type}" in model_map else name
        return model_map.get(key)

    def _evaluate(
        self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, task_type: str
    ) -> dict[str, float]:
        """Compute evaluation metrics for the trained model."""
        from sklearn.metrics import (  # type: ignore
            accuracy_score, f1_score, roc_auc_score,
            mean_squared_error, mean_absolute_error, r2_score,
        )

        y_pred = model.predict(X_test)

        if task_type == "classification":
            metrics: dict[str, float] = {
                "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
                "f1_score": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
            }
            try:
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_test)
                    if y_prob.shape[1] == 2:
                        metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_prob[:, 1])), 4)
            except Exception:  # noqa: BLE001
                pass
        elif task_type == "regression":
            metrics = {
                "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
                "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
                "r2": round(float(r2_score(y_test, y_pred)), 4),
            }
        else:
            metrics = {}

        return metrics

    def _pick_best(self, results: dict[str, Any], task_type: str) -> str:
        """Select the best model based on the primary metric."""
        if not results:
            return "none"
        if task_type == "classification":
            return max(results, key=lambda m: results[m].get("f1_score", 0))
        elif task_type == "regression":
            return max(results, key=lambda m: results[m].get("r2", -float("inf")))
        return list(results.keys())[0]

    def get_schema(self) -> dict[str, Any]:
        return {
            "file_path": {"type": "string", "description": "Path to cleaned dataset.", "required": True},
            "target_column": {"type": "string", "description": "The column to predict.", "required": True},
            "task_type": {
                "type": "string",
                "description": "'classification', 'regression', 'clustering', or 'auto'.",
                "required": False,
            },
            "models": {
                "type": "list[string]",
                "description": "List of model names to train. Defaults to all suitable models.",
                "required": False,
            },
            "test_size": {"type": "float", "description": "Train/test split ratio. Default: 0.2.", "required": False},
        }


class EvaluateModelTool(BaseTool):
    """Load a saved model and evaluate it on a dataset, generating a detailed report."""

    name = "evaluate_model"
    description = (
        "Load a previously saved model (pickle file) and perform detailed evaluation "
        "including confusion matrix (classification) or residual analysis (regression)."
    )

    def execute(  # type: ignore[override]
        self,
        model_path: str,
        file_path: str,
        target_column: str,
        task_type: str = "classification",
        **_: Any,
    ) -> dict[str, Any]:
        import pickle  # noqa: S403
        from sklearn.preprocessing import LabelEncoder  # type: ignore
        from sklearn.metrics import classification_report  # type: ignore

        path = Path(file_path)
        df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_excel(path)
        y = df[target_column]
        X = df.drop(columns=[target_column])

        for col in X.select_dtypes(include="object").columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

        with open(model_path, "rb") as f:
            model = pickle.load(f)  # noqa: S301

        y_pred = model.predict(X)

        if task_type == "classification":
            report = classification_report(y, y_pred, output_dict=True, zero_division=0)
            return {
                "summary": f"Evaluation complete. Accuracy: {report.get('accuracy', 0):.4f}.",
                "classification_report": report,
                "task_type": task_type,
            }
        else:
            from sklearn.metrics import mean_squared_error, r2_score  # type: ignore

            rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
            r2 = float(r2_score(y, y_pred))
            return {
                "summary": f"Evaluation complete. RMSE: {rmse:.4f}, R²: {r2:.4f}.",
                "rmse": round(rmse, 4),
                "r2": round(r2, 4),
                "task_type": task_type,
            }

    def get_schema(self) -> dict[str, Any]:
        return {
            "model_path": {"type": "string", "description": "Path to the .pkl model file.", "required": True},
            "file_path": {"type": "string", "description": "Path to the evaluation dataset.", "required": True},
            "target_column": {"type": "string", "description": "The target column.", "required": True},
            "task_type": {"type": "string", "description": "'classification' or 'regression'.", "required": False},
        }
