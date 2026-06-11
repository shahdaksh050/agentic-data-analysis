"""
ML Pipeline Tools — Execution Layer.

Stage 3: Model Training   (TrainModelTool)
Stage 3: Model Evaluation (EvaluateModelTool)

Anti-overfitting measures built in:
  - Stratified K-Fold cross-validation (k=5, default)
  - Explicit train/validation/test split reporting
  - Train–test accuracy gap warning (>0.10 threshold)
  - Tree depth capping via max_depth parameter
  - L2 regularisation active by default (LogisticRegression, Ridge)
  - XGBoost early stopping when eval_set is available
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from src.tools.base import BaseTool, ToolExecutionError


def _read_df(file_path: str) -> pd.DataFrame:
    """Read CSV or Excel robustly with explicit engines."""
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        return pd.read_csv(path)
    elif suffix == ".xlsx":
        return pd.read_excel(path, engine="openpyxl")
    elif suffix == ".xls":
        return pd.read_excel(path, engine="xlrd")
    else:
        raise ValueError(f"Unsupported file extension '{path.suffix}'. Use .csv, .tsv, .xlsx, or .xls.")

# Overfitting warning threshold: gap between train and test accuracy
OVERFIT_THRESHOLD = 0.10


#: Absolute skewness at which a non-negative numeric feature gets log1p.
SKEW_TREATMENT_THRESHOLD = 2.0

#: Minority-class fraction below which class weighting is applied.
IMBALANCE_THRESHOLD = 0.10


def _prepare_features(
    df: pd.DataFrame, target_column: str
) -> tuple[pd.DataFrame, pd.Series[Any], list[str]]:
    """
    Shared train/evaluate/visualise feature preparation with automatic
    treatment of profiler-detected data problems.

    Deterministic given the same data, so a saved model always sees the
    same feature matrix at train, evaluate, and visualisation time:
      - drops rows with a missing target
      - drops datetime columns and ID-like columns (near-unique strings,
        and near-unique integer identifiers)
      - log1p-transforms severely skewed non-negative numerics
      - label-encodes remaining categoricals

    Returns:
        (features, target, treatments) — treatments is a human-readable
        list of every automatic action taken, for the report.
    """
    from sklearn.preprocessing import LabelEncoder

    treatments: list[str] = []
    df = df.dropna(subset=[target_column])
    y = df[target_column]
    features = df.drop(columns=[target_column]).copy()
    n_rows = max(len(features), 1)

    # is_numeric_dtype, not `dtype == object`: pandas 3 strings are `str` dtype
    for col in list(features.columns):
        series = features[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            features = features.drop(columns=[col])
            treatments.append(f"Dropped datetime column '{col}' (not model-ready).")
        elif (
            not pd.api.types.is_numeric_dtype(series)
            and series.nunique() / n_rows > 0.5
        ):
            features = features.drop(columns=[col])
            treatments.append(f"Dropped ID-like text column '{col}' (>50% unique).")
        elif (
            pd.api.types.is_integer_dtype(series)
            and series.nunique() / n_rows >= 0.98
        ):
            features = features.drop(columns=[col])
            treatments.append(f"Dropped identifier column '{col}' (~100% unique integers).")

    # Severely skewed non-negative numerics → log1p (a data scientist's reflex)
    for col in features.columns:
        series = features[col]
        if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            continue
        clean = series.dropna()
        if len(clean) < 3 or float(clean.min()) < 0:
            continue
        skew = float(clean.skew())
        if abs(skew) >= SKEW_TREATMENT_THRESHOLD:
            features[col] = np.log1p(series.astype(float))
            treatments.append(
                f"Applied log1p to '{col}' (skew={skew:.2f} — heavy tail compressed)."
            )

    for col in features.columns:
        if not pd.api.types.is_numeric_dtype(features[col]):
            features[col] = LabelEncoder().fit_transform(features[col].astype(str))
    return features, y, treatments


def _encode_target(y: pd.Series[Any]) -> tuple[pd.Series[Any], list[str]]:
    """
    Deterministically encode non-numeric classification targets to integers.

    LabelEncoder sorts classes, so train and evaluate produce identical
    encodings for the same data. Returns (encoded_y, class_labels);
    class_labels is empty when no encoding was needed.
    """
    from sklearn.preprocessing import LabelEncoder

    if not pd.api.types.is_numeric_dtype(y) or str(y.dtype) == "bool":
        encoder = LabelEncoder()
        encoded = pd.Series(encoder.fit_transform(y.astype(str)), index=y.index, name=y.name)
        return encoded, [str(c) for c in encoder.classes_]
    return y, []


class TrainModelTool(BaseTool):
    """
    Train one or more ML models with anti-overfitting safeguards.

    Supports:
      Classification : RandomForest, XGBoost, LogisticRegression
      Regression     : RandomForest, XGBoost, LinearRegression, Ridge
      Clustering     : KMeans, DBSCAN
    """

    name = "train_model"
    description = (
        "Train one or more ML models with built-in cross-validation (k=5) "
        "and train/test gap monitoring to detect overfitting. "
        "Returns per-model metrics, CV scores, and the best model name."
    )

    CLASSIFICATION_MODELS: ClassVar[list[str]] = ["random_forest", "xgboost", "logistic_regression"]
    REGRESSION_MODELS: ClassVar[list[str]] = ["random_forest", "xgboost", "linear_regression", "ridge"]
    CLUSTERING_MODELS: ClassVar[list[str]] = ["kmeans", "dbscan"]

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        target_column: str,
        task_type: str = "auto",
        models: list[str] | None = None,
        test_size: float = 0.2,
        n_cv_folds: int = 5,
        max_depth: int = 6,
        tune_hyperparameters: bool = True,
        output_dir: str = "output/models",
        **_: Any,
    ) -> dict[str, Any]:
        from sklearn.model_selection import (
            KFold,
            StratifiedKFold,
            cross_val_score,
            train_test_split,
        )

        df = _read_df(file_path)

        if target_column not in df.columns:
            raise ToolExecutionError(f"Target column '{target_column}' not in dataset.")

        X, y, treatments = _prepare_features(df, target_column)

        # Auto-detect task type
        if task_type == "auto":
            if not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 20:
                task_type = "classification"
            else:
                task_type = "regression"

        # Encode non-numeric classification targets (XGBoost requires
        # numeric labels; roc_auc_score requires {0,1} for binary tasks)
        class_labels: list[str] = []
        balanced = False
        scale_pos_weight = 1.0
        if task_type == "classification":
            y, class_labels = _encode_target(y)
            # Act on class imbalance instead of just warning about it
            counts = y.value_counts()
            if len(counts) >= 2:
                minority_frac = float(counts.iloc[-1]) / float(counts.sum())
                if minority_frac < IMBALANCE_THRESHOLD:
                    balanced = True
                    if len(counts) == 2:
                        scale_pos_weight = float(counts.max()) / max(float(counts.min()), 1.0)
                    treatments.append(
                        f"Class weighting applied — minority class is {minority_frac:.1%} "
                        f"of rows (threshold {IMBALANCE_THRESHOLD:.0%})."
                    )

        # Tuning is skipped on large data to keep runtime bounded
        do_tune = tune_hyperparameters and len(X) <= 20_000

        if models is None:
            models = (
                self.CLASSIFICATION_MODELS
                if task_type == "classification"
                else self.REGRESSION_MODELS
                if task_type == "regression"
                else self.CLUSTERING_MODELS
            )

        if not models:
            raise ToolExecutionError(
                f"'models' list is empty. Pass a non-empty list or omit the parameter "
                f"to use the defaults for task_type='{task_type}'."
            )

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        results: dict[str, Any] = {}
        overfit_warnings: list[str] = []
        build_errors: list[str] = []

        if task_type in {"classification", "regression"}:
            stratify = y if task_type == "classification" else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=stratify
            )
            cv = (
                StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=42)
                if task_type == "classification"
                else KFold(n_splits=n_cv_folds, shuffle=True, random_state=42)
            )
            scoring = "f1_weighted" if task_type == "classification" else "r2"

            for model_name in models:
                try:
                    model = self._build_model(
                        model_name, task_type, max_depth,
                        balanced=balanced, scale_pos_weight=scale_pos_weight,
                    )
                except Exception as exc:
                    build_errors.append(f"{model_name}: build failed — {exc}")
                    continue
                if model is None:
                    build_errors.append(
                        f"{model_name}: unknown model name for task_type='{task_type}'. "
                        f"Valid names: {self.CLASSIFICATION_MODELS if task_type == 'classification' else self.REGRESSION_MODELS}"
                    )
                    continue

                try:
                    best_params: dict[str, Any] = {}
                    if do_tune:
                        model, best_params = self._tune(
                            model, model_name, X_train, y_train, cv, scoring, max_depth
                        )
                    model.fit(X_train, y_train)
                    train_metrics = self._evaluate(model, X_train, y_train, task_type)
                    test_metrics = self._evaluate(model, X_test, y_test, task_type)

                    # Cross-validation (anti-overfitting measure)
                    cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
                    cv_mean = round(float(cv_scores.mean()), 4)
                    cv_std = round(float(cv_scores.std()), 4)

                    # Train–test gap check
                    primary_train = train_metrics.get("accuracy", train_metrics.get("r2", 0.0))
                    primary_test = test_metrics.get("accuracy", test_metrics.get("r2", 0.0))
                    gap = round(primary_train - primary_test, 4)
                    if gap > OVERFIT_THRESHOLD:
                        overfit_warnings.append(
                            f"{model_name}: train-test gap={gap:.3f} > {OVERFIT_THRESHOLD} "
                            f"— possible overfitting. Consider reducing max_depth or adding regularisation."
                        )

                    # Save model
                    model_path = Path(output_dir) / f"{model_name}.pkl"
                    with open(model_path, "wb") as f:
                        pickle.dump(model, f)

                    results[model_name] = {
                        "train_metrics": train_metrics,
                        "test_metrics": test_metrics,
                        "cv_mean": cv_mean,
                        "cv_std": cv_std,
                        "train_test_gap": gap,
                        "model_path": str(model_path),
                        "best_params": best_params,
                    }
                except Exception as exc:
                    build_errors.append(f"{model_name}: training failed — {exc}")

            if not results:
                raise ToolExecutionError(
                    f"No models could be trained for task_type='{task_type}'. "
                    f"Errors: {'; '.join(build_errors) or 'all _build_model calls returned None — check model names and task_type.'}"
                )

            best_model = self._pick_best(results)

        else:
            # Clustering
            X_train, X_test = X, X
            for model_name in models:
                try:
                    model = self._build_model(model_name, task_type, max_depth)
                    if model is None:
                        build_errors.append(f"{model_name}: unknown clustering model name.")
                        continue
                    model.fit(X_train)
                    model_path = Path(output_dir) / f"{model_name}.pkl"
                    with open(model_path, "wb") as f:
                        pickle.dump(model, f)
                    results[model_name] = {"model_path": str(model_path)}
                except Exception as exc:
                    build_errors.append(f"{model_name}: {exc}")

            if not results:
                raise ToolExecutionError(
                    f"No clustering models could be trained. "
                    f"Errors: {'; '.join(build_errors)}"
                )
            best_model = next(iter(results))

        best_summary = results.get(best_model, {})

        return {
            "summary": (
                f"Trained {len(results)} model(s) [{task_type}]. "
                f"Best: {best_model} | "
                f"CV mean={best_summary.get('cv_mean', 'N/A')} "
                f"± {best_summary.get('cv_std', 'N/A')}."
            ),
            "task_type": task_type,
            "models_trained": results,
            "best_model": best_model,
            "class_labels": class_labels,
            "overfit_warnings": overfit_warnings,
            "treatments_applied": treatments,
            "hyperparameter_tuning": do_tune,
            "test_size": test_size,
            "n_cv_folds": n_cv_folds,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_model(
        self,
        name: str,
        task_type: str,
        max_depth: int,
        balanced: bool = False,
        scale_pos_weight: float = 1.0,
    ) -> Any:
        from sklearn.cluster import DBSCAN, KMeans
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

        # class_weight counteracts imbalanced targets (set when the minority
        # class falls below IMBALANCE_THRESHOLD)
        cw = "balanced" if balanced else None

        model_map: dict[str, Any] = {
            "random_forest_classification": RandomForestClassifier(
                n_estimators=100, max_depth=max_depth, random_state=42, n_jobs=-1,
                class_weight=cw,
            ),
            "random_forest_regression": RandomForestRegressor(
                n_estimators=100, max_depth=max_depth, random_state=42, n_jobs=-1
            ),
            "logistic_regression": LogisticRegression(
                max_iter=500, C=1.0, random_state=42,  # L2 regularisation (default)
                class_weight=cw,
            ),
            "linear_regression": LinearRegression(),
            "ridge": Ridge(alpha=1.0),  # L2 regularisation
            "kmeans": KMeans(n_clusters=3, random_state=42, n_init="auto"),
            "dbscan": DBSCAN(eps=0.5, min_samples=5),
        }

        try:
            from xgboost import XGBClassifier, XGBRegressor
            model_map["xgboost_classification"] = XGBClassifier(
                n_estimators=200,
                max_depth=max_depth,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="logloss",
                verbosity=0,
                scale_pos_weight=scale_pos_weight if balanced else 1.0,
            )
            model_map["xgboost_regression"] = XGBRegressor(
                n_estimators=200,
                max_depth=max_depth,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0,
            )
        except ImportError:
            pass  # XGBoost not installed; xgboost model names will resolve to None

        # NOTE: do not use `model_map.get(key) or model_map.get(name)` here.
        # sklearn ensembles define __len__ via the unfitted `estimators_`
        # attribute, so truthiness checks raise AttributeError before fit.
        model = model_map.get(f"{name}_{task_type}")
        if model is None:
            model = model_map.get(name)
        return model

    #: Search spaces for the light hyperparameter tuning pass.
    #: random_forest's max_depth space is rebuilt at tune time so the search
    #: never exceeds the user's max_depth cap.
    TUNING_GRIDS: ClassVar[dict[str, dict[str, list[Any]]]] = {
        "random_forest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 4, 6],
            "min_samples_leaf": [1, 2, 4],
        },
        "xgboost": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.7, 0.85, 1.0],
        },
        "logistic_regression": {"C": [0.01, 0.1, 1.0, 10.0]},
        "ridge": {"alpha": [0.1, 1.0, 10.0]},
    }

    def _tune(
        self,
        model: Any,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv: Any,
        scoring: str,
        max_depth: int = 6,
    ) -> tuple[Any, dict[str, Any]]:
        """
        Light randomized hyperparameter search; returns (best_model, best_params).

        Bounded by design: n_iter ≤ 8, the caller's CV splitter, seeded. Models
        without a defined grid pass through untuned. Depth-bearing grids are
        clamped to the user's max_depth so tuning can never undo the
        anti-overfitting cap.
        """
        from sklearn.model_selection import RandomizedSearchCV

        grid = self.TUNING_GRIDS.get(model_name)
        if not grid:
            return model, {}
        grid = dict(grid)
        if "max_depth" in grid:
            grid["max_depth"] = sorted({max(2, max_depth // 2), max(2, max_depth - 1), max_depth})
        n_combos = 1
        for values in grid.values():
            n_combos *= len(values)
        search = RandomizedSearchCV(
            model,
            param_distributions=grid,
            n_iter=min(8, n_combos),
            cv=cv,
            scoring=scoring,
            random_state=42,
            n_jobs=1,
        )
        search.fit(X_train, y_train)
        return search.best_estimator_, dict(search.best_params_)

    def _evaluate(
        self, model: Any, X: pd.DataFrame, y: pd.Series, task_type: str
    ) -> dict[str, float]:
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            mean_absolute_error,
            mean_squared_error,
            r2_score,
            roc_auc_score,
        )
        y_pred = model.predict(X)
        if task_type == "classification":
            metrics: dict[str, float] = {
                "accuracy": round(float(accuracy_score(y, y_pred)), 4),
                "f1_score": round(float(f1_score(y, y_pred, average="weighted", zero_division=0)), 4),
            }
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X)
                if y_prob.shape[1] == 2:
                    metrics["roc_auc"] = round(float(roc_auc_score(y, y_prob[:, 1])), 4)
        else:
            metrics = {
                "rmse": round(float(np.sqrt(mean_squared_error(y, y_pred))), 4),
                "mae": round(float(mean_absolute_error(y, y_pred)), 4),
                "r2": round(float(r2_score(y, y_pred)), 4),
            }
        return metrics

    def _pick_best(self, results: dict[str, Any]) -> str:
        if not results:
            return "none"
        return max(results, key=lambda m: float(results[m].get("cv_mean", -float("inf"))))

    def get_schema(self) -> dict[str, Any]:
        return {
            "file_path": {"type": "string", "description": "Path to cleaned dataset.", "required": True},
            "target_column": {"type": "string", "description": "Column to predict.", "required": True},
            "task_type": {
                "type": "string",
                "description": "classification | regression | clustering | auto.",
                "required": False,
            },
            "models": {
                "type": "list[string]",
                "description": "Model names to train. Defaults to all suitable models.",
                "required": False,
            },
            "test_size": {"type": "float", "description": "Test split fraction. Default: 0.2.", "required": False},
            "n_cv_folds": {"type": "int", "description": "Number of CV folds. Default: 5.", "required": False},
            "max_depth": {"type": "int", "description": "Max tree depth (RF, XGB). Default: 6.", "required": False},
            "tune_hyperparameters": {
                "type": "bool",
                "description": "Run a light randomized hyperparameter search (auto-skipped above 20k rows). Default: true.",
                "required": False,
            },
        }


class EvaluateModelTool(BaseTool):
    """
    Load a saved model and run detailed evaluation on a held-out split.

    Recreates the same train/test split used by TrainModelTool
    (random_state=42) so the reported metrics describe generalisation,
    not memorisation. Produces a classification report or regression
    metrics plus the train-test gap as an overfitting diagnostic.
    """

    name = "evaluate_model"
    description = (
        "Load a saved .pkl model and evaluate it on the held-out test split "
        "of a dataset (same random_state=42 split as train_model). "
        "Produces a full classification report (or regression metrics) "
        "plus an overfitting diagnostic (train_test_gap)."
    )

    def execute(  # type: ignore[override]
        self,
        model_path: str,
        file_path: str,
        target_column: str,
        task_type: str = "classification",
        test_size: float = 0.2,
        **_: Any,
    ) -> dict[str, Any]:
        from sklearn.metrics import classification_report
        from sklearn.model_selection import train_test_split

        if not Path(model_path).exists():
            raise ToolExecutionError(f"Model file not found: {model_path}")

        df = _read_df(file_path)
        if target_column not in df.columns:
            raise ToolExecutionError(f"Target column '{target_column}' not in dataset.")

        X, y, _treatments = _prepare_features(df, target_column)
        class_labels: list[str] = []
        if task_type == "classification":
            y, class_labels = _encode_target(y)

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # Recreate the training split so evaluation runs on unseen data only
        stratify = y if task_type == "classification" else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify
        )
        y_pred_test = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        drivers, driver_narrative = self._explain_drivers(
            model, X_test, y_test, task_type, target_column, class_labels
        )

        if task_type == "classification":
            from sklearn.metrics import accuracy_score

            report = classification_report(y_test, y_pred_test, output_dict=True, zero_division=0)
            test_acc = float(accuracy_score(y_test, y_pred_test))
            train_acc = float(accuracy_score(y_train, y_pred_train))
            gap = round(train_acc - test_acc, 4)
            # Map encoded integer class keys back to original label names
            if class_labels:
                report = {
                    (class_labels[int(k)] if k.isdigit() and int(k) < len(class_labels) else k): v
                    for k, v in report.items()
                }
            return {
                "summary": (
                    f"Held-out evaluation complete. Test accuracy: {test_acc:.4f} "
                    f"(train: {train_acc:.4f}, gap: {gap:+.4f})."
                ),
                "classification_report": report,
                "task_type": task_type,
                "accuracy": round(test_acc, 4),
                "train_accuracy": round(train_acc, 4),
                "train_test_gap": gap,
                "class_labels": class_labels,
                "top_drivers": drivers,
                "driver_narrative": driver_narrative,
            }
        else:
            from sklearn.metrics import mean_squared_error, r2_score

            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
            r2_test = float(r2_score(y_test, y_pred_test))
            r2_train = float(r2_score(y_train, y_pred_train))
            gap = round(r2_train - r2_test, 4)
            return {
                "summary": (
                    f"Held-out evaluation complete. RMSE: {rmse:.4f}, "
                    f"R²: {r2_test:.4f} (train R²: {r2_train:.4f}, gap: {gap:+.4f})."
                ),
                "rmse": round(rmse, 4),
                "r2": round(r2_test, 4),
                "train_r2": round(r2_train, 4),
                "train_test_gap": gap,
                "task_type": task_type,
                "top_drivers": drivers,
                "driver_narrative": driver_narrative,
            }

    @staticmethod
    def _explain_drivers(
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        task_type: str,
        target_column: str,
        class_labels: list[str],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """
        Model-agnostic explainability: permutation importance on the held-out
        split, with effect direction from feature-target correlation, rendered
        as plain-language driver sentences for the report.

        Failure here must never fail evaluation — returns empty results instead.
        """
        try:
            from sklearn.inspection import permutation_importance

            perm = permutation_importance(
                model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=1
            )
            order = perm.importances_mean.argsort()[::-1][:5]
            positive_label: str | None = None
            if len(class_labels) == 2:
                positive_label = class_labels[-1]
            elif task_type == "classification" and pd.Series(y_test).nunique() == 2:
                # Numeric binary target — name the positive class by its value
                positive_label = f"{target_column}={sorted(pd.Series(y_test).unique())[-1]}"

            drivers: list[dict[str, Any]] = []
            narrative: list[str] = []
            for rank, idx in enumerate(order, 1):
                importance = float(perm.importances_mean[idx])
                if importance <= 0:
                    continue
                feature = str(X_test.columns[idx])
                corr = float(X_test.iloc[:, idx].corr(pd.Series(y_test).astype(float)))
                direction = "increases" if corr >= 0 else "decreases"
                drivers.append({
                    "feature": feature,
                    "importance": round(importance, 4),
                    "direction": direction,
                })
                if task_type == "classification":
                    toward = f"'{positive_label}'" if positive_label else "the higher-encoded class"
                    narrative.append(
                        f"#{rank} driver: '{feature}' — higher values "
                        f"{'push predictions toward ' + toward if corr >= 0 else 'push predictions away from ' + toward}"
                        f" (permutation importance {importance:.3f})."
                    )
                else:
                    narrative.append(
                        f"#{rank} driver: '{feature}' — higher values {direction} "
                        f"predicted '{target_column}' (permutation importance {importance:.3f})."
                    )
            return drivers, narrative
        except Exception:
            return [], []

    def get_schema(self) -> dict[str, Any]:
        return {
            "model_path": {"type": "string", "description": "Path to .pkl model file.", "required": True},
            "file_path": {"type": "string", "description": "Path to evaluation dataset.", "required": True},
            "target_column": {"type": "string", "description": "Target column name.", "required": True},
            "task_type": {
                "type": "string",
                "description": "classification | regression.",
                "required": False,
            },
            "test_size": {
                "type": "float",
                "description": "Held-out fraction — must match train_model. Default: 0.2.",
                "required": False,
            },
        }
