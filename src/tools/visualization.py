"""
Visualization Tools — Execution Layer.

Stage 3: Chart generation (correlation heatmap, feature importance,
         distributions, ROC curve, confusion matrix).

Uses Seaborn for static PNG exports and Plotly for interactive HTML.
All charts are saved to the output directory.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from src.tools.base import BaseTool, ToolExecutionError


def _read_df(file_path: str) -> pd.DataFrame:
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


class GenerateVisualizationsTool(BaseTool):
    """
    Generate analysis charts and save them to the output directory.

    Supported chart_type values:
      - correlation_heatmap
      - feature_importance
      - distributions
      - roc_curve
      - confusion_matrix
    """

    name = "generate_visualizations"
    description = (
        "Generate visual charts for analysis results. "
        "chart_type: correlation_heatmap | feature_importance | distributions | "
        "roc_curve | confusion_matrix. "
        "Saves PNG + HTML to output_dir. Returns file paths."
    )

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        chart_type: str,
        target_column: str | None = None,
        model_path: str | None = None,
        output_dir: str = "output/visualizations",
        **_: Any,
    ) -> dict[str, Any]:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
        import seaborn as sns

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        df = _read_df(file_path)

        handlers: dict[str, Callable[..., list[str]]] = {
            "correlation_heatmap": self._heatmap,
            "feature_importance": self._feature_importance,
            "distributions": self._distributions,
            "roc_curve": self._roc_curve,
            "confusion_matrix": self._confusion_matrix,
        }

        handler = handlers.get(chart_type)
        if handler is None:
            raise ToolExecutionError(
                f"Unknown chart_type '{chart_type}'. "
                f"Valid: {list(handlers.keys())}"
            )

        saved_paths = handler(
            df=df,
            target_column=target_column,
            model_path=model_path,
            output_dir=output_dir,
            plt=plt,
            sns=sns,
        )

        return {
            "summary": f"Generated '{chart_type}' chart(s). Files: {saved_paths}",
            "chart_type": chart_type,
            "saved_paths": saved_paths,
        }

    # ------------------------------------------------------------------
    # Chart implementations
    # ------------------------------------------------------------------

    def _heatmap(self, df: pd.DataFrame, output_dir: str, plt: Any, sns: Any, **_: Any) -> list[str]:
        num_df = df.select_dtypes(include="number")
        corr = num_df.corr()
        fig, ax = plt.subplots(figsize=(max(8, len(corr) * 0.7), max(6, len(corr) * 0.6)))
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title("Feature Correlation Heatmap")
        out = str(Path(output_dir) / "correlation_heatmap.png")
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return [out]

    def _feature_importance(
        self, df: pd.DataFrame, target_column: str | None, model_path: str | None,
        output_dir: str, plt: Any, sns: Any, **_: Any
    ) -> list[str]:
        import pickle
        if not model_path or not Path(model_path).exists():
            raise ToolExecutionError("model_path is required for feature_importance chart.")
        if not target_column:
            raise ToolExecutionError("target_column is required for feature_importance chart.")

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        feature_cols = [c for c in df.columns if c != target_column]

        if hasattr(model, "feature_importances_"):
            # Tree-based models (RandomForest, XGBoost)
            raw = model.feature_importances_
            chart_title = "Top Feature Importances"
        elif hasattr(model, "coef_"):
            # Linear models (LogisticRegression, LinearRegression, Ridge)
            # Use absolute coefficient magnitude as importance proxy
            import numpy as np  # already available as a dep
            coef = model.coef_
            # coef_ shape: (n_classes, n_features) for multi-class, (n_features,) for regression
            raw = np.abs(coef[0] if coef.ndim == 2 else coef)
            chart_title = "Top Feature Importances (|coefficient|)"
        else:
            raise ToolExecutionError(
                "Model does not expose feature_importances_ or coef_. "
                "Feature importance chart requires a tree-based or linear model."
            )

        importances = pd.Series(raw, index=feature_cols)
        importances = importances.sort_values(ascending=False).head(20)

        fig, ax = plt.subplots(figsize=(10, max(4, len(importances) * 0.4)))
        sns.barplot(
            x=importances.values, y=importances.index, hue=importances.index,
            legend=False, ax=ax, palette="Blues_d",
        )
        ax.set_title(chart_title)
        ax.set_xlabel("Importance score")
        out = str(Path(output_dir) / "feature_importance.png")
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return [out]

    def _distributions(
        self, df: pd.DataFrame, output_dir: str, plt: Any, sns: Any, **_: Any
    ) -> list[str]:
        num_cols = df.select_dtypes(include="number").columns.tolist()[:12]
        saved: list[str] = []
        for col in num_cols:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(df[col].dropna(), kde=True, ax=ax, color="steelblue")
            ax.set_title(f"Distribution: {col}")
            out = str(Path(output_dir) / f"dist_{col}.png")
            fig.tight_layout()
            fig.savefig(out, dpi=120)
            plt.close(fig)
            saved.append(out)
        return saved

    def _roc_curve(
        self, df: pd.DataFrame, target_column: str | None, model_path: str | None,
        output_dir: str, plt: Any, sns: Any, **_: Any
    ) -> list[str]:
        import pickle

        from sklearn.metrics import auc, roc_curve

        if not model_path or not target_column:
            raise ToolExecutionError("model_path and target_column required for roc_curve.")

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        y = df[target_column]
        unique_classes = y.nunique()
        if unique_classes != 2:
            raise ToolExecutionError(
                f"roc_curve requires a binary classification target (2 classes). "
                f"'{target_column}' has {unique_classes} unique values. "
                f"Use feature_importance or distributions for regression tasks."
            )

        # Reuse the training-time feature/target preparation so the model
        # sees the exact column layout it was fitted on
        from src.tools.ml_pipeline import _encode_target, _prepare_features

        X, y_raw = _prepare_features(df, target_column)
        y, _class_labels = _encode_target(y_raw)

        if not hasattr(model, "predict_proba"):
            raise ToolExecutionError("Model does not support probability predictions (no predict_proba).")

        y_prob = model.predict_proba(X)[:, 1]
        try:
            fpr, tpr, _thresholds = roc_curve(y, y_prob)
        except ValueError as exc:
            raise ToolExecutionError(f"roc_curve failed: {exc}") from exc
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        out = str(Path(output_dir) / "roc_curve.png")
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return [out]

    def _confusion_matrix(
        self, df: pd.DataFrame, target_column: str | None, model_path: str | None,
        output_dir: str, plt: Any, sns: Any, **_: Any
    ) -> list[str]:
        import pickle

        from sklearn.metrics import confusion_matrix

        if not model_path or not target_column:
            raise ToolExecutionError("model_path and target_column required for confusion_matrix.")

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # Reuse the training-time feature/target preparation so the model
        # sees the exact column layout it was fitted on
        from src.tools.ml_pipeline import _encode_target, _prepare_features

        X, y_raw = _prepare_features(df, target_column)
        y, _class_labels = _encode_target(y_raw)

        y_pred = model.predict(X)
        cm = confusion_matrix(y, y_pred)
        # LabelEncoder sorts classes, so _class_labels aligns with encoded ints
        labels = _class_labels if _class_labels else sorted(y.unique())

        fig, ax = plt.subplots(figsize=(max(5, len(labels)), max(4, len(labels))))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        out = str(Path(output_dir) / "confusion_matrix.png")
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return [out]

    def get_schema(self) -> dict[str, Any]:
        return {
            "file_path": {"type": "string", "description": "Dataset path.", "required": True},
            "chart_type": {
                "type": "string",
                "description": (
                    "Type of chart: correlation_heatmap | feature_importance | "
                    "distributions | roc_curve | confusion_matrix."
                ),
                "required": True,
            },
            "target_column": {
                "type": "string",
                "description": "Target column (required for roc_curve, confusion_matrix, feature_importance).",
                "required": False,
            },
            "model_path": {
                "type": "string",
                "description": "Path to .pkl model (required for feature_importance, roc_curve, confusion_matrix).",
                "required": False,
            },
            "output_dir": {"type": "string", "description": "Output directory. Default: output/visualizations.", "required": False},
        }
