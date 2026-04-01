"""
Data Processing Tools — Execution Layer.

All tools here are deterministic and return structured JSON dicts.
These are the atomic operations the Agent Controller can dispatch.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.tools.base import BaseTool, ToolExecutionError


# ============================================================
# Tool 1: Dataset Ingestion
# ============================================================
class IngestDatasetTool(BaseTool):
    """Ingests a CSV/Excel file and extracts schema metadata."""

    name = "ingest_dataset"
    description = (
        "Load a dataset from a file path. Detects schema, column types, "
        "missing value counts, and basic statistics. Returns structured metadata."
    )

    def execute(self, file_path: str, **_: Any) -> dict[str, Any]:
        path = Path(file_path)
        if not path.exists():
            raise ToolExecutionError(f"File not found: {file_path}")

        try:
            if path.suffix.lower() in {".csv", ".tsv"}:
                df = pd.read_csv(path)
            elif path.suffix.lower() in {".xlsx", ".xls"}:
                df = pd.read_excel(path)
            else:
                raise ToolExecutionError(f"Unsupported file format: {path.suffix}")
        except Exception as exc:
            raise ToolExecutionError(f"Failed to read file: {exc}") from exc

        numerical_cols = df.select_dtypes(include="number").columns.tolist()
        categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
        missing_values: dict[str, int] = df.isnull().sum().to_dict()
        summary_stats = df.describe(include="all").to_dict()

        metadata_dict: dict[str, Any] = {
            "file_path": str(path.resolve()),
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values": {k: int(v) for k, v in missing_values.items() if v > 0},
            "numerical_cols": numerical_cols,
            "categorical_cols": categorical_cols,
            "target_column": None,
            "task_type": None,
            "summary_stats": {},  # Kept empty to avoid bloating JSON
        }

        return {
            "summary": (
                f"Dataset ingested: {len(df)} rows, {len(df.columns)} columns. "
                f"{len(numerical_cols)} numerical, {len(categorical_cols)} categorical. "
                f"{sum(missing_values.values())} missing values total."
            ),
            "metadata": metadata_dict,
            "column_list": df.columns.tolist(),
        }

    def get_schema(self) -> dict[str, Any]:
        return {
            "file_path": {
                "type": "string",
                "description": "Absolute or relative path to the CSV or Excel file.",
                "required": True,
            }
        }


# ============================================================
# Tool 2: Data Cleaning
# ============================================================
class CleanDataTool(BaseTool):
    """Handles missing value imputation and basic data cleaning."""

    name = "clean_data"
    description = (
        "Clean a dataset by handling missing values using a specified strategy. "
        "Supports: 'mean', 'median', 'mode', 'drop_rows', 'forward_fill'."
    )

    STRATEGIES = {"mean", "median", "mode", "drop_rows", "forward_fill"}

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        strategy: str = "median",
        target_column: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        if strategy not in self.STRATEGIES:
            raise ToolExecutionError(
                f"Invalid strategy '{strategy}'. Choose from: {self.STRATEGIES}"
            )

        path = Path(file_path)
        df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_excel(path)
        original_shape = df.shape
        missing_before = int(df.isnull().sum().sum())

        if strategy == "mean":
            df = df.fillna(df.mean(numeric_only=True))
        elif strategy == "median":
            df = df.fillna(df.median(numeric_only=True))
        elif strategy == "mode":
            df = df.fillna(df.mode().iloc[0])
        elif strategy == "drop_rows":
            df = df.dropna()
        elif strategy == "forward_fill":
            df = df.ffill()

        missing_after = int(df.isnull().sum().sum())
        out_path = path.parent / f"{path.stem}_cleaned{path.suffix}"
        df.to_csv(out_path, index=False)

        return {
            "summary": (
                f"Cleaned {missing_before} missing values using '{strategy}'. "
                f"Shape: {original_shape} → {df.shape}. Saved to '{out_path}'."
            ),
            "cleaned_file_path": str(out_path),
            "rows_before": original_shape[0],
            "rows_after": df.shape[0],
            "missing_before": missing_before,
            "missing_after": missing_after,
            "strategy_used": strategy,
        }

    def get_schema(self) -> dict[str, Any]:
        return {
            "file_path": {
                "type": "string",
                "description": "Path to the dataset file.",
                "required": True,
            },
            "strategy": {
                "type": "string",
                "description": "Imputation strategy: 'mean', 'median', 'mode', 'drop_rows', 'forward_fill'.",
                "required": False,
            },
            "target_column": {
                "type": "string",
                "description": "Column to exclude from imputation (the target/label column).",
                "required": False,
            },
        }


# ============================================================
# Tool 3: Outlier Detection
# ============================================================
class DetectOutliersTool(BaseTool):
    """Detects outliers in numerical columns using configurable methods."""

    name = "detect_outliers"
    description = (
        "Detect outliers in numerical features. Methods: 'zscore', 'iqr', 'isolation_forest'."
    )

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        method: str = "iqr",
        threshold: float = 3.0,
        columns: list[str] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        path = Path(file_path)
        df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_excel(path)
        num_df = df.select_dtypes(include="number")

        if columns:
            num_df = num_df[[c for c in columns if c in num_df.columns]]

        outlier_report: dict[str, Any] = {}

        if method == "zscore":
            from scipy import stats  # type: ignore[import-untyped]
            z_scores = np.abs(stats.zscore(num_df.dropna()))
            outlier_mask = (z_scores > threshold).any(axis=1)
            total_outliers = int(outlier_mask.sum())
            outlier_report = {
                "method": "zscore",
                "threshold": threshold,
                "total_outliers": total_outliers,
                "outlier_percentage": round(total_outliers / len(df) * 100, 2),
                "columns_checked": num_df.columns.tolist(),
            }

        elif method == "iqr":
            q1 = num_df.quantile(0.25)
            q3 = num_df.quantile(0.75)
            iqr = q3 - q1
            outlier_mask = ((num_df < (q1 - 1.5 * iqr)) | (num_df > (q3 + 1.5 * iqr))).any(axis=1)
            total_outliers = int(outlier_mask.sum())
            per_column = {
                col: int(((num_df[col] < (q1[col] - 1.5 * iqr[col])) |
                          (num_df[col] > (q3[col] + 1.5 * iqr[col]))).sum())
                for col in num_df.columns
            }
            outlier_report = {
                "method": "iqr",
                "total_outliers": total_outliers,
                "outlier_percentage": round(total_outliers / len(df) * 100, 2),
                "per_column_outliers": per_column,
            }

        elif method == "isolation_forest":
            from sklearn.ensemble import IsolationForest  # type: ignore[import-untyped]
            model = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
            preds = model.fit_predict(num_df.dropna())
            total_outliers = int((preds == -1).sum())
            outlier_report = {
                "method": "isolation_forest",
                "contamination": 0.05,
                "total_outliers": total_outliers,
                "outlier_percentage": round(total_outliers / len(df) * 100, 2),
            }
        else:
            raise ToolExecutionError(f"Unknown method '{method}'. Use: zscore, iqr, isolation_forest")

        return {
            "summary": (
                f"Outlier detection ({method}): {outlier_report.get('total_outliers', 0)} "
                f"outliers found ({outlier_report.get('outlier_percentage', 0)}% of data)."
            ),
            **outlier_report,
        }

    def get_schema(self) -> dict[str, Any]:
        return {
            "file_path": {"type": "string", "description": "Path to dataset.", "required": True},
            "method": {
                "type": "string",
                "description": "Detection method: 'zscore', 'iqr', or 'isolation_forest'.",
                "required": False,
            },
            "threshold": {
                "type": "float",
                "description": "Z-score threshold (only for zscore method). Default: 3.0.",
                "required": False,
            },
            "columns": {
                "type": "list[string]",
                "description": "Specific columns to check. Defaults to all numerical.",
                "required": False,
            },
        }


# ============================================================
# Tool 4: Correlation Analysis
# ============================================================
class CorrelationAnalysisTool(BaseTool):
    """Computes feature correlation matrix and identifies top correlations."""

    name = "correlation_analysis"
    description = (
        "Compute correlation matrix for numerical features and identify "
        "the strongest positive and negative correlations."
    )

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        method: str = "pearson",
        top_n: int = 10,
        target_column: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        path = Path(file_path)
        df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_excel(path)
        num_df = df.select_dtypes(include="number")

        corr_matrix = num_df.corr(method=method)  # type: ignore[call-arg]

        # Get top correlations (excluding self-correlations)
        corr_pairs: list[dict[str, Any]] = []
        for col_a in corr_matrix.columns:
            for col_b in corr_matrix.columns:
                if col_a < col_b:  # avoid duplicates
                    val = round(corr_matrix.loc[col_a, col_b], 4)
                    corr_pairs.append({"col_a": col_a, "col_b": col_b, "correlation": val})

        corr_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        top_correlations = corr_pairs[:top_n]

        target_correlations: dict[str, float] = {}
        if target_column and target_column in corr_matrix.columns:
            target_correlations = {
                col: round(corr_matrix.loc[col, target_column], 4)
                for col in corr_matrix.columns
                if col != target_column
            }
            target_correlations = dict(
                sorted(target_correlations.items(), key=lambda x: abs(x[1]), reverse=True)
            )

        return {
            "summary": (
                f"Correlation analysis ({method}) on {len(num_df.columns)} features. "
                f"Top pair: {top_correlations[0]['col_a']} ↔ {top_correlations[0]['col_b']} "
                f"(r={top_correlations[0]['correlation']})" if top_correlations else "No correlations."
            ),
            "method": method,
            "top_correlations": top_correlations,
            "target_correlations": target_correlations,
            "features_analyzed": num_df.columns.tolist(),
        }

    def get_schema(self) -> dict[str, Any]:
        return {
            "file_path": {"type": "string", "description": "Path to dataset.", "required": True},
            "method": {
                "type": "string",
                "description": "Correlation method: 'pearson', 'spearman', or 'kendall'.",
                "required": False,
            },
            "top_n": {
                "type": "int",
                "description": "Number of top correlations to return. Default: 10.",
                "required": False,
            },
            "target_column": {
                "type": "string",
                "description": "If provided, also returns correlations between all features and this target.",
                "required": False,
            },
        }
