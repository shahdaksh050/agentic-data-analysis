"""
Data Processing Tools — Execution Layer.

Stage 1: Dataset Ingestion  (IngestDatasetTool)
Stage 3: Data Cleaning      (CleanDataTool)
Stage 3: Outlier Detection  (DetectOutliersTool)
Stage 3: Correlation EDA    (CorrelationAnalysisTool)

All tools are deterministic and return structured dicts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.tools.base import BaseTool, ToolExecutionError


def _read_df(file_path: str) -> pd.DataFrame:
    """Read CSV or Excel file robustly with explicit engines."""
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


# ============================================================
# Tool 1: Dataset Ingestion — Stage 1
# ============================================================

class IngestDatasetTool(BaseTool):
    """
    Load a CSV/Excel file and extract rich schema metadata.

    Stage 1 of the agent workflow. The metadata returned here feeds
    directly into the Memory System and is the only dataset information
    passed to the LLM — raw data is never put in LLM context.
    """

    name = "ingest_dataset"
    description = (
        "Load a dataset from a file path. Detects schema, dtypes, missing values, "
        "class balance, high-cardinality columns, and basic statistics. "
        "Returns structured metadata — do NOT pass raw data to the LLM."
    )

    def execute(self, file_path: str, target_column: str | None = None, **_: Any) -> dict[str, Any]:  # type: ignore[override]
        path = Path(file_path)
        if not path.exists():
            raise ToolExecutionError(f"File not found: {file_path}")

        try:
            df = _read_df(file_path)
        except ToolExecutionError:
            raise
        except Exception as exc:
            raise ToolExecutionError(f"Failed to read file: {exc}") from exc

        numerical_cols = df.select_dtypes(include="number").columns.tolist()
        categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
        missing_values: dict[str, int] = {
            k: int(v) for k, v in df.isnull().sum().items() if v > 0
        }

        # Class balance for classification targets
        class_balance: dict[str, int] = {}
        if target_column and target_column in df.columns:
            class_balance = df[target_column].value_counts().to_dict()
            class_balance = {str(k): int(v) for k, v in class_balance.items()}

        # Unique value counts for all columns (used by target-detection confidence scoring)
        column_nunique = {col: int(df[col].nunique()) for col in df.columns}

        # High-cardinality detection (>50 unique values in a categorical col)
        high_card = [
            c for c in categorical_cols
            if df[c].nunique() > 50
        ]

        # Infer task type
        task_type: str | None = None
        if target_column and target_column in df.columns:
            dtype = str(df[target_column].dtype)
            n_unique = df[target_column].nunique()
            if "int" in dtype or "bool" in dtype or n_unique <= 20:
                task_type = "classification"
            else:
                task_type = "regression"
        elif not target_column:
            task_type = "clustering"

        metadata_dict: dict[str, Any] = {
            "file_path": str(path.resolve()),
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values": missing_values,
            "numerical_cols": numerical_cols,
            "categorical_cols": categorical_cols,
            "target_column": target_column,
            "task_type": task_type,
            "class_balance": class_balance,
            "high_cardinality_cols": high_card,
            "column_nunique": column_nunique,
            "summary_stats": {},
        }

        return {
            "summary": (
                f"Ingested: {len(df):,} rows × {len(df.columns)} cols | "
                f"{len(numerical_cols)} numerical, {len(categorical_cols)} categorical | "
                f"{sum(missing_values.values()):,} missing cells | "
                f"task={task_type or 'TBD'}"
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
            },
            "target_column": {
                "type": "string",
                "description": "Name of the label/target column if known.",
                "required": False,
            },
        }


# ============================================================
# Tool 2: Data Cleaning — Stage 3
# ============================================================

class CleanDataTool(BaseTool):
    """
    Handle missing values and basic data cleaning.

    Writes a cleaned copy to disk and returns the new file path —
    downstream tools should use `cleaned_file_path` as their input.
    """

    name = "clean_data"
    description = (
        "Clean a dataset by handling missing values using a specified strategy. "
        "Strategies: 'mean', 'median', 'mode', 'drop_rows', 'forward_fill'. "
        "Returns cleaned_file_path for use by subsequent tools."
    )

    STRATEGIES = frozenset({"mean", "median", "mode", "drop_rows", "forward_fill"})

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        strategy: str = "median",
        target_column: str | None = None,
        output_dir: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        if strategy not in self.STRATEGIES:
            raise ToolExecutionError(
                f"Invalid strategy '{strategy}'. Choose from: {sorted(self.STRATEGIES)}"
            )

        path = Path(file_path)
        df = _read_df(file_path)
        original_shape = df.shape
        missing_before = int(df.isnull().sum().sum())

        # Exclude target column from imputation
        cols_to_clean = [c for c in df.columns if c != target_column]
        subset = df[cols_to_clean]

        if strategy == "mean":
            subset = subset.fillna(subset.mean(numeric_only=True))
        elif strategy == "median":
            subset = subset.fillna(subset.median(numeric_only=True))
        elif strategy == "mode":
            subset = subset.fillna(subset.mode().iloc[0])
        elif strategy == "drop_rows":
            df = df.dropna()
        elif strategy == "forward_fill":
            subset = subset.ffill()

        if strategy != "drop_rows":
            df[cols_to_clean] = subset

        missing_after = int(df.isnull().sum().sum())
        out_dir = Path(output_dir) if output_dir else path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{path.stem}_cleaned.csv"  # always CSV
        df.to_csv(out_path, index=False)

        return {
            "summary": (
                f"Cleaned {missing_before - missing_after} missing values "
                f"using '{strategy}'. Shape: {original_shape} → {df.shape}. "
                f"Saved to '{out_path}'."
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
            "file_path": {"type": "string", "description": "Path to the dataset file.", "required": True},
            "strategy": {
                "type": "string",
                "description": "Imputation strategy: mean | median | mode | drop_rows | forward_fill.",
                "required": False,
            },
            "target_column": {
                "type": "string",
                "description": "Column to exclude from imputation (label column).",
                "required": False,
            },
        }


# ============================================================
# Tool 3: Outlier Detection — Stage 3
# ============================================================

class DetectOutliersTool(BaseTool):
    """
    Detect outliers in numerical columns using configurable methods.

    Methods: IQR (default), Z-score, Isolation Forest.
    """

    name = "detect_outliers"
    description = (
        "Detect outliers in numerical features. "
        "Methods: 'iqr' (default), 'zscore', 'isolation_forest'. "
        "Returns per-column counts and an outlier-flagged dataset path."
    )

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        method: str = "iqr",
        threshold: float = 3.0,
        columns: list[str] | None = None,
        output_dir: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        path = Path(file_path)
        df = _read_df(file_path)
        num_df = df.select_dtypes(include="number")

        if columns:
            valid = [c for c in columns if c in num_df.columns]
            num_df = num_df[valid]

        if num_df.empty:
            raise ToolExecutionError("No numerical columns found for outlier detection.")

        report: dict[str, Any] = {"method": method}

        if method == "iqr":
            q1 = num_df.quantile(0.25)
            q3 = num_df.quantile(0.75)
            iqr = q3 - q1
            mask = ((num_df < (q1 - 1.5 * iqr)) | (num_df > (q3 + 1.5 * iqr))).any(axis=1)
            per_col = {
                col: int(
                    ((num_df[col] < (q1[col] - 1.5 * iqr[col]))
                     | (num_df[col] > (q3[col] + 1.5 * iqr[col]))).sum()
                )
                for col in num_df.columns
            }
            report.update({"total_outliers": int(mask.sum()), "per_column_outliers": per_col})

        elif method == "zscore":
            from scipy import stats
            clean = num_df.dropna()
            z = np.abs(stats.zscore(clean))
            mask_idx = (z > threshold).any(axis=1)
            report.update({
                "total_outliers": int(mask_idx.sum()),
                "threshold": threshold,
                "columns_checked": num_df.columns.tolist(),
            })
            mask = pd.Series(False, index=df.index)
            mask.iloc[clean.index[mask_idx]] = True

        elif method == "isolation_forest":
            from sklearn.ensemble import IsolationForest
            model = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
            clean = num_df.dropna()
            preds = model.fit_predict(clean)
            report.update({"total_outliers": int((preds == -1).sum()), "contamination": 0.05})
            mask = pd.Series(False, index=df.index)
            mask.iloc[clean.index[preds == -1]] = True

        else:
            raise ToolExecutionError(f"Unknown method '{method}'. Use: iqr, zscore, isolation_forest")

        total = report.get("total_outliers", 0)
        report["outlier_percentage"] = round(total / max(len(df), 1) * 100, 2)

        # Save flagged dataset
        df["_is_outlier"] = mask
        out_dir = Path(output_dir) if output_dir else path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{path.stem}_outliers_flagged.csv"  # always CSV
        df.to_csv(out_path, index=False)

        return {
            "summary": (
                f"Outlier detection ({method}): {total:,} outliers "
                f"({report['outlier_percentage']}% of data). "
                f"Flagged dataset saved to '{out_path}'."
            ),
            "flagged_file_path": str(out_path),
            **report,
        }

    def get_schema(self) -> dict[str, Any]:
        return {
            "file_path": {"type": "string", "description": "Path to dataset.", "required": True},
            "method": {
                "type": "string",
                "description": "Detection method: iqr | zscore | isolation_forest.",
                "required": False,
            },
            "threshold": {
                "type": "float",
                "description": "Z-score threshold (zscore method only). Default: 3.0.",
                "required": False,
            },
            "columns": {
                "type": "list[string]",
                "description": "Subset of columns to check. Defaults to all numerical.",
                "required": False,
            },
        }


# ============================================================
# Tool 4: Correlation Analysis — Stage 3 EDA
# ============================================================

class CorrelationAnalysisTool(BaseTool):
    """
    Compute feature correlation matrix and identify top correlations.

    Supports Pearson, Spearman, and Kendall methods.
    Returns both global top pairs and target-specific correlations.
    """

    name = "correlation_analysis"
    description = (
        "Compute correlation matrix for numerical features. "
        "Returns top correlated pairs and, if target_column is given, "
        "correlations of all features with the target. "
        "Methods: 'pearson' (default), 'spearman', 'kendall'."
    )

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        method: str = "pearson",
        top_n: int = 10,
        target_column: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        df = _read_df(file_path)
        num_df = df.select_dtypes(include="number").dropna()

        if num_df.shape[1] < 2:
            raise ToolExecutionError("Need at least 2 numerical columns for correlation analysis.")

        corr = num_df.corr(method=method)

        # Top correlated pairs (exclude self-correlations)
        pairs: list[dict[str, Any]] = []
        cols = corr.columns.tolist()
        for i, ca in enumerate(cols):
            for cb in cols[i + 1 :]:
                val = float(corr.loc[ca, cb])
                if not np.isnan(val):
                    pairs.append({"col_a": ca, "col_b": cb, "correlation": round(val, 4)})
        pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        top_pairs = pairs[:top_n]

        # Target correlations
        target_corrs: dict[str, float] = {}
        if target_column and target_column in corr.columns:
            target_corrs = {
                c: round(float(corr.loc[c, target_column]), 4)
                for c in corr.columns
                if c != target_column and not np.isnan(corr.loc[c, target_column])
            }
            target_corrs = dict(
                sorted(target_corrs.items(), key=lambda x: abs(x[1]), reverse=True)
            )

        top_summary = (
            f"Top pair: {top_pairs[0]['col_a']} ↔ {top_pairs[0]['col_b']} "
            f"(r={top_pairs[0]['correlation']})"
            if top_pairs
            else "No pairs"
        )

        return {
            "summary": (
                f"Correlation ({method}) on {len(cols)} features. {top_summary}."
            ),
            "method": method,
            "top_correlations": top_pairs,
            "target_correlations": target_corrs,
            "features_analyzed": cols,
            "n_features": len(cols),
        }

    def get_schema(self) -> dict[str, Any]:
        return {
            "file_path": {"type": "string", "description": "Path to dataset.", "required": True},
            "method": {
                "type": "string",
                "description": "pearson | spearman | kendall. Default: pearson.",
                "required": False,
            },
            "top_n": {
                "type": "int",
                "description": "Number of top correlation pairs to return. Default: 10.",
                "required": False,
            },
            "target_column": {
                "type": "string",
                "description": "If given, also returns feature↔target correlations.",
                "required": False,
            },
        }
