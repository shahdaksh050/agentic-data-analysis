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
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.core.coercion import coerce_types
from src.core.io import DatasetReadError, invalidate_read_cache, read_any
from src.core.memory import DatasetMetadata
from src.tools.base import BaseTool, ToolExecutionError

if TYPE_CHECKING:
    from src.core.profiler import DatasetProfile


def _read_raw_df(file_path: str) -> pd.DataFrame:
    """Read a dataset exactly as stored, with no repair applied."""
    try:
        df, _report = read_any(file_path)
    except DatasetReadError as exc:
        raise ToolExecutionError(str(exc)) from exc
    return df


def _read_df(file_path: str) -> pd.DataFrame:
    """
    Read a dataset ready for analysis: unified reader + type coercion.

    Coercion belongs here, not at individual call sites. A retail export
    carries money as "$1,234.56" and rates as "45.3%", which read back as
    strings. `controller.load_dataset` coerces before profiling, so the
    *profile* saw them as numeric — but every tool re-read the file through
    this helper and got the strings back, so revenue was invisible to the
    entire analysis. On a real sales file that left correlation running on
    a customer ID and a quantity, and reporting r=-0.06 between them as the
    headline finding, while never once looking at revenue.

    Coercion is idempotent, and every repair is recorded and reported by
    the ingestion path (memory context "coercions" -> the report's Data
    Overview), so nothing here is silent.
    """
    df = _read_raw_df(file_path)
    repaired, _coercions = coerce_types(df)
    return repaired


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
    uses_cleaned_file = False

    def applies_to(self, profile: DatasetProfile | None, metadata: DatasetMetadata | None) -> float:
        # Stage 1 already ingests the dataset before planning starts — never
        # a candidate for the LLM's own plan.
        return 0.0

    def execute(self, file_path: str, target_column: str | None = None, **_: Any) -> dict[str, Any]:  # type: ignore[override]
        path = Path(file_path)
        if not path.exists():
            raise ToolExecutionError(f"File not found: {file_path}")

        try:
            # Raw: this tool reports what the file actually contains.
            df = _read_raw_df(file_path)
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

        metadata_dict: dict[str, Any] = {
            "file_path": str(path.resolve()),
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values": missing_values,
            "numerical_cols": numerical_cols,
            "categorical_cols": categorical_cols,
            "target_column": target_column if target_column in df.columns else None,
            "task_type": None,
            "class_balance": class_balance,
            "high_cardinality_cols": high_card,
            "column_nunique": column_nunique,
            "summary_stats": {},
        }
        # Task type has exactly one implementation: DatasetMetadata.infer_task_type().
        # A previous version duplicated this logic here with a different (wrong)
        # rule for high-cardinality object/string targets — see IMPROVEMENTS.md #1.
        task_type = DatasetMetadata(**metadata_dict).infer_task_type()
        metadata_dict["task_type"] = task_type

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
    uses_cleaned_file = False  # this IS the tool that produces cleaned_file_path
    output_subdir = "data"

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
        if df.empty:
            raise ToolExecutionError("Dataset has no rows — nothing to clean.")
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
            modes = subset.mode()
            if not modes.empty:
                subset = subset.fillna(modes.iloc[0])
        elif strategy == "drop_rows":
            df = df.dropna()
            if df.empty:
                raise ToolExecutionError(
                    "drop_rows removed every row (every row has at least one "
                    "missing value). Re-run clean_data with strategy 'median' "
                    "or 'mode' instead."
                )
        elif strategy == "forward_fill":
            subset = subset.ffill()

        if strategy != "drop_rows":
            df[cols_to_clean] = subset

        missing_after = int(df.isnull().sum().sum())
        out_dir = Path(output_dir) if output_dir else path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{path.stem}_cleaned.csv"  # always CSV
        df.to_csv(out_path, index=False)
        # This path may already be in the read cache from an earlier
        # step (a re-planned or retried run rewrites the same name).
        invalidate_read_cache(str(out_path))

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
    output_subdir = "data"

    def applies_to(self, profile: DatasetProfile | None, metadata: DatasetMetadata | None) -> float:
        if profile is None:
            return 1.0
        return 1.0 if profile.columns_of_kind("numeric") else 0.0

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
            mask.loc[clean.index[mask_idx]] = True

        elif method == "isolation_forest":
            from sklearn.ensemble import IsolationForest
            model = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
            clean = num_df.dropna()
            preds = model.fit_predict(clean)
            report.update({"total_outliers": int((preds == -1).sum()), "contamination": 0.05})
            mask = pd.Series(False, index=df.index)
            mask.loc[clean.index[preds == -1]] = True

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
        # This path may already be in the read cache from an earlier
        # step (a re-planned or retried run rewrites the same name).
        invalidate_read_cache(str(out_path))

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

    def applies_to(self, profile: DatasetProfile | None, metadata: DatasetMetadata | None) -> float:
        if profile is None:
            return 1.0
        return 1.0 if len(profile.columns_of_kind("numeric")) >= 2 else 0.0

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
        target_encoded = False
        target_corr_method = method
        if target_column and target_column in corr.columns:
            target_corrs = {
                c: round(float(corr.loc[c, target_column]), 4)
                for c in corr.columns
                if c != target_column and not np.isnan(corr.loc[c, target_column])
            }
            target_corrs = dict(
                sorted(target_corrs.items(), key=lambda x: abs(x[1]), reverse=True)
            )
        elif (
            target_column
            and target_column in df.columns
            and df[target_column].nunique(dropna=True) == 2
        ):
            # Non-numeric binary target (e.g. churn yes/no): encode to 0/1 so
            # feature↔target correlation still works (point-biserial).
            raw_target = df[target_column]
            encoded = pd.Series(
                pd.factorize(raw_target)[0], index=df.index, dtype="float64"
            ).where(raw_target.notna())
            aligned = encoded.loc[num_df.index]
            target_encoded = True
            target_corr_method = "point-biserial"
            for c in num_df.columns:
                val = float(num_df[c].corr(aligned))
                if not np.isnan(val):
                    target_corrs[c] = round(val, 4)
            target_corrs = dict(
                sorted(target_corrs.items(), key=lambda x: abs(x[1]), reverse=True)
            )
        elif (
            target_column
            and target_column in df.columns
            and df[target_column].nunique(dropna=True) > 2
        ):
            # Non-numeric target with 3+ classes: Pearson r is undefined, so
            # this used to leave target_correlations silently empty with no
            # warning (IMPROVEMENTS.md #4). Eta-squared — the ANOVA analogue
            # of R² — measures how much of each numeric feature's variance is
            # explained by target-class membership: bounded [0, 1], same
            # dict shape as a correlation magnitude, comparable across
            # features. Unlike Pearson r it has no sign (there's no single
            # "direction" across 3+ unordered classes).
            classes = df[target_column].loc[num_df.index]
            for c in num_df.columns:
                feature = num_df[c]
                groups = [
                    feature[classes == cls].to_numpy()
                    for cls in classes.dropna().unique()
                ]
                groups = [g for g in groups if len(g) >= 2]
                if len(groups) < 2:
                    continue
                overall_mean = feature.mean()
                ss_total = float(((feature - overall_mean) ** 2).sum())
                if ss_total <= 0:
                    continue
                ss_between = sum(len(g) * (g.mean() - overall_mean) ** 2 for g in groups)
                target_corrs[c] = round(float(ss_between / ss_total), 4)
            target_corrs = dict(
                sorted(target_corrs.items(), key=lambda x: x[1], reverse=True)
            )
            target_encoded = True
            target_corr_method = "eta-squared"

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
            "target_correlation_method": target_corr_method,
            "target_encoded_binary": target_encoded,
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
