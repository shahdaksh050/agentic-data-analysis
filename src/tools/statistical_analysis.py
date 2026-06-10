"""
Statistical Analysis Tools — Execution Layer.

Stage 3: Statistical hypothesis test selection and execution.

Decision logic (auto-selection):
  - Categorical feature               → Chi-Square test
  - 2 groups, normal, equal variance  → Independent T-test
  - 2 groups, normal, unequal         → Welch's T-test
  - 2 groups, non-normal              → Mann-Whitney U
  - 3+ groups, normal                 → One-Way ANOVA
  - 3+ groups, non-normal             → Kruskal-Wallis
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from scipy import stats

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


class SelectStatisticalTestTool(BaseTool):
    """
    Autonomously selects and executes the appropriate statistical test.

    Returns the test name, statistic, p-value, and a plain-English
    interpretation — ready for inclusion in the final report.
    """

    name = "select_statistical_test"
    description = (
        "Automatically select and run the correct statistical hypothesis test "
        "based on data characteristics (normality, group count, data types). "
        "Returns test name, statistic, p-value, and interpretation."
    )

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        feature_column: str,
        group_column: str,
        alpha: float = 0.05,
        **_: Any,
    ) -> dict[str, Any]:
        df = _read_df(file_path)

        if feature_column not in df.columns:
            raise ToolExecutionError(f"Feature column '{feature_column}' not found.")
        if group_column not in df.columns:
            raise ToolExecutionError(f"Group column '{group_column}' not found.")

        # Reject ID-like columns: monotonic integers or >95% unique values
        feat_series = df[feature_column].dropna()
        if pd.api.types.is_numeric_dtype(feat_series):
            uniqueness = feat_series.nunique() / max(len(feat_series), 1)
            is_monotonic = feat_series.is_monotonic_increasing or feat_series.is_monotonic_decreasing
            if uniqueness > 0.95 and is_monotonic:
                raise ToolExecutionError(
                    f"Column '{feature_column}' appears to be a row ID or index "
                    f"(monotonic, {uniqueness:.0%} unique values). "
                    "Pass a meaningful numeric feature instead."
                )

        df_clean = df[[feature_column, group_column]].dropna()
        # For continuous group columns, bin into quartiles automatically
        if df_clean[group_column].dtype in (float,) or str(df_clean[group_column].dtype).startswith("float"):
            df_clean = df_clean.copy()
            df_clean[group_column] = pd.qcut(df_clean[group_column], q=4,
                                              labels=["Q1","Q2","Q3","Q4"],
                                              duplicates="drop")

        groups = df_clean.groupby(group_column)[feature_column].apply(list)
        group_arrays = [pd.array(g) for g in groups]
        n_groups = len(group_arrays)

        if n_groups < 2:
            raise ToolExecutionError("At least 2 groups are required for hypothesis testing.")
        if n_groups > 20:
            raise ToolExecutionError(
                f"Too many groups ({n_groups}) for hypothesis testing. "
                "Pass a categorical group_column with ≤20 unique values."
            )

        # Categorical feature → Chi-Square
        # (is_numeric_dtype, not `dtype == object`: pandas 3 strings are `str` dtype)
        if not pd.api.types.is_numeric_dtype(df_clean[feature_column]):
            return self._chi_square(df_clean, feature_column, group_column, alpha)

        # Normality (Shapiro-Wilk, sub-sampled for large groups)
        is_normal = all(
            stats.shapiro(g[:5000] if len(g) > 5000 else g)[1] > alpha
            for g in group_arrays
        )

        if n_groups == 2:
            g1, g2 = group_arrays[0], group_arrays[1]
            if is_normal:
                _, p_lev = stats.levene(g1, g2)
                equal_var = p_lev > alpha
                stat, p_val = stats.ttest_ind(g1, g2, equal_var=equal_var)
                test_name = "Independent T-Test" if equal_var else "Welch's T-Test"
            else:
                stat, p_val = stats.mannwhitneyu(g1, g2, alternative="two-sided")
                test_name = "Mann-Whitney U"
        else:
            if is_normal:
                stat, p_val = stats.f_oneway(*group_arrays)
                test_name = "One-Way ANOVA"
            else:
                stat, p_val = stats.kruskal(*group_arrays)
                test_name = "Kruskal-Wallis"

        significant = bool(p_val < alpha)
        interpretation = (
            f"Statistically significant difference detected (p={p_val:.4f} < α={alpha})."
            if significant
            else f"No statistically significant difference (p={p_val:.4f} ≥ α={alpha})."
        )

        return {
            "summary": f"{test_name}: stat={stat:.4f}, p={p_val:.4f}. {interpretation}",
            "test_name": test_name,
            "statistic": round(float(stat), 6),
            "p_value": round(float(p_val), 6),
            "alpha": alpha,
            "significant": significant,
            "n_groups": n_groups,
            "normality_assumed": is_normal,
            "interpretation": interpretation,
            "feature_column": feature_column,
            "group_column": group_column,
        }

    def _chi_square(
        self, df: pd.DataFrame, feature_col: str, group_col: str, alpha: float
    ) -> dict[str, Any]:
        contingency = pd.crosstab(df[feature_col], df[group_col])
        stat, p_val, dof, _ = stats.chi2_contingency(contingency)
        significant = bool(p_val < alpha)
        return {
            "summary": (
                f"Chi-Square: chi2={stat:.4f}, p={p_val:.4f}, dof={dof}. "
                f"{'Significant association.' if significant else 'No significant association.'}"
            ),
            "test_name": "Chi-Square Test of Independence",
            "statistic": round(float(stat), 6),
            "p_value": round(float(p_val), 6),
            "degrees_of_freedom": int(dof),
            "alpha": alpha,
            "significant": significant,
            "interpretation": (
                "Significant association exists between variables."
                if significant
                else "No significant association between variables."
            ),
        }

    def get_schema(self) -> dict[str, Any]:
        return {
            "file_path": {"type": "string", "description": "Path to dataset.", "required": True},
            "feature_column": {
                "type": "string",
                "description": "The continuous or categorical feature to test.",
                "required": True,
            },
            "group_column": {
                "type": "string",
                "description": "The column defining comparison groups.",
                "required": True,
            },
            "alpha": {
                "type": "float",
                "description": "Significance level. Default: 0.05.",
                "required": False,
            },
        }
