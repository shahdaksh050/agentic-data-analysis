"""
Statistical Analysis Tools — Execution Layer.

Provides intelligent statistical test selection and execution.
Test selection is context-aware (data types, group counts, normality).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from scipy import stats  # type: ignore[import-untyped]

from src.tools.base import BaseTool, ToolExecutionError


class SelectStatisticalTestTool(BaseTool):
    """
    Autonomously selects and runs the appropriate statistical test.

    Decision logic:
      - 2 groups, normal data, equal variance → Independent T-test
      - 2 groups, normal data, unequal variance → Welch's T-test
      - 2 groups, non-normal → Mann-Whitney U
      - 3+ groups, normal → One-way ANOVA
      - 3+ groups, non-normal → Kruskal-Wallis
      - Categorical association → Chi-Square
    """

    name = "select_statistical_test"
    description = (
        "Intelligently selects and executes the appropriate statistical hypothesis test "
        "based on data characteristics (normality, group count, data types). "
        "Returns the test name, statistic, p-value, and interpretation."
    )

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        feature_column: str,
        group_column: str,
        alpha: float = 0.05,
        **_: Any,
    ) -> dict[str, Any]:
        path = Path(file_path)
        df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_excel(path)

        if feature_column not in df.columns:
            raise ToolExecutionError(f"Feature column '{feature_column}' not found.")
        if group_column not in df.columns:
            raise ToolExecutionError(f"Group column '{group_column}' not found.")

        groups = df.groupby(group_column)[feature_column].apply(list)
        group_arrays = [pd.array(g).dropna() for g in groups]  # type: ignore[attr-defined]
        n_groups = len(group_arrays)

        if n_groups < 2:
            raise ToolExecutionError("Need at least 2 groups for hypothesis testing.")

        # Check if feature is categorical
        if df[feature_column].dtype == object:
            return self._run_chi_square(df, feature_column, group_column, alpha)

        # Normality test (Shapiro-Wilk on each group, sub-sampling if large)
        normality_results = []
        for g in group_arrays:
            sample = g[:5000] if len(g) > 5000 else g
            _, p_norm = stats.shapiro(sample)
            normality_results.append(p_norm > alpha)
        is_normal = all(normality_results)

        if n_groups == 2:
            g1, g2 = group_arrays[0], group_arrays[1]
            if is_normal:
                # Levene's test for variance equality
                _, p_levene = stats.levene(g1, g2)
                equal_var = p_levene > alpha
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
            f"There IS a statistically significant difference (p={p_val:.4f} < α={alpha})."
            if significant
            else f"There is NO statistically significant difference (p={p_val:.4f} ≥ α={alpha})."
        )

        return {
            "summary": f"{test_name}: statistic={stat:.4f}, p={p_val:.4f}. {interpretation}",
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

    def _run_chi_square(
        self, df: pd.DataFrame, feature_col: str, group_col: str, alpha: float
    ) -> dict[str, Any]:
        """Run Chi-Square test for categorical feature × group association."""
        contingency = pd.crosstab(df[feature_col], df[group_col])
        stat, p_val, dof, _ = stats.chi2_contingency(contingency)
        significant = bool(p_val < alpha)
        return {
            "summary": f"Chi-Square test: chi2={stat:.4f}, p={p_val:.4f}, dof={dof}.",
            "test_name": "Chi-Square Test of Independence",
            "statistic": round(float(stat), 6),
            "p_value": round(float(p_val), 6),
            "degrees_of_freedom": int(dof),
            "alpha": alpha,
            "significant": significant,
            "interpretation": (
                "Significant association exists." if significant else "No significant association."
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
                "description": "The column defining the comparison groups.",
                "required": True,
            },
            "alpha": {
                "type": "float",
                "description": "Significance level. Default: 0.05.",
                "required": False,
            },
        }
