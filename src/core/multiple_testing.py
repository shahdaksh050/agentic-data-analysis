"""
Multiple-comparison correction (item 4 / U0.7 follow-on).

A run can call select_statistical_test repeatedly across many
feature/group pairs; uncorrected, each test keeps its own 0.05
false-positive rate and the run-wide false-discovery rate climbs with
every additional test. This corrects across every p-value a run actually
produced (accumulated by the controller's execution loop into memory
context "statistical_test_pvalues") at report time — both
src.tools.report_generator (Markdown) and src.core.html_report (HTML) use
it so the two reports agree.
"""
from __future__ import annotations

from typing import Any

#: Default significance level — matches SelectStatisticalTestTool's default
#: alpha; a run using a different per-call alpha still gets one consistent
#: correction here.
DEFAULT_ALPHA = 0.05


def apply_benjamini_hochberg(
    pvalue_tests: list[dict[str, Any]], alpha: float = DEFAULT_ALPHA
) -> list[dict[str, Any]]:
    """Each dict must have a "p_value" key; every other key is passed
    through unchanged. Adds "p_adjusted" and "significant_after_correction"."""
    if not pvalue_tests:
        return []
    from statsmodels.stats.multitest import multipletests

    pvals = [float(t["p_value"]) for t in pvalue_tests]
    reject, p_adjusted, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
    return [
        {**t, "p_adjusted": round(float(adj), 6), "significant_after_correction": bool(rej)}
        for t, adj, rej in zip(pvalue_tests, p_adjusted, reject, strict=True)
    ]
