"""
Render the 6-Section 3D Cinematic Fullpage Experience to a standalone HTML file.

Lets you test and preview the Anime.js + fullPage.js + Three.js 3D Master Architecture
without booting Streamlit.

Usage:
    python scripts/preview_cinematic_3d.py --open
    python scripts/preview_cinematic_3d.py --theme day --open
    python scripts/preview_cinematic_3d.py --scenario running -o output/preview_cinematic.html
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ui.cinematic_3d import export_cinematic_html  # noqa: E402


def _get_mock_state(scenario: str, theme: str) -> dict:
    is_complete = scenario == "complete"
    is_running = scenario == "running"

    stages = [
        {"num": "1", "name": "Dataset Ingestion", "status": "done" if not scenario == "idle" else "pending", "detail": "1,420 rows × 14 cols"},
        {"num": "2", "name": "Initial Reasoning", "status": "done" if is_complete or is_running else "pending", "detail": "6 steps planned"},
        {"num": "3", "name": "Tool Execution", "status": "done" if is_complete else ("active" if is_running else "pending"), "detail": "6 tools executed"},
        {"num": "4", "name": "Result Interpretation", "status": "done" if is_complete else "pending", "detail": "3 key findings"},
        {"num": "5", "name": "Iterative Refinement", "status": "done" if is_complete else "pending", "detail": "2 iterations"},
        {"num": "6", "name": "RLM Decomposition", "status": "done" if is_complete else "pending", "detail": "4 sub-tasks"},
        {"num": "7", "name": "Report Generation", "status": "done" if is_complete else "pending", "detail": "executive ledger saved"},
    ]

    return {
        "theme": theme,
        "dataset": {
            "name": "customer_churn_q3.csv",
            "row_count": 1420,
            "col_count": 14,
            "task_type": "classification",
            "target_col": "churned",
            "missing_cells": 18,
            "quality_score": 96,
            "column_types": {"numeric": 8, "categorical": 4, "datetime": 1, "text": 1},
        },
        "stages": stages,
        "statistics": {
            "test_name": "Two-Sample T-Test / Mann-Whitney U",
            "p_value": 0.0004,
            "significant": True,
            "top_correlations": [
                {"pair": "monthly_charges ↔ total_revenue", "val": 0.82},
                {"pair": "contract_length ↔ churn_risk", "val": -0.68},
                {"pair": "customer_service_calls ↔ churn_risk", "val": 0.61},
            ],
            "outlier_pct": 1.4,
        },
        "ml": {
            "best_model": "GradientBoostingClassifier",
            "best_cv": 93.1,
            "best_gap": 2.8,
            "models": [
                {"name": "GradientBoostingClassifier", "cv_mean": 93.1, "cv_std": 1.2, "gap": 2.8, "is_best": True, "warnings": []},
                {"name": "RandomForestClassifier", "cv_mean": 91.4, "cv_std": 1.6, "gap": 4.9, "is_best": False, "warnings": []},
                {"name": "LogisticRegression(L2)", "cv_mean": 87.2, "cv_std": 2.0, "gap": 1.5, "is_best": False, "warnings": []},
                {"name": "DecisionTreeClassifier", "cv_mean": 82.0, "cv_std": 3.1, "gap": 11.4, "is_best": False, "warnings": ["Train-test gap > 10% (overfitting)"]},
            ],
            "overfit_warnings": [],
            "cv_folds": 5,
            "regularization": "L2 Ridge / Stratified 5-Fold",
        },
        "synthesis": {
            "objective": "Identify primary drivers of quarterly customer churn and verify predictive reliability.",
            "reasoning": "Across 5-fold stratified cross-validation, GradientBoosting demonstrated optimal generalizability with 93.1% CV accuracy and a minimal 2.8% train-test gap. Customer service call frequency (>3 calls) and short contract duration are the strongest drivers of churn risk.",
            "findings": [
                "Contract duration is the primary protective factor against churn.",
                "Cross-validation envelope indicates high generalization stability (std: ±1.2%).",
                "Tree depth capped at max_depth=6 to strictly prohibit memorizing noise.",
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=["idle", "running", "complete"], default="complete")
    parser.add_argument("--theme", choices=["day", "night"], default="night", help="color theme")
    parser.add_argument("-o", "--out", type=Path, help="output file path")
    parser.add_argument("--open", action="store_true", help="open in default web browser")
    args = parser.parse_args()

    out = args.out or ROOT / "output" / f"cinematic_3d_{args.scenario}_{args.theme}.html"
    state = _get_mock_state(args.scenario, args.theme)
    export_cinematic_html(out, state_dict=state, theme=args.theme)

    print(f"Rendered {args.scenario} [{args.theme}] -> {out}")

    if args.open:
        webbrowser.open(out.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
