"""
Render the 3D pipeline rig to a standalone HTML file.

Lets you iterate on ui/assets/pipeline_3d.js without booting Streamlit or
running a real analysis — pick a scenario, open the file, refresh.

Usage:
    python scripts/preview_pipeline_3d.py                  # complete run
    python scripts/preview_pipeline_3d.py --scenario idle
    python scripts/preview_pipeline_3d.py --scenario running -o /tmp/rig.html
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ui.pipeline_3d import Stage, build_document  # noqa: E402

#: Mirrors STAGE_DEFS in app.py.
STAGE_NAMES = [
    "Dataset Ingestion",
    "Initial Reasoning",
    "Tool Execution",
    "Result Interpretation",
    "Iterative Refinement",
    "RLM Decomposition",
    "Report Generation",
]


def _stages(statuses: list[str], details: dict[int, str] | None = None) -> list[Stage]:
    notes = details or {}
    return [
        Stage(num=str(i + 1), name=name, status=status, detail=notes.get(i, ""))  # type: ignore[arg-type]
        for i, (name, status) in enumerate(zip(STAGE_NAMES, statuses, strict=True))
    ]


SCENARIOS = {
    "idle": lambda: _stages(["pending"] * 7),
    "running": lambda: _stages(
        ["done", "done", "active", "pending", "pending", "pending", "pending"],
        {0: "1,000 rows x 12 cols", 1: "4 steps planned", 2: "correlation_analysis"},
    ),
    "complete": lambda: _stages(
        ["done"] * 7,
        {
            0: "1,000 rows x 12 cols",
            1: "6 steps planned",
            2: "6 tools run",
            3: "3 insights",
            4: "2 iterations",
            5: "4 sub-queries",
            6: "report.md written",
        },
    ),
    "error": lambda: _stages(
        ["done", "done", "error", "pending", "pending", "pending", "pending"],
        {0: "1,000 rows x 12 cols", 2: "train_model failed"},
    ),
    "skipped": lambda: _stages(
        ["done", "done", "done", "done", "skipped", "skipped", "done"],
        {4: "converged early", 5: "not needed"},
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="complete")
    parser.add_argument("-o", "--out", type=Path, help="output path")
    parser.add_argument("--open", action="store_true", help="open in the default browser")
    args = parser.parse_args()

    out = args.out or ROOT / "output" / f"pipeline_3d_{args.scenario}.html"
    out.parent.mkdir(parents=True, exist_ok=True)

    # The component fills its iframe, so give the standalone page a real height.
    document = build_document(SCENARIOS[args.scenario]()).replace(
        "<body>", '<body style="height:420px;background:#dcdbd3">', 1
    )
    out.write_text(document, encoding="utf-8")
    print(f"{args.scenario} -> {out}")

    if args.open:
        webbrowser.open(out.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
