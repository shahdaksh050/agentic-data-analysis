"""
Agentic Data Analysis System — Entry Point.

Usage:
    python main.py --dataset path/to/data.csv
    python main.py --dataset path/to/data.csv --provider anthropic --target churn
    python main.py --dataset path/to/data.csv --no-rlm --max-iterations 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv()

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agentic AI Powered Autonomous Data Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dataset", "-d", type=str, required=True, help="Path to CSV or Excel file.")
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "anthropic"],
        help="LLM provider (default: openai).",
    )
    parser.add_argument("--model", type=str, default=None, help="Override LLM model name.")
    parser.add_argument("--target", type=str, default=None, help="Target/label column name.")
    parser.add_argument(
        "--objective",
        type=str,
        default=None,
        help='Natural-language analysis goal, e.g. "what drives customer churn?".',
    )
    parser.add_argument("--max-iterations", type=int, default=15, help="Max reasoning-execution cycles.")
    parser.add_argument("--no-rlm", action="store_true", help="Disable RLM inference (flat loop).")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Root directory for all outputs (default: output/).",
    )
    parser.add_argument("--persist", type=str, default=None, help="Path to persist memory JSON.")
    return parser.parse_args()


def _print_banner() -> None:
    console.print(
        Panel(
            "[bold cyan]Agentic Data Analysis System[/]\n"
            "[dim]Powered by Recursive Language Model Inference (Zhang et al., 2024)[/]\n"
            "[dim]Separation of reasoning ↔ execution — RLM context offloading enabled[/]",
            border_style="cyan",
            expand=False,
        )
    )


def _print_final_report(report: dict) -> None:  # type: ignore[type-arg]
    status = report.get("status", "unknown")
    insights = report.get("insights", [])
    recommendations = report.get("recommendations", [])
    best_model = report.get("best_model")
    key_metrics = report.get("key_metrics", {})

    panel_lines = [f"[bold green]Status:[/] {status}"]

    if best_model:
        panel_lines.append(f"[bold]Best model:[/] {best_model}")

    if key_metrics:
        metrics_str = "  ".join(f"{k}: {v}" for k, v in key_metrics.items())
        panel_lines.append(f"[bold]Key metrics:[/] {metrics_str}")

    console.print(
        Panel(
            "\n".join(panel_lines),
            title="[bold green]Final Report Summary",
            border_style="green",
        )
    )

    if insights:
        table = Table(title="Key Insights", show_header=False, padding=(0, 1))
        table.add_column("", style="cyan")
        table.add_column("")
        for i, insight in enumerate(insights, 1):
            table.add_row(f"[bold]{i}.[/]", str(insight))
        console.print(table)

    if recommendations:
        table = Table(title="Recommendations", show_header=False, padding=(0, 1))
        table.add_column("", style="yellow")
        table.add_column("")
        for rec in recommendations:
            table.add_row("[bold]→[/]", str(rec))
        console.print(table)

    # Save raw JSON report
    output_dir = os.getenv("OUTPUT_DIR", "output")
    report_path = Path(output_dir) / "reports" / "final_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str))
    console.print(f"\n[dim]Full report saved to: {report_path}[/]")


def main() -> None:
    args = parse_args()
    _print_banner()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        console.print(f"[bold red]Error:[/] Dataset not found: {dataset_path}")
        sys.exit(1)

    # Apply CLI overrides to environment
    if args.model:
        os.environ["LLM_MODEL"] = args.model
    os.environ["LLM_PROVIDER"] = args.provider
    os.environ["MAX_ITERATIONS"] = str(args.max_iterations)
    os.environ["OUTPUT_DIR"] = args.output_dir
    if args.no_rlm:
        os.environ["ENABLE_RLM_INFERENCE"] = "false"
    if args.target:
        os.environ["TARGET_COLUMN_HINT"] = args.target
    if args.objective:
        os.environ["USER_OBJECTIVE"] = args.objective

    from src.core.controller import AgentController

    agent = AgentController(
        max_iterations=args.max_iterations,
        enable_rlm=not args.no_rlm,
        memory_persist_path=args.persist,
    )

    # ---- Stage 1: Dataset Ingestion ----
    metadata = agent.load_dataset(str(dataset_path), target_hint=args.target)
    console.print(
        f"\n[bold]Dataset loaded:[/] {metadata.row_count} rows × "
        f"{metadata.column_count} cols  |  "
        f"task: [cyan]{metadata.task_type or 'TBD'}[/]"
    )

    # ---- Stages 2-7: Full Autonomous Analysis ----
    final_report = agent.analyze()

    _print_final_report(final_report)


if __name__ == "__main__":
    main()
