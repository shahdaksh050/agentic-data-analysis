"""
Agentic Data Analysis System — Entry Point.

Usage:
    python main.py --dataset path/to/data.csv
    python main.py --dataset path/to/data.csv --provider anthropic
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

load_dotenv()

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agentic AI Powered Autonomous Data Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", "-d", type=str, required=True, help="Path to dataset file (CSV or Excel).")
    parser.add_argument("--provider", type=str, default="openai", choices=["openai", "anthropic"], help="LLM provider.")
    parser.add_argument("--model", type=str, default=None, help="LLM model name (overrides .env).")
    parser.add_argument("--max-iterations", type=int, default=15, help="Max reasoning-execution cycles.")
    parser.add_argument("--no-rlm", action="store_true", help="Disable RLM inference (use flat agent loop).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    console.print(Panel(
        "[bold cyan]Agentic Data Analysis System[/]\n"
        "[dim]Powered by Recursive Language Model Inference (Zhang et al., 2024)[/]",
        border_style="cyan",
        expand=False,
    ))

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        console.print(f"[bold red]Error:[/] Dataset not found: {dataset_path}")
        sys.exit(1)

    # Override env vars from CLI args if provided
    import os
    if args.model:
        os.environ["LLM_MODEL"] = args.model
    os.environ["LLM_PROVIDER"] = args.provider
    os.environ["MAX_ITERATIONS"] = str(args.max_iterations)
    if args.no_rlm:
        os.environ["ENABLE_RLM_INFERENCE"] = "false"

    from src.core.controller import AgentController

    agent = AgentController(
        max_iterations=args.max_iterations,
        enable_rlm=not args.no_rlm,
    )

    agent.load_dataset(str(dataset_path))
    final_report = agent.analyze()

    console.print(Panel(
        "[bold green]Analysis Complete![/]\n"
        f"Status: {final_report.get('status')}\n"
        f"Insights: {len(final_report.get('insights', []))} generated",
        title="Final Report Summary",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
