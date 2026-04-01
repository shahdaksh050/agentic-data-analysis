"""
RLM Engine — Recursive Language Model Inference Layer.

Implements the RLM paradigm from Zhang et al. (2024):
- Treats the full analysis context as an 'external environment' (REPL state).
- Enables the LLM to programmatically decompose tasks and recursively invoke
  sub-calls with minimal, targeted context.
- Manages REPL state across the entire analysis lifecycle.

This is the heart of what differentiates this system from naive LLM chains.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from rich.console import Console
from rich.tree import Tree

console = Console()

# Type alias for the recursive LLM callable signature
LLMCallable = Callable[[str, str], dict[str, Any]]  # (system_prompt, user_prompt) -> parsed_json


@dataclass
class REPLEnvironment:
    """
    Simulates the persistent REPL state used in RLM inference.

    In Zhang et al. (2024), the 'external environment' stores variables
    that the LLM code can read without consuming context tokens.
    """

    variables: dict[str, Any] = field(default_factory=dict)
    call_log: list[dict[str, Any]] = field(default_factory=list)
    depth: int = 0

    def set(self, key: str, value: Any) -> None:
        """Store a variable in the REPL environment."""
        self.variables[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a variable from the REPL environment."""
        return self.variables.get(key, default)

    def log_call(self, depth: int, prompt_summary: str, response_summary: str) -> None:
        """Record an LLM sub-call for the Reasoning Trace visualizer."""
        self.call_log.append({
            "depth": depth,
            "prompt_summary": prompt_summary[:120],
            "response_summary": response_summary[:120],
        })


@dataclass
class RLMSubTask:
    """Represents a decomposed sub-task for recursive LLM invocation."""

    task_id: str
    description: str
    context: dict[str, Any]
    max_depth: int = 3
    result: dict[str, Any] | None = None


class RLMEngine:
    """
    Implements the Recursive Language Model inference scaffold.

    Key responsibilities:
    1. Context Offloading: Store analysis state in REPLEnvironment,
       not in LLM context window.
    2. Task Decomposition: Break complex workflows into RLMSubTasks.
    3. Recursive Invocation: Invoke LLM on each sub-task with minimal context.
    4. Result Aggregation: Combine sub-task outputs programmatically.
    5. Trace Recording: Log every recursive call for visualization.
    """

    def __init__(
        self,
        llm_callable: LLMCallable,
        system_prompt: str,
        max_depth: int = 5,
    ) -> None:
        """
        Initialize the RLM Engine.

        Args:
            llm_callable: A function that calls the LLM and returns parsed JSON.
            system_prompt: The core system prompt (tool descriptions, format rules).
            max_depth: Maximum allowed recursion depth.
        """
        self.llm = llm_callable
        self.system_prompt = system_prompt
        self.max_depth = max_depth
        self.repl_env = REPLEnvironment()
        self._trace_tree: Tree | None = None

    # ------------------------------------------------------------------ #
    # Primary Entry Point
    # ------------------------------------------------------------------ #

    def invoke(self, user_prompt: str, depth: int = 0) -> dict[str, Any]:
        """
        Invoke the LLM with context offloading via the REPL environment.

        This is the core RLM primitive. The LLM only sees a compact user_prompt
        — full state lives in self.repl_env.

        Args:
            user_prompt: Compact prompt injecting only what the LLM needs now.
            depth: Current recursion depth.

        Returns:
            Parsed JSON response from the LLM.
        """
        if depth > self.max_depth:
            console.print(
                f"[yellow]⚠ RLM max depth ({self.max_depth}) reached. Returning partial result.[/]"
            )
            return {"status": "max_depth_reached", "depth": depth}

        indent = "  " * depth
        console.print(f"{indent}[bold blue]🔁 RLM Call[/] [dim](depth={depth})[/]")

        response = self.llm(self.system_prompt, user_prompt)

        # Log call to trace
        self.repl_env.log_call(
            depth=depth,
            prompt_summary=user_prompt[:120],
            response_summary=json.dumps(response)[:120],
        )

        return response

    # ------------------------------------------------------------------ #
    # Task Decomposition
    # ------------------------------------------------------------------ #

    def decompose_and_invoke(
        self,
        sub_tasks: list[RLMSubTask],
        prompt_builder: Callable[[RLMSubTask], str],
        depth: int = 0,
    ) -> dict[str, dict[str, Any]]:
        """
        Implement Principle 2 from Zhang et al. (2024): Programmatic Recursion.

        For each sub-task, build a focused prompt and invoke the LLM recursively.
        This enables parallel conceptual processing of dataset feature groups.

        Args:
            sub_tasks: List of decomposed sub-tasks.
            prompt_builder: Function that produces a focused prompt per sub-task.
            depth: Current recursion depth.

        Returns:
            Dict mapping task_id → LLM result for each sub-task.
        """
        results: dict[str, dict[str, Any]] = {}
        console.print(
            f"\n[bold cyan]🔀 RLM Decomposition:[/] {len(sub_tasks)} sub-tasks at depth {depth}"
        )

        for task in sub_tasks:
            console.print(f"  [dim]→ Sub-task: {task.task_id} — {task.description}[/]")
            # Store sub-task context in REPL env (not in LLM prompt)
            self.repl_env.set(f"subtask_{task.task_id}", task.context)
            focused_prompt = prompt_builder(task)
            result = self.invoke(focused_prompt, depth=depth + 1)
            task.result = result
            results[task.task_id] = result

        return results

    def aggregate_sub_results(
        self,
        sub_results: dict[str, dict[str, Any]],
        synthesis_prompt: str,
        depth: int = 0,
    ) -> dict[str, Any]:
        """
        Implement Principle 3: Aggregate sub-task results via a root LLM call.

        Sub-results are stored in REPL env; only a compact summary is passed
        to the LLM for synthesis.

        Args:
            sub_results: Mapping of task_id → result from decompose_and_invoke.
            synthesis_prompt: Compact synthesis instructions for the LLM.
            depth: Current recursion depth.

        Returns:
            Final aggregated LLM response.
        """
        self.repl_env.set("sub_results", sub_results)
        summaries = {
            task_id: json.dumps(result)[:200] for task_id, result in sub_results.items()
        }
        summary_str = json.dumps(summaries, indent=2)
        full_prompt = f"{synthesis_prompt}\n\n## Sub-Task Summaries\n{summary_str}"
        return self.invoke(full_prompt, depth=depth)

    # ------------------------------------------------------------------ #
    # Reasoning Trace Visualizer
    # ------------------------------------------------------------------ #

    def print_reasoning_trace(self) -> None:
        """Render the full RLM call tree to the terminal using Rich."""
        if not self.repl_env.call_log:
            console.print("[dim]No reasoning trace available yet.[/]")
            return

        trace_tree = Tree("[bold magenta]🧠 RLM Reasoning Trace[/]")
        depth_nodes: dict[int, Any] = {-1: trace_tree}

        for entry in self.repl_env.call_log:
            depth = entry["depth"]
            parent_node = depth_nodes.get(depth - 1, trace_tree)
            node_label = (
                f"[cyan]Depth {depth}[/] → [dim]{entry['response_summary']}[/]"
            )
            child_node = parent_node.add(node_label)
            depth_nodes[depth] = child_node

        console.print(trace_tree)
