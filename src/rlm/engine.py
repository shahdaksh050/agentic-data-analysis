"""Recursive inference layer — task decomposition and context offloading."""
from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.table import Table

console = Console()

#: Bounded worker pool for parallel sub-task invocation in
#: decompose_and_invoke(). These are I/O-bound HTTP calls, so threads are
#: correct and the GIL is irrelevant; bounded by policy (not hardware) so a
#: wide decomposition doesn't slam the provider's own rate limits.
_MAX_DECOMPOSE_WORKERS = 6


# ---------------------------------------------------------------------------
# REPL Environment — external state store (RLM paradigm)
# ---------------------------------------------------------------------------

class REPLEnvironment:
    """
    Persistent key-value store representing the external REPL environment
    from the recursive-decomposition paradigm.

    The controller writes sub-task context here so each recursive invocation
    can read prior results without inflating the main LLM context window.
    """

    def __init__(self) -> None:
        self.variables: dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        self.variables[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.variables.get(key, default)

    def summary(self) -> str:
        if not self.variables:
            return "(empty)"
        lines = [f"  {k}: {str(v)[:80]}" for k, v in self.variables.items()]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# RLMSubTask — unit of decomposed work for Stage 6
# ---------------------------------------------------------------------------

@dataclass
class RLMSubTask:
    """A single decomposed reasoning unit passed to decompose_and_invoke()."""

    task_id: str
    description: str
    context: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Reasoning trace entry
# ---------------------------------------------------------------------------

@dataclass
class _TraceEntry:
    iteration: int
    depth: int
    stage: str
    user_prompt_snippet: str
    response_snippet: str
    latency_ms: float


# ---------------------------------------------------------------------------
# RLM Engine
# ---------------------------------------------------------------------------

class RLMEngine:
    """
    Recursive inference engine.

    Wraps a provider-agnostic LLM callable and adds:
      - Depth-bounded recursive invocation (max_depth guard)
      - External REPL environment for cross-call state sharing
      - Full reasoning trace for post-run inspection
      - Sub-task decomposition (Stage 6)

    The engine never imports memory, tools, or the controller — it only
    depends on the injected llm_callable.
    """

    def __init__(
        self,
        llm_callable: Callable[[str, str], dict[str, Any]],
        system_prompt: str,
        max_depth: int = 5,
    ) -> None:
        self._llm = llm_callable
        self._system_prompt = system_prompt
        self.max_depth = max_depth
        self.repl_env = REPLEnvironment()
        self._trace: list[_TraceEntry] = []
        self._iteration: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_iteration(self, iteration: int) -> None:
        """Tell the engine which outer reasoning cycle we are in."""
        self._iteration = iteration

    @property
    def trace(self) -> list[_TraceEntry]:
        """Read-only copy of the reasoning trace (one entry per LLM call)."""
        return list(self._trace)

    def invoke(
        self,
        user_prompt: str,
        depth: int = 0,
        stage: str = "",
    ) -> dict[str, Any]:
        """
        Call the LLM with the injected system prompt and the given user prompt.

        Args:
            user_prompt: The fully-formed user message.
            depth:       Current recursion depth (0 = top-level call).
            stage:       Human-readable label for the reasoning trace.

        Returns:
            Parsed JSON dict from the LLM, or an error dict on failure.
        """
        if depth > self.max_depth:
            raise RecursionError(
                f"RLM max recursion depth ({self.max_depth}) exceeded at stage '{stage}'."
            )

        t0 = time.perf_counter()
        response = self._llm(self._system_prompt, user_prompt)
        latency_ms = (time.perf_counter() - t0) * 1000

        self._trace.append(
            _TraceEntry(
                iteration=self._iteration,
                depth=depth,
                stage=stage,
                user_prompt_snippet=user_prompt[:120],
                response_snippet=str(response)[:120],
                latency_ms=latency_ms,
            )
        )
        return response

    def decompose_and_invoke(
        self,
        sub_tasks: list[RLMSubTask],
        prompt_builder: Callable[[RLMSubTask], str],
        depth: int = 1,
    ) -> dict[str, dict[str, Any]]:
        """
        Stage 6: run one LLM call per sub-task and aggregate results.

        Sub-tasks are independent — prompt_builder only ever reads a
        sub-task's own task_id/description/context, never a sibling's
        REPL-stored result — so they run concurrently on a small bounded
        thread pool instead of one LLM round trip at a time. Each
        sub-task's context is stored in the REPL environment under
        ``subtask_ctx_<task_id>`` before any call starts, and every
        result under ``subtask_result_<task_id>`` once all calls finish
        (written back in ``sub_tasks`` order, not completion order, so the
        REPL env's final state is deterministic regardless of which
        thread happened to finish first).

        Args:
            sub_tasks:      List of RLMSubTask instances to process.
            prompt_builder: Callable that turns an RLMSubTask into a prompt string.
            depth:          Recursion depth to pass to invoke().

        Returns:
            Dict mapping task_id -> LLM response dict.
        """
        for sub_task in sub_tasks:
            self.repl_env.set(f"subtask_ctx_{sub_task.task_id}", sub_task.context)

        def _run(sub_task: RLMSubTask) -> dict[str, Any]:
            prompt = prompt_builder(sub_task)
            return self.invoke(
                prompt, depth=depth, stage=f"stage6:decompose:{sub_task.task_id}"
            )

        if len(sub_tasks) <= 1:
            responses = [_run(t) for t in sub_tasks]
        else:
            with ThreadPoolExecutor(max_workers=min(_MAX_DECOMPOSE_WORKERS, len(sub_tasks))) as pool:
                responses = list(pool.map(_run, sub_tasks))

        results: dict[str, dict[str, Any]] = {}
        for sub_task, response in zip(sub_tasks, responses, strict=True):
            results[sub_task.task_id] = response
            self.repl_env.set(f"subtask_result_{sub_task.task_id}", response)

        return results

    def print_reasoning_trace(self) -> None:
        """Render the full reasoning trace as a Rich table."""
        if not self._trace:
            console.print("[dim]No reasoning trace recorded.[/]")
            return

        table = Table(
            title="RLM Reasoning Trace",
            show_lines=True,
            highlight=True,
        )
        table.add_column("Iter", style="cyan", width=4)
        table.add_column("Depth", width=5)
        table.add_column("Stage", style="magenta", width=24)
        table.add_column("Latency (ms)", width=12)
        table.add_column("Response snippet", style="dim")

        for entry in self._trace:
            table.add_row(
                str(entry.iteration),
                str(entry.depth),
                entry.stage,
                f"{entry.latency_ms:.0f}",
                entry.response_snippet,
            )

        console.print(table)
