"""
Isolated Compute Sandbox — safe execution of LLM-generated Python code
against the current dataset.

Additive to the existing profile-driven tool architecture (see
docs/superpowers/specs/2026-09-11-isolated-compute-sandbox-design.md):
the agent reaches for this when no BaseTool in the registry answers a
question, not as a replacement for the deterministic tool library.

Process-level trust, not a kernel/VM boundary — the threat model is
buggy or wasteful LLM-generated code, not a deliberate sandbox-escape
adversary. Revisit with a container/VM boundary before ever executing
code from untrusted external users.

Pure orchestration in this file; the restricted execution itself
happens in a subprocess running _sandbox_worker.py (Task 3).
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Literal

#: Modules the sandboxed code is allowed to import. Enforced both
#: statically (_static_check, below) and at runtime inside the worker's
#: import hook (src/core/_sandbox_worker.py) as defense in depth.
ALLOWED_MODULES: frozenset[str] = frozenset({
    "pandas", "numpy", "scipy", "sklearn", "duckdb", "polars",
    "math", "statistics", "json", "datetime", "re", "collections",
    "itertools",
})

ALLOWED_MODULES_TEXT: str = ", ".join(sorted(ALLOWED_MODULES))

#: Builtins referenced by name are blocked even before exec() — matches
#: the restricted __builtins__ dict the worker installs.
BLOCKED_BUILTIN_NAMES: frozenset[str] = frozenset({
    "open", "eval", "exec", "compile", "__import__", "input", "exit", "quit",
})

#: Result-variable contract the generated code must satisfy.
RESULT_VAR_NAME: str = "RESULT"

#: Captured stdout is capped so it never balloons a future LLM prompt.
STDOUT_CAP_CHARS: int = 8_000


@dataclass
class SandboxResult:
    """Outcome of one sandboxed code execution."""

    status: Literal["ok", "error"]
    result: Any | None
    stdout: str
    #: "static_check" | "import_blocked" | "syntax" | "runtime"
    #: | "timeout" | "memory" | "output_invalid"
    error_type: str | None
    traceback: str | None
    hint: str | None
    duration_ms: float


def _static_check(code: str) -> tuple[str, str] | None:
    """
    Validate `code` before any subprocess is spawned.

    Returns (error_type, hint) if the code fails validation, else None.
    Catches the mistakes a lightweight LLM makes most often — a missing
    RESULT assignment or an import outside the allowlist — in
    milliseconds, without burning a timeout cycle on a subprocess.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return "syntax", f"Code has a syntax error: {exc.msg} (line {exc.lineno})."

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_level = alias.name.split(".")[0]
                if top_level not in ALLOWED_MODULES:
                    return (
                        "static_check",
                        f"Only {ALLOWED_MODULES_TEXT} are available. "
                        f"Remove the import for '{alias.name}'.",
                    )
        elif isinstance(node, ast.ImportFrom):
            top_level = (node.module or "").split(".")[0]
            if top_level not in ALLOWED_MODULES:
                return (
                    "static_check",
                    f"Only {ALLOWED_MODULES_TEXT} are available. "
                    f"Remove the import for '{node.module}'.",
                )
        elif isinstance(node, ast.Name) and node.id in BLOCKED_BUILTIN_NAMES:
            return (
                "static_check",
                f"'{node.id}' is not available in the sandbox. "
                "Use only pandas/numpy/scipy/sklearn/duckdb/polars operations.",
            )

    has_result_assignment = any(
        isinstance(stmt, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == RESULT_VAR_NAME for t in stmt.targets)
        for stmt in tree.body
    )
    if not has_result_assignment:
        return (
            "static_check",
            f"Your code must assign the final answer to a variable named "
            f"{RESULT_VAR_NAME} at the top level.",
        )
    return None
