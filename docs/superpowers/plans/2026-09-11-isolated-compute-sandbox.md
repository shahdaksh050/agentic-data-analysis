# Isolated Compute Sandbox Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a safe, subprocess-isolated Python code execution capability so the agent can answer questions no existing `BaseTool` covers, without touching the existing profile-driven tool architecture.

**Architecture:** A static AST pre-check rejects obviously-bad code before any process spawns. Valid code runs in a fresh, restricted subprocess per call (no persistent worker) with a pre-loaded `df`/`SCHEMA`, capped wall-clock time and polled memory, and returns a structured `SandboxResult` — success or one of seven typed failure modes, each with a plain-language hint. A thin `BaseTool` wraps it so the existing planner can select it like any other tool.

**Tech Stack:** Python 3.11 stdlib (`ast`, `subprocess`, `tempfile`, `json`), `psutil` (new dependency, soft memory-limit polling — no Windows equivalent of `resource.setrlimit`), existing `pandas`/`src.core.io.read_any`/`src.core.profiler.profile_dataframe`.

**Spec:** [docs/superpowers/specs/2026-09-11-isolated-compute-sandbox-design.md](../specs/2026-09-11-isolated-compute-sandbox-design.md)

## Global Constraints

- Python >= 3.11 (`pyproject.toml`).
- `ruff check .` clean (select: E, W, F, I, UP, N, B, C4, RUF; line-length 100).
- `mypy src/` clean under `strict = true` (`disallow_untyped_defs`, `disallow_any_generics`) — every function fully typed, `from __future__ import annotations` at the top of every new file.
- All existing tests keep passing (`pytest tests/` — 232+ passing before this work per HANDOVER.md); nothing here modifies existing files' behavior except the one registration line in `controller.py`.
- Tool Contract (AGENTS.md): subclass `BaseTool`; `execute()` returns a dict with a `"summary"` key; raise `ToolExecutionError` on expected failures; implement `get_schema()`; at least one unit test.
- Layer rules (AGENTS.md): `src/core/sandbox.py` and `src/core/_sandbox_worker.py` depend only on stdlib + data libs (pandas/numpy) + each other — never on `memory.py`, `engine.py`, or `controller.py`. `src/tools/dynamic_code.py` imports `src.core.sandbox` directly, the same established precedent as `src/tools/statistical_analysis.py` importing `src.core.io`/`src.core.profiler` (documented exception to the tools/* -> core/* rule; see HANDOVER.md Item 2).
- No Windows `resource` module — memory enforcement is a soft, parent-side poll via `psutil`, documented as such, not a hard kernel limit.
- Process-level trust, not a sandbox-escape-proof boundary — the threat model is buggy/wasteful generated code, not a deliberate adversary.

---

### Task 1: SandboxResult + static pre-check

**Files:**
- Create: `src/core/sandbox.py` (this task: module docstring, constants, `SandboxResult`, `_static_check` only — `run_sandboxed` comes in Task 3)
- Test: `tests/test_sandbox.py`

**Interfaces:**
- Produces: `ALLOWED_MODULES: frozenset[str]`, `ALLOWED_MODULES_TEXT: str`, `BLOCKED_BUILTIN_NAMES: frozenset[str]`, `RESULT_VAR_NAME: str = "RESULT"`, `STDOUT_CAP_CHARS: int = 8_000`, `SandboxResult` dataclass (fields: `status: Literal["ok", "error"]`, `result: Any | None`, `stdout: str`, `error_type: str | None`, `traceback: str | None`, `hint: str | None`, `duration_ms: float`), `_static_check(code: str) -> tuple[str, str] | None` (returns `(error_type, hint)` or `None`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sandbox.py
"""Tests for src/core/sandbox.py."""
from __future__ import annotations

from src.core import sandbox


class TestStaticCheck:
    def test_valid_code_passes(self) -> None:
        code = "RESULT = 1\n"
        assert sandbox._static_check(code) is None

    def test_syntax_error_caught(self) -> None:
        code = "RESULT = (\n"
        result = sandbox._static_check(code)
        assert result is not None
        error_type, hint = result
        assert error_type == "syntax"
        assert "syntax error" in hint.lower()

    def test_missing_result_assignment_caught(self) -> None:
        code = "x = 1\n"
        result = sandbox._static_check(code)
        assert result is not None
        error_type, hint = result
        assert error_type == "static_check"
        assert "RESULT" in hint

    def test_result_assignment_nested_in_if_not_recognized(self) -> None:
        # Only a module-level assignment counts — matches the spec's "at
        # the top level" requirement; the worker's own runtime check
        # (Task 2) is what catches an unreachable nested assignment.
        code = "if False:\n    RESULT = 1\n"
        result = sandbox._static_check(code)
        assert result is not None
        assert result[0] == "static_check"

    def test_disallowed_import_caught(self) -> None:
        code = "import os\nRESULT = 1\n"
        result = sandbox._static_check(code)
        assert result is not None
        error_type, hint = result
        assert error_type == "static_check"
        assert "os" in hint

    def test_disallowed_import_from_caught(self) -> None:
        code = "from socket import socket\nRESULT = 1\n"
        result = sandbox._static_check(code)
        assert result is not None
        assert result[0] == "static_check"

    def test_allowed_import_passes(self) -> None:
        code = "import numpy as np\nRESULT = 1\n"
        assert sandbox._static_check(code) is None

    def test_blocked_builtin_name_caught(self) -> None:
        code = "f = open('x.txt')\nRESULT = 1\n"
        result = sandbox._static_check(code)
        assert result is not None
        error_type, hint = result
        assert error_type == "static_check"
        assert "open" in hint
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_sandbox.py -v`
Expected: FAIL/ERROR — `src.core.sandbox` module does not exist yet.

- [ ] **Step 3: Write the implementation**

```python
# src/core/sandbox.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_sandbox.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Lint and type-check**

Run: `ruff check src/core/sandbox.py tests/test_sandbox.py && mypy src/core/sandbox.py`
Expected: no violations, no type errors.

- [ ] **Step 6: Commit**

```bash
git add src/core/sandbox.py tests/test_sandbox.py
git commit -m "feat(sandbox): add SandboxResult contract and static pre-check"
```

---

### Task 2: Restricted worker script

**Files:**
- Create: `src/core/_sandbox_worker.py`
- Test: `tests/test_sandbox_worker.py`

**Interfaces:**
- Consumes: `ALLOWED_MODULES`, `ALLOWED_MODULES_TEXT`, `BLOCKED_BUILTIN_NAMES`, `RESULT_VAR_NAME`, `STDOUT_CAP_CHARS` from `src.core.sandbox` (Task 1). `src.core.io.read_any(file_path: str) -> tuple[pd.DataFrame, ReadReport]`. `src.core.profiler.profile_dataframe(df: pd.DataFrame) -> DatasetProfile` (per-column `.kind: str`).
- Produces: `_execute(code: str, dataset_ref: str) -> dict[str, Any]` (a JSON-serializable dict matching `SandboxResult`'s fields as plain keys: `status`, `result`, `stdout`, `error_type`, `traceback`, `hint`, `duration_ms`), and a `main()` entry point that Task 3's `run_sandboxed` invokes as a subprocess: `python _sandbox_worker.py <input_json_path> <result_json_path>`, where `input_json_path` contains `{"code": ..., "dataset_ref": ...}` and `main()` writes `_execute(...)`'s return value to `result_json_path`.

**Reuses `tests/fixtures/single_column(tmp_path)`** (existing factory in `tests/fixtures/__init__.py`) — writes a 1-column, 50-row numeric CSV at `<tmp_path>/single_column.csv` with a `value` column.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sandbox_worker.py
"""
Tests for src/core/_sandbox_worker.py's execution logic, called directly
(no subprocess) for speed. Some cases here deliberately bypass
src.core.sandbox._static_check to verify the worker enforces its own
restrictions independently — defense in depth, not redundancy: a
RESULT assignment can be present in source but never actually execute
(e.g. inside `if False:`), which only a runtime check catches.
"""
from __future__ import annotations

from pathlib import Path

from tests.fixtures import single_column
from src.core import _sandbox_worker as worker


class TestExecute:
    def test_scalar_result(self, tmp_path: Path) -> None:
        dataset = single_column(tmp_path)
        payload = worker._execute("RESULT = float(df['value'].mean())\n", str(dataset))
        assert payload["status"] == "ok"
        assert isinstance(payload["result"], float)
        assert payload["error_type"] is None

    def test_dataframe_result_converted_to_records(self, tmp_path: Path) -> None:
        dataset = single_column(tmp_path)
        payload = worker._execute("RESULT = df.head(3)\n", str(dataset))
        assert payload["status"] == "ok"
        assert isinstance(payload["result"], list)
        assert len(payload["result"]) == 3
        assert "value" in payload["result"][0]

    def test_series_result_converted(self, tmp_path: Path) -> None:
        dataset = single_column(tmp_path)
        payload = worker._execute("RESULT = df['value'].head(2)\n", str(dataset))
        assert payload["status"] == "ok"
        assert isinstance(payload["result"], list)
        assert len(payload["result"]) == 2

    def test_ndarray_result_converted(self, tmp_path: Path) -> None:
        dataset = single_column(tmp_path)
        code = "import numpy as np\nRESULT = np.array([1, 2, 3]).tolist()\n"
        payload = worker._execute(code, str(dataset))
        assert payload["status"] == "ok"
        assert payload["result"] == [1, 2, 3]

    def test_schema_is_prebuilt_and_usable(self, tmp_path: Path) -> None:
        dataset = single_column(tmp_path)
        payload = worker._execute("RESULT = SCHEMA['value']\n", str(dataset))
        assert payload["status"] == "ok"
        assert payload["result"] == "numeric"

    def test_print_output_captured(self, tmp_path: Path) -> None:
        dataset = single_column(tmp_path)
        payload = worker._execute("print('hello sandbox')\nRESULT = 1\n", str(dataset))
        assert payload["status"] == "ok"
        assert "hello sandbox" in payload["stdout"]

    def test_stdout_capped(self, tmp_path: Path) -> None:
        dataset = single_column(tmp_path)
        code = "print('x' * 20000)\nRESULT = 1\n"
        payload = worker._execute(code, str(dataset))
        assert payload["status"] == "ok"
        assert len(payload["stdout"]) < 20000
        assert "truncated" in payload["stdout"]

    def test_result_never_assigned_at_runtime(self, tmp_path: Path) -> None:
        # Static-check-shaped source, but the assignment never runs.
        dataset = single_column(tmp_path)
        payload = worker._execute("if False:\n    RESULT = 1\n", str(dataset))
        assert payload["status"] == "error"
        assert payload["error_type"] == "static_check"

    def test_import_blocked_at_runtime(self, tmp_path: Path) -> None:
        # Bypasses src.core.sandbox._static_check on purpose — this test
        # is specifically about the worker's own import hook.
        dataset = single_column(tmp_path)
        payload = worker._execute("import os\nRESULT = 1\n", str(dataset))
        assert payload["status"] == "error"
        assert payload["error_type"] == "import_blocked"

    def test_syntax_error_reported(self, tmp_path: Path) -> None:
        # Bypasses src.core.sandbox._static_check on purpose (that gate
        # rejects this before a worker would ever see it in production) —
        # this test is specifically about the worker's own fallback.
        dataset = single_column(tmp_path)
        payload = worker._execute("RESULT = (\n", str(dataset))
        assert payload["status"] == "error"
        assert payload["error_type"] == "syntax"
        assert payload["hint"] is not None

    def test_runtime_exception_reported(self, tmp_path: Path) -> None:
        dataset = single_column(tmp_path)
        payload = worker._execute("RESULT = 1 / 0\n", str(dataset))
        assert payload["status"] == "error"
        assert payload["error_type"] == "runtime"
        assert "ZeroDivisionError" in payload["traceback"]
        assert payload["hint"] is not None  # extracted from the exception message

    def test_output_invalid_for_non_serializable_result(self, tmp_path: Path) -> None:
        dataset = single_column(tmp_path)
        code = "class Thing:\n    pass\nRESULT = Thing()\n"
        payload = worker._execute(code, str(dataset))
        assert payload["status"] == "error"
        assert payload["error_type"] == "output_invalid"

    def test_dataset_load_failure_reported_as_runtime(self, tmp_path: Path) -> None:
        payload = worker._execute("RESULT = 1\n", str(tmp_path / "does_not_exist.csv"))
        assert payload["status"] == "error"
        assert payload["error_type"] == "runtime"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_sandbox_worker.py -v`
Expected: FAIL/ERROR — `src.core._sandbox_worker` module does not exist yet.

- [ ] **Step 3: Write the implementation**

```python
# src/core/_sandbox_worker.py
"""
Sandbox worker — runs as a standalone subprocess (see src/core/sandbox.py
run_sandboxed). Loads the dataset, builds a restricted execution
namespace, execs the caller's code, and writes a JSON result.

Not imported by anything outside tests and sandbox.py's subprocess
spawn — this is the untrusted-code boundary, kept minimal on purpose.
"""
from __future__ import annotations

import builtins as _builtins_module
import contextlib
import datetime
import io
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.io import read_any
from src.core.profiler import profile_dataframe
from src.core.sandbox import (
    ALLOWED_MODULES,
    ALLOWED_MODULES_TEXT,
    BLOCKED_BUILTIN_NAMES,
    RESULT_VAR_NAME,
    STDOUT_CAP_CHARS,
)

#: `src.*` is importable here because run_sandboxed (src/core/sandbox.py)
#: sets PYTHONPATH to the repo root on this subprocess's environment —
#: no sys.path manipulation needed in this file. The subprocess itself
#: still runs with cwd set to a scratch temp directory.

#: DataFrame/Series results are capped at this many rows when converted
#: to RESULT — matches the spec's proposed default.
RESULT_ROW_CAP = 1_000


def _cap(text: str) -> str:
    if len(text) <= STDOUT_CAP_CHARS:
        return text
    hidden = len(text) - STDOUT_CAP_CHARS
    return text[:STDOUT_CAP_CHARS] + f"\n...[truncated, {hidden} more characters]"


def _build_schema(df: pd.DataFrame) -> dict[str, str]:
    profile = profile_dataframe(df)
    return {col.name: col.kind for col in profile.columns}


def _convert_result(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        truncated = len(value) > RESULT_ROW_CAP
        records = value.head(RESULT_ROW_CAP).to_dict(orient="records")
        return {"__truncated__": True, "rows": records} if truncated else records
    if isinstance(value, pd.Series):
        return _convert_result(value.to_frame())
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _restricted_import(
    real_import: Any,
) -> Any:
    def _import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        top_level = name.split(".")[0]
        if top_level not in ALLOWED_MODULES:
            raise ImportError(
                f"Only {ALLOWED_MODULES_TEXT} are available. "
                f"Import of '{name}' is not permitted in the sandbox."
            )
        return real_import(name, globals, locals, fromlist, level)

    return _import


def _build_restricted_builtins() -> dict[str, Any]:
    safe = {
        name: obj
        for name, obj in vars(_builtins_module).items()
        if name not in BLOCKED_BUILTIN_NAMES
    }
    safe["__import__"] = _restricted_import(_builtins_module.__import__)
    return safe


def _build_restricted_globals(df: pd.DataFrame, schema: dict[str, str]) -> dict[str, Any]:
    return {
        "__builtins__": _build_restricted_builtins(),
        "df": df,
        "SCHEMA": schema,
    }


def _error_payload(
    error_type: str,
    tb: str | None,
    hint: str | None,
    t0: float,
    stdout: str,
) -> dict[str, Any]:
    return {
        "status": "error",
        "result": None,
        "stdout": _cap(stdout),
        "error_type": error_type,
        "traceback": tb,
        "hint": hint,
        "duration_ms": (time.perf_counter() - t0) * 1000,
    }


def _execute(code: str, dataset_ref: str) -> dict[str, Any]:
    t0 = time.perf_counter()

    try:
        df, _report = read_any(dataset_ref)
    except Exception:
        return _error_payload(
            "runtime", traceback.format_exc(), "Failed to load the dataset.", t0, ""
        )

    schema = _build_schema(df)
    restricted_globals = _build_restricted_globals(df, schema)
    stdout_buf = io.StringIO()

    try:
        with contextlib.redirect_stdout(stdout_buf):
            exec(compile(code, "<sandboxed_code>", "exec"), restricted_globals)
    except ImportError as exc:
        return _error_payload(
            "import_blocked", traceback.format_exc(), str(exc), t0, stdout_buf.getvalue()
        )
    except SyntaxError as exc:
        return _error_payload(
            "syntax", traceback.format_exc(),
            f"Code has a syntax error: {exc.msg} (line {exc.lineno}).",
            t0, stdout_buf.getvalue(),
        )
    except Exception as exc:
        # Every failure should carry an actionable hint, not just a raw
        # traceback — lightweight models are markedly worse at
        # self-diagnosing tracebacks unassisted. Fall back to the
        # exception's own message when it's short enough to stand alone;
        # a very long message is more noise than help as a one-line hint.
        message = str(exc)
        hint = message if message and len(message) <= 200 else None
        return _error_payload("runtime", traceback.format_exc(), hint, t0, stdout_buf.getvalue())

    if RESULT_VAR_NAME not in restricted_globals:
        return _error_payload(
            "static_check", None,
            f"Your code must assign the final answer to a variable named "
            f"{RESULT_VAR_NAME} at the top level.",
            t0, stdout_buf.getvalue(),
        )

    raw_result = restricted_globals[RESULT_VAR_NAME]
    converted = _convert_result(raw_result)
    try:
        json.dumps(converted, default=_json_default)
    except TypeError:
        return _error_payload(
            "output_invalid", None,
            f"RESULT must be a plain value (number, string, list, dict) or a "
            f"DataFrame/array — got {type(raw_result).__name__}.",
            t0, stdout_buf.getvalue(),
        )

    return {
        "status": "ok",
        "result": converted,
        "stdout": _cap(stdout_buf.getvalue()),
        "error_type": None,
        "traceback": None,
        "hint": None,
        "duration_ms": (time.perf_counter() - t0) * 1000,
    }


def main() -> None:
    input_path, result_path = sys.argv[1], sys.argv[2]
    payload_in = json.loads(Path(input_path).read_text(encoding="utf-8"))
    result = _execute(payload_in["code"], payload_in["dataset_ref"])
    Path(result_path).write_text(
        json.dumps(result, default=_json_default), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_sandbox_worker.py -v`
Expected: PASS (13 tests)

Note: these tests call `_execute` directly in-process, so they don't exercise
the PYTHONPATH-via-subprocess-env mechanism described in the comment at the
top of `_sandbox_worker.py` — that only matters when the script actually runs
as a subprocess, which Task 3's `run_sandboxed` tests cover end-to-end.

- [ ] **Step 5: Lint and type-check**

Run: `ruff check src/core/_sandbox_worker.py tests/test_sandbox_worker.py && mypy src/core/_sandbox_worker.py`
Expected: no violations, no type errors.

- [ ] **Step 6: Commit**

```bash
git add src/core/_sandbox_worker.py tests/test_sandbox_worker.py
git commit -m "feat(sandbox): add restricted worker script for sandboxed execution"
```

---

### Task 3: run_sandboxed orchestration (subprocess, timeout, memory)

**Files:**
- Modify: `src/core/sandbox.py` (append `run_sandboxed` and its subprocess-management helpers to the file from Task 1)
- Modify: `requirements.txt` (add `psutil`)
- Test: `tests/test_sandbox.py` (append a new test class)

**Interfaces:**
- Consumes: `SandboxResult`, `_static_check` (Task 1, same file). `_WORKER_SCRIPT` resolves to `src/core/_sandbox_worker.py` (Task 2) via `Path(__file__).with_name(...)`.
- Produces: `run_sandboxed(code: str, dataset_ref: str, timeout_s: float = 20.0, memory_limit_mb: int = 512) -> SandboxResult` — the public entry point Task 4's tool calls.

- [ ] **Step 1: Add the psutil dependency**

Add to `requirements.txt` under `--- Core Data Processing ---` (alongside pandas/numpy):

```
psutil>=5.9.0        # src/core/sandbox.py soft memory-limit polling (no Windows resource module)
```

Run: `pip install psutil>=5.9.0`

- [ ] **Step 2: Write the failing tests**

First, add this import to the **top** import block of `tests/test_sandbox.py`
(next to the existing `from src.core import sandbox` line) — appending it
after the `TestStaticCheck` class would trip ruff's E402:

```python
from tests.fixtures import single_column
```

Then append the following new test class to the end of `tests/test_sandbox.py`:

```python
class TestRunSandboxed:
    def test_success_end_to_end(self, tmp_path) -> None:
        dataset = single_column(tmp_path)
        result = sandbox.run_sandboxed(
            "RESULT = float(df['value'].mean())\n", str(dataset)
        )
        assert result.status == "ok"
        assert isinstance(result.result, float)
        assert result.error_type is None
        assert result.duration_ms > 0

    def test_static_check_failure_short_circuits_before_subprocess(
        self, tmp_path, monkeypatch
    ) -> None:
        dataset = single_column(tmp_path)

        def _fail_if_called(*args, **kwargs):
            raise AssertionError("subprocess.Popen should not be called")

        monkeypatch.setattr(sandbox.subprocess, "Popen", _fail_if_called)
        result = sandbox.run_sandboxed("x = 1\n", str(dataset))
        assert result.status == "error"
        assert result.error_type == "static_check"

    def test_timeout_kills_process(self, tmp_path) -> None:
        dataset = single_column(tmp_path)
        code = "RESULT = 0\nwhile True:\n    pass\n"
        result = sandbox.run_sandboxed(code, str(dataset), timeout_s=0.3)
        assert result.status == "error"
        assert result.error_type == "timeout"

    def test_memory_limit_kills_process(self, tmp_path, monkeypatch) -> None:
        class _FakeMemInfo:
            rss = 999 * 1024 * 1024

        class _FakePsProcess:
            def __init__(self, pid: int) -> None:
                pass

            def memory_info(self) -> _FakeMemInfo:
                return _FakeMemInfo()

        monkeypatch.setattr(sandbox.psutil, "Process", _FakePsProcess)
        dataset = single_column(tmp_path)
        # Long-running so the parent's poll fires before natural completion.
        code = "RESULT = 0\ntotal = 0\nfor i in range(10**9):\n    total += i\nRESULT = total\n"
        result = sandbox.run_sandboxed(code, str(dataset), timeout_s=5.0, memory_limit_mb=50)
        assert result.status == "error"
        assert result.error_type == "memory"

    def test_worker_crash_without_result_file_reports_runtime_error(
        self, tmp_path, monkeypatch
    ) -> None:
        crashing_worker = tmp_path / "crashing_worker.py"
        crashing_worker.write_text("import sys\nsys.exit(3)\n", encoding="utf-8")
        monkeypatch.setattr(sandbox, "_WORKER_SCRIPT", crashing_worker)

        dataset = single_column(tmp_path)
        result = sandbox.run_sandboxed("RESULT = 1\n", str(dataset))
        assert result.status == "error"
        assert result.error_type == "runtime"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_sandbox.py::TestRunSandboxed -v`
Expected: FAIL/ERROR — `run_sandboxed` does not exist yet.

- [ ] **Step 4: Write the implementation**

First, update the import block at the **top** of `src/core/sandbox.py` (the
one Task 1 wrote) to add the new stdlib and third-party imports this task
needs — do not add these mid-file, or ruff's E402 will flag them:

```python
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import psutil
```

Then append the following **code** (constants and functions, no more
imports) to `src/core/sandbox.py`, after `_static_check`:

```python
_WORKER_SCRIPT = Path(__file__).with_name("_sandbox_worker.py")

#: Repo root, so the worker subprocess can `import src.core...` despite
#: running with cwd set to a scratch temp directory (see _worker_env below).
_REPO_ROOT = Path(__file__).resolve().parents[2]

#: How often the parent polls the child's RSS / checks for completion.
_POLL_INTERVAL_S = 0.2


def _worker_env() -> dict[str, str]:
    """Child process environment with the repo root prepended to
    PYTHONPATH, so `src.core.io` etc. import cleanly regardless of the
    worker's cwd — without any sys.path manipulation inside the worker
    script itself."""
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(_REPO_ROOT) if not existing else f"{_REPO_ROOT}{os.pathsep}{existing}"
    )
    return env


def run_sandboxed(
    code: str,
    dataset_ref: str,
    timeout_s: float = 20.0,
    memory_limit_mb: int = 512,
) -> SandboxResult:
    """
    Execute `code` against the dataset at `dataset_ref` in an isolated
    subprocess and return a structured result.

    Never raises for a failure of the *sandboxed code* — syntax errors,
    blocked imports, timeouts, and runtime exceptions all come back as
    SandboxResult(status="error", ...).
    """
    t0 = time.perf_counter()

    static_error = _static_check(code)
    if static_error is not None:
        error_type, hint = static_error
        return SandboxResult(
            status="error", result=None, stdout="",
            error_type=error_type, traceback=None, hint=hint,
            duration_ms=(time.perf_counter() - t0) * 1000,
        )

    with tempfile.TemporaryDirectory(prefix="sandbox_") as scratch_dir:
        input_path = Path(scratch_dir) / "input.json"
        result_path = Path(scratch_dir) / "result.json"
        input_path.write_text(
            json.dumps({"code": code, "dataset_ref": dataset_ref}), encoding="utf-8"
        )

        proc = subprocess.Popen(
            [sys.executable, str(_WORKER_SCRIPT), str(input_path), str(result_path)],
            cwd=scratch_dir,
            env=_worker_env(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        try:
            ps_proc: psutil.Process | None = psutil.Process(proc.pid)
        except psutil.NoSuchProcess:
            ps_proc = None

        killed_as: str | None = None
        while True:
            try:
                proc.wait(timeout=_POLL_INTERVAL_S)
                break
            except subprocess.TimeoutExpired:
                elapsed = time.perf_counter() - t0
                if elapsed > timeout_s:
                    killed_as = "timeout"
                elif ps_proc is not None:
                    try:
                        rss_mb = ps_proc.memory_info().rss / (1024 * 1024)
                        if rss_mb > memory_limit_mb:
                            killed_as = "memory"
                    except psutil.NoSuchProcess:
                        pass
                if killed_as is not None:
                    proc.kill()
                    proc.wait()
                    break

        duration_ms = (time.perf_counter() - t0) * 1000

        if killed_as == "timeout":
            return SandboxResult(
                status="error", result=None, stdout="",
                error_type="timeout",
                traceback=None,
                hint=(
                    f"Execution exceeded {timeout_s:.0f}s. Avoid unbounded "
                    "loops; operate on df directly instead of iterating rows."
                ),
                duration_ms=duration_ms,
            )
        if killed_as == "memory":
            return SandboxResult(
                status="error", result=None, stdout="",
                error_type="memory",
                traceback=None,
                hint=(
                    f"Execution used more than {memory_limit_mb}MB. Avoid "
                    "materializing full copies of large intermediate results."
                ),
                duration_ms=duration_ms,
            )

        if not result_path.exists():
            return SandboxResult(
                status="error", result=None, stdout="",
                error_type="runtime",
                traceback=f"Worker exited with code {proc.returncode} and wrote no result.",
                hint="The sandboxed process crashed before producing a result.",
                duration_ms=duration_ms,
            )

        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            return SandboxResult(
                status="error", result=None, stdout="",
                error_type="runtime",
                traceback=str(exc),
                hint="The sandboxed process produced an unreadable result.",
                duration_ms=duration_ms,
            )

        return SandboxResult(
            status=payload["status"],
            result=payload.get("result"),
            stdout=payload.get("stdout", ""),
            error_type=payload.get("error_type"),
            traceback=payload.get("traceback"),
            hint=payload.get("hint"),
            duration_ms=duration_ms,
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_sandbox.py -v`
Expected: PASS (13 tests — 8 from Task 1 + 5 from this task)

- [ ] **Step 6: Lint and type-check**

Run: `ruff check src/core/sandbox.py tests/test_sandbox.py && mypy src/core/sandbox.py`
Expected: no violations, no type errors.

- [ ] **Step 7: Full-suite regression check**

Run: `python -m pytest tests/ -v`
Expected: all previously-passing tests still pass; new sandbox tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/core/sandbox.py requirements.txt tests/test_sandbox.py
git commit -m "feat(sandbox): add run_sandboxed subprocess orchestration with timeout/memory limits"
```

---

### Task 4: DynamicCodeExecutionTool + registration

**Files:**
- Create: `src/tools/dynamic_code.py`
- Modify: `src/core/controller.py:437-470` (`ToolRegistry._register_builtin_tools`) — add the import and one line to the registration tuple
- Test: `tests/test_dynamic_code_tool.py`

**Interfaces:**
- Consumes: `run_sandboxed`, `SandboxResult` from `src.core.sandbox` (Task 3). `BaseTool`, `ToolExecutionError` from `src.tools.base`.
- Produces: `DynamicCodeExecutionTool` class, `name = "execute_dynamic_code"`, registered in `ToolRegistry`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_dynamic_code_tool.py
"""Tests for src/tools/dynamic_code.py."""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.fixtures import single_column
from src.tools.base import ToolExecutionError
from src.tools.dynamic_code import DynamicCodeExecutionTool


class TestDynamicCodeExecutionTool:
    def test_execute_success(self, tmp_path: Path) -> None:
        dataset = single_column(tmp_path)
        tool = DynamicCodeExecutionTool()
        output = tool.execute(file_path=str(dataset), code="RESULT = float(df['value'].mean())\n")
        assert "summary" in output
        assert output["status"] == "ok"
        assert isinstance(output["result"], float)

    def test_execute_reports_sandbox_failure_without_raising(self, tmp_path: Path) -> None:
        dataset = single_column(tmp_path)
        tool = DynamicCodeExecutionTool()
        output = tool.execute(file_path=str(dataset), code="x = 1\n")  # no RESULT
        assert output["status"] == "error"
        assert output["error_type"] == "static_check"
        assert "summary" in output

    def test_run_wraps_sandbox_failure_as_tool_success(self, tmp_path: Path) -> None:
        # The tool did its job (ran the sandbox and reported faithfully);
        # the *generated code* failing is not a tool-infrastructure error.
        dataset = single_column(tmp_path)
        tool = DynamicCodeExecutionTool()
        tool_result = tool.run(file_path=str(dataset), code="x = 1\n")
        assert tool_result.status == "success"
        assert tool_result.output["status"] == "error"

    def test_empty_code_raises(self, tmp_path: Path) -> None:
        dataset = single_column(tmp_path)
        tool = DynamicCodeExecutionTool()
        with pytest.raises(ToolExecutionError):
            tool.execute(file_path=str(dataset), code="")

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        tool = DynamicCodeExecutionTool()
        with pytest.raises(ToolExecutionError):
            tool.execute(file_path=str(tmp_path / "nope.csv"), code="RESULT = 1\n")

    def test_get_schema(self) -> None:
        tool = DynamicCodeExecutionTool()
        schema = tool.get_schema()
        assert schema["file_path"]["required"] is True
        assert schema["code"]["required"] is True


class TestRegistration:
    def test_tool_is_registered(self) -> None:
        from src.core.controller import ToolRegistry

        registry = ToolRegistry()
        assert registry.has("execute_dynamic_code")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_dynamic_code_tool.py -v`
Expected: FAIL/ERROR — `src.tools.dynamic_code` does not exist yet.

- [ ] **Step 3: Write the tool implementation**

```python
# src/tools/dynamic_code.py
"""
Dynamic Code Execution Tool — Execution Layer.

Wraps src.core.sandbox.run_sandboxed as a BaseTool so the existing
profile-driven planner can select it like any other tool. Additive:
use only for questions no other tool in the registry answers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.core.sandbox import run_sandboxed
from src.tools.base import BaseTool, ToolExecutionError


class DynamicCodeExecutionTool(BaseTool):
    """Execute custom Python code against the dataset in an isolated subprocess."""

    name = "execute_dynamic_code"
    description = (
        "Execute custom Python code against the dataset when no other tool "
        "answers the question. A DataFrame `df` and column-kind `SCHEMA` "
        "are pre-loaded; assign the answer to RESULT. Runs in an isolated, "
        "restricted subprocess — no file/network access, no imports outside "
        "pandas/numpy/scipy/sklearn/duckdb/polars/math/statistics/json/"
        "datetime/re/collections/itertools. Use only for questions the "
        "other analysis tools cannot address."
    )

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        code: str,
        **_: Any,
    ) -> dict[str, Any]:
        if not code or not code.strip():
            raise ToolExecutionError("No code was provided to execute.")
        if not Path(file_path).exists():
            raise ToolExecutionError(f"Dataset file not found: {file_path}")

        sandbox_result = run_sandboxed(code=code, dataset_ref=file_path)

        if sandbox_result.status == "ok":
            return {
                "summary": f"Executed successfully in {sandbox_result.duration_ms:.0f} ms.",
                "status": "ok",
                "result": sandbox_result.result,
                "stdout": sandbox_result.stdout,
                "error_type": None,
                "traceback": None,
                "hint": None,
                "duration_ms": sandbox_result.duration_ms,
            }

        return {
            "summary": (
                f"Execution failed ({sandbox_result.error_type}): "
                f"{sandbox_result.hint or 'see traceback for details.'}"
            ),
            "status": "error",
            "result": None,
            "stdout": sandbox_result.stdout,
            "error_type": sandbox_result.error_type,
            "traceback": sandbox_result.traceback,
            "hint": sandbox_result.hint,
            "duration_ms": sandbox_result.duration_ms,
        }

    def get_schema(self) -> dict[str, Any]:
        return {
            "file_path": {
                "type": "string",
                "description": "Path to the (cleaned) dataset.",
                "required": True,
            },
            "code": {
                "type": "string",
                "description": (
                    "Python code to execute against the dataset. Must assign "
                    "the final answer to a variable named RESULT at the top "
                    "level. A pandas DataFrame `df` and a `SCHEMA` dict "
                    "(column name -> semantic kind) are pre-loaded — do not "
                    "read or parse any file. Only pandas, numpy, scipy, "
                    "sklearn, duckdb, polars, math, statistics, json, "
                    "datetime, re, collections, itertools are available."
                ),
                "required": True,
            },
        }
```

- [ ] **Step 4: Register the tool**

In `src/core/controller.py`, inside `ToolRegistry._register_builtin_tools` (around line 437-470):

Add to the import block (after the `src.tools.dimensionality` import, alphabetical among the `src.tools.*` imports):
```python
        from src.tools.dynamic_code import DynamicCodeExecutionTool
```

Add to the registration tuple (after `DimensionalityAnalysisTool(),`):
```python
            DynamicCodeExecutionTool(),
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_dynamic_code_tool.py -v`
Expected: PASS (7 tests)

- [ ] **Step 6: Lint and type-check**

Run: `ruff check src/tools/dynamic_code.py src/core/controller.py tests/test_dynamic_code_tool.py && mypy src/tools/dynamic_code.py src/core/controller.py`
Expected: no violations, no type errors.

- [ ] **Step 7: Full-suite regression check**

Run: `python -m pytest tests/ -v`
Expected: all previously-passing tests still pass; all new sandbox + tool tests pass (33 new tests total across Tasks 1-4).

- [ ] **Step 8: Commit**

```bash
git add src/tools/dynamic_code.py src/core/controller.py tests/test_dynamic_code_tool.py
git commit -m "feat(sandbox): add DynamicCodeExecutionTool and register it in ToolRegistry"
```
