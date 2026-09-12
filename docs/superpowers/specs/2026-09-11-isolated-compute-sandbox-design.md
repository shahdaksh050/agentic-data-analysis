# Isolated Compute Sandbox — Design Spec

**Date:** 2026-09-11
**Status:** Approved for planning
**Sub-project:** 1 of 4 in the "Universal Dynamic Analyst" hybrid extension (see Context)

## Context

The product goal is a domain-agnostic data analyst agent that can answer
plain-English questions about any uploaded dataset. The existing system
(`src/core/controller.py`, `src/rlm/engine.py`, `src/tools/*`) already does
this via a profile-driven planner that selects from a fixed library of
deterministic, tested `BaseTool` classes — see `AGENTS.md` for the 7-stage
workflow and layer rules, and `HANDOVER.md` for the in-flight "Round 5"
ingestion/rigor hardening, which is out of scope here and proceeds
independently.

What that architecture cannot do is answer a question nobody anticipated a
tool for. This sub-project adds that capability: a sandbox that safely
executes LLM-generated Python code against the actual loaded dataset. It is
purely additive — nothing in the existing planner, tool registry, or
profiler is removed or altered.

**Decisions already made (do not re-litigate):**
- Hybrid direction: the sandbox is an additional tool the agent reaches
  for; the existing tool library, profiler, and RLM planner stay as-is.
- Isolation mechanism: an OS-level subprocess sandbox (restricted
  builtins/imports, no network, resource caps enforced by the parent) —
  not Docker (not installed on the dev machine) and not WASM/Pyodide
  (partial scipy/sklearn/duckdb support would block existing use cases).
- Fresh subprocess per execution, not a persistent worker — clean
  interpreter state per attempt matters more than startup latency at the
  scale of a few self-correction attempts.
- The contract must be usable by lightweight/smaller LLMs, not only
  frontier models — this drives several choices below (auto-injected
  `df`, static pre-check, plain-language hints, minimal surface).

**Explicitly out of scope for this spec** (separate sub-projects):
- The LLM prompt/self-correction retry loop that calls this sandbox
  repeatedly on failure (sub-project 2).
- Translating a plain-English question into what code to generate
  (sub-project 3).
- Formatting sandbox results into report/dashboard output, chart specs
  (sub-project 4).

## Scope

A single capability: given a Python code string and a reference to the
current dataset, execute the code under restriction and return either a
structured success result or a structured, actionable failure. Stateless
across calls — every invocation is independent.

## Execution Flow

1. Caller (a new `DynamicCodeExecutionTool` in `src/tools/dynamic_code.py`,
   satisfying the existing Tool Contract in `AGENTS.md`) invokes
   `src/core/sandbox.py` with: `code: str`, `dataset_ref: str` (the
   already-cleaned file path from `MemorySystem`'s `cleaned_file_path`
   context — never a DataFrame serialized inline), `timeout_s`,
   `memory_limit_mb`.
2. **Static pre-check** (in the parent process, no subprocess spawned yet):
   `ast.parse` the code and verify (a) it parses, (b) it contains a
   module-level assignment to `RESULT`, (c) it contains no references to
   names outside the allowlist (`import os`, `open(`, `eval(`, etc. are
   caught lexically). Failure here returns immediately as
   `error_type: "static_check"` with a hint — no process spawned, no
   timeout consumed.
3. The sandbox spawns a fresh subprocess running
   `src/core/_sandbox_worker.py`, passing `code` and `dataset_ref` via a
   temp file (not argv/stdin, to avoid shell-escaping and length limits).
4. The worker process: loads `dataset_ref` into `df` (pandas), builds
   `SCHEMA` (column name → dtype, reusing `profiler.py`'s existing type
   labels so the LLM sees the same vocabulary the rest of the system
   uses), installs the restricted builtins/import hook, `exec()`s the code
   with `{"df": df, "SCHEMA": SCHEMA}` as globals, then serializes
   `RESULT` to JSON on stdout as its last action.
5. The parent enforces `timeout_s` via `subprocess.communicate(timeout=...)`
   and polls the child's RSS via `psutil` at a short interval, killing on
   breach of `memory_limit_mb` (soft limit — see Isolation section).
6. Parent parses the worker's stdout; any deviation from the expected
   JSON contract (crash, non-zero exit, unparseable output) is normalized
   into the same structured failure shape the worker itself would have
   produced.

## Data Contract

**Input:**
```python
def run_sandboxed(
    code: str,
    dataset_ref: str,
    timeout_s: float = 20.0,
    memory_limit_mb: int = 512,
) -> SandboxResult: ...
```

**Output** (`SandboxResult`, JSON-serializable dataclass):
```python
@dataclass
class SandboxResult:
    status: Literal["ok", "error"]
    result: Any | None              # JSON-serializable value of RESULT; None on error
    stdout: str                     # captured print() output, capped (see Limits)
    error_type: str | None          # "static_check" | "import_blocked" | "timeout"
                                     # | "memory" | "syntax" | "runtime" | "output_invalid"
    traceback: str | None           # full traceback text, only on error
    hint: str | None                # plain-language, actionable suggestion, only on error
    duration_ms: float
```

`RESULT` must be JSON-serializable (dict/list/str/int/float/bool/None). If
the code assigns a DataFrame or ndarray to `RESULT`, the worker converts it
via a fixed, documented rule (`DataFrame.to_dict(orient="records")` capped
at a row limit; `ndarray.tolist()`) rather than erroring — this is common
enough for lightweight models to produce that silently rejecting it would
waste an attempt.

## Isolation & Restrictions

Windows dev target, so no `resource` module (Unix-only, unavailable) and no
`RLIMIT_*`. This is process-level trust, not a kernel/VM boundary — the
threat model is buggy or wasteful LLM-generated code, not a deliberate
sandbox-escape adversary. Revisit with a container/VM boundary if the
system ever executes code from untrusted external users rather than an
LLM generating from the operator's own prompt.

- **Import allowlist**, enforced via a custom `__import__` hook installed
  before `exec()`: `pandas`, `numpy`, `scipy`, `sklearn`, `duckdb`,
  `polars`, `math`, `statistics`, `json`, `datetime`, `re`, `collections`,
  `itertools`. Anything else raises `import_blocked` with a hint naming
  the offending module and the allowlist.
- **Restricted builtins**: a curated `__builtins__` dict excludes `open`,
  `eval`, `exec`, `compile`, `__import__` (replaced by the hook),
  `input`, `exit`, `quit`. `print` is kept but redirected to a
  length-capped buffer.
- **No network reachability is required by the contract.** Not
  independently enforced at the OS level on Windows without extra
  tooling — the import allowlist (no `socket`, `urllib`, `requests`)
  is the actual enforcement mechanism. Documented as such, not
  oversold as a kernel-level guarantee.
- **Filesystem**: worker's cwd is a per-execution `tempfile.mkdtemp()`,
  removed after the call whether it succeeded or failed. `open()` being
  absent from builtins is the primary control; the scratch dir is
  defense in depth in case a library does file I/O internally (e.g. a
  plotting library's temp cache).
- **Timeout**: hard-enforced by the parent via `subprocess` timeout +
  `Popen.kill()`. Default 20s, caller-overridable per call.
- **Memory**: soft-enforced by the parent polling `psutil.Process(pid)
  .memory_info().rss` on a ~200ms interval and killing on breach.
  Documented as soft (a fast allocation spike between polls can
  overshoot briefly) — acceptable given the threat model above.
- **stdout cap**: captured print output truncated at a fixed size (e.g.
  8KB) to bound what flows back into a future LLM prompt.

## Module Placement

- `src/core/sandbox.py` — parent-side orchestration: static pre-check,
  subprocess lifecycle, timeout/memory enforcement, result parsing.
  Lives alongside `profiler.py`/`security.py` as infra `core` reuses,
  consistent with the existing precedent that `tools/base.py` already
  imports from `core.memory`.
- `src/core/_sandbox_worker.py` — the script the subprocess actually
  runs: restricted-namespace construction, `df`/`SCHEMA` injection,
  `exec()`, result serialization. Kept separate from `sandbox.py` so the
  worker's own import surface (what it needs at parse time before the
  restriction hook is installed) is minimal and auditable.
- `src/tools/dynamic_code.py` — `DynamicCodeExecutionTool(BaseTool)`,
  the thin adapter satisfying the Tool Contract (`name`, `description`,
  `execute()` returning a `"summary"` key, `get_schema()`,
  `ToolExecutionError` on failure). This is what the controller/planner
  actually sees and selects.

This is a new tool and touches infra `core` modules — per `AGENTS.md`
("Ask First: before adding a new tool", "before modifying the RLM core
engine or task decomposition logic"), confirm before implementation
begins that `DynamicCodeExecutionTool` fits the Execution Layer contract
as described, and that `src/core/sandbox.py` placement (vs. a
`src/tools/`-local alternative) is acceptable — same shape of question as
the `io.py` placement decision already resolved for Round 5 item 2.

## Error Taxonomy & Lightweight-LLM Accommodations

| `error_type` | Trigger | `hint` example |
| :--- | :--- | :--- |
| `static_check` | No `RESULT` assignment found, or a disallowed name referenced lexically | "Your code must assign the final answer to a variable named RESULT at the top level." |
| `import_blocked` | Import hook rejects a module at exec time | "Only pandas, numpy, scipy, sklearn, duckdb, polars, math, statistics, json, datetime, re, collections, itertools are available. Remove the import for 'os'." |
| `syntax` | `SyntaxError` during `exec()` | Raw syntax error message, already fairly actionable as-is |
| `runtime` | Any other exception during execution | Full traceback; hint extracts the exception's own message when short enough to stand alone |
| `timeout` | Wall-clock exceeded | "Execution exceeded Ns. Avoid unbounded loops; operate on df directly instead of iterating rows." |
| `memory` | RSS breach | "Execution used too much memory. Avoid materializing full copies of large intermediate results." |
| `output_invalid` | `RESULT` not JSON-serializable and not a DataFrame/ndarray | "RESULT must be a plain value (number, string, list, dict) or a DataFrame/array — got a &lt;type&gt;." |

Design choices that specifically target lightweight-model reliability:
- `df` and `SCHEMA` are pre-loaded — no file I/O code required at all.
- The static pre-check catches the single most common lightweight-model
  mistake (forgetting `RESULT =`) in milliseconds, before a timeout is
  ever at stake.
- Every failure carries a `hint`, not just a traceback — smaller models
  are measurably worse at diagnosing raw Python errors unassisted.
- The allowed-library list and the `RESULT`/`df` contract are the entire
  surface a generated program needs to know about, keeping sub-project
  2's few-shot examples short enough to be reliably imitated.

## Testing Strategy

Unit tests against `src/core/sandbox.py` directly, no LLM involved:
- Valid code assigning `RESULT` (scalar, dict, DataFrame, ndarray) returns
  `status="ok"` with the correctly-converted `result`.
- Code missing `RESULT` is caught by the static pre-check, not executed
  (assert no subprocess side effect / fast return).
- `import os` / `open(...)` / `eval(...)` each caught, both statically
  where lexically detectable and at runtime via the import hook for any
  that slip past static analysis (e.g. dynamic import strings).
- An infinite loop hits `timeout_s` and is killed; result is `error_type
  == "timeout"`.
- A large-allocation loop hits the memory cap; result is `error_type ==
  "memory"`.
- A `SyntaxError` and a runtime `ZeroDivisionError`-style exception each
  produce `error_type` `"syntax"` / `"runtime"` with a populated
  `traceback`.
- `RESULT` set to a non-serializable object (e.g. an open file handle,
  a custom class instance) returns `"output_invalid"`.
- stdout longer than the cap is truncated, not dropped entirely.
- Temp working directory is removed after both success and failure paths.

Integration test: `DynamicCodeExecutionTool.run()` end-to-end against a
real fixture dataset (reuse `tests/fixtures/` from the Round 5 corpus),
asserting the tool's `"summary"` key is present per the Tool Contract.

## Open Questions For Implementation (resolve before coding)

1. Confirm `src/core/sandbox.py` / `src/core/_sandbox_worker.py` /
   `src/tools/dynamic_code.py` placement (see Module Placement).
2. Confirm default `timeout_s` (20s proposed) and `memory_limit_mb` (512MB
   proposed) are reasonable for interactive Streamlit use — a user is
   waiting synchronously.
3. Confirm the DataFrame/ndarray auto-conversion row cap for `RESULT`
   (proposed: 1,000 rows via `to_dict(orient="records")`, with a note in
   the result if truncated).
