# HANDOVER: state as of 2026-09-12

## Isolated Compute Sandbox — COMPLETE (all 4 tasks committed)

Worktree `.worktrees/isolated-compute-sandbox`, branch
`isolated-compute-sandbox`, branched from `master` at `1ae9461`. **Not yet
merged to master.**

- Task 1 `965031f` — SandboxResult + static pre-check. Reviewed, clean.
- Task 2 `d9154bb` — restricted worker script. Reviewed: spec compliant,
  quality approved. 3 Minor findings deferred (see the SDD ledger); the one
  worth promoting is **no regression test for the dotted-submodule import
  gate** (`name.split(".")[0]`), the most security-relevant line in the file.
- Task 3 `7a68c2e` — `run_sandboxed` subprocess orchestration.
- Task 4 `45a22cb` — `DynamicCodeExecutionTool` + registration.

Tasks 3 and 4 were written directly (user directed "just coding, no
testing"). Their 12 tests have since run green as part of the full suite
(**317 passed**, up from the 305 baseline — exactly those 12). Deviations
from the plan text: `ignore_cleanup_errors=True` on the scratch
TemporaryDirectory (Windows holds the killed child's cwd; without it a
PermissionError escapes `run_sandboxed` and breaks its "never raises"
contract), and longer timeouts in the two kill tests (the worker imports
pandas first, so the plan's 0.3s budget fired during startup and never
reached the busy loop it claimed to test).

## Domain layer — NEW, complete

`src/core/domains.py` adds semantic domain inference on top of the
structural profiler: what the data is *about*, not just its shape.

- Extensible registry — `register_domain(DomainSpec)` adds a domain without
  touching the module or any tool.
- Three built-ins: `financial`, `transactional`, `workforce`.
- Name-hint role resolution **plus a structural check**, because names alone
  cannot separate these domains (`date`/`amount`/`price`/`id` appear in all
  of them). Cardinality is the discriminator: one row per date = price
  series; many rows per customer = transaction log; one row per person =
  roster. A structural check returns a negative delta to veto a name-only
  coincidence.
- Returns a **ranked list with confidences**, not one winner — real files are
  mixed, and `applies_to` returns a float, so a plausible second domain still
  contributes.
- `DatasetProfile.domains` is a defaulted field (populated by the controller
  after profiling, since inference needs the dataframe too). Defaulted so
  `applies_to` always finds the attribute — a missing one would silently gate
  every domain tool out.
- Surfaced to the planner via `DatasetProfile.to_prompt_string()`.

New tools, each gated by `domain_confidence(profile, <domain>)`:
`src/tools/financial_analysis.py` (returns, annualised volatility, drawdown,
Sharpe, autocorrelation; single series or multi-symbol panel),
`src/tools/cohort_analysis.py` (RFM, repeat rate, AOV, revenue
concentration, top products), `src/tools/workforce_analysis.py` (headcount,
tenure, attrition, pay distribution, **unadjusted** pay gap).

Full suite after all of this: **317 passed, 1 deselected, 1 warning** (the
warning is a pre-existing sklearn FitFailedWarning from a one-class CV fold,
present in the baseline). `ruff check .` and `mypy src/` clean (34 files).

**Verified manually** (the domain layer has no committed tests yet): domain inference and
exclusive gating across stock/shop/HR fixtures; financial volatility,
max-drawdown and total return cross-checked against independent pandas
calculations; attrition and pay gap cross-checked; no false positives on
generic, IoT-sensor or single-column data; a currency-string retail export
(`$1,234.56`) flows read_any -> coerce_types -> profile -> domain -> tool.

**Ordering constraint that makes this work:** `coerce_types` must run BEFORE
`profile_dataframe`, or money-as-string columns stay non-numeric and the
transactional domain never matches. `controller.load_dataset` already does
this; do not reorder it.

## Generic robustness pass (after the domain layer)

- **Domain surfaced in both reports.** Markdown gets a "Recognised as
  <domain> data (confidence X)" block in Data Overview listing the resolved
  role->column mapping and the structural evidence, so a reader can
  challenge the classification rather than wonder why an RFM table appeared.
  HTML gets the matching badge. (Item 6's rule: mirror or the two diverge.)
- **`app.py` upload preview now uses the one reader.** It previously used a
  bare `pd.read_csv`/`pd.read_excel` in 4 places — a sixth reader knowing
  nothing about delimiters or encodings — so a `;`-delimited cp1252 export
  previewed as one mangled column while the analysis behind it was correct.
  New `io.read_any_bytes(raw, filename)` spills to a temp file so the full
  detection chain serves the in-memory upload path too. Verified: a cp1252
  `;` file with accents now previews as 3 clean columns.
- **Read-once cache in `io.read_any`**, keyed on (resolved path, mtime_ns,
  size), bounded to 4 frames. Measured on a 36MB/300k-row file: cold 270ms,
  warm 3.7ms (72x); a 9-tool run drops ~2.4s -> ~0.3s. Callers get a
  defensive `.copy()` because tools mutate what they read.

  **Two traps that were closed — do not undo these:**
  1. `invalidate_read_cache(path)` is called by both `to_csv` writers in
     `data_processing.py`. The (path, mtime, size) key alone is NOT safe for
     a rewrite: Windows `st_mtime_ns` has ~10-15ms granularity despite the
     name, so a same-size rewrite inside one tick would serve the previous
     frame — a silently wrong analysis, the exact failure class Round 5
     exists to remove. Any new code that writes a dataset to a path that may
     already have been read must call it too.
  2. `read_any_bytes` calls `_read_uncached`, not `read_any`. Caching under
     a temp path that is deleted immediately would hold a whole DataFrame
     under a key nothing can hit again and evict the real dataset (max 4
     entries).

Deferred / known gaps: no non-CSV formats (JSON/JSONL/Parquet/.gz) — the
user explicitly deferred this; `src/core/io.SUPPORTED_EXTENSIONS` is still
`{.csv,.tsv,.xlsx,.xls}`. Item 9's sampling half is untouched (it changes
results and the plan marks it ASK FIRST); only the uncontroversial
read-once cache landed. The domain layer and these changes have **no
committed tests** — verification was manual and cross-checked, but it lives
in the session transcript, not the repo. The 317 passing tests include zero
domain-layer tests.

## Round 5 — SUBSTANTIALLY COMPLETE (the section below is STALE)

Everything from "Round 5 remediation plan" down was written before the work
landed and says "Nothing below is implemented." That is **no longer true**.
Verified present and wired: item 1 (fixtures corpus, 19 factories +
`tests/test_data_shapes.py`), item 2 (`src/core/io.py`, all 5 call sites),
item 3 (`src/core/coercion.py`, called before profiling), item 4 (effect
sizes + `src/core/multiple_testing.py`), item 5 (public
`profiler.is_identifier_like`, `is_sufficient`), item 6 (report Data
Overview / Methodology / Limitations), item 7 (`profile_status`), item 10
(`src/core/degradations.py`). Items 8 and 9 remain open. Read the section
below for the original detail, not for current status.

---

# Session Handover") is a separate, ongoing piece of
work (Round 5 hardening) — see "Relationship to Round 5" at the end of this
section for how the two connect.

## What this is

Sub-project 1 of 4 in a "Universal Dynamic Analyst" hybrid extension: an
isolated subprocess sandbox letting the agent execute LLM-generated Python
code against the current dataset. Purely additive — the existing
profile-driven tool architecture (profiler, RLM planner, `BaseTool`
library) is untouched.

- Spec: `docs/superpowers/specs/2026-09-11-isolated-compute-sandbox-design.md`
- Plan: `docs/superpowers/plans/2026-09-11-isolated-compute-sandbox.md`
  (4 tasks, TDD, complete code given per task)
- Being executed via the `superpowers:subagent-driven-development` skill
  (fresh implementer subagent per task, task review after each, final
  whole-branch review at the end).

## Where the work lives

**Worktree:** `.worktrees/isolated-compute-sandbox` (this directory) on
branch `isolated-compute-sandbox`, branched from local `master` at commit
`1ae9461` — **not** from `origin/master`, which is stale at `935eaae`
(predates the checkpoint this plan depends on; see below).

**Ledger — read this first, it's the authoritative progress record:**
`.superpowers/sdd/2026-09-11-isolated-compute-sandbox/progress.md`
(pre-flight conflict scan, both tasks' implementer/reviewer results,
deferred Minor findings).

**Shared venv:** this worktree has no `.venv` of its own — reuse the main
repo's: `D:\(Dev2)DSA_AGENT\.venv\Scripts\python.exe`. Run
`ruff`/`mypy`/`pytest` via that interpreter (e.g.
`D:\(Dev2)DSA_AGENT\.venv\Scripts\python.exe -m pytest tests/ -q`) — not a
bare `python`/`pytest` on PATH (nothing is activated).

## Exact state when the session ended

- **Task 1** (SandboxResult + static pre-check): complete. Commit
  `965031f`. Reviewed — spec ✅ compliant, task quality Approved. Two Minor
  findings deferred in the ledger (relative-import hint text says "None",
  small import-branch duplication) — not blocking.
- **Task 2** (restricted worker script, `src/core/_sandbox_worker.py`):
  implementer finished, DONE, commit `d9154bb`. Implementer-reported
  13/13 task tests + 305/305 full suite passing, ruff/mypy clean — **not
  yet independently verified by a reviewer.**
- Review package for Task 2 was already generated:
  `.superpowers/sdd/2026-09-11-isolated-compute-sandbox/review-965031f..d9154bb.diff`
  — but **the task-reviewer subagent dispatch was interrupted before it
  ran** (session ended first). This is the exact resume point.

## Resume: dispatch the Task 2 reviewer

1. Work from this worktree directory (do not `cd` here from the main repo
   checkout — it's a separate working tree on its own branch).
2. Sanity check: `git log --oneline -3` should show `d9154bb` at HEAD;
   `git status --short` should be clean.
3. Dispatch a task-reviewer subagent (model: sonnet — this file is
   security-relevant: a restricted-exec namespace + import hook for
   untrusted LLM-generated code) using the template at
   `C:\Users\daksh\.claude\plugins\cache\claude-plugins-official\superpowers\6.3.0\skills\subagent-driven-development\task-reviewer-prompt.md`,
   filled in with:
   - Brief: `.superpowers/sdd/2026-09-11-isolated-compute-sandbox/task-2-brief.md`
   - Report: `.superpowers/sdd/2026-09-11-isolated-compute-sandbox/task-2-report.md`
   - Base: `965031faedb7360b2533001c397a30490b7f529d`
   - Head: `d9154bb0a51421fe6e1b5afaed3576bfe2fad1c5`
   - Diff file: `.superpowers/sdd/2026-09-11-isolated-compute-sandbox/review-965031f..d9154bb.diff`
   - Global constraints: copy from the plan's Global Constraints section
     plus Task 2's own scope notes (file list, layer rules, "no
     `sys.path` manipulation — that's a later task's job via subprocess
     env injection").
   - Security framing to include verbatim: "This is a security-relevant
     file: it builds a restricted execution namespace for untrusted,
     LLM-generated code (blocked builtins, an import allowlist enforced
     via a custom `__import__` hook). Give the restriction logic real
     scrutiny — verify `_build_restricted_builtins` actually excludes
     every `BLOCKED_BUILTIN_NAMES` entry, verify the import hook gates on
     the top-level module name correctly (`import os.path` blocked,
     `import numpy.random` not), verify `__builtins__` is actually wired
     into the globals dict passed to `exec()`. The documented threat
     model is buggy/wasteful LLM-generated code, not a deliberate
     sandbox-escape adversary — flag concrete, nameable bypasses, not
     theoretical concerns."
4. Handle the verdict per the skill: clean → ledger
   `Task 2: complete (commits 965031f..d9154bb, review clean)` and move to
   Task 3. Findings → the fix loop (resume implementer agent id
   `abcc7cc798aa74445` for rounds 1-3; ledger every round).
5. Continue with **Task 3** (`run_sandboxed` orchestration — subprocess
   spawn, timeout/memory polling, `PYTHONPATH` env injection) and
   **Task 4** (`DynamicCodeExecutionTool` + `ToolRegistry` registration in
   `controller.py`) the same way each: `task-brief` → dispatch implementer
   → `review-package` → dispatch reviewer → ledger. Task 3 needs `psutil`
   added to `requirements.txt` (in the brief already). Task 4 touches
   `src/core/controller.py` — no other task touches that file.
6. After Task 4 review is clean: dispatch the **final whole-branch code
   review** (most capable available model) per the skill, using
   `review-package PLAN_FILE 1ae9461 HEAD` (MERGE_BASE = `1ae9461`, this
   branch's point of divergence from master) and
   `superpowers:requesting-code-review`'s `code-reviewer.md` template.
   Point it at the ledger's deferred-Minor list to triage what must be
   fixed before merge.
7. When that's clean (or its one allowed fix-wave is applied and
   re-reviewed): delete the SDD workspace
   (`rm -rf .superpowers/sdd/2026-09-11-isolated-compute-sandbox/`) and
   use `superpowers:finishing-a-development-branch` — the branch
   `isolated-compute-sandbox` still needs to be merged/PR'd back to
   `master`; nothing has done that yet.

## Why a checkpoint commit exists (`19f5c9a` on master)

This plan's Task 2 depends on Round 5 files (`src/core/io.py`,
`profiler.py`'s sufficiency-gate additions) that only existed as
uncommitted working-tree state when this work started, and
`EnterWorktree`'s default `fresh` mode branches from `origin/<default
branch>` — which was stale and missing those files. User-approved ruling:
commit the in-progress Round 5 state as one checkpoint (deliberately
deviating once from Round 5's own "one item per commit" rule, for this
single snapshot only), then branch the sandbox worktree from that commit
via manual `git worktree add` (not `EnterWorktree`, to control the base
ref). Round 5 itself was **not** "finished" by this — it continues
independently below.

## Relationship to Round 5 (below)

Unrelated, parallel work. The Round 5 remediation plan below this section
is unaffected by the sandbox work and continues on `master` per its own
one-item-per-commit sequencing (see "How to work this plan" below) — the
checkpoint commit above just captured Round 5's in-progress state as of
2026-09-11 so the sandbox worktree could build on it without waiting for
Round 5 to finish first.

---

# Session Handover

## State at end of session

App boots (`streamlit run app.py`, port 8501, confirmed HTTP 200). Quality
gates all green: `ruff check .` clean, `mypy src/` clean (23 files),
`pytest tests/` 232 passed, `python scripts/validate.py` 68/68. Nothing is
committed — everything below is uncommitted working-tree state.

Modified files: `src/tools/ml_pipeline.py`, `src/tools/visualization.py`,
`tests/test_ml_enhancements.py` (Round 4 refactor), plus `IMPROVEMENTS.md`
and `HANDOVER.md` (Round 5 audit — docs only, no code).

---

## What landed this session — Round 4 (P0.1 / P0.5 / P0.6)

The `Pipeline`/`ColumnTransformer` refactor, the last open item from Round 2.
Full detail in `IMPROVEMENTS.md` "Round 4". Summary:

- `_prepare_features` now does only structural feature engineering (datetime
  expansion, ID-column dropping) — no statistic is fit there.
- New `_SkewLog1pTransformer` (`BaseEstimator`/`TransformerMixin`/
  `OneToOneFeatureMixin`) decides log1p columns at `fit` time; wiring it into
  a `Pipeline` fit on `X_train` is what makes the decision training-fold-only.
- New `_build_preprocessor(X, encoding)`: `OneHotEncoder(handle_unknown=
  "ignore")` for `LINEAR_MODELS`, `OrdinalEncoder(handle_unknown=
  "use_encoded_value", unknown_value=-1)` for tree/ensemble/clustering.
- Every model — including the clustering branch — is wrapped as
  `Pipeline([("prep", ...), ("model", ...)])` and the whole pipeline pickled.
- `_tune` prefixes grid keys `model__` and strips the prefix off `best_params`.
- `visualization.py::_feature_importance` reads
  `named_steps["model"]` + `named_steps["prep"].get_feature_names_out()`.
- `EvaluateModelTool._explain_drivers` guards `.corr()` on `is_numeric_dtype`
  (raw categoricals now reach it; `.corr()` on those was silently dropping
  *every* driver inside the blanket `except`).

New tests: `TestSkewLog1pTransformer`, `TestUnseenCategoryHandling`, and
`test_log1p_decision_uses_training_fold_only_not_full_frame` (the actual
leakage regression guard — fails against pre-refactor behaviour).

---

# Round 5 remediation plan — what needs to be done

Full audit with evidence in `IMPROVEMENTS.md` "Round 5". **Nothing below is
implemented.** This section is the executable detail so a fresh session does
not have to re-derive any of it.

## How to work this plan

- Work **one numbered item per commit**. Items 2–6 each touch multiple files
  by design; do not split an item across commits (the same no-piecemeal rule
  that made Round 4 land cleanly).
- Item 1 first, always — it is the regression harness the rest are validated
  against. Several of its fixtures are *expected to fail on day one*; those
  failures are the acceptance criteria for items 2–5.
- After each item: `ruff check .`, `mypy src/`, `pytest tests/`,
  `python scripts/validate.py` — all green before moving on (AGENTS.md
  success criteria 1–3).
- Every tool stays deterministic (AGENTS.md tool contract #6): seed anything
  random with `random_state=42`.

## Don't rewrite these — they already work

Stated because "handle any data" invites rebuilding what is sound:

- `src/core/profiler.py` — genuine semantic column typing (numeric /
  categorical / datetime / boolean / identifier / constant / text) **plus**
  dataset-nature facts (`is_time_series`, `text_cols`, `geo_lat_col`/
  `geo_lon_col`, `is_high_dimensional`, `panel_group_cols`). Check ordering is
  already deliberate — datetime before identifier, free-text before
  identifier. Extend it; don't replace it.
- `BaseTool.applies_to()` + `ToolRegistry.candidate_tools`
  (`controller.py:495-513`) — analysis selection is already profile-driven and
  the planner only sees tools that fit the data.
- `dashboard.py:607-652` — chart choice already derives from column kinds plus
  which tools produced output.
- `ToolRegistry.register()` — the pluggable-tool requirement is already met.

The gap is the **front door** (bytes → DataFrame) and the **back door**
(rigor/honesty of output), not the middle.

---

## Item 1 — Edge-case dataset corpus  `[0.5 d]` `[no deps]`

**Do this first.** It is the harness every later item is verified against.

**Files**
- Create: `tests/fixtures/__init__.py` — programmatic frame/file factories.
- Create: `tests/test_data_shapes.py` — the matrix test.
- Do **not** commit binary/large fixture files; generate them into `tmp_path`.

**What to build.** A factory per shape, each returning a written file path:

| Factory | Shape | Currently |
| :--- | :--- | :--- |
| `tsv_file` | 3×3 tab-separated | **parses as 3×1, reports success** |
| `semicolon_csv` | `;`-delimited (EU Excel) | **parses as 3×1, success** |
| `cp1252_csv` | latin-1 accented text | **UnicodeDecodeError** |
| `json_records` | `[{...},{...}]` | unsupported |
| `empty_file` | 0 bytes | `EmptyDataError` |
| `headers_only` | header row, 0 data rows | **success, quality 81/100** |
| `single_row` | 1 row | **success, quality 81/100** |
| `single_column` | 1 col × 50 | success |
| `all_categorical` | no numeric cols | success |
| `all_text` | varied prose, 2 cols | success (classifies `text` correctly) |
| `mixed_type_column` | `1,2,NOT_A_NUMBER,4` | col → object → `identifier` |
| `duplicate_headers` | `a,a,b` | silently mangled to `a`,`a.1`,`b` |
| `high_missing` | 60% NaN | success |
| `wide_frame` | 200 numeric cols | untested |
| `million_rows` | 1M × 4 (mark `slow`) | works: 0.64 s read, 0.57 s profile |
| `time_series`, `panel`, `geo`, `high_cardinality` | nature fixtures | work |

**The assertion pattern** — this is the whole point of the item:

> For every fixture, the pipeline must **either succeed on correctly-parsed
> data, or fail with an actionable error. It must never report success on
> corrupt input.**

So `tsv_file` asserts `df.shape[1] == 3` (fails today), and `headers_only`
asserts either a raised error or a profile flagged insufficient (fails today).

**Gotchas**
- pandas 3 gives string columns `str` dtype, not `object` — assert with
  `pd.api.types.is_numeric_dtype(...)`, never `dtype == object`.
- Mark `million_rows` with `@pytest.mark.slow` and keep it out of the default
  run, or the suite's 60 s runtime balloons.

**Done when** the matrix runs, and every currently-broken row is an
`xfail`-or-failing test that items 2–5 will flip green.

---

## Item 2 — Unified reader  `[1 d]` `[dep: 1]` `[ASK FIRST]`

*Closes U0.1 (delimiter corruption), U0.2 (encoding), U1.1 (5× duplication),
U1.2 (inconsistent format allowlists).*

**Why it's the highest-impact item.** U0.1 is the only finding where the
system produces a confident, complete, plausible analysis *of data that does
not exist* — and it fires on Excel's default European export format.

**The Ask First question — resolve before writing code.** AGENTS.md layer
rules say `tools/*` may not import from `core/`. But `src/tools/base.py:20`
**already** does (`from src.core.memory import MemorySystem, ToolResult`), so
the rule is already bent by the base class itself. Two options:
`src/core/io.py` (consistent with that precedent, groups with `profiler.py`/
`security.py`) or `src/tools/_io.py` (strictly layer-clean). Recommend
`src/core/io.py`. **Confirm with the user before creating the module.**

**Files**
- Create: `src/core/io.py`
- Replace the reader in all five call sites:
  `src/tools/data_processing.py:26-37`, `src/tools/ml_pipeline.py:32-42`,
  `src/tools/statistical_analysis.py:29-39`, `src/tools/visualization.py:24-34`,
  `src/core/controller.py:42-49` *(this fifth one is the divergent copy — it
  doesn't know `.tsv`, which is what makes profiling silently skip)*.
- Reconcile allowlists: `src/core/security.py:25` (`ALLOWED_EXTENSIONS`) and
  `app.py:1413` (Streamlit `type=[...]`) must import the one constant.

**Interface**

```python
@dataclass
class ReadReport:
    path: str
    format: str              # csv | tsv | excel | json | parquet
    encoding: str            # what was used
    encoding_confident: bool # False when guessed via fallback
    delimiter: str | None    # what was sniffed/assumed
    delimiter_sniffed: bool  # False when it came from the extension
    duplicate_headers: list[str]
    notes: list[str]         # human-readable, flows into the report

def read_any(file_path: str) -> tuple[pd.DataFrame, ReadReport]: ...
```

**Detection chain** (in order):
1. **Extension** → format. `.tsv` forces `sep="\t"` — this alone fixes half of
   U0.1.
2. **Encoding**: try UTF-8 → check BOM (`utf-8-sig`) → `charset_normalizer` →
   cp1252 fallback. Record which won and whether it was a guess.
   **Verified:** `charset_normalizer` 3.4.7 is installed but is a *transitive*
   dependency — it is **not** in `requirements.txt`. If item 2 uses it, add it
   explicitly; relying on a transitive dep will break on a clean install.
   (`chardet` is not installed.)
3. **Delimiter**: for delimited text, `csv.Sniffer().sniff()` on a ~64 KB
   sample against candidates `,`, `;`, `\t`, `|`. Record whether sniffed.
4. **Header validation**: detect duplicate and `Unnamed: N` columns, record
   them (item 3 acts on them; this item only reports).

**Gotchas**
- `csv.Sniffer` **raises `_csv.Error`** on genuinely single-column files.
  Catch it and fall back to `,` — do not let it propagate.
- Sniffing on a tiny sample of quoted text mis-detects. Prefer the extension
  when it is unambiguous (`.tsv`), and only sniff `.csv`.
- A BOM makes the first column name `﻿col` — strip it, or every
  downstream column-name lookup misses by one character.
- Keep `ToolExecutionError` as the raised type from tool call sites so
  `BaseTool.run()` still packages failures normally.

**Tests.** Item 1's `tsv_file`, `semicolon_csv`, `cp1252_csv` flip green.
Add: a single-column CSV still reads (Sniffer fallback), a BOM'd CSV yields
clean column names, and `ReadReport.encoding_confident is False` when the
cp1252 fallback fires.

**Done when** all five call sites use `read_any`, the three allowlists are one
constant, and `.tsv` profiles instead of silently skipping.

---

## Item 3 — Type coercion / repair pass  `[1 d]` `[dep: 2]`

*Closes U0.7 (numerics trapped in strings), part of U1.4 (mixed-type cols).*

**Evidence.** Probed on routine real-world columns: `$123.45` → classified
**identifier**; `45.3%` → **categorical, high_cardinality**. Both are dropped
from every numeric analysis *and* counted against the quality score.

**Files**
- Create: `src/core/coercion.py`
- Call it once at ingestion in `controller.load_dataset`, **before**
  `profile_dataframe(...)` (`controller.py:694-699`).

**Interface**

```python
@dataclass
class Coercion:
    column: str
    from_kind: str          # "string"
    to_kind: str            # "numeric" | "boolean" | "datetime"
    rule: str               # "currency" | "percent" | "thousands" | "yes_no"
    n_converted: int
    n_failed: int
    failed_examples: list[str]   # capped at 5 — the contaminating values

def coerce_types(df: pd.DataFrame) -> tuple[pd.DataFrame, list[Coercion]]: ...
```

**Rules** (apply only when ≥95% of non-null values match, so one stray value
can't flip a genuine text column):
- Currency: leading/trailing `$ € £ ¥`, strip thousands `,` / `_` → float.
- Percent: trailing `%` → float **/100** (decide and document: store 0.453,
  not 45.3 — say which in the `Coercion.rule` text so the report is honest).
- Thousands-separated integers: `1,234,567` → int.
- Booleans: `Y/N`, `yes/no`, `true/false`, `T/F` (case-insensitive) → bool.
- Mixed-type: when a column is ≥95% numeric-parseable, coerce and record the
  failures in `failed_examples` rather than leaving the whole column object.

**Non-negotiable.** Every coercion is **recorded and reported**, never silent
— the same discipline as `treatments_applied` in the ML path. Store the list
in memory context (`memory.set_context("coercions", ...)`) so item 6 can
render it.

**Gotchas**
- Run coercion **before** profiling or the profiler still sees the old dtypes
  and nothing changes.
- Don't coerce columns the profiler would call `identifier` by name hint (a
  zero-padded `zipcode` must not become `4521`). Check the name-hint list at
  `profiler.py:31` (`_ID_NAME_HINTS`) first, and never strip leading zeros.
- European decimals (`4,5` meaning 4.5) collide with thousands separators.
  Only treat `,` as a decimal when the delimiter sniffed in item 2 was `;` —
  this is exactly why item 3 depends on item 2's `ReadReport`.

**Tests.** `$1,234.56` → 1234.56 numeric; `45.3%` → 0.453; `Y/N` → bool;
`zipcode` `04521` stays string; a 50/50 mixed column is *not* coerced; every
coercion appears in the returned list.

---

## Item 4 — Statistical rigor  `[1.5 d]` `[dep: 1]`

*Closes U0.4. Tied with item 2 for highest impact — this is the one most
likely to be actively misleading a user today.*

**Evidence.** Two groups of 50,000, Cohen's **d = 0.038** (negligible), the
tool reports: *"Statistically significant difference detected (p=0.0000 <
α=0.05)."* Nothing in the output lets a reader tell that apart from a real
effect.

**Files:** `src/tools/statistical_analysis.py` (whole tool),
`tests/test_statistical_analysis.py`.

**Add to every branch**

| Test | Effect size | Also add |
| :--- | :--- | :--- |
| Independent / Welch's t | Cohen's d (Welch → Hedges' g) | 95% CI on mean difference |
| Mann-Whitney U | rank-biserial correlation | — |
| One-Way ANOVA | η² (eta-squared) | — |
| Kruskal-Wallis | ε² (epsilon-squared) | — |
| Chi-Square | Cramér's V | **expected-frequency validity check** |

Plus, for all branches:
- `practical_significance`: bool — distinct from `significant`. Use
  conventional thresholds (d/g: 0.2 negligible; V: 0.1; η²: 0.01) and state
  the threshold used in the output so it is auditable.
- `sample_size_note`: flag both directions — underpowered (n per group < ~20)
  *and* the large-n trap ("with n=100,000, p<0.05 corresponds to a negligible
  effect — see effect_size").
- Rewrite `interpretation` to lead with the effect, not the p-value.

**Three existing defects to fix in the same pass**
1. `statistical_analysis.py:123` — `g[:5000]` takes the **first** 5000 values,
   not a random sample. On sorted data Shapiro sees a truncated tail and
   mis-answers. Use a seeded random sample (`random_state=42`).
2. Chi-square (`_chi_square`, `:166-188`) reports a statistic that is invalid
   when expected cell counts fall below ~5 and says nothing. Compute expected
   counts, and either warn or switch to Fisher's exact for 2×2.
3. No multiple-comparison correction, though the tool is designed to be called
   repeatedly across feature/group pairs in one run. Add Benjamini-Hochberg
   across tests within a run (needs the per-run test list — store p-values in
   memory context and correct at report time, item 6).

**Gotchas**
- `scipy.stats` has no Cohen's d; compute it (pooled SD for equal-var, Welch
  correction otherwise) and unit-test the formula directly against a
  hand-computed value.
- Don't change the existing output keys (`statistic`, `p_value`,
  `significant`) — `_flag_unverified_claims` and the report both read them.
  Add alongside.

**Tests.** The n=100k / d=0.038 case asserts `significant is True` **and**
`practical_significance is False`. A real effect (d≈0.8) asserts both True.
A 2×2 chi-square with an expected count of 2 emits the validity warning.

---

## Item 5 — ID-guard fix + data-sufficiency gate  `[0.5 d]` `[dep: 1]`

*Closes U0.3 (sorted continuous rejected as ID), U1.3 (no row-count floor).*

**Evidence.** `statistical_analysis.py:81-91` rejects a feature when
`uniqueness > 0.95 and is_monotonic`. Continuous measurements are ~100%
unique, so this reduces to "is it sorted?":

| n | sorted | result |
| :-- | :-- | :-- |
| 6 / 500 / 5000 | yes | **error** — "appears to be a row ID or index" |
| 5000 | no (same data shuffled) | success |

Row order alone decides whether the system's only hypothesis-testing tool
runs. Data exported grouped by key is sorted by construction.

**Fix A — the guard.** Delete the local heuristic and reuse
`profiler._is_identifier_like` (`profiler.py:188-201`), which is stricter and
already correct: it requires a **name hint** alongside near-uniqueness, or
uniqueness == 1.0 on a **non-float** dtype. A sorted float measurement passes
that. Promote it to a public helper rather than importing an underscore name,
and delete the duplicate logic so there is one identifier rule in the codebase.

**Fix B — sufficiency gate.** `profiler.py:341-343` penalises `row_count < 100`
by a flat 10 points, so 0 rows and 1 row both score **81/100**. Add:
- `DatasetProfile.is_sufficient: bool` and a `sufficiency_reason: str | None`.
- Hard floor: `row_count < 2` → `is_sufficient = False`, quality capped at a
  low value, blocking warning emitted.
- Graduated: `< 30` rows → warn that no inferential result will be reliable.
- Surface it in `to_prompt_string()` so the planner sees it, and in the report
  limitations section (item 6).

**Gotchas**
- Don't gate *execution* on `is_sufficient` — tools should still run and
  report, but every result must carry the caveat. Silently refusing to analyse
  is as unhelpful as silently over-claiming.
- `profiler.py` has no `src.tools` import today and must keep it that way
  (layer rules). Put the shared helper in `profiler.py` and import *into* the
  tool, not the reverse.

**Tests.** Sorted continuous column at n=6/500/5000 → `success`. A genuine
`customer_id` integer column → still rejected. 0-row and 1-row frames →
`is_sufficient is False`.

---

## Item 6 — Report restructure  `[1 d]` `[dep: 3]`

*Closes U0.6. This is what makes the report explain **why these analyses, for
this dataset**.*

**Evidence.** `GenerateReportTool.execute` (`report_generator.py:130-137`)
takes only `dataset_name`, `tool_results_json`, `llm_insights`, `output_dir`.
The `DatasetProfile` never reaches it — quality score, warnings, missingness
and every caveat land in the Streamlit UI and one HTML badge
(`html_report.py:154`) but never the Markdown report the user keeps.

**The methodology gap is self-inflicted.** The planning prompt *demands* a
rationale per step — "Provide a `rationale` for EVERY step — this is a
research-grade system" (`prompt_manager.py:85`), "rationale tied to the
profile evidence" (`:131`). The LLM supplies it, it is parsed
(`controller.py:889`), stored on `AnalysisStep.rationale` (`memory.py:273`),
rendered **truncated to 60 characters in a terminal panel**
(`controller.py:1210`), and dropped. The narrative the brief asks for is
already being generated and thrown away.

**How to inject the profile — precedent already exists.** Do **not** fight the
signature. `GenerateReportTool.prepare_params` (`report_generator.py:116-128`)
already holds `memory` and already injects `tool_results_json`. Add two lines
there pulling `memory.get_context("data_profile")` and the item-2/3 artifacts.
The HTML report does exactly this at `controller.py:1056`
(`profile=self.memory.get_context("data_profile")`) — copy that pattern.

**Plumb the rationale.** `AnalysisStep` objects live in memory; serialise
`{step_number, tool_name, rationale}` into a new
`memory.set_context("plan_rationales", ...)` (or extend the tool-results
payload) so the report can pair each executed tool with why it was chosen.

**Three new Markdown sections**

1. **Data Overview** — shape, column-kind counts, quality score, missingness,
   duplicates; **what was detected vs assumed at read time** (item 2's
   `ReadReport`: encoding, delimiter, whether either was guessed); **what was
   coerced** (item 3's `Coercion` list).
2. **Methodology** — each tool that ran, with the planner's own `rationale`,
   so the report answers "why this analysis for this data".
3. **Limitations & Caveats** — profile warnings, `sufficiency_reason` (item 5),
   effect-size / practical-significance caveats and the BH correction note
   (item 4), plus anything `_flag_unverified_claims` (P0.7) marked
   `[unverified: ...]`.

**Gotchas**
- `execute()` must keep working when the new params are absent — `validate.py`
  and several tests call this tool directly with the old parameter set.
  Default them to `None` and skip the section.
- Mirror the sections into `src/core/html_report.py`, or the two reports
  diverge in content.

**Tests.** A report generated from a profile with warnings contains a
"Limitations" heading and the warning text; a run with rationales contains a
"Methodology" section naming each tool; calling `execute()` with only the old
four params still succeeds (back-compat).

---

## Item 7 — Loud profiling failure  `[0.5 d]` `[dep: 2]`

*Closes U0.5.*

**Evidence.** `controller.py:694-705` swallows profiling failure and continues
with `last_profile = None`. Measured effect on tool gating
(`candidate_tools(profile, meta)` vs `candidate_tools(None, meta)`):
**lost `time_series_analysis`; gained nothing.** So a profiling failure
quietly removes exactly the dataset-nature tools that make the analysis fit
the data — and says nothing. Keep it non-fatal; make it visible.

**Changes**
- `memory.set_context("profile_status", ...)` = `"ok"` / `"failed: <reason>"`.
- Surface "running in degraded mode — dataset-nature tools unavailable
  because X" in the Streamlit UI, the report's Limitations section (item 6),
  and the console.
- Item 2 removes the most common cause (controller's reader not knowing
  `.tsv`); this item makes the *remaining* causes visible.

---

## Item 8 — Format expansion  `[1 d]` `[dep: 2]` `[ASK FIRST]`

JSON, JSONL, Parquet, and `.gz`/`.zip`-compressed CSV via `read_any`'s
dispatch table.

**Needs a product decision before coding:** nested/semi-structured JSON —
flatten with `json_normalize` (depth-capped, which invents column names), or
reject with a clear message? Flattening a deeply nested API dump produces
hundreds of sparse columns and can make the analysis worse, not better.
**Ask the user.** Also confirm whether upload (`security.ALLOWED_EXTENSIONS`,
Streamlit `type=[...]`) should accept these or whether they stay CLI-only —
that widens the product's supported-input promise and the attack surface
`security.py` validates.

---

## Item 9 — Scale policy  `[1.5 d]` `[dep: 2]` `[ASK FIRST]`

**Measured first — this is not a current performance problem:**

| Rows | File | read | profile | in-memory |
| ---: | ---: | ---: | ---: | ---: |
| 10,000 | 0.6 MB | 0.03 s | 0.01 s | 0.5 MB |
| 200,000 | 11.6 MB | 0.15 s | 0.09 s | 9.9 MB |
| 1,000,000 | 58.0 MB | 0.64 s | 0.57 s | 49.6 MB |

Two real risks, both unmeasured beyond 1M rows:
1. **No row cap, chunking, or sampling.** Every read is a full load; memory is
   the binding constraint and nothing degrades gracefully when it binds.
2. **Every tool re-reads from disk independently.** A 9-tool run on a 1M-row
   file pays the read nine times. `dashboard.py` caps *rendering* via
   `MAX_POINTS`; nothing caps *analysis input*.

**Ask First:** sampling changes results, so whether it is acceptable and at
what default threshold is a product call. The read-once cache (keyed on
path + mtime + size — the same key shape `AgentController._step_cache` already
uses) is uncontroversial and could ship alone if sampling is rejected.

---

## Item 10 — Observability  `[0.5 d]` `[dep: 7]`

A structured `degradations` list in memory that every silent fallback appends
to: encoding guessed, delimiter sniffed, profiling skipped, coercion applied,
sampling applied, tool gated out, claim flagged unverified. Rendered in the
report and UI. Turns every "silent" in the Round 5 audit into "visible".

---

## Sequencing

```
1 (corpus) ──> 2 (reader) ──> 3 (coercion) ──> 6 (report)
          └──> 4 (rigor) ────────────────────> 6
          └──> 5 (guards) ───────────────────> 6
                2 ──> 7 (loud failure) ──> 10 (observability)
                2 ──> 8 (formats)   [ask first]
                2 ──> 9 (scale)     [ask first]
```

Items 1–6 are the core of the brief, in order. 7–10 are independent
follow-ons.

**If only one thing gets done:** item 2, with item 1 as its harness. U0.1 is
the only finding where the system produces a confident, complete analysis of
data that does not exist, and it fires on Excel's default European export.

**If the user wants the most user-visible honesty win instead:** item 4 — it
is the finding most likely to be misleading someone *today*.

## Items needing user approval before coding

| Item | Question |
| :-- | :--- |
| 2 | `src/core/io.py` vs `src/tools/_io.py` — AGENTS.md layer rules vs the precedent that `tools/base.py:20` already imports from `core.memory` |
| 8 | Nested JSON: flatten (depth-capped) or reject? Do new formats reach the uploader or stay CLI-only? |
| 9 | Is result-changing sampling acceptable, and at what row threshold? |

AGENTS.md also gates "before adding a new tool" and "before changing the
`MemorySystem` schema". Items 3, 6, 7 and 10 add **context keys**, not schema
fields — confirm that reading is acceptable, or treat them as Ask First too.

## Verification protocol (per item)

```bash
ruff check .
mypy src/
pytest tests/                 # 232 passing today — must not regress
python scripts/validate.py    # 68/68
```

Plus AGENTS.md criteria 4–5: the 7-stage workflow runs end-to-end on a sample
CSV and `output/reports/` contains a populated Markdown report. For items 2,
3 and 6, also re-run a representative dataset and **read the resulting
report** — the point of those items is output quality, which a green test
suite does not measure.

---

## Environment gotchas (cost time last session)

- **pandas 3**: string columns are `str` dtype, not `object`. Use
  `pd.api.types.is_numeric_dtype(...)`; `dtype == object` checks silently miss.
- **Console is cp1252**: probe scripts printing `α`, `σ`, `→` crash with
  `UnicodeEncodeError`. Run one-off scripts with `PYTHONIOENCODING=utf-8`.
- **sklearn 1.8**: every `Pipeline` step needs `__sklearn_tags__`; only
  `BaseEstimator` provides it. Duck-typed transformers fail at `fit`.
- **venv**: `.venv/` (note the dot) — `python -m pytest` works from the repo
  root without activation.
- Scratch scripts belong in the session scratchpad, not the repo.

---

## Also still open (pre-Round-5 backlog, in `IMPROVEMENTS.md`)

- P1.6 / P2.5 (latency, API shape) and P2.1–P2.7 structural cleanup. The
  entire P0 tier is now clear (P0.7 and P1.5 landed in earlier sessions).
- P4.1–P4.2: no tests for `time_series`, `text_analysis`, `geospatial`,
  `dimensionality`. Also `visualization.py`'s `_roc_curve` /
  `_confusion_matrix` success paths are untested (only the non-binary-target
  rejection is) — verified manually in Round 4. Item 1's corpus is the
  natural place to close these.
