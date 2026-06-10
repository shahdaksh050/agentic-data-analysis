# PLAN.md — End-to-End "AI Data Scientist" Build Plan
**Created**: 2026-06-10 | **Branch**: `agent-updates` | **Supersedes**: HANDOVER.md (previous session)

---

## 1. Product Vision

A system where the user **only uploads a dataset** and optionally types **what they want to know in plain English**. The software then autonomously:

1. Validates & ingests the upload (security-checked).
2. Profiles the data like a data scientist's "first look" (nature, size, quality, semantics).
3. Cleans and processes it.
4. Plans and executes statistical / ML analysis via LLM agents (with deterministic non-AI fallbacks).
5. Hands results to a **dashboard agent** that builds charts *fitted to the data's nature* (not hardcoded).
6. Generates a final report answering the user's stated objective.
7. Uses the RLM (Recursive Language Model) pattern for context offloading: full data stays in the Memory System; agents see only compact summaries.

**Replace-the-data-scientist principle**: every AI step must have a deterministic fallback so the pipeline completes even with no API key.

---

## 2. Current State (as of this session)

### Committed (baseline, 4 commits on `agent-updates`)
- `f0ab9a1` bugfix: RF/XGB training, pandas-3 dtypes, eval data leakage
- `929a5be` test: 96/96 tests passing across 9 suites
- `210ea3c` feat/docs: Vega-Lite charts in app.py, sample data, README rewrite
- `876e124` chore: untrack generated CSV artifacts

All quality gates were green at baseline: `ruff check .`, `mypy src/`, `pytest tests/` (96/96), `scripts/validate.py` (68/68), `scripts/dry_run.py` (40/40).

### NEW this session — ALL COMPLETE, tested, gates green (159/159 tests)
| Item | Status |
|------|--------|
| `src/core/security.py` — upload validation, filename sanitisation, magic-byte sniffing, safe output paths | ✅ + 19 tests |
| `src/core/profiler.py` — per-column semantics, quality score, warnings, prompt summary | ✅ + 18 tests |
| `src/core/dashboard.py` — dynamic chart selection (8 chart types), Vega-Lite specs, `dashboard.json` | ✅ + 12 tests |
| NL objective: `main.py --objective` → `USER_OBJECTIVE` env → memory context → all prompts → deterministic fallback | ✅ + tests |
| Controller: profiling at Stage 1, `_generate_dashboard()` after Stage 7 | ✅ |
| `app.py`: secure upload, objective textarea, 📊 Dashboard + 🔬 Profile tabs | ✅ |
| README updated (features, CLI, UI, config table) | ✅ |

**Verified E2E**: `main.py --dataset data/sample_customer_churn.csv --target churn --objective "..."` with no API key → complete report, objective echoed, 10-chart dashboard (box plot correctly picked `support_calls`, the planted signal).

Sections 3.1–3.6 below were the build spec and are now DONE; kept for reference. Remaining: section 3.7 commits (if not yet pushed) and section 4 stretch goals.

---

## 3. Remaining Work (ordered, with file-level specs)

### 3.1 Dynamic Dashboard Agent — `src/core/dashboard.py` (NEXT)
Pure-Python module (no streamlit import → testable, mypy-clean) that builds Vega-Lite specs **adapted to the data**:

```python
@dataclass
class ChartSpec:
    chart_id: str; title: str; description: str
    spec: dict[str, Any]          # full Vega-Lite spec with inline "values"

def build_dashboard(df, profile, metadata, tool_results) -> list[ChartSpec]
```

**Chart selection rules** (the "thinks like a data scientist" logic):
- Target categorical → class-balance bar.
- Numeric columns → histograms for the top-N most informative (by correlation with target, else variance). Skip identifiers/constants (use `profile.columns_of_kind`).
- Low-cardinality categorical → count bars (top 12 categories).
- Top correlated numeric pair(s) → scatter (sample max 1,000 rows, colored by target if classification).
- Datetime column + numeric → monthly aggregated line chart.
- Numeric vs categorical target → per-class boxplot for the strongest feature.
- `train_model` result present → grouped Train/Test/CV bar (port from app.py).
- `correlation_analysis` result present → diverging bars + heatmap (rect mark).

**Constraints**: inline data only (no file refs); aggregate/sample to keep specs < ~100KB; dark-theme config injected by the UI, not the module; deterministic sampling (`random_state=42`).
Also expose `dashboard_to_json(specs) -> str` so the controller can save `output/dashboard.json`.

### 3.2 Natural-Language Objective Plumbing
Touch points (all small):
- `main.py`: add `--objective "..."` arg → `os.environ["USER_OBJECTIVE"]`.
- `src/core/controller.py`: `AgentController.__init__` reads `USER_OBJECTIVE` env (or new `objective` param); store via `self.memory.set_context("user_objective", obj)`.
- `src/core/prompt_manager.py`: when `memory.get_context("user_objective")` is set, inject a `## User Objective` section into `get_initial_user_prompt`, `get_iteration_user_prompt`, and `get_final_interpretation_prompt` with instruction: *"Prioritise analyses that answer this objective; the final insights MUST directly address it."*
- `_deterministic_final()` in controller: prepend the objective to `reasoning` so non-LLM runs still echo it.
- `app.py`: sidebar `st.text_area("What do you want to learn from this data?")` → env var before run.

### 3.3 Profiler Integration
- `controller.load_dataset()`: after metadata stored, read the df (it's already read inside IngestDatasetTool — simplest decoupled approach: `pd.read_csv/read_excel` once more here, or refactor ingest tool to return the profile; prefer running `profile_dataframe` in `load_dataset` and `self.memory.set_context("data_profile", profile.to_dict())` + keep the object on `self.last_profile`).
- `prompt_manager.get_initial_user_prompt()`: append `profile.to_prompt_string()` (warnings + quality score) so the planner knows about skew/IDs/imbalance.
- `app.py`: new "🔬 Profile" tab — quality-score gauge, per-column table (kind, missing %, flags), warnings list.

### 3.4 app.py UI Overhaul
- Use `validate_upload()` + sanitised name when writing the temp file (fixes path-traversal; show `UploadValidationError` message nicely).
- Sidebar: objective text area (3.2).
- New results tab "📊 Dashboard": `for cs in build_dashboard(...): st.vega_lite_chart(cs.spec | dark config)` with title/description captions; download button for `dashboard.json`.
- New "🔬 Profile" tab (3.3).
- Keep existing tabs (Overview/Dataset/Models/Insights/Tool Log/Report/Downloads).
- Optional polish: replace emoji metric tiles with consistent cards; empty-state copy mentions objective box.

### 3.5 Tests (target ≥ 96 + ~25 new, all deterministic)
- `tests/test_security.py`: traversal (`../../etc/passwd`, `..\\..\\x.csv`), reserved names (`con.csv`), bad extension, empty file, oversize (monkeypatch `MAX_UPLOAD_MB=1`), PNG-bytes-as-csv rejection, valid csv/xlsx pass-through, `resolve_output_path` escape rejection.
- `tests/test_profiler.py`: kind detection per column type (numeric/categorical/datetime strings/bool/constant/id), skew flag, high-missing flag, duplicate counting, imbalance warning, quality-score monotonicity (dirty < clean), `to_prompt_string` content.
- `tests/test_dashboard.py`: classification df with target → expect class-balance + histogram + model-comparison specs; no-target EDA df → no class-balance; identifiers excluded; specs JSON-serialisable; sampling cap respected.
- `tests/test_prompt_manager.py`: add objective-injection cases (present in all three prompts; absent → unchanged).
- `tests/test_controller.py`: objective stored in memory context; profile context key set after `load_dataset`.

### 3.6 Quality Gates + Docs
```powershell
.venv/Scripts/python.exe -m ruff check .
.venv/Scripts/python.exe -m mypy src/
.venv/Scripts/python.exe -m pytest tests/ -q
.venv/Scripts/python.exe scripts/validate.py
.venv/Scripts/python.exe scripts/dry_run.py
```
- README: add "How it works" (upload → profile → plan → analyze → dashboard → report), objective flag, security section (limits, sniffing), dashboard docs.
- E2E smoke: `python main.py --dataset data/sample_customer_churn.csv --target churn --objective "what drives churn?" --max-iterations 3 --output-dir output_test` (works with no API key via fallback).

### 3.7 Commit Strategy for this work
1. `feat(security): upload validation, filename sanitisation, safe output paths`
2. `feat(profiler): automated dataset profiling with quality score`
3. `feat(dashboard): dynamic data-aware Vega-Lite dashboard agent`
4. `feat(objective): natural-language analysis objective plumbed end-to-end`
5. `feat(ui): profile + dashboard tabs, secure upload, objective input`
6. `test/docs: new suites + README`

Then (with user approval): merge `agent-updates` → `master`.

---

## 4. Future / Stretch (not this milestone)
- FastAPI service layer wrapping `AgentController` (REST: POST /analyze, GET /status) → decouples UI from pipeline; enables React frontend later.
- More sample datasets: regression (housing), multiclass, time-series.
- PDF export of report (weasyprint) + dashboard PNG snapshots.
- Per-session result caching keyed by dataset hash.
- Plotly/altair theme toggle; drill-down filters in dashboard.
- Auth + rate limiting if ever hosted publicly.

---

## 5. Environment Gotchas (do not relearn these)
- **venv**: `.venv/Scripts/python.exe` (Python 3.13, pandas 3.0.3).
- **pandas 3.0**: string columns are `str` dtype, NOT `object`. Use `pd.api.types.is_numeric_dtype()` negation / `is_*_dtype` helpers. `select_dtypes(include="object")` finds nothing.
- **sklearn**: never truthiness-check an unfitted estimator (`__len__` → AttributeError).
- **Windows console**: cp1252 — set `PYTHONIOENCODING=utf-8` when running scripts via Bash; PowerShell is the default shell.
- **mypy strict** on `src/` only (tests excluded). Every new src file needs full annotations, no bare `dict`/`list` generics.
- **ruff**: line-length 100 ignored (E501), isort enforced, `RUF001-3` ignored (unicode in strings OK).
- **CI**: `.github/workflows/ci.yml` watches master/main/dev/agent-updates.

## 6. Architecture Invariants (preserve these)
- Raw data NEVER enters LLM prompts — only `to_prompt_string()` summaries (RLM context offloading).
- Tools only via `ToolRegistry`; LLM only via `RLMEngine`; prompts only via `PromptManager`.
- Every tool returns `ToolResult` with `summary`; failures become `status: error`, never exceptions.
- Tool failure budget: 2 attempts, then `skipped`.
- All randomness seeded (`random_state=42`).
- API keys from env vars only; never logged, never committed.
