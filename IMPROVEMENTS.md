# Backend Audit — Improvements (Round 2)

Deep audit of the backend: orchestration (`src/core/controller.py`,
`src/rlm/engine.py`, `src/core/memory.py`, `src/core/prompt_manager.py`,
`src/core/profiler.py`, `src/core/security.py`), execution
(`src/tools/*.py`), presentation builders (`src/core/dashboard.py`,
`src/core/html_report.py`), entry points (`main.py`, `scripts/*`), plus a
scoped review of `app.py` where it constrains the backend API.

Findings are ranked by impact on the system's core promise: **correct,
fast, autonomous analysis and interpretation that a user can trust.**

Every claim below is anchored to `file:line` and, where it is a
performance claim, to a measurement taken on this machine (method in
[Appendix A](#appendix-a--measurement-method)).

---

## Round 1 status — verified fixed

The previous `IMPROVEMENTS.md` (deleted in the working tree) listed ten
items. Nine are confirmed landed in the current code, most with in-code
comments citing the original item number:

| # | Item | Verified at |
| :-- | :--- | :--- |
| 1 | Task-type inference unified | `src/tools/data_processing.py:117` delegates to `DatasetMetadata.infer_task_type()` |
| 2 | `.iloc` → `.loc` on label indices | `src/tools/data_processing.py:320`, `:329` |
| 3 | Field-wise shrink instead of blind slice | `src/core/memory.py:41-86` (`_shrink_to_fit`) |
| 4 | Multi-class target correlation | `src/tools/data_processing.py:470-494` (eta-squared branch) |
| 5 | Overfit-penalised best-model pick | `src/tools/ml_pipeline.py:532-550` (`_pick_best`) |
| 6 | `test_size` forced to match training | `src/tools/ml_pipeline.py:618-621` |
| 7 | Shapiro `n>=3` guard | `src/tools/statistical_analysis.py:121-126` |
| 8 | String-sniffed path substitution replaced | `src/tools/base.py:69-105` (`prepare_params` + declarative `uses_cleaned_file`) |
| 9 | Dead `dtype in (float,)` clause removed | `src/tools/statistical_analysis.py:95` |

Item 10 was `app.py`-scoped and is out of this round's scope.

**Nothing below repeats those.** This round is a fresh pass.

---

## 0. Current state of the quality gates

Measured on the working tree at audit time:

| Gate | Result |
| :--- | :--- |
| `ruff check .` | ✅ clean |
| `mypy src/` | ✅ clean, 23 files |
| `pytest tests/` | ❌ **5 failed, 199 passed in 124.22s** |
| `python scripts/validate.py` | ✅ 68/68, 2.59s |

The five failures are **stale tests, not product defects** — but they
still break AGENTS.md success criterion #3, so they block any claim that
the tree is green:

- `tests/test_controller.py:78,86,95,105` — four `TestParseSteps` cases call
  `AgentController._parse_steps({...})` unbound. `_parse_steps` became an
  instance method when the hallucinated-tool-name guard started needing
  `self.tool_registry` (`src/core/controller.py:784-834`). The test was
  never updated.
- `tests/test_pipeline_3d.py:135` — asserts `PALETTE["ink"] == "#171c1f"`;
  the palette is now `#3a2b1e` after the Drafting Table → Ledger theme
  change in `DESIGN.md`.

**Fix:** bind the four `_parse_steps` calls to a controller instance (or
a lightweight fake registry), and either re-point the palette assertion at
`DESIGN.md` as the source of truth or update the literal. ~20 minutes,
and it restores the ability to trust CI.

---

## P0 — Correctness and trustworthiness

These are the findings that most directly undermine "analysis you can
trust." A wrong number delivered confidently is worse than a slow one.

### P0.1 — Train/test leakage chain: imputation, encoding and transforms are all fit on the full dataset

The pipeline's split happens *after* every preprocessing decision has
already seen the test rows.

1. `CleanDataTool` computes imputation statistics over the whole file and
   writes a cleaned CSV: `subset.fillna(subset.median(numeric_only=True))`
   at `src/tools/data_processing.py:195` (same for `mean` at `:193`,
   `mode` at `:197-199`, `ffill` at `:209`).
2. `_prepare_features` then applies `log1p` using a skew computed on all
   rows (`src/tools/ml_pipeline.py:114-120`) and fits a fresh
   `LabelEncoder` per categorical column on all rows
   (`src/tools/ml_pipeline.py:124-125`).
3. Only then does `train_test_split` run
   (`src/tools/ml_pipeline.py:251-253`).

Every "held-out" metric the system reports — `test_metrics`,
`train_test_gap`, `evaluate_model`'s accuracy — is therefore computed on
rows whose imputed values, log transform and category codes were derived
with knowledge of those same rows. The bias is small for median
imputation on large data and can be large for `mode`/`ffill` on small or
high-missingness data. Either way, the *number the agent prints as its
generalisation estimate is not a generalisation estimate.*

**Fix:** move preprocessing inside a `sklearn.pipeline.Pipeline` fitted
only on the training fold — `ColumnTransformer(SimpleImputer +
OrdinalEncoder/OneHotEncoder + FunctionTransformer(np.log1p))`. This also
fixes P0.5 (the saved model becomes self-contained) and P0.6 (ordinal
codes stop being fed to linear models) in the same change. `CleanDataTool`
stays useful as an *EDA* step and as the thing that reports what was
missing — it just stops being the thing that decides model inputs.

### P0.2 — Cross-validation runs on the full dataset, including the test split

`src/tools/ml_pipeline.py:288`:

```python
cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
```

`X, y` here is the *entire* dataset — `X_test` rows are inside those CV
folds. `cv_mean` is then used as the primary ranking metric by
`_pick_best` (`src/tools/ml_pipeline.py:532-550`), surfaced as the
headline number in `_deterministic_final`
(`src/core/controller.py:1012-1018`), and charted next to "Test" in the
dashboard's model comparison (`src/core/dashboard.py:372-411`) — where
the chart's own description tells the reader that a train-vs-test gap
signals overfitting.

Compounding it: when tuning is on, hyperparameters are selected by
`RandomizedSearchCV` over `X_train` (`src/tools/ml_pipeline.py:279-282`,
`:491-500`), and the resulting estimator is then CV-scored over data that
includes the rows tuning did *not* see. The two numbers the pipeline
places side by side are not measured on comparable partitions.

**Fix:** `cross_val_score(model, X_train, y_train, ...)`. When tuning ran,
don't even call it — `RandomizedSearchCV` already computed exactly this
quantity with the same splitter and scorer; read `search.best_score_` and
`np.std(search.cv_results_["mean_test_score"])`. That makes the fix
*free* and removes 5 model fits per tuned model (see P1.2).

### P0.3 — The profiler detects time-series and panel structure; the splitter never hears about it

`profile_dataframe` computes `is_time_series`
(`src/core/profiler.py:350`) and `panel_group_cols`
(`src/core/profiler.py:371-378`). Grepping every consumer of those two
fields shows they reach exactly two places: tool *gating*
(`src/tools/time_series.py:70`) and *prompt text*
(`src/core/profiler.py:153-162`). They never reach `TrainModelTool`.

So on a dataset the profiler has just labelled time-series,
`train_test_split(X, y, random_state=42, stratify=...)` at
`src/tools/ml_pipeline.py:251` shuffles rows — the model trains on the
future and is tested on the past, and `StratifiedKFold`
(`:255-259`) does the same five more times. On panel data (repeated
observations per customer/store/device), the same entity lands in both
train and test, and the reported accuracy is substantially inflated.

The detection already exists. Only the wiring is missing — and
`prepare_params` already receives `memory`
(`src/tools/base.py:69-71`), which already holds the serialised profile
at `memory.get_context("data_profile")` (`src/core/controller.py:634`).

**Fix:** have `TrainModelTool.prepare_params` read `data_profile` and pass
a `split_strategy` through to `execute`:

| Profile fact | Splitter | CV |
| :--- | :--- | :--- |
| `is_time_series` | chronological `train_test_split(shuffle=False)` on sorted time column | `TimeSeriesSplit` |
| `panel_group_cols` non-empty | `GroupShuffleSplit` on the group key | `GroupKFold` |
| neither | current behaviour | current behaviour |

Report the chosen strategy in the tool output so it reaches the report and
the LLM's synthesis. This is the single highest-value change in this
document.

### P0.4 — Datetime columns are discarded rather than engineered

`src/tools/ml_pipeline.py:91-94`:

```python
if pd.api.types.is_datetime64_any_dtype(series):
    features = features.drop(columns=[col])
    treatments.append(f"Dropped datetime column '{col}' (not model-ready).")
```

On a dataset whose profile says `is_time_series: true`, *all* temporal
signal is thrown away before modelling. The agent then tells the user, in
the report's "treatments applied" list, that it did so — which is honest
but not what a data scientist would do.

**Fix:** replace the drop with expansion —
`year, month, day, dayofweek, hour, is_weekend`, plus elapsed-days-since-min
as a trend term. Roughly ten lines, and it turns the system's best-detected
data nature from a liability into a feature set. Note the interaction: do
this *after* P0.3, so the raw time column is still available to order the
chronological split.

### P0.5 — Saved models are not self-contained, so they cannot score new data

`pickle.dump(model, f)` (`src/tools/ml_pipeline.py:305`, `:339`) saves the
bare estimator. The `LabelEncoder`s fitted in `_prepare_features`
(`src/tools/ml_pipeline.py:124-125`) and the target encoder from
`_encode_target` (`:129-143`) are discarded.

`EvaluateModelTool` gets away with it only by re-deriving identical
encodings from the identical file
(`src/tools/ml_pipeline.py:640-644`) — a coincidence, not a contract.
Point the saved `.pkl` at any new data with a different category set, or
even the same categories in a different order, and the codes silently
differ. There is no `predict` path in the system today, which is precisely
why this hasn't bitten yet; it will the moment scoring is added.

`ClusterDataTool` shows the correct pattern already —
`pickle.dump({"scaler": ..., "kmeans": ..., "features": ...})` at
`src/tools/clustering.py:186`. Apply it uniformly, or better, fold the
encoders into the `Pipeline` from P0.1.

### P0.6 — `LabelEncoder` on nominal categoricals feeds fake ordinality to linear models

`src/tools/ml_pipeline.py:124-125` label-encodes every remaining
categorical column into `0..n-1`. Trees can recover from arbitrary
integer codes; `LogisticRegression`, `Ridge` and `LinearRegression`
(`src/tools/ml_pipeline.py:401-406`) cannot — they read
`plan=basic(0) < plan=premium(1) < plan=enterprise(2)` as a real distance.
Two of the four regression models and one of the three classification
models are affected.

`LabelEncoder` is also documented for *targets*, not features; its use
here is why the column-order dependence in P0.5 exists at all.

**Fix:** `OneHotEncoder(handle_unknown="ignore", max_categories=...)` for
linear models inside the per-model pipeline, keeping ordinal codes for
tree models. The profiler already flags `high_cardinality`
(`src/core/profiler.py:265`) so the cases where one-hot would explode are
already known.

### P0.7 — "Cite only verbatim metrics" is a prompt rule with no enforcement

`SYSTEM_PROMPT_CORE` states it as a hard rule
(`src/core/prompt_manager.py:82-84`):

> In Form 2, cite ONLY metric values that appear verbatim in the results
> provided to you. If a number is not in the results, do not state it —
> never estimate, extrapolate, or invent values.

Nothing checks it. The LLM's `insights`, `recommendations` and
`key_metrics` flow unvalidated from
`RLMEngine.invoke` → `_generate_final_report`
(`src/core/controller.py:1266-1302`) → `report.md`, `report.html` and the
dashboard. For a system whose entire value proposition is *interpretation*,
the one rule that keeps interpretation honest is enforced by hope.

**Fix — the highest-leverage interpretation change in this document:** add
a post-synthesis validator in the controller, between `analyze()`'s loop
exit and `_generate_final_report`:

1. Extract every numeric literal from the LLM's `insights` /
   `recommendations` / `key_metrics`.
2. Build the set of numbers that actually appear in
   `[r.to_dict() for r in memory.tool_results]` (recursively, rounded to
   the same precision).
3. Any literal with no match is either stripped, or the sentence carrying
   it is annotated `[unverified]` in the report and logged to
   `memory.set_context("unverified_claims", ...)`.

This converts the system's central promise from a prompt instruction into
a mechanism, and it produces a *measurable* hallucination rate you can
track across model/provider changes. It also makes the deterministic
fallback (`_deterministic_final`, `src/core/controller.py:998-1078`) and
the LLM path directly comparable for the first time.

### P0.8 — `max_iterations` constructor argument is silently overridden by the environment

`src/core/controller.py:482`:

```python
self.max_iterations = int(os.getenv("MAX_ITERATIONS", str(max_iterations or 15)))
```

The env var wins over the explicit argument. Its sibling three lines
below gets the precedence right:

```python
self.enable_rlm = (
    enable_rlm if enable_rlm is not None
    else os.getenv("ENABLE_RLM_INFERENCE", "true").lower() == "true"
)
```

Both current callers work around it by mutating the environment first
(`app.py:1608`, `main.py:143`), which is why nobody has noticed. Any
programmatic caller — an API wrapper, a test, a notebook — that passes
`max_iterations=3` with a `.env` present silently gets 15, and with each
iteration costing an LLM round trip plus a full tool pass, that is a 5×
cost overrun with no error.

**Fix:** mirror the `enable_rlm` pattern. Two lines.

### P0.9 — A step that already succeeded is re-executed on every replanning cycle

`_execute_steps` (`src/core/controller.py:1092-1195`) tracks
`_tool_failure_counts` so a *failing* tool is skipped after
`MAX_STEP_RETRIES`. There is no equivalent for *successes*. The LLM
re-plans from scratch each iteration (`src/core/controller.py:727-733`),
and any step it repeats — `clean_data` on the same file,
`correlation_analysis` with the same parameters — runs again in full.

Correctness impact, not just speed: `train_model` re-run with a different
tuning draw can overwrite `output/models/random_forest.pkl`
(`src/tools/ml_pipeline.py:303-305`) *after* `best_model_path` was already
written to memory context (`src/core/controller.py:1174-1180`), so
`evaluate_model` can report on a different fitted model than the one whose
metrics reached the report. With `max_iterations` defaulting to 15, this
is not a corner case.

**Fix:** a content-addressed result cache in the controller, keyed on
`(tool_name, sorted resolved params, input file mtime+size)`. On a hit,
re-append the cached `ToolResult` and skip execution. This is the same
mechanism as P1.4 and it is the cheapest large win in the document.

---

## P1 — Speed

### Measured baseline

50,000 rows × 16 columns (12 numeric, 2 categorical, 1 datetime, 1 binary
target), 13.2 MB CSV, warm OS cache, single run:

| Step | Wall time |
| :--- | ---: |
| `_read_df` — one CSV parse | 0.12 s |
| `profile_dataframe` | 0.08 s |
| `clean_data` | 0.53 s |
| `detect_outliers` (iqr) | 0.55 s |
| `correlation_analysis` | 0.11 s |
| **`train_model`** (tuning **off**) | **5.71 s** |
| `evaluate_model` (permutation importance) | 2.34 s |
| `cluster_data` (k search 2..8) | 3.60 s |

Per-model breakdown at 20,000 rows (the threshold below which tuning is
**on by default** — `src/tools/ml_pipeline.py:227`):

| Model | `.fit()` | `cross_val_score` n_jobs=1 | `cross_val_score` n_jobs=-1 | `_tune` (n_iter ≤ 8) |
| :--- | ---: | ---: | ---: | ---: |
| random_forest | 0.20 s | 1.21 s | **2.66 s** | **14.03 s** |
| xgboost | 0.25 s | 0.92 s | **1.92 s** | **6.99 s** |
| logistic_regression | 0.01 s | 0.09 s | **1.24 s** | 0.30 s |

Two results drive everything below.

### P1.1 — Re-reading CSVs is *not* the bottleneck. Don't optimise it first.

Five separate copies of `_read_df` exist
(`src/core/controller.py:41`, `src/tools/data_processing.py:26`,
`src/tools/ml_pipeline.py:31`, `src/tools/statistical_analysis.py:29`,
`src/tools/visualization.py:24`) and the file is re-parsed by every tool.
That is a real design problem — see P2.1 — but at 50k rows it costs
**0.12 s × ~7 calls ≈ 0.8 s**, under 5% of a run. Fixing it first would be
optimising the wrong thing. It matters at 10× the row count; it does not
matter now.

### P1.2 — Hyperparameter tuning is ~95% of model-training time, and it is on by default

At 20k rows, tuning costs **21.3 s** against **0.46 s** of actual fitting —
a 46× multiplier. `do_tune = tune_hyperparameters and len(X) <= 20_000`
(`src/tools/ml_pipeline.py:227`) means the default path for any dataset a
user is likely to upload interactively pays it, and P0.9 means a
replanning cycle can pay it twice.

The arithmetic: `n_iter=min(8, n_combos)` × `n_splits=5` = up to **40 fits
per model** in `_tune` (`src/tools/ml_pipeline.py:492-500`), *plus* 1
final fit, *plus* 5 more in the separate `cross_val_score` at `:288`.
**46 fits to report one model.** Three models → ~138 fits.

Three fixes, in order of value:

1. **Delete the redundant CV entirely when tuning ran.** `search.best_score_`
   is the cross-validated score of the selected configuration, computed by
   the same splitter with the same scorer. Reading it instead of calling
   `cross_val_score` at `:288` removes 5 fits per tuned model — **~11% of
   training time, for free** — *and* fixes the leakage in P0.2. Do this one
   first; it is strictly a win on both axes.
2. **Switch to successive halving.**
   `sklearn.model_selection.HalvingRandomSearchCV` evaluates many
   configurations on small data subsets and promotes only survivors,
   typically reaching the same optimum in 3–5× less time on this search
   space shape.
3. **Make the tuning budget explicit and adaptive.** `n_iter=8` on a
   3-parameter random-forest grid of 27 combinations is a coin flip
   dressed as a search. Either raise it and accept the cost knowingly, or
   drop to a 2-point grid for the interactive path and expose
   `tune_hyperparameters` in the UI. Right now the user pays 21 s for a
   search they cannot see or control.

### P1.3 — `n_jobs=-1` on the CV loop is a pessimisation here — measured, not theorised

The obvious move on seeing three `n_jobs=1` sites
(`src/tools/ml_pipeline.py:288`, `:498`, `:728`) is to flip them to `-1`.
**Measured, that makes it slower on every model** — 2.2× for
random_forest, 2.1× for xgboost, 13× for logistic_regression (table
above).

Two causes: joblib's `loky` backend spawns processes on Windows and must
re-pickle the feature matrix per worker, and `RandomForestClassifier` /
`XGBClassifier` are *already* constructed with `n_jobs=-1`
(`src/tools/ml_pipeline.py:395`, `:399`), so outer parallelism oversubscribes
the CPU against inner parallelism.

**Recommendation:** leave the CV `n_jobs=1` sites alone. If nesting is
ever wanted, it must come with inner `n_jobs=1` and a measurement on the
target platform. The genuinely parallelisable work in this codebase is at
a coarser grain — see P1.5. This item exists mainly so the next person
doesn't "fix" it.

### P1.4 — No caching anywhere in the backend

A grep for `lru_cache`, `joblib.Memory`, `st.cache_data`, `st.cache_resource`
across `src/`, `app.py` and `main.py` returns **zero hits**. Concretely:

- `_generate_dashboard` (`src/core/controller.py:941-975`) re-reads the
  dataframe at `:951` and calls `profile_dataframe` again at `:952`,
  although `load_dataset` already profiled the same data at `:632` and
  stored it at `:634`. It re-profiles because it wants the *cleaned* file
  — a legitimate reason that a keyed cache handles and an unconditional
  recompute does not.
- Every tool re-parses the CSV (P1.1).
- Every replanned step re-executes (P0.9).

**Fix, one mechanism at the right layer:** a `ResultCache` in the
controller keyed on `(tool_name, resolved params, input mtime+size)`,
plus a small `DataFrameCache` keyed on `(path, mtime, size)` behind the
single shared `_read_df` from P2.1. The orchestration-level cache subsumes
the I/O-level one for repeated steps; the I/O one still helps within a
single planning cycle where several distinct tools read the same file.

### P1.5 — RLM sub-tasks are provably independent and are run strictly serially

`decompose_and_invoke` (`src/rlm/engine.py:175-190`) loops one LLM call at
a time. The comment at `:187` explains the sequencing — "Store the result
too so later sub-tasks can reference it" — but grepping every reader of
`repl_env` shows the only consumers outside the engine are
`scripts/validate.py:409` and `tests/test_rlm_engine.py:70-71`, both
assertions. The actual prompt builder,
`controller._run_rlm_decomposition.build_prompt`
(`src/core/controller.py:1240-1247`), reads only `task.task_id`,
`task.description` and `json.dumps(task.context)`. **No sub-task has ever
read another's result.** The data dependency the serial loop protects does
not exist.

A wide dataset partitions into groups of 8 numeric columns plus one
categorical group (`src/core/controller.py:1222-1226`), so a 40-column
dataset is 6 sub-tasks — 6 sequential LLM round trips, typically 2–5 s
each, on the critical path.

**Fix:** `concurrent.futures.ThreadPoolExecutor` over `sub_tasks` with a
small bounded pool (4–6; these are I/O-bound HTTP calls, so threads are
right and the GIL is irrelevant). Preserve deterministic ordering by
writing results back into a dict keyed by `task_id` — which
`decompose_and_invoke` already returns. Expected saving: `(n_groups − 1) ×
round-trip`, commonly 10–25 s. If a future sub-task genuinely needs a
predecessor's output, the dependency becomes explicit rather than
accidental.

### P1.6 — The iteration prompt re-sends the entire accumulated result set every cycle

`get_iteration_user_prompt` (`src/core/prompt_manager.py:310-360`) calls
`memory.get_results_summary()` with the default
`max_chars_per_result=1200` (`src/core/memory.py:367`), which serialises
**every** tool result accumulated so far. By iteration 8 with 6 tools per
cycle, that is up to 48 entries — tens of thousands of tokens re-sent on
every call, growing linearly, with `MAX_ITERATIONS` defaulting to 15.

There is also no token accounting anywhere (see P3.1), so this cost is
invisible.

**Fix:** two cheap changes. (a) Pass only the *latest* iteration's results
in full and a one-line-per-tool digest for older ones — the LLM's job on
iteration N is to react to what just happened. (b) On providers that
support it, mark the static system prompt for caching (Anthropic
`cache_control`, OpenAI automatic prefix caching) — the tool-description
block from `get_system_prompt` (`src/core/prompt_manager.py:238-239`) is
byte-identical across all 15 calls and is currently re-billed each time.

### P1.7 — A new HTTP client is constructed on every LLM call

`src/core/controller.py:184` calls `OpenAI(**client_kwargs)` inside
`_call_openai_compat`, i.e. once per invocation; `_call_anthropic` does
the same at `:272`. Each construction builds a fresh `httpx` client and a
fresh connection pool, so every call pays TLS handshake and TCP setup
— typically 100–300 ms against a cloud endpoint, × (iterations +
sub-tasks + 1 ping), so commonly 2–6 s per run of pure avoidable latency.

**Fix:** build the client once in `LLMClient.__init__` (or a cached
property keyed on provider), and reuse it. The SDK clients are designed to
be long-lived and are thread-safe — which P1.5 needs anyway.

### P1.8 — The test suite takes 124 s, which is why it stops being run

199 passing tests at ~0.6 s each is dominated by real sklearn fits on
generated frames. At two minutes, the suite falls outside the
edit-run-edit loop and gets skipped locally — which is a plausible reason
the five stale failures in §0 survived.

**Fix:** mark the genuinely slow model-fitting tests
`@pytest.mark.slow`, add `-m "not slow"` to the default `addopts` in
`pyproject.toml:62`, and run the full set in CI. Shrink the synthetic
frames in the ML tests — they exist to check plumbing and output shape,
not to measure accuracy. Target: under 15 s for the default suite.

---

## P2 — Architecture and extensibility

### P2.1 — Five copies of `_read_df`, and the tool layer imports across itself to avoid a sixth

`_read_df` is defined five times with near-identical bodies
(`src/core/controller.py:41`, `src/tools/data_processing.py:26`,
`src/tools/ml_pipeline.py:31`, `src/tools/statistical_analysis.py:29`,
`src/tools/visualization.py:24`). The controller's copy silently differs —
it omits the explicit `openpyxl`/`xlrd` engines and rejects `.tsv`, which
the other four accept.

The newer tools avoided a sixth copy by importing sideways between peers:
`src/tools/time_series.py:19`, `src/tools/text_analysis.py:16`,
`src/tools/geospatial.py:18` and `src/tools/dimensionality.py:19` all
import `_read_df` from `data_processing`; `src/tools/clustering.py:25` imports it
from `ml_pipeline`; `src/tools/dimensionality.py:18` imports
`_select_cluster_features` from `clustering`. Six tools now depend on two
sibling tools' private helpers. Deleting or renaming a leading-underscore
function in `data_processing.py` breaks four unrelated tools.

**Fix:** `src/tools/io.py` (or `src/core/dataio.py`) exporting one public
`read_dataframe`, with the caching from P1.4 behind it. Every tool imports
from there; nothing imports from a peer. Add `_select_cluster_features`
to a `src/tools/features.py` for the same reason.

### P2.2 — AGENTS.md's layer rules are violated by the code they describe

AGENTS.md states: `tools/*` may depend on `base.py`, stdlib and data libs,
and must **not** import `memory`. But:

- `src/tools/base.py:20` — `from src.core.memory import MemorySystem, ToolResult`
  at module level.
- `src/tools/data_processing.py:19` — `from src.core.memory import DatasetMetadata`.
- `src/tools/base.py:23-25` — `DatasetMetadata` and `DatasetProfile` under
  `TYPE_CHECKING`, which is the honest version of the same dependency.

The imports aren't wrong — `prepare_params` genuinely needs `MemorySystem`
and `ToolResult` is genuinely the tool-layer return type. The *document*
is stale, and a stale architecture doc is worse than none: it stops being
checked.

**Fix — pick one and commit:**
(a) Extract `ToolResult`, `DatasetMetadata`, `AnalysisStep`, `DatasetProfile`
into `src/core/contracts.py` that both layers may import, leaving
`memory.py` as behaviour only. This makes the stated rule true again and
is the better end state.
(b) Amend the AGENTS.md table to permit `memory` *types* (not the
`MemorySystem` instance) in the tool layer.

Either way, add a CI check — `import-linter` with a contract file, or a
ten-line `tests/test_architecture.py` walking the AST — so the rules can't
drift again silently.

### P2.3 — Two tools write derived data outside the output root, and the guard that would prevent it is dead code

`CleanDataTool` and `DetectOutliersTool` declare no `output_subdir`
(only `clustering`, `ml_pipeline`, `report_generator` and `visualization`
do — `src/tools/base.py:92-93` is the injection point). So both fall back
to `out_dir = Path(output_dir) if output_dir else path.parent`
(`src/tools/data_processing.py:215`, `:339`) and write
`*_cleaned.csv` / `*_outliers_flagged.csv` **next to the input file** —
i.e. into `data/`, or into the user-upload directory. `.gitignore:30-31`
exists specifically to paper over this.

Meanwhile `src/core/security.py:197-212` defines `resolve_output_path`,
whose entire purpose is "refuse any escape from the output root." Grepping
every call site: `tests/test_security.py` only. The same is true of
`escape_csv_formulas` (`src/core/security.py:179-194`) — tested, never
called, so every CSV the pipeline writes is still formula-injectable when
opened in Excel.

Both derived files also flow to the LLM as `file_path` values for
downstream tools, and `output_dir` is an LLM-supplied parameter, so the
planner currently chooses where the pipeline writes.

**Fix:** give both tools `output_subdir = "data"`, and route *every*
tool's file write through `resolve_output_path(output_root, ...)`. Apply
`escape_csv_formulas` to user-facing CSV exports only — its own docstring
correctly warns not to apply it to files the pipeline reads back.

### P2.4 — Runs share one output directory, so concurrent runs corrupt each other

Every run writes to fixed paths: `output/models/random_forest.pkl`
(`src/tools/ml_pipeline.py:303`), `output/reports/dashboard.json`
(`src/core/controller.py:962`), `output/reports/report.html`
(`src/core/controller.py:993`). Two analyses in flight — two Streamlit
sessions, or a CLI run alongside the app — overwrite each other's models
and reports mid-flight, and `evaluate_model` can load a `.pkl` written by
the other run.

**Fix:** `MemorySystem` already generates `self.session_id`
(`src/core/memory.py:303`). Make the controller's `_output_dir` default to
`output/runs/{session_id}/` and symlink or copy `output/latest`. This is a
prerequisite for anything multi-user, and it gives run history for free.

### P2.5 — `AgentController.analyze()` has no non-blocking or streaming interface

`analyze()` (`src/core/controller.py:649-797`) runs the whole pipeline —
up to 15 LLM round trips and every tool execution — in one synchronous
call, driving a Rich `Progress` bar it owns (`:675-679`). The only
extension points are two fire-and-forget callbacks,
`on_step_callback` / `on_iteration_callback`
(`src/core/controller.py:507-509`).

That forces every non-CLI caller to block. `app.py:1707` calls
`agent.analyze()` inline in the Streamlit script run, so the UI freezes
for the full duration, the callbacks can only append to a list that is
rendered afterwards (`app.py:946-987`), and there is no way to cancel a
run. An HTTP API in front of this would have the same problem.

This is a backend API gap, not UI work: the controller offers no way to
observe or interrupt a run in progress.

**Fix:** add `analyze_iter()` as a generator yielding structured progress
events (`stage`, `iteration`, `tool`, `status`, `payload`), and implement
`analyze()` as `deque(self.analyze_iter(), maxlen=0)` plus a return value.
Accept an optional `cancel_token` checked at iteration and step
boundaries. Move the Rich `Progress` out of the controller and into
`main.py`, where the CLI owns its own presentation — the controller
currently imports `rich.progress` and prints emoji directly
(`src/core/controller.py:29-30`, `:675-679`), which is presentation logic in
the orchestration layer.

### P2.6 — `prepare_params` cannot recover a missing required parameter

`BaseTool.prepare_params` fills a parameter from memory context only when
the planner left it empty (`src/tools/base.py:94-98`), and
`uses_cleaned_file` redirects `file_path` only `if ... "file_path" in params`
(`src/tools/base.py:88-90`). If the LLM omits `file_path` *entirely* on a
later iteration — plausible, since the prompt tells it the controller
substitutes paths automatically (`src/core/prompt_manager.py:299`) —
the tool raises a `TypeError`, burns a retry, and the failure text the LLM
sees back is a Python signature error rather than actionable guidance.

Similarly `SelectStatisticalTestTool` maps
`{"target_column": "group_column"}` (`src/tools/statistical_analysis.py:56`)
but `feature_column` is required with no fallback
(`:193-197`), so an omitted `feature_column` is a hard error where the profiler
could nominate the highest-variance numeric column.

**Fix:** make `requires_context` fill unconditionally-required params
whether absent or empty, and have `BaseTool.run` catch `TypeError` on
signature mismatch and convert it into a `ToolExecutionError` naming the
missing parameter and its schema description — so the next planning cycle
gets a usable correction.

### P2.7 — Dashboard specs inline raw rows, so artifact size scales with chart count

`_records` inlines up to `MAX_POINTS = 1_000` rows per chart
(`src/core/dashboard.py:36`, `:92-101`), and `build_dashboard` can emit
4 histograms + 3 category charts + scatter + box + time-series + results
charts (`src/core/dashboard.py:489-511`). Each of those Vega-Lite specs
carries its own copy of the data, and `build_html_report` embeds all of
them into a single `report.html`. Existing artifacts already show the
shape of this: `output/reports/*_raw.json` are ~124 KB each, dominated by
`cluster_data`'s 1,000 `pca_points` (`src/tools/clustering.py:170-177`).

**Fix:** pre-aggregate server-side rather than shipping rows —
histograms become bin counts, box plots become five-number summaries,
the time series is already resampled (`src/core/dashboard.py:341-347`) and
should be the model for the rest. Scatter is the one chart that genuinely
needs points; cap it lower (250–400 is visually indistinguishable at
typical opacity). Expect a 5–10× reduction in `report.html` size and a
correspondingly faster first paint.

---

## P3 — Observability, cost control, reproducibility

### P3.1 — No token, cost, or latency accounting

`RLMEngine` records per-call latency and 120-character snippets
(`src/rlm/engine.py:62-69`, `:141-150`), which is useful for a trace table
and nothing else. Nowhere does the system read `response.usage` from the
provider SDK, so there is no record of prompt tokens, completion tokens,
or spend — for a loop that can make 15+ calls with a linearly growing
prompt (P1.6), that is the one number an operator most wants.

**Fix:** capture `usage` in `LLMClient._dispatch` (all three branches
expose it), accumulate it on the engine's trace entries, expose
`RLMEngine.usage_summary()`, and surface tokens + estimated cost in the
report footer and the trace table. Add an optional `max_total_tokens`
budget that ends the loop gracefully via `_deterministic_final` rather
than by exhausting iterations.

### P3.2 — `print`-based diagnostics via Rich, no structured logging

The orchestration layer writes user-facing prose with emoji directly to a
module-level `Console` (`src/core/controller.py:38` and ~40 `console.print`
calls; same pattern in `src/core/memory.py:24`, `src/rlm/engine.py:12`).
There is no `logging` usage anywhere in `src/`. Consequences: output can't
be redirected, filtered by level, or captured as JSON lines; a failed run
leaves no artifact to diagnose from; and `scripts/validate.py:52-66` has to
stub out `rich` entirely just to import the modules under test.

**Fix:** `logging.getLogger(__name__)` for diagnostics, Rich only in the
presentation layer (`main.py`, `app.py`) via `RichHandler`. Write a
`run.log` alongside each run's outputs (pairs naturally with P2.4).

### P3.3 — `arize-phoenix` is a declared dependency with zero imports

`requirements.txt:44` pins `arize-phoenix>=3.0.0` under "Observability."
Grepping `src/`, `app.py`, `main.py` and `scripts/` for `phoenix`: no
hits. It is a large dependency tree (FastAPI, SQLAlchemy, Alembic,
OpenTelemetry — all visible in `.venv/Scripts/`) that every install and
every CI run pays for and nothing uses.

**Fix:** either wire it up — it is a genuinely good fit for P3.1, since
the RLM trace is already structured for it — or remove it. Do not leave it
declared and unused.

### P3.4 — No dependency lockfile, and `pyproject.toml` declares no dependencies at all

`pyproject.toml:6-12` has no `[project.dependencies]`; `requirements.txt`
is 100% `>=` constraints with no upper bounds and no lock. CI installs
whatever PyPI serves that morning across a 3.11/3.12 matrix
(`.github/workflows/ci.yml:29-31`).

This matters more here than in most projects because behaviour is
version-sensitive in ways the codebase already knows about: comments at
`src/tools/ml_pipeline.py:89` and `src/tools/statistical_analysis.py:114`
both work around pandas 3's `str` dtype, `n_init="auto"`
(`src/tools/ml_pipeline.py:407`) is sklearn-version-dependent, and
`src/tools/ml_pipeline.py:436-442` documents a real sklearn truthiness
trap. A silent minor-version bump can change a reported metric with no
test failure.

**Fix:** move the runtime list into `[project.dependencies]` with sensible
upper bounds, generate `requirements.lock` via `uv pip compile` or
`pip-tools`, install from the lock in CI, and keep the loose file for
development. Add a scheduled job that re-resolves and runs the suite, so
upstream drift surfaces as a PR rather than as a wrong number.

---

## P4 — Test coverage

### P4.1 — Four tools ship with no tests at all

`tests/` has no `test_time_series.py`, `test_text_analysis.py`,
`test_geospatial.py` or `test_dimensionality.py`, though all four tools
are registered and reachable by the planner
(`src/core/controller.py:413-416`). AGENTS.md's tool contract requires
"at least one unit test in `tests/`" for every tool.

These four are the *most* likely to need tests: each auto-detects its own
input columns from heuristics
(`src/tools/time_series.py:35-55`, `src/tools/text_analysis.py:37-58`,
`src/tools/geospatial.py:31-54`) and each is gated by an `applies_to`
score that decides whether the planner ever sees it. A silent regression
in a detector makes the tool invisible rather than broken — the failure
mode no one notices.

**Fix:** one test per tool covering (a) the happy path on a small
synthetic frame, (b) `applies_to` returning 0.0 on unsuitable data and
1.0 on suitable, (c) the auto-detection path with the column omitted.

### P4.2 — No coverage measurement

Nothing in `pyproject.toml` or `ci.yml` measures coverage, so the gap in
P4.1 is invisible to CI and the next one will be too.

**Fix:** `pytest-cov` with `--cov=src --cov-report=term-missing`, and a
`--cov-fail-under` floor set just below today's actual number so it
ratchets up rather than blocking immediately.

### P4.3 — No test asserts the anti-leakage or anti-overfitting invariants

`tests/test_ml_enhancements.py` covers the tuning and imbalance features,
but nothing asserts the properties the system's credibility rests on: that
CV never sees test rows, that a time-series dataset gets a chronological
split, that `_pick_best` prefers a lower-CV model when the higher one is
badly overfit, that reported metrics appear verbatim in tool outputs.

**Fix:** as each P0 item lands, add the invariant test with it. These are
cheap property-style tests (construct a frame with a known leak signal,
assert the metric does *not* detect it) and they are what keeps P0 fixed.

---

## Suggested order of work

Sequenced so each step is independently shippable and earlier steps make
later ones easier.

| Step | Items | Why here | Rough size |
| :--- | :--- | :--- | :--- |
| **1** | §0 stale tests | Can't verify anything else until the suite is green | 20 min |
| **2** | P0.2 + P1.2(1) | Same one-line change fixes the leakage *and* removes 11% of training time | 1 h |
| **3** | P0.8, P2.3, P1.7 | Small, isolated, high value-per-line | 2 h |
| **4** | P0.9 + P1.4 | One cache mechanism; also closes the model-overwrite race | 4 h |
| **5** | P0.3 + P0.4 | The flagship correctness fix; wiring already exists in memory context | 1 day |
| **6** | P0.1 + P0.5 + P0.6 | The `Pipeline` refactor — one change, three findings | 2 days |
| **7** | P0.7 | Verbatim-metric validator; makes the core promise measurable | 1 day |
| **8** | P1.5, P1.6, P2.5 | Latency and API shape; P1.7 from step 3 is a prerequisite for P1.5 | 2 days |
| **9** | P2.1, P2.2, P2.4 | Structural cleanup, best done once the above have settled | 2 days |
| **10** | P3.x, P4.x | Observability, lockfile, coverage — the ratchet that keeps it all fixed | 2 days |

If only one thing gets done: **step 5** (P0.3). It is the largest gap
between what the system already knows about the data and what it does with
it, and it is the difference between a plausible number and a correct one.

If only one *hour* is available: **step 2**. A one-line change that is
simultaneously a correctness fix and a speed win is rare enough to take
immediately.

---

## Appendix A — Measurement method

All timings from this machine (Windows 11, Python 3.13 in `.venv`), warm
OS file cache, single run each — treat them as order-of-magnitude, not
benchmarks.

- **Gates:** `ruff check .`, `mypy src/`, `pytest tests/ -q`,
  `python scripts/validate.py`, each timed end to end.
- **Pipeline table (P1):** synthetic frame, 50,000 rows × 16 columns
  (12 `np.random.normal` numerics, 2 categoricals at cardinality 5 and 20,
  1 hourly datetime, 1 binary target derived from `num_0` plus noise),
  seed 0, written to CSV and driven through the real tool classes via
  `BaseTool.run()` in pipeline order.
- **Per-model table (P1.2/P1.3):** first 20,000 rows of the cleaned frame
  — deliberately at the `do_tune` threshold
  (`src/tools/ml_pipeline.py:227`) — through `_prepare_features` /
  `_encode_target`, then each of `fit`, `cross_val_score` at `n_jobs=1`
  and `n_jobs=-1`, and `TrainModelTool._tune`, timed separately with the
  same `StratifiedKFold(5, shuffle=True, random_state=42)` and
  `f1_weighted` scorer the tool uses.
- **Call-site claims** ("never called", "no readers") are from
  `grep -rn` across `src/`, `app.py`, `main.py`, `scripts/` and `tests/`,
  excluding `.venv/`, `.py/` and `__pycache__/`.

Benchmark scripts were written to the session scratchpad and are not part
of the repository; the parameters above are sufficient to reproduce them.

---

*Filename note: written to `IMPROVEMENTS.md` rather than a new
`Improvement.md`. Git has that path staged as deleted, so restoring it
records this as a modification with the previous audit's history intact,
rather than a delete-plus-add of a near-identical name.*
