# Backend Audit — Improvements

Audit of the orchestration layer (`controller.py`, `rlm/engine.py`, `memory.py`,
`profiler.py`, `prompt_manager.py`, `security.py`) and execution layer
(`tools/*.py`, `dashboard.py`), plus a scoped diff-review of `app.py`.
Findings are ranked by impact on the system's core promise: correct,
autonomous, interpretable analysis.

## High — correctness bugs

### 1. Three disagreeing implementations of task-type inference

- `src/tools/data_processing.py:89-99` (`IngestDatasetTool.execute`):
  ```python
  if "int" in dtype or "bool" in dtype or n_unique <= 20:
      task_type = "classification"
  else:
      task_type = "regression"
  ```
  For an **object/string** target with >20 unique values (e.g. a `city` or
  free-text column mistakenly picked as target), this falls to `else` →
  **`regression`**.
- `src/core/memory.py:136-157` (`DatasetMetadata.infer_task_type()`) handles
  the same decision correctly (string dtype → `classification` by default).
- `src/tools/ml_pipeline.py:184-188` (`TrainModelTool`'s own `"auto"` branch)
  also gets it right.

**Why it matters:** `src/core/controller.py:520-521` (`load_dataset`) only
calls the *correct* `infer_task_type()` when auto-detection ran. If a
target is supplied explicitly (`--target`, `TARGET_COLUMN_HINT`, or a user
typing a column at the interactive prompt), the *wrong* ingest-tool value
survives untouched and is threaded straight into `train_model`'s
`task_type` parameter. Since `task_type != "auto"` there, `TrainModelTool`
skips its own (correct) re-detection and tries to fit regression models on
raw strings — every model's `.fit()` throws, and the tool surfaces a
generic "No models could be trained" instead of running classification.

**Fix:** delete the duplicate logic in `IngestDatasetTool` and delegate to
`DatasetMetadata.infer_task_type()` everywhere (single source of truth).

### 2. `.iloc` indexed with label values, not positions

`src/tools/data_processing.py:310-311` (zscore) and `:319-320`
(isolation_forest):
```python
mask.iloc[clean.index[mask_idx]] = True
```
`clean.index[mask_idx]` yields index **labels**; `.iloc[...]` expects
**positions**. It only works today because `_read_df` always produces a
fresh `RangeIndex` (label == position by coincidence). It's a latent
correctness bug — any future change that reads a DataFrame with a
non-default index would silently flag the wrong rows as outliers with no
error.

**Fix:** use `.loc[...]` instead.

## Medium — logic gaps that degrade interpretation quality

### 3. `get_results_summary()` truncates tool JSON mid-structure

`src/core/memory.py:319-337` serializes each tool's full output dict and
slices to `max_chars_per_result=350`. `train_model`'s output alone
(multiple models × `{train_metrics, test_metrics, cv_mean, cv_std, gap,
best_params, ...}`) routinely exceeds that, so the summary the LLM sees is
often a syntactically broken JSON fragment — potentially cut off before
`best_model` or a model's actual metric values.

Given the system prompt's hard rule "cite ONLY metric values that appear
verbatim in the results," feeding it truncated JSON undermines the one
thing this agent is supposed to get right: correct citation of real
numbers.

**Fix:** raise the cap, or truncate per-field (e.g. keep `models_trained`
summarized to just `cv_mean`/`gap` per model) instead of a blind character
slice.

### 4. `correlation_analysis` drops target correlations for multi-class targets

`src/tools/data_processing.py:413-444` (`CorrelationAnalysisTool.execute`)
only computes `target_corrs` when the target is numeric, or (via the
`elif`) when it's non-numeric **and exactly 2 classes**. A 3+ class
categorical target (a genuinely common classification case) gets an empty
`target_correlations` with no warning — the LLM has no signal that
feature-target relationships were never computed for that run.

**Fix:** extend the point-biserial-style encoding path to multi-class
targets (e.g. one-vs-rest correlation per class, or ANOVA F-value per
feature) instead of silently returning nothing.

### 5. `_pick_best` selects purely by `cv_mean`, blind to overfitting

`src/tools/ml_pipeline.py:516-519`:
```python
def _pick_best(self, results: dict[str, Any]) -> str:
    if not results:
        return "none"
    return max(results, key=lambda m: float(results[m].get("cv_mean", -float("inf"))))
```
No penalty for `train_test_gap`. A model that scores highest on CV but is
flagged in `overfit_warnings` two lines earlier can still be crowned "best"
and propagated as `best_model_path`/`best_model_name` throughout the rest
of the pipeline (evaluate_model, feature_importance chart, final report).

**Fix:** a composite score, e.g.
`cv_mean - λ·max(0, gap - OVERFIT_THRESHOLD)`, to keep model selection
consistent with the system's own anti-overfitting stance.

### 6. `evaluate_model` doesn't enforce the `test_size` used at training time

`src/tools/ml_pipeline.py:564-595` (`EvaluateModelTool.execute`) recreates
the train/test split with a `test_size` parameter that defaults to `0.2`
but isn't automatically synced with whatever `train_model` actually used
(the docstring says "must match" but nothing enforces it). If a plan step
trains with a non-default `test_size` and evaluates with the default, the
"held-out" data silently overlaps the training data, invalidating the
overfitting diagnostic.

**Fix:** `controller._execute_steps` already auto-injects
`best_model_path`/`best_model_name` into memory context
(`src/core/controller.py:1074-1081`) — give `test_size` the same
treatment.

### 7. `select_statistical_test` crashes on tiny groups instead of degrading gracefully

`src/tools/statistical_analysis.py:105-109` calls `scipy.stats.shapiro(g)`
for every group with no size guard; Shapiro requires n≥3 and raises
otherwise. Any `group_column` that produces a group smaller than 3 rows
(easy with a fine-grained categorical) kills the whole test instead of
treating undersized groups as non-normal and falling back to
Mann-Whitney/Kruskal.

**Fix:** guard group size before calling `shapiro`, fall back to the
non-parametric branch when any group is too small.

## Low — fragility / cleanup

### 8. String-sniffing file-path substitution

`src/core/controller.py:999-1015` (`_execute_steps`) decides whether to
override a plan step's `file_path` by checking placeholder strings and
extension suffixes rather than a structured signal. It works, but it's the
kind of heuristic that breaks quietly when the LLM's phrasing drifts (e.g.
different casing, or a path with an unexpected suffix).

**Suggested improvement:** replace with an explicit parameter/flag the
planner sets (e.g. `use_cleaned: true`) rather than inferring intent from
string shape.

### 9. Dead condition in statistical test tool

`src/tools/statistical_analysis.py:82`:
```python
if df_clean[group_column].dtype in (float,) or str(df_clean[group_column].dtype).startswith("float"):
```
`dtype in (float,)` never evaluates `True` (pandas dtype objects don't
equal the builtin `float`); only the adjacent `str(...).startswith("float")`
clause does any work.

**Fix:** drop the dead first clause.

### 10. Issues already flagged in the uncommitted `app.py` diff

From a scoped diff-review pass:

- A second, disagreeing train-test-gap threshold check (`>=0.10` vs the
  tool's own `>0.10`) that can make the KPI gauge and the comparison table
  disagree on the same model.
- Self-XSS via the unescaped custom-model-name text input rendered with
  `unsafe_allow_html=True`.
- The 3D pipeline rig remounting on every Streamlit rerun, not just after
  an analysis run.

## Suggested priority order

1. **#1** (task-type divergence) — can outright break training on a
   plausible real dataset.
2. **#2** (`.iloc`/`.loc`) — cheap, removes a landmine.
3. **#3** (truncation) and **#5** (best-model selection) — both directly
   touch the "autonomous interpretation" value proposition.
4. **#4, #6, #7** — real gaps but narrower blast radius.
5. **#8, #9, #10** — cleanup, do opportunistically.
