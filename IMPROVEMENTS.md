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

## Round 2 status — landed this pass

Items below are fixed in the working tree, each with an invariant test
added alongside it (`tests/test_controller.py`, `tests/test_ml_enhancements.py`,
`tests/test_pipeline_3d.py`). `ruff check .`, `mypy src/`, and `pytest tests/`
are green (217 passed).

| # | Item | What changed |
| :-- | :--- | :--- |
| §0 | Stale tests | `_parse_steps` calls bound to a controller instance; palette assertion re-pointed at the current Ledger tokens (`ink=#3a2b1e`, `pen=#a34f20`, `risk=#a33526`, plus `accent`/`grid`). |
| P0.2 | CV leakage | `cross_val_score` now fits `X_train`/`y_train`, never the full `X`/`y`. |
| P1.2(1) | Redundant CV after tuning | `_tune` returns `cv_mean`/`cv_std` from `search.best_score_`/`cv_results_`; the separate `cross_val_score` call is skipped when tuning ran. |
| P0.8 | `max_iterations` env override | Explicit constructor argument now wins over `MAX_ITERATIONS`, mirroring `enable_rlm`. |
| P2.3 | `clean_data`/`detect_outliers` write next to input | Both now declare `output_subdir = "data"`, routing writes under the run's output root by default. |
| P1.7 | New HTTP client per LLM call | `LLMClient` builds the SDK client once (lazily) and reuses it. |
| P0.9 / P1.4 | Re-executing already-succeeded steps | `AgentController._step_cache` keyed on `(tool_name, resolved params, input file mtime+size)` — an identical re-planned step is served from cache instead of re-run (closes the `train_model` overwrite race). |
| P0.3 | Time-series/panel structure never reached the splitter | `TrainModelTool.prepare_params` reads `is_time_series`/`panel_group_cols` from `data_profile` and sets `split_strategy`; `execute` branches (via shared `_resolve_split_strategy`/`_split_train_test` helpers) to a chronological split + `TimeSeriesSplit`, or `GroupShuffleSplit`/`GroupKFold` on the entity column, which is also dropped from the feature matrix (it's a split key, not a feature). Reported in output as `split_strategy`/`time_column`/`group_column`, and **persisted to memory context so `EvaluateModelTool` recreates the identical partition** — the two tools now share one split implementation instead of `evaluate_model` silently reverting to a random split on the exact datasets this fix targets. |
| P0.4 | Datetime columns dropped before modelling | `_prepare_features` now expands them into `year/month/day/dayofweek/hour/is_weekend/days_since_min` instead of dropping (ordered *after* P0.3's chronological sort, so the raw column still orders the split first). |
| P0.7 | "Cite only verbatim metrics" unenforced | `AgentController._flag_unverified_claims` runs between the reasoning loop and Stage 7: any numeric literal in `insights`/`recommendations`/`key_metrics` not traceable to an actual tool result is annotated `[unverified: ...]` in place and logged to `memory.set_context("unverified_claims", ...)`. Verified the deterministic fallback path can't trip its own validator (it only ever echoes real tool numbers) with an explicit test. |

**Caught in self-review before landing (worth recording — both were introduced
by the P0.2/P1.2 and P0.3 fixes above, not pre-existing):**
- `_tune`'s `cv_std` initially computed `std(mean_test_score)` — the spread
  *between candidate configurations* — instead of `std_test_score[best_index_]`,
  the fold-to-fold variability of the actually-selected model. Fixed to read
  the latter, which is what the untuned path's `cross_val_score(...).std()`
  has always meant.
- `EvaluateModelTool` still called a bare `train_test_split` after P0.3 landed
  in `TrainModelTool` — meaning evaluate's "held-out" rows on a time-series or
  panel dataset would include rows the model *had* trained on, inflating
  exactly the metric P0.3 was fixing. Extracted `_resolve_split_strategy`/
  `_split_train_test` as shared helpers and wired `EvaluateModelTool` to use
  the same persisted `split_strategy` train_model resolved.

**Not done this pass — deliberately deferred, not silently dropped:**

- **P0.1 / P0.5 / P0.6** (the `Pipeline`/`ColumnTransformer` refactor —
  imputation/encoding/log-transform fit on train-only, self-contained
  saved models, one-hot for linear models). This is the single largest
  remaining correctness gap and the right next target, but it's a
  multi-file refactor touching `CleanDataTool`, `_prepare_features`,
  every model's save/load path, and `EvaluateModelTool`'s re-derivation
  logic — too large to land safely in the same pass as everything above.
  **Landed in Round 4, below** (attempted and reverted once mid-Round-2
  session before that; see Round 4 for what actually shipped).
- **P1.5** (parallelise RLM sub-tasks) — `src/rlm/engine.py` is
  AGENTS.md "Ask First" territory.
- **P2.1, P2.2, P2.4, P2.5, P2.6, P2.7** — structural cleanup, best done
  once the above settle.
- **P3.x, P4.1, P4.2** — observability, lockfile, coverage measurement,
  tests for the four untested tools.
- **P2.3's second half** — `resolve_output_path` is still not called from
  any tool's write path; only the default-location half of the finding
  (`output_subdir`) is fixed. An LLM-supplied `output_dir` can still point
  outside the output root.

---

## Round 3 — data-flow/tool-value audit, landed items

Separate exercise from the correctness audit above: traced what each of
the 14 registered tools produces against what actually reaches a user
(Streamlit tabs in `app.py`, the Vega dashboard in `dashboard.py`, the
HTML/Markdown reports). Two real findings, both fixed and covered by tests
(`tests/test_dashboard.py::test_time_series_chart_uses_tool_columns_and_findings`):

1. **`generate_visualizations`'s PNGs reached nobody.** Grepped `.png` /
   `chart_path` / `image_path` across `app.py`, `html_report.py`,
   `report_generator.py` — zero consumers; the interactive Vega dashboard
   already covers the same ground better. Yet the deterministic no-target
   fallback plan still called it every run
   (`controller.py`, `_build_fallback_plan`). **Fixed:** removed that step;
   the tool stays registered for the LLM planner to call deliberately.
2. **Five tools' findings had no narrated UI surface.** `cluster_data`,
   `time_series_analysis`, `text_analysis`, `geospatial_analysis`,
   `dimensionality_analysis` results were reachable only via the raw
   "Full Technical Log" JSON dump — no dedicated tab section, and only
   `cluster_data` got a dashboard chart. **Fixed:**
   - `app.py`: `_render_other_findings()` — a generic "Other Analyses"
     card in the Full Details tab covering all five, so a future tool
     without a bespoke renderer is never JSON-only again.
   - `dashboard.py`: the time-series chart now reads
     `time_series_analysis`'s actual `date_column`/`value_column` (instead
     of independently re-guessing) and its description states the tool's
     trend/stationarity/seasonality findings, not just the axis labels.

Deferred from the same audit (lower value/effort, natural follow-ons once
the above shapes existed) — not implemented, ranked below the two items
above: a dedicated geospatial chart, a PCA scree-plot chart for
`dimensionality_analysis`, promoting these findings into the Summary tab
when they're a dataset's dominant story, and pointing
`report_generator.py`'s Markdown log at the same richer formatting instead
of a one-line-per-tool log.

---

## Round 4 — Pipeline/ColumnTransformer refactor (P0.1, P0.5, P0.6), landed

Closes the single largest deferred item from Round 2. `_prepare_features`
now does only structural feature engineering (datetime expansion, ID-column
dropping) — no statistic is fit there any more. Skew-decision and
categorical encoding moved into a `Pipeline([("prep", ColumnTransformer),
("model", estimator)])` built by `_build_preprocessor` and fit exclusively
on `X_train`:

- `_SkewLog1pTransformer` (`src/tools/ml_pipeline.py`) decides, at `fit`,
  which numeric columns are skewed enough for log1p — from whichever frame
  it's fit on, so wiring it into the Pipeline is what makes the decision
  train-fold-only (closes P0.1). Clips at transform so a test-fold negative
  in a column the training fold saw as non-negative doesn't produce NaN.
- `_build_preprocessor(X, encoding)` — `OneHotEncoder(handle_unknown=
  "ignore")` for `LINEAR_MODELS = {logistic_regression, linear_regression,
  ridge}`, `OrdinalEncoder(handle_unknown="use_encoded_value",
  unknown_value=-1)` for tree/ensemble and clustering models (closes P0.6).
  `SimpleImputer` (median for numeric, most-frequent for categorical) runs
  per branch — this is *new* behavior, not a port: `_prepare_features`
  never imputed (`CleanDataTool` did that, upstream, on its own separate
  full-dataset pass); a model trained on an uncleaned file now silently
  imputes inside the pipeline instead of failing or carrying NaNs through.
  Named per the P0.1/P0.5 fix IMPROVEMENTS.md already specified.
- The whole `Pipeline` is pickled (closes P0.5 — a saved model is now
  self-contained; scoring new data no longer depends on `EvaluateModelTool`
  coincidentally re-deriving identical `LabelEncoder` state from the same
  file).
- `TrainModelTool._tune` tunes the whole `Pipeline` — grid keys are
  prefixed `model__` for `RandomizedSearchCV` and the prefix is stripped
  from the returned `best_params` before it reaches the report.
- `TrainModelTool`'s clustering branch (`kmeans`/`dbscan`) gets its own
  `_build_preprocessor(X_train, "ordinal")` wrap — it stopped getting free
  imputation/encoding from `_prepare_features` once that moved out.
- `visualization.py::_feature_importance` pulls the estimator via
  `pipeline.named_steps["model"]` and post-encoding names via
  `pipeline.named_steps["prep"].get_feature_names_out()` — `Pipeline`
  doesn't delegate `feature_importances_`/`coef_`, and one-hot expands
  column count past raw `X.columns` length.
- `EvaluateModelTool._explain_drivers`'s direction-from-correlation
  calculation now guards on `is_numeric_dtype` before calling `.corr()` —
  `X_test` carries raw string categoricals post-refactor, and `.corr()` on
  those raised inside the method's blanket `except Exception`, silently
  dropping every driver rather than just the categorical one. Categoricals
  now report importance without a direction. `permutation_importance` and
  `model.predict()`/`.predict_proba()` needed no changes — they work on a
  `Pipeline` via duck typing, confirmed for `_roc_curve`/`_confusion_matrix`
  too (no automated test exercised those two chart types' success path;
  verified manually against a trained Pipeline before landing this).

Tests: `_SkewLog1pTransformer` covered in isolation
(`TestSkewLog1pTransformer`); an invariant test builds a frame where a
categorical level exists only in the test-split rows and asserts `predict`
succeeds (`TestUnseenCategoryHandling`) — proves `handle_unknown="ignore"`/
`use_encoded_value` is doing its job, the thing the old whole-dataset
`LabelEncoder` was hiding; and — the actual regression guard, not just
mechanism — `test_log1p_decision_uses_training_fold_only_not_full_frame`
builds a column whose training-fold skew is below threshold but whose
full-frame skew is above threshold (outliers concentrated in the test
tail) and asserts no log1p line appears for it, which fails against the
pre-refactor whole-frame decision. `ruff check .`, `mypy src/`,
`pytest tests/` (232 passed), and `python scripts/validate.py` (68/68) all
green.

---

# Round 5 — Universal Data Handling Audit

**Question asked:** can this backend ingest *any* dataset — any format, schema,
size, or quality level — and produce the best analysis, dashboard and report
available for it, without hardcoded assumptions?

**Method.** Read every module in the data path (ingestion → profiling →
cleaning → analysis → dashboard → report), then probed the running system with
adversarial inputs rather than reasoning from the source alone. Every finding
below cites either a `file:line` or a probe result. Probes are reproducible
from the descriptions given; they were throwaway scripts, not committed.

**Scope note.** Rounds 2–4 audited *correctness of the ML path*. This round
audits *generality of the data path*. Overlap is deliberate only where a
Round 2–4 fix created the surface being judged here.

## Phase 1 — Findings

### What already works (do not rewrite these)

Worth stating plainly, because the temptation with a brief like "handle any
data" is to rebuild what is already sound:

- **`DatasetProfile` is a real semantic profiler**, not a `df.describe()`
  wrapper (`src/core/profiler.py`). It classifies each column as numeric /
  categorical / datetime / boolean / identifier / constant / text, and derives
  dataset-*nature* facts — `is_time_series`, `text_cols`, `geo_lat_col` /
  `geo_lon_col`, `is_high_dimensional`, `panel_group_cols`. Ordering of the
  checks is already thought through (datetime before identifier so a daily
  index isn't an ID; free-text before identifier so 100%-unique prose isn't
  an ID).
- **Analysis selection is already data-driven, not fixed.**
  `BaseTool.applies_to(profile, metadata) -> float` scores each tool against
  the profile and `ToolRegistry.candidate_tools` drops anything scoring 0.0
  (`controller.py:495-513`). The planner is only *shown* tools that fit the
  data's nature. This is the right architecture for the brief — it needs
  extending, not replacing.
- **Dashboard chart choice is already data-driven** (`dashboard.py:607-652`):
  charts are selected from profile column-kinds plus which tools actually
  produced output, not from a fixed list.
- **The pluggable-tool requirement is already met.** `ToolRegistry.register()`
  is generic; adding a tool needs no registry edit.

The gap is therefore **not** "the system is hardcoded to one shape of data."
It is that the *front door* (reading bytes into a DataFrame) is far narrower
and more fragile than everything behind it, and that the *rigor and honesty*
of what comes out the back door lags what the profiler already knows.

### U0.1 — Delimiter is assumed to be a comma; non-comma files are silently corrupted

`_read_df` dispatches on file extension and calls bare `pd.read_csv(path)`
for both `.csv` **and `.tsv`** — with no `sep` argument
(`data_processing.py:26-37`, and four identical copies, see U1.1).

A tab-separated file is therefore parsed with a comma delimiter. Probed with a
3-row × 3-column TSV:

| Input | Parsed as | `IngestDatasetTool.run` | Profiler verdict |
| :--- | :--- | :--- | :--- |
| 3×3 `.tsv` | **3 rows × 1 column** | `success` | quality **90/100** |
| 3×3 `;`-delimited `.csv` (EU Excel default) | **3 rows × 1 column** | `success` | quality **72/100** |

This is the worst failure shape in the system: not a crash, but a confident
success. Every column collapses into one, the profiler classifies that single
mangled column as an identifier, and the run proceeds to produce an analysis
and a report about a dataset that does not exist. Semicolon-delimited CSV is
the *default export format of Excel in most of Europe*, so this is not an
exotic input.

No delimiter sniffing exists anywhere in the codebase.

### U0.2 — Encoding is assumed UTF-8; any other encoding is a hard failure

No `encoding=` argument is passed at any read site. A cp1252/latin-1 CSV —
again, a routine Excel export — fails with `UnicodeDecodeError` at
`tools/_read_df`, at `controller._read_dataframe`, and at the Streamlit
preview path. `IngestDatasetTool` converts it to `error: Failed to read file`.
No encoding detection or fallback chain is attempted.

### U0.3 — The ID-guard rejects any *sorted* continuous feature, at any sample size

`statistical_analysis.py:81-91` rejects a feature column when
`uniqueness > 0.95 and is_monotonic`. Continuous measurements are naturally
~100% unique, so the guard reduces to "is it sorted?".

Probed with a legitimate continuous measurement, sorted (the natural layout of
data exported grouped by key):

| n | sorted | result |
| :--- | :--- | :--- |
| 6 | yes | **error** — "appears to be a row ID or index" |
| 500 | yes | **error** — same |
| 5000 | yes | **error** — same |
| 5000 | no (identical data, shuffled) | **success** |

Row order alone decides whether the system's *only* hypothesis-testing tool
will run. This is not a small-sample artifact — it fires at every size tested.

### U0.4 — Statistical significance is reported with no effect size, interval, or power

`select_statistical_test` returns `statistic`, `p_value`, `significant`, and a
prose `interpretation`. It returns no effect size, no confidence interval, and
no sample-size or power caveat.

Probed with two groups of 50,000 differing by 0.6 on SD 15 — **Cohen's
d = 0.038**, a negligible difference no one should act on:

> `Independent T-Test: stat=-5.9670, p=0.0000. Statistically significant
> difference detected (p=0.0000 < α=0.05).`

The report will tell a user there is a meaningful difference between two
practically identical groups. Given this repo's own stated standard —
"*a wrong number delivered confidently is worse than a slow one*" — this is a
P0-class honesty defect, not a nice-to-have.

Related rigor gaps in the same tool:
- **Chi-square validity unchecked** — no expected-cell-frequency test; the
  statistic is invalid when expected counts fall below ~5, and nothing says so.
- **No effect size for any branch** — no Cohen's d, no Cramér's V, no η².
- **Normality subsample is order-dependent** — `g[:5000]` takes the *first*
  5000 values, not a random sample (`statistical_analysis.py:123`), so on
  sorted data the Shapiro test sees a truncated tail and mis-answers.
- **No multiple-comparison correction**, though the tool is designed to be
  called repeatedly across feature/group pairs in one run.

### U0.5 — Profiling failure is swallowed, and silently narrows the toolset

`controller.py:694-705` wraps profiling in `try/except` and continues with
`last_profile = None` on any failure — correct as a resilience choice, but the
consequence is invisible. `controller._read_dataframe` (`controller.py:42-49`)
recognises only `.csv/.xlsx/.xls` — **it does not know `.tsv`**, though the
tools' `_read_df` does. A `.tsv` upload therefore ingests (badly, per U0.1)
and then profiles not at all.

With `profile=None`, `applies_to` falls back to per-tool defaults that differ
by tool. Measured on a time-series fixture, `candidate_tools(profile, meta)`
vs `candidate_tools(None, meta)`:

- **Lost:** `time_series_analysis`
- **Gained:** *(none)*

So a profiling failure does not corrupt the plan — it *quietly removes exactly
the dataset-nature tools that make the analysis fit the data*, leaving generic
EDA, and says nothing to the user. The degradation is real but narrower than
"the gating breaks"; the defect is that it is silent, not that it is chaotic.

### U0.6 — The Markdown report has no data overview, methodology, or limitations

`GenerateReportTool.execute` accepts `dataset_name`, `tool_results_json`,
`llm_insights`, `output_dir` (`report_generator.py:130-137`). **The
`DatasetProfile` is never passed to it.** Quality score, per-column warnings,
missingness, class imbalance and every caveat the profiler computed reach the
Streamlit UI and a single badge in the HTML report (`html_report.py:154`) —
but never the Markdown report a user actually keeps.

Against the seven-section report structure this brief asks for:

| Required section | Present? |
| :--- | :--- |
| Executive summary | ✅ (LLM `reasoning`) |
| Data overview / profile | ❌ **absent** |
| Methodology — which analyses ran and *why* | ❌ absent — see below |
| Key findings | ✅ |
| Visualizations | ⚠️ dashboard only; PNGs reach no report (Round 3, item 1) |
| Limitations / caveats | ❌ **absent** |
| Recommendations | ✅ |

The system computes the honesty material and then discards it at the last step.

**The methodology gap is self-inflicted and cheap to close.** The planning
prompt *demands* a rationale for every step — "Provide a `rationale` for EVERY
step — this is a research-grade system" (`prompt_manager.py:85`), "rationale
tied to the profile evidence" (`:131`) — and the LLM supplies one. It is
parsed (`controller.py:889`), stored on `AnalysisStep.rationale`
(`memory.py:273`), rendered **truncated to 60 characters in a terminal
panel** (`controller.py:1210`), and then dropped. The exact "why this analysis
for this dataset" narrative the brief asks for is already being generated and
thrown away.

### U0.7 — Numerics trapped in strings are never recovered

No type coercion or repair pass exists. The profiler classifies whatever dtype
pandas inferred. Probed on routine real-world columns:

| Column | Values | Classified as | Consequence |
| :--- | :--- | :--- | :--- |
| `price_with_symbol` | `$123.45` | **identifier** | dropped from all numeric analysis |
| `pct_as_string` | `45.3%` | **categorical**, `high_cardinality` | one-hot candidate; flagged as a data-quality problem |
| `zipcode` | `04521` | identifier | reasonable, but never offered as a geo/categorical key |
| `bool_yn` | `Y`/`N` | categorical | never becomes boolean |
| `date_iso`, `date_us` | ISO and `MM/DD/YYYY` | datetime ✅ | (correct — date handling is already good) |

Currency, percentages and thousands separators are among the most common
real-world CSV contents. Today every such column is excluded from correlation,
modelling, and outlier detection — silently, and while being counted against
the dataset's quality score.

### U1.1 — `_read_df` is duplicated five times

Identical (or near-identical) reader implementations at
`data_processing.py:26`, `ml_pipeline.py:32`, `statistical_analysis.py:29`,
`visualization.py:24`, plus a *divergent* fifth at `controller.py:42` that
supports fewer formats. Any ingestion fix — U0.1, U0.2, format coverage — must
currently be made in five places and has already drifted in one. This is the
single biggest structural obstacle to everything else in this round.

### U1.2 — Format coverage is narrow, and inconsistent about its own limits

Supported: `.csv`, `.tsv` (broken, U0.1), `.xlsx`, `.xls`. Unsupported: JSON,
JSONL, Parquet, SQL, compressed CSV, nested/semi-structured data of any kind.

The supported set is also *declared inconsistently* across three places:

| Site | Accepts |
| :--- | :--- |
| `security.ALLOWED_EXTENSIONS:25` | `.csv`, `.xlsx`, `.xls` |
| Streamlit uploader (`app.py:1413`) | `csv`, `xlsx`, `xls` |
| tools' `_read_df` | `.csv`, **`.tsv`**, `.xlsx`, `.xls` |
| `controller._read_dataframe:42` | `.csv`, `.xlsx`, `.xls` |

`.tsv` is a phantom format: reachable by direct tool/CLI invocation, rejected
by upload validation, unprofileable, and corrupt when it does load.

### U1.3 — The quality score has no row-count floor

`profile_dataframe` penalises `row_count < 100` by a flat 10 points. Probed:

- **0 rows** (headers only) → quality **81/100**, `success`
- **1 row** → quality **81/100**, `success`

A dataset that cannot support any inference at all is scored "good". Nothing
downstream gates on insufficient data; tools fail individually and
idiosyncratically further down instead (U0.3's guard, Shapiro's n≥3, etc.).

### U1.4 — Structural corruption is neither detected nor reported

- **Mixed-type columns:** `1, 2, NOT_A_NUMBER, 4` → whole column becomes
  object → classified `identifier` → excluded from analysis. The single
  contaminating cell is never surfaced.
- **Duplicate headers:** `a,a,b` → pandas silently mangles to `a`, `a.1`, `b`.
  No warning.

### U1.5 — Scale: adequate where measured, but unbounded and re-read per tool

Measured on this machine (read + profile):

| Rows | File | `pd.read_csv` | `profile_dataframe` | In-memory |
| ---: | ---: | ---: | ---: | ---: |
| 10,000 | 0.6 MB | 0.03 s | 0.01 s | 0.5 MB |
| 200,000 | 11.6 MB | 0.15 s | 0.09 s | 9.9 MB |
| 1,000,000 | 58.0 MB | 0.64 s | 0.57 s | 49.6 MB |

Profiling is **not** a bottleneck at these sizes and needs no optimisation.
Two architectural risks remain, both unmeasured beyond 1M rows and stated here
as risks rather than defects:

1. **No row cap, no chunking, no sampling policy.** Every read is a full load.
   Memory is the binding constraint and nothing degrades gracefully when it
   binds.
2. **Every tool re-reads the file from disk independently.** A 9-tool run on a
   1M-row file pays the ~0.64 s read nine times and holds nine transient
   copies. `dashboard.py` has a `MAX_POINTS` cap for rendering, but nothing
   equivalent governs analysis input.

### U1.6 — No edge-case test corpus

232 tests pass, but the suite is organised by *module*, not by *data shape*.
There is no fixture for: empty file, headers-only, single row, single column,
all-categorical, all-text, wrong delimiter, non-UTF-8 encoding, mixed-type
column, duplicate headers, or high-missingness data. Every finding in this
round was found by probing, because nothing in CI probes.

---

## Phase 2 — Improvement Roadmap

Ordered so each item is independently shippable and earlier items make later
ones cheaper. Effort estimates are rough and assume the existing quality gates
(`ruff`, `mypy src/`, `pytest tests/`, `scripts/validate.py`) must stay green.

**Deviation from `superpowers:writing-plans`, stated deliberately:** that skill
produces bite-sized TDD task bodies with literal code for each step. Writing
that for all ten items would be thousands of lines of speculative code for
items you may reorder or reject. This is written at roadmap altitude —
numbered, ranked, with dependencies and breaking-change flags. I'll expand the
**one item you approve first** into a full `writing-plans` task breakdown at
that point.

### Prioritised items

| # | Item | Impact | Effort | Depends on | Flags |
| :-- | :--- | :--- | :--- | :--- | :--- |
| **1** | Edge-case dataset corpus + fixtures | High | 0.5 d | — | |
| **2** | Unified reader: one `read_any()` behind all five call sites | **Highest** | 1 d | 1 | **Ask First** |
| **3** | Type coercion / repair pass | High | 1 d | 2 | |
| **4** | Statistical rigor: effect sizes, CIs, power, validity checks | **Highest** | 1.5 d | 1 | |
| **5** | ID-guard fix + data-sufficiency gate | High | 0.5 d | 1 | |
| **6** | Report restructure: overview, methodology, limitations | High | 1 d | 3 | |
| **7** | Profiling failure becomes loud + degraded-mode flag | Medium | 0.5 d | 2 | |
| **8** | Format expansion: JSON / JSONL / Parquet / compressed | Medium | 1 d | 2 | **Ask First** |
| **9** | Scale policy: row cap, sampling, read-once cache | Medium | 1.5 d | 2 | **Ask First** |
| **10** | Observability: structured degradation log | Low-Med | 0.5 d | 7 | |

---

**1. Edge-case dataset corpus** *(do this first — it is the regression harness
every later item is validated against)*

`tests/fixtures/` + a `conftest.py` factory producing: empty, headers-only,
single-row, single-column, all-categorical, all-text, wrong-delimiter (TSV and
`;`), cp1252-encoded, mixed-type column, duplicate headers, high-missingness,
1M-row synthetic, time-series, panel, geo, and high-cardinality frames. Plus a
`test_data_shapes.py` that asserts, for each, that the pipeline either
succeeds *or* fails with a clear actionable error — never succeeds on corrupt
input. Several will fail immediately: that is the point, and they become the
acceptance criteria for items 2–5.

**2. Unified reader** — *closes U0.1, U0.2, U1.1, U1.2*

New `src/core/io.py` exposing `read_any(path) -> tuple[pd.DataFrame, ReadReport]`:
extension dispatch → delimiter sniffing (`csv.Sniffer` on a byte sample, with
explicit `sep` for `.tsv`) → encoding detection (UTF-8, then BOM check, then
`charset_normalizer`, then cp1252 fallback) → header validation (duplicate
names, unnamed columns). `ReadReport` carries what was detected and what was
assumed, so the report can say so (feeds item 6). Replace all five call sites.
Reconcile the three disagreeing extension allowlists into one constant.

*Breaking-change risk:* every tool's read path changes at once. Mitigated by
item 1 landing first. *Ask First:* AGENTS.md gates new shared modules in
`src/core/` and this sits on the boundary between the execution and core
layers — I'd confirm placement (`src/core/io.py` vs `src/tools/_io.py`) with
you before writing it, since the layer rules forbid `tools/*` importing from
`core/` except `memory`.

**3. Type coercion / repair pass** — *closes U0.7, part of U1.4*

A `coerce_types(df) -> tuple[pd.DataFrame, list[Coercion]]` step run once at
ingestion, before profiling: strip currency symbols/thousands separators →
numeric; `45.3%` → 0.453 numeric; `Y/N`, `yes/no`, `true/false` → boolean;
detect mixed-type columns and report the contaminating values rather than
letting the column degrade to object. Every coercion is *recorded and
reported*, never silent — same discipline as `treatments_applied`. Profiler
then classifies the repaired frame.

**4. Statistical rigor** — *closes U0.4*

For every branch of `select_statistical_test`: add effect size (Cohen's d /
Cramér's V / η² / rank-biserial as the test dictates), bootstrap or analytic
confidence interval on the effect, a sample-size/power note, and an explicit
`practical_significance` verdict distinct from `significant` so the p<0.05 /
d=0.04 case reports honestly. Add chi-square expected-frequency validity check.
Make the Shapiro subsample random (seeded), not positional. Add
Benjamini-Hochberg correction across tests within one run. Surface all of it
in the report's limitations section (item 6).

**5. ID-guard fix + data-sufficiency gate** — *closes U0.3, U1.3*

Replace `uniqueness > 0.95 and is_monotonic` with a check that cannot be
tripped by sort order: require integer dtype **and** near-perfect uniqueness
**and** (name hint **or** consecutive-integer spacing) — reuse
`profiler._is_identifier_like` rather than maintaining a second, worse copy.
Separately, add a dataset-sufficiency gate in the profiler: below a row-count
floor, cap the quality score and emit a blocking warning that the report must
carry, so 0-row and 1-row datasets stop scoring 81/100.

**6. Report restructure** — *closes U0.6*

Pass the `DatasetProfile` and the `ReadReport` into `GenerateReportTool`
(`prepare_params` already pulls from memory — `data_profile` is in context, so
this is an injection change, not a signature fight). Add three sections: **Data
Overview** (shape, column kinds, quality score, missingness, what was
detected/assumed at read time, what was coerced), **Methodology** (each tool
that ran, with the planner's own `rationale` for it — currently computed and
discarded), **Limitations & Caveats** (profile warnings, sufficiency flags,
effect-size caveats from item 4, anything the unverified-claims validator from
P0.7 flagged). This is what makes the report explain *why* these analyses, for
*this* dataset.

**7. Loud profiling failure** — *closes U0.5*

Keep profiling non-fatal, but record `profile_status` in memory context and
surface "running in degraded mode — dataset-nature tools unavailable, because
X" in the UI, the report's limitations section, and the log. Add `.tsv` (and
whatever item 2 supports) to the controller's reader so the most common cause
disappears.

**8. Format expansion** — JSON/JSONL (with `json_normalize` flattening for
nested records, depth-capped), Parquet, `.gz`/`.zip` CSV. Needs a decision
from you on nested data: flatten, or reject with a clear message? **Ask First**
— this expands the product's supported-input promise, which is a product
decision, not a code one.

**9. Scale policy** — a configurable row cap with *reported* reservoir
sampling above it, and a read-once cache keyed on path+mtime so a 9-tool run
reads once rather than nine times. **Ask First** — sampling changes results,
so whether that is acceptable (and the default threshold) is your call.
Note the measurements in U1.5: this is about robustness beyond 1M rows and
wasted I/O, not a current performance problem.

**10. Observability** — a structured `degradations` list in memory that every
silent fallback appends to (encoding guessed, delimiter sniffed, profiling
skipped, sampling applied, tool gated out), rendered in the report and the UI.
Turns every "silent" in this document into "visible".

### Sequencing

```
1 (corpus) ──> 2 (reader) ──> 3 (coercion) ──> 6 (report)
          └──> 4 (rigor) ────────────────────> 6
          └──> 5 (guards) ───────────────────> 6
                2 ──> 7 (loud failure) ──> 10 (observability)
                2 ──> 8 (formats)
                2 ──> 9 (scale)
```

Items 1–6 are the core of the brief and are worth doing in order. 7–10 are
independent follow-ons.

### If only one thing gets done

**Item 2 (unified reader)**, with item 1 as its test harness. U0.1 is the only
finding in this round where the system produces a confident, complete,
plausible-looking analysis *of data that does not exist* — and it triggers on
a file format Excel produces by default across most of Europe.

---

---

# Round 6 — domain layer + capability toggles, residual backlog

Round 6 was not a planned audit round. It is the residue of the session that
added the semantic domain layer (`src/core/domains.py` + three domain tools) and
made LLM and ML independently switchable. That session ran the full pipeline
end-to-end on a retail export and found **4 bugs and 2 flags**.

**The four bugs are fixed and in `master`** (`1bcb739`):

| Bug observed | Fix, and where it lives |
| :--- | :--- |
| dd/mm/yyyy dates silently destroyed — 63% of rows to `NaT`, day/month swapped on the survivors | `_detect_date_convention` in `src/core/coercion.py`, returning `date_iso` / `date_dayfirst` / `date_monthfirst` / `date_ambiguous`; the datetime branch runs *before* the numeric rules |
| A tautological model reported as the best result (CV 0.9969, accuracy 1.0000) | `_detect_target_leakage` in `src/tools/ml_pipeline.py` — per-feature purity (`_LEAKAGE_PURITY = 0.99`) plus a near-perfect-score heuristic |
| "Trend is increasing" asserted on pure noise (R² = 0.0008) | `_TREND_MIN_R_SQUARED = 0.05` in `src/tools/time_series.py` |
| AOV reported as 563.20 against a true 76.03 — the order-id role matched `order_date`, so revenue aggregated per *day* | sequential role claiming with an `exclude` set and most-specific-first candidates, consolidated into `domains.resolve_column` |

**The two flags were deliberate non-fixes.** `cross_val_score` keeps `n_jobs=1`
— measured **4× faster** than `n_jobs=-1` on 16 cores (0.52 s vs 2.06 s), see
P1.3, which reached the same conclusion independently. The second flag is IQR
over-flagging skewed data, carried below as item 6.4.

**Naming warning:** the session that produced these called them **P1–P4**, and
the user refers to them that way. They are renumbered 6.1–6.4 here because
`P1`–`P4` in this document are priority *tiers* (P1 = Speed, P4 = Test
coverage). The session labels are noted on each item so both handles resolve.

### Prioritised items

| # | Item | Impact | Effort | Depends on | Flags |
| :-- | :--- | :--- | :--- | :--- | :--- |
| **6.1** | Chart panels for `cohort_analysis` / `financial_analysis` / `workforce_analysis` | **Highest** | 0.5 d | — | user-requested |
| **6.2** | Tests for the domain layer, date coercion, leakage detector, toggles, read cache | High | 1 d | — | |
| **6.3** | Surface `date_ambiguous` as an explicit warning | Medium | 1 h | — | cheapest |
| **6.4** | Distribution-aware outlier detection | Medium | 0.5 d | — | **Ask First** |

### 6.1 — Three domain tools compute results that are never charted  *(session "P1")*

`build_dashboard` resolves exactly six tool outputs
(`src/core/dashboard.py:631-636`): `train_model`, `correlation_analysis`,
`cluster_data`, `time_series_analysis`, `geospatial_analysis`,
`dimensionality_analysis`. `cohort_analysis`, `financial_analysis` and
`workforce_analysis` are absent, so RFM segments and drawdown curves reach the
reports as prose and tables and never become a chart.

Ranked top because it is the gap the user named directly ("real charts with real
value"), and because the analysis behind the charts already exists and is
verified — this is presentation wiring, not new computation.

One `_*_chart` builder per panel, appended to `candidates`: drawdown area +
cumulative-return line (financial), RFM segment bar + revenue-by-month line
(cohort), tenure histogram + headcount-by-department bar (workforce). Cap rows
at `MAX_POINTS` and mind P2.7 — dashboard specs inline raw rows, so each new
panel adds to artifact size.

### 6.2 — The new code has no tests at all  *(session "P2")*

The suite is green (**326 passed, 1 deselected, 1 warning**, 69 s) and covers
none of the last two sessions' work. Evidence that does not rot with the count:
grepping `tests/` for
`domains|infer_domains|_detect_target_leakage|date_dayfirst|read_cache|use_ml|use_llm`
matches **zero files**. No `tests/test_domains.py` exists;
`tests/test_coercion.py` predates the date work and never mentions a convention.

Risk order: date-convention detection (silently rewrites data), the leakage
detector (needs a true positive *and* a true negative), domain inference
(structural discriminators + the `resolve_column` exclusion order that fixed
AOV), the capability toggles (`use_ml=False` must exclude every `requires_ml`
tool), and `invalidate_read_cache` firing on rewrite — the Windows
mtime-granularity trap has no test holding it shut.

**Do not merge this with P4.1–P4.2.** Those cover the older untested tools
(`time_series`, `text_analysis`, `geospatial`, `dimensionality`). 6.2 is
new-code coverage; the two are separate debts with separate scopes.

### 6.3 — `date_ambiguous` is computed, then rendered as if it were a success  *(session "P3")*

`_detect_date_convention` returns `"date_ambiguous"` (`src/core/coercion.py:172`)
when no day in the column exceeds 12 — dd/mm and mm/dd are indistinguishable
from the data, and the parser picks one silently. That verdict reaches the user
only through the generic coercion line in `src/core/degradations.py:46-50`:
`Column 'x' repaired from string to datetime (date_ambiguous rule): N converted, 0 left unparsed`
— which reads as a clean repair. Nothing warns that the dates may be wrong.

Fix: a dedicated branch in the degradation log for datetime coercions carrying
the ambiguous rule, naming the column and stating the convention could not be
determined. Worst failure mode in this round (a confidently wrong date axis on
every chart) against the smallest fix.

### 6.4 — Outlier detection ignores the skew flag the profiler already sets  *(session "P4")*

`detect_outliers` (`src/tools/data_processing.py:286`) applies IQR, z-score or
isolation-forest with no reference to the column's distribution. On the retail
fixture it flagged **~22% of revenue rows** — revenue is right-skewed by nature,
so the tail is the business, not an anomaly.

The profiler already computes what is needed: skewness per column and a
`severe_skew` flag (`src/core/profiler.py:317-318`, `SEVERE_SKEW_THRESHOLD`),
surfaced in `to_prompt_string` (`profiler.py:182-184`). `detect_outliers` never
reads it.

Fix: for a `severe_skew` column, apply IQR to log-transformed values or switch
to a robust alternative (MAD-based, or asymmetric fences), and report which rule
was used per column. **Ask First** — the default changes reported outlier counts
on existing datasets.

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
