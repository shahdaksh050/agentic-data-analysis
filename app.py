"""
Streamlit UI — Agentic Data Analysis System.

Run:
    streamlit run app.py

Key fixes vs previous version
------------------------------
* NO background thread + st.rerun() loop.  The pipeline runs synchronously
  inside st.status() so Streamlit renders live progress without fighting its
  own execution model.
* Dataset preview is saved to session_state on file upload and rendered from
  there — no dependency on sidebar scope surviving a rerun.
* OpenRouter support added (any model string, OpenAI-compatible endpoint).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
import types
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# ── Project root on sys.path ─────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Agentic Data Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Rich stub ─────────────────────────────────────────────────────────────────
def _stub_rich() -> None:
    """Silence rich so src/ imports work without the package installed."""
    for mod_name in [
        "rich", "rich.console", "rich.panel",
        "rich.table", "rich.tree", "rich.progress",
    ]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)

    class _C:
        def print(self, *a: Any, **k: Any) -> None: pass
    class _P:
        def __init__(self, *a: Any, **k: Any): pass
    class _T:
        def __init__(self, *a: Any, **k: Any): pass
        def add_column(self, *a: Any, **k: Any) -> None: pass
        def add_row(self, *a: Any, **k: Any) -> None: pass
    class _Tr:
        def __init__(self, *a: Any, **k: Any): pass
        def add(self, *a: Any, **k: Any) -> "_Tr": return self
    class _Pr:
        def __init__(self, *a: Any, **k: Any): pass
        def __enter__(self) -> "_Pr": return self
        def __exit__(self, *a: Any) -> None: pass
        def add_task(self, *a: Any, **k: Any) -> int: return 0
        def update(self, *a: Any, **k: Any) -> None: pass
    class _Sp:
        def __init__(self, *a: Any, **k: Any): pass
    class _Tx:
        def __init__(self, *a: Any, **k: Any): pass

    sys.modules["rich"].Console = _C            # type: ignore[attr-defined]
    sys.modules["rich.console"].Console = _C    # type: ignore[attr-defined]
    sys.modules["rich.panel"].Panel = _P        # type: ignore[attr-defined]
    sys.modules["rich.table"].Table = _T        # type: ignore[attr-defined]
    sys.modules["rich.tree"].Tree = _Tr         # type: ignore[attr-defined]
    sys.modules["rich.progress"].Progress = _Pr         # type: ignore[attr-defined]
    sys.modules["rich.progress"].SpinnerColumn = _Sp    # type: ignore[attr-defined]
    sys.modules["rich.progress"].TextColumn = _Tx       # type: ignore[attr-defined]


_stub_rich()


# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
footer { visibility: hidden; }
.main  { background: #0f1117; }
section[data-testid="stSidebar"] { background: #0d0f18; }

/* Stage progress cards */
.sc {
    display: flex; align-items: center; gap: 10px;
    padding: .65rem 1rem; margin-bottom: .4rem;
    border-radius: 8px; border: 1px solid #2a2d3e;
    background: #1a1d27; font-size: .88rem; color: #ccc;
}
.sc.done   { border-color: #1a7a4a; background: #0c1c14; color: #7ed9a5; }
.sc.active { border-color: #3b5bdb; background: #0c1228; color: #93b4f7; }
.sc.skip   { border-color: #444;    background: #181a22; color: #555; }
.sc.err    { border-color: #a33;    background: #1e0c0c; color: #f08080; }
.sc .detail { margin-left: auto; font-size: .78rem; opacity: .7; }

/* Metric tiles */
.mt { background:#1a1d27; border:1px solid #2a2d3e; border-radius:8px;
      padding:.9rem 1rem; text-align:center; }
.mt .lbl { font-size:.75rem; color:#888; margin-bottom:3px; }
.mt .val { font-size:1.5rem; font-weight:700; color:#e0e0e0; }
.mt .sub { font-size:.72rem; color:#666; margin-top:2px; }

/* Card variants */
.ic { background:#141820; border-left:3px solid #3b5bdb; border-radius:0 6px 6px 0;
      padding:.55rem .9rem; margin-bottom:.35rem; font-size:.88rem; color:#c0cce0; }
.rc { background:#141820; border-left:3px solid #1a7a4a; border-radius:0 6px 6px 0;
      padding:.55rem .9rem; margin-bottom:.35rem; font-size:.88rem; color:#a8ddb8; }
.wc { background:#1f1a0d; border-left:3px solid #e67e22; border-radius:0 6px 6px 0;
      padding:.55rem .9rem; margin-bottom:.35rem; font-size:.85rem; color:#f0c080; }
</style>
""", unsafe_allow_html=True)


# ── Session-state initialisation ──────────────────────────────────────────────
_DEFAULTS: dict[str, Any] = {
    "preview_df":     None,   # pd.DataFrame
    "preview_name":   "",
    "preview_bytes":  None,   # raw bytes
    "stage_log":      [],
    "analysis_done":  False,
    "analysis_error": None,
    "final_report":   None,
    "tool_results":   [],
    "metadata":       None,
    "tmp_dir":        None,
    "progress_lines": [],
    "llm_warning":    None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Constants ─────────────────────────────────────────────────────────────────
STAGE_DEFS = [
    ("1", "Dataset Ingestion",     "📥"),
    ("2", "Initial Reasoning",     "🧠"),
    ("3", "Tool Execution",        "⚙️"),
    ("4", "Result Interpretation", "🔍"),
    ("5", "Iterative Refinement",  "🔄"),
    ("6", "RLM Decomposition",     "🔀"),
    ("7", "Report Generation",     "📄"),
]

OR_MODELS = [
    "openai/gpt-4o",
    "openai/gpt-4-turbo",
    "anthropic/claude-sonnet-4-6",
    "anthropic/claude-opus-4-6",
    "meta-llama/llama-3.3-70b-instruct",
    "google/gemini-2.0-flash-001",
    "mistralai/mistral-large",
    "deepseek/deepseek-chat",
    "cohere/command-r-plus",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _reset_pipeline() -> None:
    for k in ("stage_log", "analysis_done", "analysis_error",
              "final_report", "tool_results", "metadata", "tmp_dir",
              "progress_lines", "llm_warning"):
        st.session_state[k] = _DEFAULTS[k]  # type: ignore[assignment]


def _set_stage(num: str, status: str, detail: str = "") -> None:
    log: list[tuple[str, str, str]] = [
        e for e in st.session_state["stage_log"] if e[0] != num
    ]
    log.append((num, status, detail))
    st.session_state["stage_log"] = log


def _stage_card(num: str, name: str, icon: str,
                status: str, detail: str = "") -> str:
    cls = {"done": "done", "active": "active",
           "skipped": "skip", "error": "err"}.get(status, "")
    ico = {"done": "✅", "active": "⏳", "error": "❌",
           "skipped": "⏭️", "pending": "⬜"}.get(status, "⬜")
    det = f'<span class="detail">{detail}</span>' if detail else ""
    return (f'<div class="sc {cls}">'
            f'<span>{ico}</span><span>{icon}</span>'
            f'<b>Stage {num}</b>&nbsp;—&nbsp;{name}{det}</div>')


def _mt(label: str, value: str, sub: str = "",
        color: str = "#e0e0e0") -> str:
    return (f'<div class="mt"><div class="lbl">{label}</div>'
            f'<div class="val" style="color:{color}">{value}</div>'
            f'<div class="sub">{sub}</div></div>')


def _gap_color(g: float) -> str:
    return "#2ecc71" if g < 0.05 else "#e67e22" if g < 0.10 else "#e74c3c"


def _safe_df(df: pd.DataFrame) -> pd.DataFrame:
    """Convert any datetime/Timestamp columns to strings so PyArrow can serialise them."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].astype(str)
        elif out[col].dtype == object:
            # Mixed types that may include Timestamps
            try:
                if out[col].dropna().apply(lambda x: hasattr(x, "strftime")).any():
                    out[col] = out[col].astype(str)
            except Exception:
                out[col] = out[col].astype(str)
    return out


def _find_tool(tool_results: list[dict[str, Any]],
               name: str) -> dict[str, Any] | None:
    for r in tool_results:
        if r.get("tool_name") == name and r.get("status") == "success":
            return r.get("output", {})
    return None


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 Agentic Data Analysis")
    st.caption("RLM-Powered Autonomous Pipeline")
    st.divider()

    # ── Upload ────────────────────────────────────────────────────────────────
    st.markdown("### 📂 Dataset")
    uploaded = st.file_uploader(
        "CSV or Excel",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed",
    )

    # Persist to session_state immediately on upload / clear on removal
    if uploaded is not None:
        if uploaded.name != st.session_state.get("preview_name", ""):
            _reset_pipeline()
            raw_bytes = uploaded.read()
            st.session_state["preview_bytes"] = raw_bytes
            st.session_state["preview_name"]  = uploaded.name
            _fname = uploaded.name.lower()
            try:
                buf = pd.io.common.BytesIO(raw_bytes)
                if _fname.endswith(".csv"):
                    st.session_state["preview_df"] = pd.read_csv(buf)
                elif _fname.endswith(".xlsx"):
                    st.session_state["preview_df"] = pd.read_excel(
                        pd.io.common.BytesIO(raw_bytes), engine="openpyxl")
                elif _fname.endswith(".xls"):
                    st.session_state["preview_df"] = pd.read_excel(
                        pd.io.common.BytesIO(raw_bytes), engine="xlrd")
                else:
                    st.error(f"Unsupported file type: {uploaded.name}")
                    st.session_state["preview_df"] = None
            except Exception as _e:
                st.session_state["preview_df"] = None
                st.error(f"Could not read file: {_e}")
    else:
        if st.session_state.get("preview_name"):
            for _k2, _v2 in _DEFAULTS.items():
                st.session_state[_k2] = _v2

    target_col = st.text_input(
        "Target column",
        placeholder="e.g. churn, price, label  (blank = clustering)",
    )

    # ── LLM Provider ──────────────────────────────────────────────────────────
    st.markdown("### 🤖 LLM Provider")
    provider = st.selectbox("Provider", ["openai", "anthropic", "openrouter"])

    if provider == "openai":
        model_list = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
        key_ph     = "sk-..."
    elif provider == "anthropic":
        model_list = ["claude-sonnet-4-6", "claude-opus-4-6",
                      "claude-haiku-4-5-20251001"]
        key_ph     = "sk-ant-..."
    else:
        model_list = OR_MODELS
        key_ph     = "sk-or-..."

    model_sel = st.selectbox("Model", model_list)
    if provider == "openrouter":
        custom_m = st.text_input(
            "Custom model string (overrides above)",
            placeholder="e.g. cohere/command-r-plus",
        )
        final_model = custom_m.strip() if custom_m.strip() else model_sel
    else:
        final_model = model_sel

    api_key = st.text_input(
        ("OpenAI" if provider == "openai"
         else "Anthropic" if provider == "anthropic"
         else "OpenRouter") + " API Key",
        type="password",
        placeholder=key_ph,
    )

    # ── Analysis Settings ─────────────────────────────────────────────────────
    st.markdown("### ⚙️ Analysis Settings")
    max_iter   = st.slider("Max iterations", 3, 25, 10)
    enable_rlm = st.toggle("Enable RLM decomposition (Stage 6)", value=True)

    st.markdown("### 🛡️ Anti-Overfitting")
    max_depth = st.slider("Max tree depth", 2, 15, 6,
                          help="Lower = less overfitting for tree-based models")
    test_pct  = st.slider("Test split %", 10, 40, 20, step=5)
    n_cv      = st.slider("CV folds (k)", 3, 10, 5)

    st.divider()

    # ── Buttons ───────────────────────────────────────────────────────────────
    has_file = st.session_state["preview_df"] is not None
    has_key  = bool(api_key.strip())
    can_run  = has_file and has_key and not st.session_state["analysis_done"]

    run_clicked = st.button(
        "▶ Run Analysis",
        disabled=not can_run,
        width='stretch',
        type="primary",
    )
    if st.session_state["analysis_done"] or st.session_state["analysis_error"]:
        if st.button("🔄 New Analysis", width='stretch'):
            _reset_pipeline()
            st.rerun()

    if not has_file:
        st.caption("⬆ Upload a CSV or Excel file first.")
    elif not has_key:
        st.caption("⬆ Enter your API key to continue.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA — header
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div style="display:flex;align-items:center;gap:12px;'
    'padding-bottom:1rem;border-bottom:1px solid #2a2d3e;margin-bottom:1.4rem">'
    '<span style="font-size:2rem">🧠</span>'
    '<div><h2 style="margin:0;color:#e0e0e0">Agentic Data Analysis System</h2>'
    '<p style="margin:0;font-size:.82rem;color:#555">'
    'Autonomous · RLM-Powered · Anti-Overfitting Built-in</p></div></div>',
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# DATASET PREVIEW — always visible once a file is loaded
# ══════════════════════════════════════════════════════════════════════════════
preview_df: pd.DataFrame | None = st.session_state["preview_df"]

if preview_df is not None and not st.session_state["analysis_done"]:
    st.markdown(
        f"### 📊 Dataset Preview — `{st.session_state['preview_name']}`"
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows",          f"{len(preview_df):,}")
    c2.metric("Columns",       len(preview_df.columns))
    c3.metric("Missing cells", int(preview_df.isnull().sum().sum()))
    c4.metric("Numeric cols",
              len(preview_df.select_dtypes(include="number").columns))

    with st.expander("First 10 rows", expanded=True):
        st.dataframe(_safe_df(preview_df.head(10)), width='stretch')

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Column types & missing**")
        dtype_df = pd.DataFrame(
            [(c, str(t), int(preview_df[c].isnull().sum()))
             for c, t in preview_df.dtypes.items()],
            columns=["Column", "Type", "Missing"],
        )
        st.dataframe(_safe_df(dtype_df), width='stretch', height=200)
    with col_r:
        st.markdown("**Descriptive statistics**")
        st.dataframe(_safe_df(preview_df.describe()), width='stretch', height=200)
    st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE — runs synchronously inside st.status() on button click
# ══════════════════════════════════════════════════════════════════════════════
if run_clicked:
    _reset_pipeline()
    for num, _, _ in STAGE_DEFS:
        _set_stage(num, "pending")

    # Save dataset to a temp file
    tmp = tempfile.mkdtemp()
    st.session_state["tmp_dir"] = tmp
    dpath  = str(Path(tmp) / st.session_state["preview_name"])
    outdir = str(Path(tmp) / "output")
    with open(dpath, "wb") as _f:
        _f.write(st.session_state["preview_bytes"])

    # Set env vars before importing src
    os.environ["LLM_PROVIDER"]          = provider
    os.environ["LLM_MODEL"]             = final_model
    os.environ["MAX_ITERATIONS"]        = str(max_iter)
    os.environ["ENABLE_RLM_INFERENCE"]  = "true" if enable_rlm else "false"
    os.environ["OUTPUT_DIR"]            = outdir
    {
        "openai":     lambda: os.environ.__setitem__("OPENAI_API_KEY",    api_key.strip()),
        "anthropic":  lambda: os.environ.__setitem__("ANTHROPIC_API_KEY", api_key.strip()),
        "openrouter": lambda: os.environ.__setitem__("OPENROUTER_API_KEY", api_key.strip()),
    }[provider]()

    # ── Spinner placeholder — replaced after run completes ───────────────
    _spinner_ph = st.empty()
    _spinner_ph.info("🚀 Running analysis… this may take 1–3 minutes depending on dataset size and model.")

    # ── Collect progress lines into session state (no st.write during run) ─
    _progress_lines: list[str] = []

    def _upd(num: str, s: str, detail: str = "") -> None:
        _set_stage(num, s, detail)
        _ico = {"done": "✅", "active": "⏳", "error": "❌", "skipped": "⏭️"}.get(s, "⬜")
        _nm  = next(n for no, n, _ in STAGE_DEFS if no == num)
        _progress_lines.append(f"{_ico} Stage {num}: {_nm}" + (f" — {detail}" if detail else ""))

    try:
        from src.core.controller import AgentController  # noqa: PLC0415

        _upd("1", "active", "ingesting…")
        agent = AgentController(
            max_iterations=max_iter,
            enable_rlm=enable_rlm,
        )
        meta = agent.load_dataset(
            dpath,
            target_hint=target_col.strip() or None,
            interactive=False,
        )
        st.session_state["metadata"] = meta
        _upd("1", "done",
             f"{meta.row_count:,} rows × {meta.column_count} cols · task={meta.task_type} · target={meta.target_column}")

        _upd("2", "active", "calling LLM for analysis plan…")
        _upd("3", "pending")
        _upd("4", "pending")
        _upd("5", "pending")
        _upd("6", "pending" if enable_rlm else "skipped",
             "" if enable_rlm else "disabled")
        _upd("7", "pending")

        # ── Lightweight callbacks — only update stage_log, no st.write ────
        def _on_step(tool_name: str, status: str, detail: str) -> None:
            _set_stage("3", "active" if status in ("running", "success") else "active", detail)
            _progress_lines.append(f"  {'✓' if status=='success' else '→'} {detail}")

        def _on_iter(iteration: int, stage: str) -> None:
            if "stage2" in stage:
                _set_stage("2", "active", f"iter {iteration} — LLM reasoning…")
                _progress_lines.append(f"⏳ Iteration {iteration}: LLM reasoning…")
            elif "stage4" in stage or "stage5" in stage:
                _set_stage("4", "active", f"iter {iteration} — interpreting results…")
                _set_stage("5", "active", f"iter {iteration} — refining plan…")
                _progress_lines.append(f"⏳ Iteration {iteration}: interpreting & refining…")

        agent.on_step_callback      = _on_step  # type: ignore[attr-defined]
        agent.on_iteration_callback = _on_iter  # type: ignore[attr-defined]

        final = agent.analyze()

        # ── Mark all stages done ──────────────────────────────────────────
        _upd("2", "done", "plan generated & executed")
        tool_names_run = list({r.get("tool_name","") for r in [t.to_dict() for t in agent.memory.tool_results]})
        _upd("3", "done", f"{len(agent.memory.tool_results)} tools executed: {', '.join(tool_names_run[:5])}")
        _upd("4", "done", "results interpreted")
        _upd("5", "done", f"{agent.memory.iteration_count} iteration(s)")
        if enable_rlm:
            sub = agent.memory.get_context("rlm_sub_results")
            _upd("6", "done",
                 f"{len(sub)} sub-tasks" if sub else "no decomposition needed")
        _upd("7", "done", "report saved")

        st.session_state["tool_results"]  = [r.to_dict() for r in agent.memory.tool_results]
        st.session_state["final_report"]  = final
        st.session_state["analysis_done"] = True
        st.session_state["progress_lines"] = _progress_lines
        llm_err = agent.memory.get_context("llm_error")
        if llm_err:
            st.session_state["llm_warning"] = f"Fallback plan was used (LLM issue): {llm_err[:300]}"
        _spinner_ph.empty()

    except Exception:
        err = traceback.format_exc()
        st.session_state["analysis_error"] = err
        st.session_state["progress_lines"] = _progress_lines
        for _n, _s, _d in reversed(st.session_state["stage_log"]):
            if _s == "active":
                _set_stage(_n, "error", "failed")
                break
        _spinner_ph.empty()
        st.error("❌ Pipeline error — see traceback below.")
        st.code(err, language="python")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE PROGRESS CARDS — shown once pipeline has started or finished
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state["stage_log"]:
    st.markdown("### 🔄 Pipeline Stages")
    log_map = {n: (s, d) for n, s, d in st.session_state["stage_log"]}
    cols = st.columns(2)
    for i, (num, name, icon) in enumerate(STAGE_DEFS):
        s, d = log_map.get(num, ("pending", ""))
        with cols[i % 2]:
            st.markdown(
                _stage_card(num, name, icon, s, d),
                unsafe_allow_html=True,
            )


# ── Progress log ─────────────────────────────────────────────────────────────
if st.session_state.get("progress_lines"):
    with st.expander("📋 Execution log", expanded=False):
        st.code("\n".join(st.session_state["progress_lines"]), language=None)


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS — shown after a successful run
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state["analysis_done"] and st.session_state["final_report"]:
    report: dict[str, Any]             = st.session_state["final_report"]
    tool_results: list[dict[str, Any]] = st.session_state["tool_results"]
    meta                               = st.session_state["metadata"]
    tmp_dir: str                       = st.session_state.get("tmp_dir", "")

    st.divider()
    st.markdown("## ✅ Analysis Complete")

    (tab_ov, tab_ds, tab_ml, tab_ins,
     tab_log, tab_rep, tab_dl) = st.tabs([
        "📈 Overview", "🗂 Dataset", "🤖 Models",
        "💡 Insights", "🔧 Tool Log", "📄 Report", "⬇ Downloads",
    ])

    train_out   = _find_tool(tool_results, "train_model")
    eval_out    = _find_tool(tool_results, "evaluate_model")
    corr_out    = _find_tool(tool_results, "correlation_analysis")
    outlier_out = _find_tool(tool_results, "detect_outliers")
    stat_out    = _find_tool(tool_results, "select_statistical_test")
    clean_out   = _find_tool(tool_results, "clean_data")

    # ── Overview ─────────────────────────────────────────────────────────────
    with tab_ov:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        best_model   = report.get("best_model") or "N/A"
        best_cv      = "—"
        best_gap_str = "—"
        gap_val: float | None = None

        if train_out:
            _mt_map = train_out.get("models_trained", {})
            _best   = train_out.get("best_model", "")
            if _best and _best in _mt_map:
                _bm      = _mt_map[_best]
                best_cv  = f"{_bm.get('cv_mean', 0)*100:.1f}%"
                gap_val  = _bm.get("train_test_gap")
                best_gap_str = f"{gap_val*100:.1f}%" if gap_val is not None else "—"

        task_type = (
            (train_out.get("task_type") if train_out else None)
            or (meta.task_type if meta else "—") or "—"
        )
        outlier_pct = (
            f"{outlier_out.get('outlier_percentage','—')}%"
            if outlier_out else "—"
        )

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.markdown(_mt("Best model", best_model),
                    unsafe_allow_html=True)
        m2.markdown(_mt("CV score", best_cv, "generalisation"),
                    unsafe_allow_html=True)
        m3.markdown(
            _mt("Train–test gap", best_gap_str, "overfit signal",
                _gap_color(gap_val) if gap_val is not None else "#e0e0e0"),
            unsafe_allow_html=True,
        )
        m4.markdown(_mt("Outliers", outlier_pct, "of dataset"),
                    unsafe_allow_html=True)
        m5.markdown(_mt("Task", task_type), unsafe_allow_html=True)

        for _w in (train_out.get("overfit_warnings", []) if train_out else []):
            st.markdown(f'<div class="wc">⚠ {_w}</div>',
                        unsafe_allow_html=True)

        # Model comparison bar chart
        if train_out:
            _mt_map2 = train_out.get("models_trained", {})
            if _mt_map2:
                st.markdown("#### Model Comparison")
                _names = list(_mt_map2.keys())
                _task  = train_out.get("task_type", "classification")
                _pk    = "accuracy" if _task == "classification" else "r2"
                _tr    = [_mt_map2[m].get("train_metrics", {}).get(_pk, 0)*100 for m in _names]
                _te    = [_mt_map2[m].get("test_metrics",  {}).get(_pk, 0)*100 for m in _names]
                _cv    = [_mt_map2[m].get("cv_mean", 0)*100 for m in _names]
                _x     = np.arange(len(_names))
                _w2    = 0.25
                _fig, _ax = plt.subplots(figsize=(max(6, len(_names)*2.5), 4))
                _fig.patch.set_facecolor("#0f1117")
                _ax.set_facecolor("#1a1d27")
                _ax.bar(_x-_w2, _tr, _w2, label="Train",   color="#5b8dee", alpha=.85)
                _ax.bar(_x,     _te, _w2, label="Test",    color="#2ecc71", alpha=.85)
                _ax.bar(_x+_w2, _cv, _w2, label="CV mean", color="#e67e22", alpha=.85)
                _ax.set_xticks(_x)
                _ax.set_xticklabels(_names, color="#bbb", fontsize=9)
                _ax.set_ylabel(f"{_pk} %", color="#bbb", fontsize=9)
                _ax.tick_params(colors="#bbb")
                for _sp in _ax.spines.values():
                    _sp.set_color("#333")
                _ax.spines["top"].set_visible(False)
                _ax.spines["right"].set_visible(False)
                _ax.legend(fontsize=8, labelcolor="#bbb",
                           facecolor="#1a1d27", edgecolor="#333")
                _ax.set_ylim(0, 110)
                plt.tight_layout()
                st.pyplot(_fig, width='stretch')
                plt.close(_fig)

        # Correlation chart
        if corr_out:
            _top = corr_out.get("top_correlations", [])[:10]
            if _top:
                st.markdown("#### Top Feature Correlations")
                _pairs  = [f"{r['col_a']} ↔ {r['col_b']}" for r in _top]
                _vals   = [r["correlation"] for r in _top]
                _clrs   = ["#2ecc71" if v >= 0 else "#e74c3c" for v in _vals]
                _fig2, _ax2 = plt.subplots(
                    figsize=(8, max(3, len(_pairs)*0.42)))
                _fig2.patch.set_facecolor("#0f1117")
                _ax2.set_facecolor("#1a1d27")
                _ax2.barh(_pairs[::-1], _vals[::-1],
                          color=_clrs[::-1], alpha=.85)
                _ax2.set_xlabel("Correlation coefficient",
                                color="#bbb", fontsize=9)
                _ax2.tick_params(colors="#bbb", labelsize=8)
                _ax2.axvline(0, color="#555", lw=0.8)
                _ax2.set_xlim(-1.1, 1.1)
                for _sp2 in _ax2.spines.values():
                    _sp2.set_color("#333")
                plt.tight_layout()
                st.pyplot(_fig2, width='stretch')
                plt.close(_fig2)

    # ── Dataset ───────────────────────────────────────────────────────────────
    with tab_ds:
        if meta:
            _c1, _c2, _c3 = st.columns(3)
            _c1.metric("Rows",      f"{meta.row_count:,}")
            _c2.metric("Columns",   meta.column_count)
            _c3.metric("Task type", meta.task_type or "—")

            _col_rows = [{
                "Column":  col,
                "Type":    dtype,
                "Kind":    "numerical" if col in meta.numerical_cols
                           else "categorical",
                "Missing": meta.missing_values.get(col, 0),
                "Note":    ("🎯 target" if col == meta.target_column else "")
                           + (" ⚠ high card." if col in meta.high_cardinality_cols else ""),
            } for col, dtype in meta.columns.items()]
            st.dataframe(_safe_df(pd.DataFrame(_col_rows)), width='stretch')

            if meta.missing_values:
                st.markdown("#### Missing Values")
                _miss = pd.DataFrame(
                    [(c, v) for c, v in meta.missing_values.items()],
                    columns=["Column", "Count"],
                ).sort_values("Count", ascending=False)
                st.bar_chart(_safe_df(_miss.set_index("Column")))

            if meta.class_balance:
                st.markdown("#### Class Balance")
                _cb = pd.DataFrame(
                    [(str(k), v) for k, v in meta.class_balance.items()],
                    columns=["Class", "Count"],
                )
                st.bar_chart(_safe_df(_cb.set_index("Class")))

        if clean_out:
            st.markdown("#### Cleaning Summary")
            _c1, _c2, _c3 = st.columns(3)
            _c1.metric("Strategy",       clean_out.get("strategy_used", "—"))
            _c2.metric("Missing before", clean_out.get("missing_before", "—"))
            _c3.metric("Missing after",  clean_out.get("missing_after", "—"))

        if outlier_out:
            st.markdown("#### Outlier Detection")
            _c1, _c2 = st.columns(2)
            _c1.metric("Total outliers", outlier_out.get("total_outliers", "—"))
            _c2.metric("Outlier %",
                       f"{outlier_out.get('outlier_percentage','—')}%")
            _pc = outlier_out.get("per_column_outliers", {})
            if _pc:
                _pc_df = pd.DataFrame(
                    [(c, v) for c, v in _pc.items() if v > 0],
                    columns=["Column", "Outliers"],
                ).sort_values("Outliers", ascending=False)
                if not _pc_df.empty:
                    st.dataframe(_safe_df(_pc_df), width='stretch')

    # ── Models ────────────────────────────────────────────────────────────────
    with tab_ml:
        if train_out:
            _mt2   = train_out.get("models_trained", {})
            _best2 = train_out.get("best_model", "")
            _task2 = train_out.get("task_type", "classification")
            _pk2   = "accuracy" if _task2 == "classification" else "r2"
            _sk2   = "f1_score" if _task2 == "classification" else "rmse"

            st.markdown(
                f"**Task:** {_task2} &nbsp;|&nbsp; **Best:** 🏆 `{_best2}` "
                f"&nbsp;|&nbsp; **CV folds:** {train_out.get('n_cv_folds',5)} "
                f"&nbsp;|&nbsp; **Test split:** "
                f"{int(train_out.get('test_size', 0.2)*100)}%"
            )
            _rows = []
            for _nm, _m in _mt2.items():
                _tr2 = _m.get("train_metrics", {})
                _te2 = _m.get("test_metrics",  {})
                _g   = _m.get("train_test_gap")
                _rows.append({
                    "Model": f"🏆 {_nm}" if _nm == _best2 else _nm,
                    f"Train {_pk2}": f"{_tr2.get(_pk2,0)*100:.1f}%",
                    f"Test {_pk2}":  f"{_te2.get(_pk2,0)*100:.1f}%",
                    "CV mean":  f"{_m.get('cv_mean',0)*100:.1f}%",
                    "CV std":   f"±{_m.get('cv_std',0)*100:.1f}%",
                    "Gap": (f"{_g*100:.1f}%" + (" ⚠" if _g and _g > .10 else "")
                            if _g is not None else "—"),
                    _sk2.replace("_", " "): (
                        f"{_te2.get(_sk2,0)*100:.1f}%"
                        if _sk2 != "rmse" else f"{_te2.get(_sk2,0):.4f}"
                    ),
                })
            st.dataframe(_safe_df(pd.DataFrame(_rows)), width='stretch')

            _warn2 = train_out.get("overfit_warnings", [])
            if _warn2:
                st.markdown("#### ⚠️ Overfitting Warnings")
                for _w2 in _warn2:
                    st.markdown(f'<div class="wc">⚠ {_w2}</div>',
                                unsafe_allow_html=True)
            else:
                st.success("✅ No overfitting — gap within range for all models.")

        if eval_out:
            st.markdown("#### Full Evaluation")
            _cr = eval_out.get("classification_report", {})
            if _cr:
                _cr_rows = [
                    {"Class": _lbl,
                     "Precision": f"{_v.get('precision',0):.3f}",
                     "Recall":    f"{_v.get('recall',0):.3f}",
                     "F1":        f"{_v.get('f1-score',0):.3f}",
                     "Support":   int(_v.get("support", 0))}
                    for _lbl, _v in _cr.items() if isinstance(_v, dict)
                ]
                st.dataframe(_safe_df(pd.DataFrame(_cr_rows)), width='stretch')

        if stat_out:
            st.markdown("#### Statistical Test")
            _c1, _c2, _c3 = st.columns(3)
            _c1.metric("Test",        stat_out.get("test_name", "—"))
            _c2.metric("p-value",     f"{stat_out.get('p_value', 0):.4f}")
            _c3.metric("Significant",
                       "Yes ✅" if stat_out.get("significant") else "No ❌")
            st.info(stat_out.get("interpretation", ""))

    # ── Insights ──────────────────────────────────────────────────────────────
    with tab_ins:
        if report.get("reasoning"):
            st.markdown("#### Agent Reasoning")
            st.markdown(
                f'<div style="background:#1a1d27;border-radius:8px;padding:1rem;'
                f'font-size:.9rem;color:#c0cce0;line-height:1.75">'
                f'{report["reasoning"]}</div>',
                unsafe_allow_html=True,
            )
        for _ins in report.get("insights", []):
            st.markdown(f'<div class="ic">💡 {_ins}</div>',
                        unsafe_allow_html=True)
        if report.get("recommendations"):
            st.markdown("#### Recommendations")
        for _rec in report.get("recommendations", []):
            st.markdown(f'<div class="rc">→ {_rec}</div>',
                        unsafe_allow_html=True)
        if report.get("key_metrics"):
            st.markdown("#### Key Metrics")
            st.dataframe(
                _safe_df(pd.DataFrame(
                    [(k, str(v)) for k, v in report["key_metrics"].items()],
                    columns=["Metric", "Value"],
                )),
                width='stretch',
                hide_index=True,
            )

    # ── Tool Log ──────────────────────────────────────────────────────────────
    with tab_log:
        if tool_results:
            st.dataframe(
                _safe_df(pd.DataFrame([{
                    "Tool":      r.get("tool_name", "?"),
                    "Status":    r.get("status", "?"),
                    "Time (ms)": f"{r.get('execution_time_ms',0):.0f}",
                    "Summary":   r.get("output", {}).get(
                                     "summary", r.get("error", ""))[:140],
                } for r in tool_results])),
                width='stretch',
            )
        with st.expander("Raw JSON"):
            st.json(tool_results)

    # ── Report ────────────────────────────────────────────────────────────────
    with tab_rep:
        if tmp_dir:
            _rdir = Path(tmp_dir) / "output" / "reports"
            _mds  = sorted(_rdir.glob("*.md")) if _rdir.exists() else []
            if _mds:
                st.markdown(_mds[0].read_text(encoding="utf-8"))
            else:
                st.json(report)

    # ── Downloads ─────────────────────────────────────────────────────────────
    with tab_dl:
        if tmp_dir:
            _out = Path(tmp_dir) / "output"

            _rdir2 = _out / "reports"
            if _rdir2.exists():
                st.markdown("**Reports**")
                for _f in sorted(_rdir2.iterdir()):
                    st.download_button(
                        f"⬇ {_f.name}", _f.read_bytes(), _f.name,
                        mime=("text/markdown" if _f.suffix == ".md"
                              else "application/json"),
                        key=f"dlr_{_f.name}",
                    )

            _mdir = _out / "models"
            if _mdir.exists() and any(_mdir.iterdir()):
                st.markdown("**Trained Models (.pkl)**")
                for _f in sorted(_mdir.iterdir()):
                    st.download_button(
                        f"⬇ {_f.name}", _f.read_bytes(), _f.name,
                        mime="application/octet-stream",
                        key=f"dlm_{_f.name}",
                    )

            _vdir = _out / "visualizations"
            if _vdir.exists() and any(_vdir.iterdir()):
                st.markdown("**Visualizations**")
                _vcols = st.columns(2)
                for _i, _f in enumerate(sorted(_vdir.glob("*.png"))):
                    with _vcols[_i % 2]:
                        st.image(str(_f), caption=_f.name,
                                 width='stretch')
                        st.download_button(
                            f"⬇ {_f.name}", _f.read_bytes(), _f.name,
                            mime="image/png",
                            key=f"dlv_{_f.name}",
                        )

            st.markdown("**Full JSON report**")
            st.download_button(
                "⬇ final_report.json",
                json.dumps(report, indent=2, default=str),
                "final_report.json",
                mime="application/json",
                key="dl_json_final",
            )


# ══════════════════════════════════════════════════════════════════════════════
# EMPTY STATE
# ══════════════════════════════════════════════════════════════════════════════
if (preview_df is None
        and not st.session_state["analysis_done"]
        and not st.session_state["stage_log"]):
    st.markdown(
        '<div style="text-align:center;padding:4rem 2rem;color:#444">'
        '<div style="font-size:3.5rem">🧠</div>'
        '<h2 style="color:#666;font-weight:400;margin:.6rem 0">'
        'Upload a dataset to get started</h2>'
        '<p style="max-width:460px;margin:.4rem auto;line-height:1.8;color:#555">'
        'Drop a <b style="color:#777">CSV</b> or '
        '<b style="color:#777">Excel</b> file in the sidebar, '
        'enter your API key, configure settings, then click '
        '<b style="color:#777">▶ Run Analysis</b>.'
        '</p>'
        '<p style="color:#3b5bdb;font-size:.82rem;margin-top:1.2rem">'
        'Stage 1: Ingest → 2: Reason → 3: Execute → '
        '4: Interpret → 5: Refine → 6: RLM → 7: Report'
        '</p></div>',
        unsafe_allow_html=True,
    )
