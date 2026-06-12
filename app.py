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
from io import BytesIO
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
        def add(self, *a: Any, **k: Any) -> _Tr: return self
    class _Pr:
        def __init__(self, *a: Any, **k: Any): pass
        def __enter__(self) -> _Pr: return self
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


# ── CSS — Sauce Labs design system (DESIGN.md): obsidian canvas, neon pulse ───
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500&family=Inter+Tight:wght@400;500&display=swap');

:root {
    --obsidian: #132322;
    --abyss:    #0e1a19;
    --charcoal: #070f0f;
    --neon:     #3ddc91;
    --mint:     #97ddbc;
    --yellow:   #ffcd48;
    --slate:    #828786;
    --danger:   #ff5c5c;   /* functional exception: failure states only */
    --line:        rgba(255,255,255,.07);
    --line-strong: rgba(255,255,255,.22);
    --dim:   rgba(255,255,255,.55);
    --faint: rgba(255,255,255,.38);
    --font-body:    'Inter', ui-sans-serif, system-ui, sans-serif;
    --font-display: 'Inter Tight', 'Inter', ui-sans-serif, system-ui, sans-serif;
}

/* ── Chrome ── */
#MainMenu, footer, .stAppDeployButton { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.stApp { background: var(--obsidian); }
section[data-testid="stSidebar"] { background: var(--abyss); border-right: 1px solid var(--line); }
.block-container { max-width: 1200px; }
hr { border-color: var(--line) !important; }
a { color: var(--neon) !important; }
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: #2c403d; border-radius: 8px;
                            border: 2px solid var(--obsidian); }
::-webkit-scrollbar-track { background: transparent; }

/* ── Typography ── */
h1, h2, h3, h4, h5, h6 { font-family: var(--font-display) !important;
                         font-weight: 500 !important; letter-spacing: -0.01em; }
.stMarkdown p { letter-spacing: -0.08px; }

/* Eyebrow — small tracked-out caps above headlines */
.eyebrow { font-family: var(--font-display); font-size: 10px; font-weight: 500;
           letter-spacing: 1.6px; text-transform: uppercase;
           color: var(--neon); margin: 0 0 .45rem; }
.eyebrow.side { margin: 1.15rem 0 .45rem; }
.sect { margin: .3rem 0 1rem; }
.sect h2, .sect h3 { margin: 0; padding: 0; }

/* Live-signal pulse dot */
.pulse-dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%;
             background: var(--neon); flex: none;
             animation: pulse 2.4s ease-in-out infinite; }
@keyframes pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(61,220,145,.45); }
    50%      { box-shadow: 0 0 0 8px rgba(61,220,145,0); }
}

/* ── Hero header ── */
.hero { display: flex; align-items: flex-start; gap: 14px;
        padding: .2rem 0 1.6rem; border-bottom: 1px solid var(--line);
        margin-bottom: 1.6rem; }
.hero .pulse-dot { margin-top: 8px; }
.hero h1 { font-size: 34px; line-height: 1.1; margin: .25rem 0 0; color: #fff; }
.hero .hero-sub { color: var(--dim); font-size: 15px; margin: .5rem 0 0; max-width: 640px; }

/* ── Sidebar brand ── */
.side-brand { display: flex; align-items: center; gap: 11px; margin: .2rem 0 .5rem; }
.side-title { font-family: var(--font-display); font-weight: 500; font-size: 16px;
              color: #fff; line-height: 1.25; }
.side-sub { font-family: var(--font-display); font-size: 9px; letter-spacing: 1.4px;
            text-transform: uppercase; color: var(--faint); margin-top: 3px; }

/* ── Buttons — pill language ── */
.stButton button, .stDownloadButton button {
    font-weight: 400; letter-spacing: -0.08px;
    transition: filter .15s ease, border-color .15s ease, color .15s ease;
}
.stButton button[kind="primary"], .stDownloadButton button[kind="primary"] {
    background: var(--neon); color: var(--obsidian); border: none;
}
.stButton button[kind="primary"]:hover:enabled,
.stDownloadButton button[kind="primary"]:hover:enabled {
    background: var(--neon); color: var(--obsidian); filter: brightness(1.12);
}
.stButton button[kind="primary"]:disabled {
    background: rgba(61,220,145,.14); color: rgba(255,255,255,.32); border: none;
}
.stButton button[kind="secondary"], .stDownloadButton button[kind="secondary"] {
    background: transparent; border: 1.5px solid var(--line-strong); color: #fff;
}
.stButton button[kind="secondary"]:hover:enabled,
.stDownloadButton button[kind="secondary"]:hover:enabled {
    border-color: var(--neon); color: var(--neon); background: transparent;
}

/* ── Tabs — pill switcher ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px; background: var(--abyss); border: 1px solid var(--line);
    border-radius: 56px; padding: 5px; width: max-content; max-width: 100%;
}
.stTabs [data-baseweb="tab"] { border-radius: 56px; padding: 4px 16px;
                               background: transparent; }
.stTabs [data-baseweb="tab"] p { font-size: 14.5px; color: var(--slate); }
.stTabs [data-baseweb="tab"]:hover p { color: #fff; }
.stTabs [aria-selected="true"] { background: var(--neon) !important; }
.stTabs [aria-selected="true"] p { color: var(--obsidian) !important; font-weight: 500; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.1rem; }

/* ── File uploader ── */
[data-testid="stFileUploaderDropzone"] {
    background: var(--obsidian); border: 1.5px dashed rgba(61,220,145,.35);
    border-radius: 20px;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--neon); }

/* ── st.metric — stat block ── */
[data-testid="stMetric"] { background: var(--abyss); border: 1px solid var(--line);
                           border-radius: 20px; padding: 14px 18px; }
[data-testid="stMetricValue"] { font-family: var(--font-display);
                                font-weight: 500; color: var(--neon); }
[data-testid="stMetricLabel"] p { font-family: var(--font-display); font-size: 10px;
                                  letter-spacing: 1.4px; text-transform: uppercase;
                                  color: var(--faint); }

/* ── Expanders / code / alerts ── */
[data-testid="stExpander"] details { background: var(--abyss);
    border: 1px solid var(--line) !important; border-radius: 20px; }
[data-testid="stExpander"] summary:hover { color: var(--neon); }
[data-testid="stCode"] pre { background: var(--charcoal) !important;
    border: 1px solid var(--line); border-radius: 16px; }
[data-testid="stAlert"] { border-radius: 16px; }

/* ── Stage progress cards ── */
.sc { display: flex; align-items: center; gap: 12px;
      padding: .8rem 1.1rem; margin-bottom: .5rem;
      border-radius: 16px; border: 1px solid var(--line);
      background: var(--abyss); font-size: .88rem; color: rgba(255,255,255,.85); }
.sc .dot { width: 8px; height: 8px; border-radius: 50%; flex: none;
           background: transparent; border: 1.5px solid rgba(255,255,255,.25); }
.sc .sc-num { font-family: var(--font-display); font-size: 10px;
              letter-spacing: 1.2px; color: var(--faint); }
.sc .detail { margin-left: auto; font-size: .76rem; color: var(--faint);
              text-align: right; }
.sc.done   { border-color: rgba(61,220,145,.28); }
.sc.done .dot   { background: var(--neon); border-color: var(--neon); }
.sc.active { border-color: var(--neon); background: rgba(61,220,145,.06); color: #fff; }
.sc.active .dot { background: var(--neon); border-color: var(--neon);
                  animation: pulse 1.6s ease-in-out infinite; }
.sc.skip   { color: var(--faint); }
.sc.skip .dot   { background: rgba(255,255,255,.18); border-color: transparent; }
.sc.err    { border-color: rgba(255,92,92,.5); }
.sc.err .dot    { background: var(--danger); border-color: var(--danger); }

/* ── Metric tiles (custom) — stat block pattern ── */
.mt { background: var(--abyss); border: 1px solid var(--line); border-radius: 20px;
      padding: 1.05rem .9rem .95rem; text-align: center; }
.mt .lbl { font-family: var(--font-display); font-size: 10px; letter-spacing: 1.3px;
           text-transform: uppercase; color: var(--faint); margin-bottom: 6px; }
.mt .val { font-family: var(--font-display); font-size: 1.55rem; font-weight: 500;
           line-height: 1.15; overflow-wrap: anywhere; }
.mt .sub { font-size: .7rem; color: var(--faint); margin-top: 4px; }

/* ── Content cards: insight / recommendation / warning ── */
.ic, .rc, .wc { background: var(--abyss); border: 1px solid var(--line);
                border-left-width: 3px; border-radius: 12px;
                padding: .7rem 1rem; margin-bottom: .5rem;
                font-size: .9rem; line-height: 1.55; }
.ic { border-left-color: var(--mint);   color: #e6f3ee; }
.rc { border-left-color: var(--neon);   color: #dff5ea; }
.wc { border-left-color: var(--yellow); color: #f5e8c8; }

/* Agent reasoning panel */
.reason { background: var(--abyss); border: 1px solid var(--line);
          border-radius: 20px; padding: 1.1rem 1.3rem;
          font-size: .92rem; color: rgba(255,255,255,.82); line-height: 1.75; }

/* Run-in-progress banner — mint whisper wash */
.run-banner { display: flex; align-items: center; gap: 12px;
              background: rgba(151,221,188,.08); border: 1px solid rgba(151,221,188,.3);
              border-radius: 16px; padding: .85rem 1.2rem;
              color: #d9efe6; font-size: .92rem; margin: .4rem 0 1rem; }

/* ── Empty state ── */
.empty { text-align: center; padding: 5rem 2rem 4rem; }
.empty .pulse-dot { width: 12px; height: 12px; margin-bottom: 1.5rem; }
.empty h2 { font-size: 30px; color: #fff; margin: .4rem 0; }
.empty p { color: var(--dim); max-width: 500px; margin: .6rem auto 0;
           line-height: 1.8; font-size: 15px; }
.empty b { color: #fff; font-weight: 500; }
.empty .stages { margin-top: 2rem; font-family: var(--font-display); font-size: 10px;
                 letter-spacing: 1.6px; text-transform: uppercase; color: var(--faint); }
.empty .stages span { color: var(--mint); }
</style>
""", unsafe_allow_html=True)


# ── Session-state initialisation ──────────────────────────────────────────────
_DEFAULTS: dict[str, Any] = {
    "preview_df":     None,   # pd.DataFrame
    "preview_name":   "",     # sanitised filename (safe for filesystem)
    "orig_name":      "",     # exact name as uploaded (change detection)
    "preview_bytes":  None,   # raw bytes
    "stage_log":      [],
    "analysis_done":  False,
    "analysis_error": None,
    "final_report":   None,
    "tool_results":   [],
    "metadata":       None,
    "profile":        None,   # DatasetProfile.to_dict()
    "dashboard":      None,   # list of ChartSpec dicts
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

# Shared Vega-Lite config so dashboard charts match the obsidian theme (DESIGN.md)
VEGA_DARK_CONFIG = {
    "font": "Inter, sans-serif",
    "axis": {
        "labelColor": "rgba(255,255,255,0.62)",
        "titleColor": "rgba(255,255,255,0.62)",
        "gridColor": "rgba(255,255,255,0.07)",
        "domainColor": "rgba(255,255,255,0.18)",
        "tickColor": "rgba(255,255,255,0.18)",
        "labelFont": "Inter, sans-serif",
        "titleFont": "Inter, sans-serif",
    },
    "legend": {
        "labelColor": "rgba(255,255,255,0.72)",
        "titleColor": "rgba(255,255,255,0.72)",
        "labelFont": "Inter, sans-serif",
        "titleFont": "Inter, sans-serif",
    },
    "view": {"stroke": "transparent"},
    "range": {"category": ["#3ddc91", "#ffcd48", "#97ddbc",
                           "#1c8f5c", "#d6f0b2", "#62b5a4"]},
}

# OpenRouter slugs use DOT version notation for Claude (claude-sonnet-4.6,
# not claude-sonnet-4-6). Every entry below is verified against the live
# /api/v1/models catalog — an invalid slug makes every call fail with 404.
OR_MODELS = [
    "openai/gpt-4o",
    "openai/gpt-4.1",
    "anthropic/claude-sonnet-4.6",
    "anthropic/claude-opus-4.8",
    "meta-llama/llama-3.3-70b-instruct",
    "google/gemini-2.5-flash",
    "mistralai/mistral-large",
    "deepseek/deepseek-chat",
    "cohere/command-r-plus-08-2024",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _reset_pipeline() -> None:
    for k in ("stage_log", "analysis_done", "analysis_error",
              "final_report", "tool_results", "metadata", "profile",
              "dashboard", "tmp_dir", "progress_lines", "llm_warning"):
        st.session_state[k] = _DEFAULTS[k]  # type: ignore[assignment]


def _set_stage(num: str, status: str, detail: str = "") -> None:
    log: list[tuple[str, str, str]] = [
        e for e in st.session_state["stage_log"] if e[0] != num
    ]
    log.append((num, status, detail))
    st.session_state["stage_log"] = log


def _stage_card(num: str, name: str,
                status: str, detail: str = "") -> str:
    cls = {"done": "done", "active": "active",
           "skipped": "skip", "error": "err"}.get(status, "")
    det = f'<span class="detail">{detail}</span>' if detail else ""
    return (f'<div class="sc {cls}"><span class="dot"></span>'
            f'<span class="sc-num">{num.zfill(2)}</span>'
            f'{name}{det}</div>')


def _mt(label: str, value: str, sub: str = "",
        color: str = "#3ddc91") -> str:
    return (f'<div class="mt"><div class="lbl">{label}</div>'
            f'<div class="val" style="color:{color}">{value}</div>'
            f'<div class="sub">{sub}</div></div>')


def _gap_color(g: float) -> str:
    return "#3ddc91" if g < 0.05 else "#ffcd48" if g < 0.10 else "#ff5c5c"


def _section(eyebrow: str, title: str, level: str = "h3") -> None:
    """Eyebrow label + headline — the DESIGN.md section-header pattern."""
    st.markdown(
        f'<div class="sect"><div class="eyebrow">{eyebrow}</div>'
        f'<{level}>{title}</{level}></div>',
        unsafe_allow_html=True,
    )


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
    st.markdown(
        '<div class="side-brand"><span class="pulse-dot"></span>'
        '<div><div class="side-title">Agentic Data Analysis</div>'
        '<div class="side-sub">RLM-powered autonomous pipeline</div></div></div>',
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Upload ────────────────────────────────────────────────────────────────
    st.markdown('<div class="eyebrow side">Dataset</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "CSV or Excel",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed",
    )

    # Persist to session_state immediately on upload / clear on removal.
    # Every upload passes through src.core.security before touching disk:
    # extension allowlist, size ceiling, magic-byte sniffing, safe filename.
    if uploaded is not None:
        if uploaded.name != st.session_state.get("orig_name", ""):
            _reset_pipeline()
            raw_bytes = uploaded.read()
            st.session_state["orig_name"] = uploaded.name
            from src.core.security import UploadValidationError, validate_upload
            try:
                safe_name = validate_upload(uploaded.name, raw_bytes)
            except UploadValidationError as _ve:
                st.session_state["preview_df"]    = None
                st.session_state["preview_bytes"] = None
                st.session_state["preview_name"]  = ""
                st.error(f"🛡️ Upload rejected: {_ve}")
            else:
                st.session_state["preview_bytes"] = raw_bytes
                st.session_state["preview_name"]  = safe_name
                _fname = safe_name.lower()
                try:
                    if _fname.endswith(".csv"):
                        st.session_state["preview_df"] = pd.read_csv(BytesIO(raw_bytes))
                    elif _fname.endswith(".xlsx"):
                        st.session_state["preview_df"] = pd.read_excel(
                            BytesIO(raw_bytes), engine="openpyxl")
                    elif _fname.endswith(".xls"):
                        st.session_state["preview_df"] = pd.read_excel(
                            BytesIO(raw_bytes), engine="xlrd")
                except Exception as _e:
                    st.session_state["preview_df"] = None
                    st.error(f"Could not read file: {_e}")
    else:
        if st.session_state.get("orig_name"):
            for _k2, _v2 in _DEFAULTS.items():
                st.session_state[_k2] = _v2

    target_col = st.text_input(
        "Target column",
        placeholder="e.g. churn, price, label  (blank = clustering)",
    )

    objective = st.text_area(
        "Analysis objective (optional, plain English)",
        placeholder="e.g. What drives customer churn? Which customers should "
                    "we focus retention efforts on?",
        height=90,
        help="The agents will prioritise analyses that answer this question "
             "and address it directly in the final report.",
    )

    # ── LLM Provider ──────────────────────────────────────────────────────────
    st.markdown('<div class="eyebrow side">LLM Provider</div>', unsafe_allow_html=True)
    provider = st.selectbox("Provider", ["openai", "anthropic", "openrouter", "nvidia"])

    NVIDIA_MODELS = [
        "openai/gpt-oss-120b",
        "meta/llama-3.1-70b-instruct",
        "meta/llama-3.3-70b-instruct",
        "mistralai/mistral-large-2-instruct",
        "microsoft/phi-3-medium-128k-instruct",
        "google/gemma-2-27b-it",
        "deepseek-ai/deepseek-r1",
    ]

    if provider == "openai":
        model_list = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
        key_ph     = "sk-..."
    elif provider == "anthropic":
        model_list = ["claude-sonnet-4-6", "claude-opus-4-8",
                      "claude-haiku-4-5-20251001"]
        key_ph     = "sk-ant-..."
    elif provider == "nvidia":
        model_list = NVIDIA_MODELS
        key_ph     = "nvapi-..."
    else:
        model_list = OR_MODELS
        key_ph     = "sk-or-..."

    model_sel = st.selectbox("Model", model_list)
    if provider in ("openrouter", "nvidia"):
        custom_m = st.text_input(
            "Custom model string (overrides above)",
            placeholder=(
                "e.g. cohere/command-r-plus"
                if provider == "openrouter"
                else "e.g. nvidia/llama-3.1-nemotron-70b-instruct"
            ),
        )
        final_model = custom_m.strip() if custom_m.strip() else model_sel
    else:
        final_model = model_sel

    _key_label = {
        "openai": "OpenAI",
        "anthropic": "Anthropic",
        "openrouter": "OpenRouter",
        "nvidia": "NVIDIA",
    }.get(provider, provider)
    api_key = st.text_input(
        f"{_key_label} API Key",
        type="password",
        placeholder=key_ph,
    )

    # ── Analysis Settings ─────────────────────────────────────────────────────
    st.markdown('<div class="eyebrow side">Analysis Settings</div>', unsafe_allow_html=True)
    max_iter   = st.slider("Max iterations", 3, 25, 10)
    enable_rlm = st.toggle("Enable RLM decomposition (Stage 6)", value=True)

    st.markdown('<div class="eyebrow side">Anti-Overfitting</div>', unsafe_allow_html=True)
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
        "Run Analysis",
        disabled=not can_run,
        width='stretch',
        type="primary",
    )
    if st.session_state["analysis_done"] or st.session_state["analysis_error"]:
        if st.button("New Analysis", width='stretch'):
            _reset_pipeline()
            st.rerun()

    if not has_file:
        st.caption("Upload a CSV or Excel file first.")
    elif not has_key:
        st.caption("Enter your API key to continue.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA — header
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="hero"><span class="pulse-dot"></span>'
    '<div><div class="eyebrow">Autonomous · RLM-powered · Anti-overfitting built-in</div>'
    '<h1>Agentic Data Analysis</h1>'
    '<p class="hero-sub">An engineering console that profiles, models and explains '
    'your dataset — end to end, on its own.</p></div></div>',
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# DATASET PREVIEW — always visible once a file is loaded
# ══════════════════════════════════════════════════════════════════════════════
preview_df: pd.DataFrame | None = st.session_state["preview_df"]

if preview_df is not None and not st.session_state["analysis_done"]:
    _section("Dataset preview", st.session_state["preview_name"])
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
    if objective.strip():
        os.environ["USER_OBJECTIVE"] = objective.strip()
    else:
        os.environ.pop("USER_OBJECTIVE", None)
    {
        "openai":     lambda: os.environ.__setitem__("OPENAI_API_KEY",     api_key.strip()),
        "anthropic":  lambda: os.environ.__setitem__("ANTHROPIC_API_KEY",  api_key.strip()),
        "openrouter": lambda: os.environ.__setitem__("OPENROUTER_API_KEY", api_key.strip()),
        "nvidia":     lambda: os.environ.__setitem__("NVIDIA_API_KEY",     api_key.strip()),
    }[provider]()

    # ── Spinner placeholder — replaced after run completes ───────────────
    _spinner_ph = st.empty()
    _spinner_ph.markdown(
        '<div class="run-banner"><span class="pulse-dot"></span>'
        'Running analysis — this may take 1–3 minutes depending on dataset '
        'size and model.</div>',
        unsafe_allow_html=True,
    )

    # ── Collect progress lines into session state (no st.write during run) ─
    _progress_lines: list[str] = []

    def _upd(num: str, s: str, detail: str = "") -> None:
        _set_stage(num, s, detail)
        _ico = {"done": "✅", "active": "⏳", "error": "❌", "skipped": "⏭️"}.get(s, "⬜")
        _nm  = next(n for no, n, _ in STAGE_DEFS if no == num)
        _progress_lines.append(f"{_ico} Stage {num}: {_nm}" + (f" — {detail}" if detail else ""))

    # ── LLM preflight — fail fast with the REAL error instead of running
    #    the whole pipeline on the deterministic fallback ──────────────────
    from src.core.controller import AgentController, LLMClient

    _ok, _ping_err = LLMClient().ping()
    if not _ok:
        _spinner_ph.empty()
        _set_stage("2", "error", "LLM unreachable")
        st.session_state["analysis_error"] = _ping_err
        st.error(
            f"🛑 LLM connection check failed — analysis was not started.\n\n"
            f"Provider: `{provider}` · Model: `{final_model}`"
        )
        st.code(_ping_err, language=None)
        st.info(
            "Check that the model ID exists on the provider, the API key is "
            "valid, and your account has credits. Then click Run Analysis again."
        )
        st.stop()

    try:
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
            _set_stage("3", "active", detail)
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
        if agent.last_profile is not None:
            st.session_state["profile"] = agent.last_profile.to_dict()
        _dash_path = Path(outdir) / "reports" / "dashboard.json"
        if _dash_path.exists():
            try:
                st.session_state["dashboard"] = json.loads(
                    _dash_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                st.session_state["dashboard"] = None
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
    _section("Live progress", "Pipeline stages")
    log_map = {n: (s, d) for n, s, d in st.session_state["stage_log"]}
    cols = st.columns(2)
    for i, (num, name, _icon) in enumerate(STAGE_DEFS):
        s, d = log_map.get(num, ("pending", ""))
        with cols[i % 2]:
            st.markdown(
                _stage_card(num, name, s, d),
                unsafe_allow_html=True,
            )


# ── Progress log ─────────────────────────────────────────────────────────────
if st.session_state.get("progress_lines"):
    with st.expander("Execution log", expanded=False):
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
    _section("Results", "Analysis complete", "h2")

    if st.session_state.get("llm_warning"):
        st.warning(f"⚠️ {st.session_state['llm_warning']}")

    (tab_ov, tab_dash, tab_prof, tab_ds, tab_ml, tab_ins,
     tab_log, tab_rep, tab_dl) = st.tabs([
        "Overview", "Dashboard", "Profile", "Dataset", "Models",
        "Insights", "Tool Log", "Report", "Downloads",
    ])

    train_out   = _find_tool(tool_results, "train_model")
    eval_out    = _find_tool(tool_results, "evaluate_model")
    corr_out    = _find_tool(tool_results, "correlation_analysis")
    outlier_out = _find_tool(tool_results, "detect_outliers")
    stat_out    = _find_tool(tool_results, "select_statistical_test")
    clean_out   = _find_tool(tool_results, "clean_data")

    # ── Overview ─────────────────────────────────────────────────────────────
    with tab_ov:
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
                _gap_color(gap_val) if gap_val is not None else "#ffffff"),
            unsafe_allow_html=True,
        )
        m4.markdown(_mt("Outliers", outlier_pct, "of dataset"),
                    unsafe_allow_html=True)
        m5.markdown(_mt("Task", task_type), unsafe_allow_html=True)

        for _w in (train_out.get("overfit_warnings", []) if train_out else []):
            st.markdown(f'<div class="wc">⚠ {_w}</div>',
                        unsafe_allow_html=True)

        # Model comparison — interactive Vega-Lite grouped bars
        if train_out:
            _mt_map2 = train_out.get("models_trained", {})
            if _mt_map2:
                st.markdown("#### Model Comparison")
                _task = train_out.get("task_type", "classification")
                _pk   = "accuracy" if _task == "classification" else "r2"
                _rows_v = []
                for _name, _m in _mt_map2.items():
                    _rows_v += [
                        {"model": _name, "metric": "Train",
                         "score": round(_m.get("train_metrics", {}).get(_pk, 0) * 100, 2)},
                        {"model": _name, "metric": "Test",
                         "score": round(_m.get("test_metrics", {}).get(_pk, 0) * 100, 2)},
                        {"model": _name, "metric": "CV mean",
                         "score": round(_m.get("cv_mean", 0) * 100, 2)},
                    ]
                st.vega_lite_chart(
                    pd.DataFrame(_rows_v),
                    {
                        "mark": {"type": "bar", "cornerRadiusEnd": 2},
                        "height": 300,
                        "background": "transparent",
                        "config": VEGA_DARK_CONFIG,
                        "encoding": {
                            "x": {"field": "model", "type": "nominal",
                                  "axis": {"labelAngle": 0, "title": None}},
                            "xOffset": {"field": "metric"},
                            "y": {"field": "score", "type": "quantitative",
                                  "title": f"{_pk} %",
                                  "scale": {"domain": [0, 110]}},
                            "color": {
                                "field": "metric",
                                "scale": {
                                    "domain": ["Train", "Test", "CV mean"],
                                    "range": ["#97ddbc", "#3ddc91", "#ffcd48"],
                                },
                                "legend": {"orient": "top", "title": None},
                            },
                            "tooltip": [
                                {"field": "model"},
                                {"field": "metric"},
                                {"field": "score", "title": f"{_pk} %"},
                            ],
                        },
                    },
                    use_container_width=True,
                )

        # Correlation chart — interactive Vega-Lite diverging bars
        if corr_out:
            _top = corr_out.get("top_correlations", [])[:10]
            if _top:
                st.markdown("#### Top Feature Correlations")
                _corr_df = pd.DataFrame(
                    [{"pair": f"{r['col_a']} ↔ {r['col_b']}",
                      "correlation": r["correlation"]} for r in _top]
                )
                st.vega_lite_chart(
                    _corr_df,
                    {
                        "mark": {"type": "bar", "cornerRadiusEnd": 2},
                        "height": max(160, len(_top) * 30),
                        "background": "transparent",
                        "config": VEGA_DARK_CONFIG,
                        "encoding": {
                            "y": {"field": "pair", "type": "nominal",
                                  "sort": "-x", "title": None},
                            "x": {"field": "correlation", "type": "quantitative",
                                  "scale": {"domain": [-1.1, 1.1]},
                                  "title": "Correlation coefficient"},
                            "color": {
                                "condition": {"test": "datum.correlation >= 0",
                                              "value": "#3ddc91"},
                                "value": "#ffcd48",
                            },
                            "tooltip": [
                                {"field": "pair"},
                                {"field": "correlation"},
                            ],
                        },
                    },
                    use_container_width=True,
                )

    # ── Dashboard — dynamic, data-aware charts from the Dashboard Agent ──────
    with tab_dash:
        dashboard: list[dict[str, Any]] | None = st.session_state.get("dashboard")
        if dashboard:
            st.caption(
                "Charts selected automatically by the Dashboard Agent to fit "
                "this dataset's nature and the analysis results."
            )
            _full_width_ids = {"model_comparison", "top_correlations",
                               "scatter_top_pair", "time_series"}
            _grid_charts: list[dict[str, Any]] = []

            def _render_chart(_ch: dict[str, Any]) -> None:
                st.markdown(f"**{_ch.get('title', '')}**")
                _spec = dict(_ch.get("spec", {}))
                _spec.setdefault("background", "transparent")
                _spec.setdefault("config", VEGA_DARK_CONFIG)
                st.vega_lite_chart(_spec, use_container_width=True)
                if _ch.get("description"):
                    st.caption(_ch["description"])

            for _ch in dashboard:
                if _ch.get("chart_id") in _full_width_ids:
                    _render_chart(_ch)
                else:
                    _grid_charts.append(_ch)
            if _grid_charts:
                _dcols = st.columns(2)
                for _i, _ch in enumerate(_grid_charts):
                    with _dcols[_i % 2]:
                        _render_chart(_ch)
        else:
            st.info("No dashboard was generated for this run.")

    # ── Profile — automated data-quality first look ──────────────────────────
    with tab_prof:
        prof: dict[str, Any] | None = st.session_state.get("profile")
        if prof:
            _q = int(prof.get("quality_score", 0))
            _qc = "#3ddc91" if _q >= 80 else "#ffcd48" if _q >= 60 else "#ff5c5c"
            p1, p2, p3, p4 = st.columns(4)
            p1.markdown(_mt("Quality score", f"{_q}/100", "0–100", _qc),
                        unsafe_allow_html=True)
            p2.markdown(_mt("Duplicate rows", f"{prof.get('duplicate_rows', 0):,}"),
                        unsafe_allow_html=True)
            p3.markdown(_mt("Memory", f"{prof.get('memory_mb', 0)} MB"),
                        unsafe_allow_html=True)
            p4.markdown(_mt("Columns profiled", str(prof.get('column_count', 0))),
                        unsafe_allow_html=True)

            for _w in prof.get("warnings", []):
                st.markdown(f'<div class="wc">⚠ {_w}</div>', unsafe_allow_html=True)

            st.markdown("#### Column Semantics")
            _prows = [{
                "Column":    c.get("name"),
                "Kind":      c.get("kind"),
                "Dtype":     c.get("dtype"),
                "Missing %": c.get("missing_pct"),
                "Unique":    c.get("nunique"),
                "Flags":     ", ".join(c.get("flags", [])),
            } for c in prof.get("columns", [])]
            st.dataframe(_safe_df(pd.DataFrame(_prows)), width='stretch')
        else:
            st.info("No profile available for this run.")

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
        if report.get("llm_fallback"):
            st.warning(
                "⚠️ The LLM became unreachable mid-run, so these insights were "
                "synthesised deterministically from tool outputs. The error is "
                "shown above — fix it and re-run for narrative interpretation."
            )
        if report.get("reasoning"):
            st.markdown("#### Agent Reasoning")
            st.markdown(
                f'<div class="reason">{report["reasoning"]}</div>',
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
            _html = _rdir / "report.html"
            if _html.exists():
                st.download_button(
                    "Download shareable HTML report (interactive charts)",
                    _html.read_bytes(), "report.html", mime="text/html",
                    key="dl_html_top", type="primary",
                )
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
                _mimes = {".md": "text/markdown", ".html": "text/html",
                          ".json": "application/json"}
                for _f in sorted(_rdir2.iterdir()):
                    st.download_button(
                        f"⬇ {_f.name}", _f.read_bytes(), _f.name,
                        mime=_mimes.get(_f.suffix, "application/octet-stream"),
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
        '<div class="empty">'
        '<span class="pulse-dot"></span>'
        '<div class="eyebrow">Awaiting dataset</div>'
        '<h2>Upload a dataset to get started</h2>'
        '<p>Drop a <b>CSV</b> or <b>Excel</b> file in the sidebar, '
        'optionally describe <b>what you want to learn</b> in plain English, '
        'enter your API key, then click <b>Run Analysis</b>.</p>'
        '<div class="stages">Ingest <span>→</span> Reason <span>→</span> '
        'Execute <span>→</span> Interpret <span>→</span> Refine '
        '<span>→</span> RLM <span>→</span> Report</div>'
        '</div>',
        unsafe_allow_html=True,
    )
