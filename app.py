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
from typing import Any, cast

import pandas as pd
import streamlit as st

# ── Project root on sys.path ─────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Agentic Data Analysis",
    page_icon="📐",
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


# ── CSS — "Drafting Table" design system (DESIGN.md) ─────────────────────────
# Mineral drafting stock, ink linework, two plotter pens. Nothing that carries
# data is rounded; structure comes from ruled hairlines, not from radius.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@75..125,400..800&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --stock:    #dcdbd3;   /* mineral drafting stock — the page itself */
    --sheet:    #efeee8;   /* the lifted sheet — plates, tables, panels */
    --ink:      #171c1f;   /* drawing ink — type, rules, linework */
    --graphite: #54585b;   /* soft pencil — secondary type */
    --pen:      #12467e;   /* measurement pen — what the agent measured */
    --risk:     #b5271a;   /* risk pen — overfit, failure, warning. Nothing else. */

    --rule:        #b6b4a9;   /* ruled hairline */
    --rule-faint:  #c8c6bc;
    --quadrille:   rgba(23,28,31,.045);
    --lift: 3px 3px 0 rgba(23,28,31,.09);   /* a sheet lying on the table */

    --sans: 'Archivo', ui-sans-serif, 'Segoe UI', system-ui, sans-serif;
    --mono: 'IBM Plex Mono', 'Cascadia Code', ui-monospace, Consolas, monospace;
}

/* ── The drafting sheet ── */
#MainMenu, footer, .stAppDeployButton { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.stApp {
    background-color: var(--stock);
    background-image:
        repeating-linear-gradient(to right,  var(--quadrille) 0 1px, transparent 1px 28px),
        repeating-linear-gradient(to bottom, var(--quadrille) 0 1px, transparent 1px 28px);
}
.block-container { max-width: 1240px; padding-top: 2.4rem; }
html, body, .stApp, [class*="css"] { font-family: var(--sans); color: var(--ink); }
hr { border: none; border-top: 1px solid var(--rule) !important; }
a { color: var(--pen) !important; text-underline-offset: 3px; }

section[data-testid="stSidebar"] {
    background: var(--sheet);
    border-right: 1px solid var(--rule);
    background-image: none;
}
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] label p { font-size: 13px; color: var(--graphite); }

::-webkit-scrollbar { width: 11px; height: 11px; }
::-webkit-scrollbar-thumb { background: var(--rule); border: 3px solid var(--stock); }
::-webkit-scrollbar-track { background: transparent; }

/* ── Type: an expanded grotesque against a mono for every measured number ── */
h1, h2, h3, h4, h5, h6 {
    font-family: var(--sans) !important;
    font-variation-settings: 'wdth' 112;
    letter-spacing: -.025em;
    color: var(--ink);
}
h1 { font-weight: 800 !important; }
h2 { font-weight: 700 !important; font-size: 30px !important; line-height: 1.05; }
h3 { font-weight: 700 !important; font-size: 19px !important; }
h4 { font-weight: 600 !important; font-size: 15.5px !important;
     font-variation-settings: 'wdth' 100; letter-spacing: -.01em; }
.stMarkdown p, .stMarkdown li { font-size: 15px; line-height: 1.62; max-width: 72ch; }
code, kbd, pre, .stCode, [data-testid="stMetricValue"] { font-family: var(--mono) !important; }

/* Datum line — a ruled measurement bar, cells divided by hairlines.
   Replaces the tracked-caps eyebrow: the rule carries the label, not a label
   floating above a heading. */
.datum { display: flex; align-items: stretch; flex-wrap: wrap;
         border-top: 1px solid var(--ink); border-bottom: 1px solid var(--rule);
         margin: 0 0 1.5rem; }
.datum .cell { padding: .5rem 1.1rem .5rem 0; margin-right: 1.1rem;
               border-right: 1px solid var(--rule-faint); }
.datum .cell:last-child { border-right: none; margin-right: 0; }
.datum .k { font-size: 12px; color: var(--graphite); }
.datum .v { font-family: var(--mono); font-size: 13px; font-weight: 500; color: var(--ink); }

/* Section head — the title sits on its own rule, sentence case, no eyebrow. */
.sect { margin: 2.2rem 0 1rem; border-bottom: 1px solid var(--ink);
        padding-bottom: .4rem; }
.sect:first-child { margin-top: .4rem; }
.sect h2, .sect h3 { margin: 0; padding: 0; }
.sect .note { font-size: 12.5px; color: var(--graphite); font-family: var(--mono);
              margin-top: .25rem; }

/* ── Hero — asymmetric: the headline holds the left, the plate bleeds right ── */
.hero { padding: .2rem 0 1.1rem; }
.hero h1 {
    font-size: clamp(40px, 6.6vw, 78px);
    font-variation-settings: 'wdth' 118;
    font-weight: 800;
    line-height: .93;
    letter-spacing: -.038em;
    margin: 0;
    max-width: 14ch;
    /* The one page-load moment: the headline is struck onto the sheet. */
    animation: strike 900ms cubic-bezier(.16,.84,.34,1) both;
}
@keyframes strike {
    from { clip-path: inset(0 100% 0 0); }
    to   { clip-path: inset(0 0 0 0); }
}
.hero .hero-sub {
    color: var(--graphite); font-size: 16px; line-height: 1.55;
    margin: 1.1rem 0 0; max-width: 54ch;
}
@media (prefers-reduced-motion: reduce) { .hero h1 { animation: none; } }

/* The 3D plate sits in the right column and runs past the container edge. */
.st-key-plate { border-left: 1px solid var(--ink); padding-left: 16px;
                margin-right: -3.4rem; }
@media (max-width: 900px) { .st-key-plate { margin-right: 0; border-left: none;
                                            padding-left: 0; } }

/* ── Sidebar masthead ── */
.side-brand { margin: .1rem 0 .2rem; }
.side-title { font-family: var(--sans); font-variation-settings: 'wdth' 118;
              font-weight: 800; font-size: 17px; line-height: 1.05;
              letter-spacing: -.03em; color: var(--ink); }
.side-sub { font-size: 12px; color: var(--graphite); margin-top: 4px;
            max-width: 26ch; line-height: 1.4; }
.side-head { font-family: var(--sans); font-weight: 700; font-size: 13px;
             color: var(--ink); border-bottom: 1px solid var(--ink);
             padding-bottom: .3rem; margin: 1.5rem 0 .7rem; }
.side-head:first-of-type { margin-top: .6rem; }

/* ── Buttons — struck rectangles, not pills ── */
.stButton button, .stDownloadButton button {
    font-family: var(--sans); font-weight: 600; font-size: 14px;
    border-radius: 0 !important; letter-spacing: -.01em;
    transition: transform .12s ease, box-shadow .12s ease, background .12s ease;
}
.stButton button[kind="primary"], .stDownloadButton button[kind="primary"] {
    background: var(--pen); color: var(--sheet); border: 1px solid var(--pen);
    box-shadow: var(--lift);
}
.stButton button[kind="primary"]:hover:enabled,
.stDownloadButton button[kind="primary"]:hover:enabled {
    background: #0d3660; border-color: #0d3660; color: var(--sheet);
    transform: translate(1px, 1px); box-shadow: 2px 2px 0 rgba(23,28,31,.09);
}
.stButton button[kind="primary"]:disabled {
    background: transparent; color: var(--graphite);
    border: 1px dashed var(--rule); box-shadow: none;
}
.stButton button[kind="secondary"], .stDownloadButton button[kind="secondary"] {
    background: var(--sheet); border: 1px solid var(--ink); color: var(--ink);
    box-shadow: var(--lift);
}
.stButton button[kind="secondary"]:hover:enabled,
.stDownloadButton button[kind="secondary"]:hover:enabled {
    background: var(--stock); color: var(--ink); border-color: var(--ink);
    transform: translate(1px, 1px); box-shadow: 2px 2px 0 rgba(23,28,31,.09);
}
:focus-visible { outline: 2px solid var(--pen) !important; outline-offset: 2px; }
.stButton button:focus-visible, .stDownloadButton button:focus-visible {
    outline: 2px solid var(--pen) !important; outline-offset: 3px;
}

/* ── Tabs — an index strip ruled off the content below it ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0; background: transparent; border-bottom: 1px solid var(--ink);
    padding: 0; overflow-x: auto;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 0; padding: 5px 15px; background: transparent;
    border-right: 1px solid var(--rule-faint);
}
.stTabs [data-baseweb="tab"] p { font-size: 14px; font-weight: 600;
                                 color: var(--graphite); letter-spacing: -.01em; }
.stTabs [data-baseweb="tab"]:hover p { color: var(--ink); }
.stTabs [aria-selected="true"] { background: var(--ink) !important; }
.stTabs [aria-selected="true"] p { color: var(--sheet) !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.3rem; }

/* ── Inputs ── */
[data-testid="stFileUploaderDropzone"] {
    background: var(--stock); border: 1px dashed var(--graphite); border-radius: 0;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--pen);
                                               background: var(--sheet); }
.stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {
    border-radius: 0 !important; border-color: var(--rule) !important;
    background: var(--stock) !important;
}
.stTextInput input:focus, .stTextArea textarea:focus { border-color: var(--pen) !important; }

/* ── Gauge strip — one ruled band, not a row of identical cards ── */
.gauge { border-top: 2px solid var(--ink); border-bottom: 1px solid var(--ink);
         padding: .75rem 0 .7rem; height: 100%;
         /* A cell with no sub-label must still rule out on the same line as
            one that has it, or the strip stops reading as a single band. */
         min-height: 96px; }
.gauge .v { font-family: var(--mono); font-size: 27px; font-weight: 500;
            line-height: 1.05; letter-spacing: -.03em; color: var(--ink);
            overflow-wrap: anywhere; }
.gauge.long .v   { font-size: 19px; letter-spacing: -.02em; }
.gauge.longer .v { font-size: 14.5px; letter-spacing: -.01em; line-height: 1.2; }
.gauge .k { font-size: 12.5px; color: var(--graphite); margin-top: .4rem; }
.gauge .s { font-family: var(--mono); font-size: 11px; color: var(--graphite);
            margin-top: 2px; }
.gauge.flag { border-top-color: var(--risk); }
.gauge.flag .v { color: var(--risk); }

/* st.metric picks up the same instrument readout */
[data-testid="stMetric"] { background: transparent; border: none;
                           border-top: 2px solid var(--ink);
                           border-bottom: 1px solid var(--ink);
                           border-radius: 0; padding: .7rem 0; }
[data-testid="stMetricValue"] { font-family: var(--mono) !important; font-weight: 500;
                                color: var(--ink); letter-spacing: -.03em; }
[data-testid="stMetricLabel"] p { font-size: 12.5px; color: var(--graphite);
                                  text-transform: none; letter-spacing: 0; }

/* ── Plates — sheets laid on the table, square, hard-shadowed ── */
[data-testid="stExpander"] details {
    background: var(--sheet); border: 1px solid var(--ink) !important;
    border-radius: 0; box-shadow: var(--lift);
}
[data-testid="stExpander"] summary { font-weight: 600; font-size: 14px; }
[data-testid="stExpander"] summary:hover { color: var(--pen); }
[data-testid="stCode"] pre, pre {
    background: #e4e3dc !important; border: 1px solid var(--rule);
    border-radius: 0; font-size: 12.5px;
}
[data-testid="stAlert"] { border-radius: 0; }
[data-testid="stAlertContainer"] {
    background: transparent !important; border-radius: 0;
    border-left: 3px solid var(--graphite);
    padding: .4rem .5rem .4rem 1rem; color: var(--ink) !important;
}
[data-testid="stAlertContainer"] p { color: inherit !important; font-size: 14.5px; }
[data-testid="stAlertContainer"] svg { fill: currentColor; }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) {
    border-left-color: var(--pen); color: var(--pen) !important;
}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]),
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) {
    border-left-color: var(--risk); color: var(--risk) !important;
}
[data-testid="stDataFrame"], [data-testid="stTable"] {
    border-radius: 0; box-shadow: var(--lift);
}

/* ── Stage ledger — the pipeline as numbered rows, because it IS a sequence ── */
.sc { display: flex; align-items: baseline; gap: 12px;
      padding: .5rem 0; border-bottom: 1px solid var(--rule-faint);
      font-size: 14px; color: var(--graphite); }
.sc .sc-num { font-family: var(--mono); font-size: 12px; color: var(--graphite);
              flex: none; width: 2.2em; }
.sc .nm { color: var(--ink); font-weight: 600; }
.sc .detail { margin-left: auto; font-family: var(--mono); font-size: 11.5px;
              color: var(--graphite); text-align: right; padding-left: 1rem; }
.sc.done  { border-left: 3px solid var(--pen); padding-left: .6rem; }
.sc.active { border-left: 3px solid var(--pen); padding-left: .6rem;
             background: rgba(18,70,126,.07); }
.sc.active .nm::after { content: " — running"; font-weight: 400;
                        color: var(--pen); font-size: 12.5px; }
.sc.skip  { border-left: 3px solid var(--rule); padding-left: .6rem; }
.sc.skip .nm { color: var(--graphite); font-weight: 400; }
.sc.err   { border-left: 3px solid var(--risk); padding-left: .6rem; }
.sc.err .nm { color: var(--risk); }

/* ── Annotations — a reviewer's marginal note, not another rounded card ── */
.ic, .rc, .wc {
    border-left: 3px solid var(--rule); padding: .3rem 0 .3rem 1rem;
    margin: 0 0 .75rem; font-size: 15px; line-height: 1.6; max-width: 74ch;
    color: var(--ink);
}
.ic { border-left-color: var(--graphite); }
.rc { border-left-color: var(--pen); }
.wc { border-left-color: var(--risk); color: var(--risk); }
.ic .mk, .rc .mk, .wc .mk {
    font-family: var(--mono); font-size: 11px; color: var(--graphite);
    display: block; margin-bottom: 1px;
}
.rc .mk { color: var(--pen); }
.wc .mk { color: var(--risk); }

/* Agent reasoning — a written finding, given room to read */
.reason { background: var(--sheet); border: 1px solid var(--ink);
          box-shadow: var(--lift); padding: 1.2rem 1.4rem;
          font-size: 15px; color: var(--ink); line-height: 1.72; max-width: 72ch; }

/* Run banner — a strip of tape across the sheet */
.run-banner { border-top: 2px solid var(--pen); border-bottom: 1px solid var(--pen);
              background: rgba(18,70,126,.07); padding: .7rem 1rem;
              color: var(--pen); font-size: 14px; font-weight: 600;
              margin: .4rem 0 1.2rem; }
.run-banner .sub { display: block; font-weight: 400; color: var(--graphite);
                   font-size: 13px; margin-top: 2px; }

/* ── Empty state — a blank sheet with its own instruction ── */
.empty { padding: 3.5rem 0 4rem; max-width: 58ch; }
.empty h2 { font-size: clamp(30px, 4.4vw, 46px); font-variation-settings: 'wdth' 118;
            font-weight: 800; line-height: .98; letter-spacing: -.035em;
            margin: 0 0 1rem; }
.empty p { color: var(--graphite); font-size: 16px; line-height: 1.62; margin: 0; }
.empty .steps { display: flex; flex-wrap: wrap; margin-top: 2rem;
                border-top: 1px solid var(--ink); }
.empty .steps div { font-family: var(--mono); font-size: 12px; color: var(--graphite);
                    padding: .5rem .9rem .5rem 0; margin-right: .9rem;
                    border-right: 1px solid var(--rule-faint); }
.empty .steps div:last-child { border-right: none; }
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
    ("1", "Dataset Ingestion"),
    ("2", "Initial Reasoning"),
    ("3", "Tool Execution"),
    ("4", "Result Interpretation"),
    ("5", "Iterative Refinement"),
    ("6", "RLM Decomposition"),
    ("7", "Report Generation"),
]

# Shared Vega-Lite config — charts are plotted on the sheet in the same two
# inks as the rest of the console (DESIGN.md). Blue is the measured series;
# red is reserved for the series that carries risk.
PLOT_INK = "#171c1f"
PLOT_GRAPHITE = "#54585b"
PLOT_RULE = "#c8c6bc"
PEN_BLUE = "#12467e"
PEN_RED = "#b5271a"

VEGA_PLOT_CONFIG = {
    "font": "Archivo, 'Segoe UI', sans-serif",
    "axis": {
        "labelColor": PLOT_GRAPHITE,
        "titleColor": PLOT_GRAPHITE,
        "gridColor": PLOT_RULE,
        "gridDash": [2, 3],
        "domainColor": PLOT_INK,
        "tickColor": PLOT_INK,
        "labelFont": "IBM Plex Mono, monospace",
        "labelFontSize": 11,
        "titleFont": "Archivo, sans-serif",
        "titleFontWeight": 600,
    },
    "legend": {
        "labelColor": PLOT_INK,
        "titleColor": PLOT_GRAPHITE,
        "labelFont": "Archivo, sans-serif",
        "titleFont": "Archivo, sans-serif",
        "symbolType": "square",
    },
    "view": {"stroke": "transparent"},
    "range": {"category": [PEN_BLUE, PEN_RED, "#7a8b99",
                           "#c08a2e", "#3f6f5b", "#8e6e9e"]},
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
        st.session_state[k] = _DEFAULTS[k]


def _set_stage(num: str, status: str, detail: str = "") -> None:
    log: list[tuple[str, str, str]] = [
        e for e in st.session_state["stage_log"] if e[0] != num
    ]
    log.append((num, status, detail))
    st.session_state["stage_log"] = log


def _stage_card(num: str, name: str,
                status: str, detail: str = "") -> str:
    """One row of the stage ledger. Numbered: the pipeline is a real sequence."""
    cls = {"done": "done", "active": "active",
           "skipped": "skip", "error": "err"}.get(status, "")
    det = f'<span class="detail">{detail}</span>' if detail else ""
    return (f'<div class="sc {cls}">'
            f'<span class="sc-num">{num.zfill(2)}</span>'
            f'<span class="nm">{name}</span>{det}</div>')


def _datum(cells: list[tuple[str, str]]) -> str:
    """A ruled measurement bar. Each reading gets its own cell and hairline."""
    body = "".join(
        f'<div class="cell"><div class="k">{k}</div><div class="v">{v}</div></div>'
        for k, v in cells
    )
    return f'<div class="datum">{body}</div>'


def _draw_pipeline_rig(slot: Any) -> list[Any]:
    """Render the 3D pipeline rig into `slot`; return the stages it drew.

    Called from two places — the normal position at the end of the script, and
    just before `st.stop()` on an aborted run, since otherwise the hero would
    keep a blank gap where the rig should be.

    Imported lazily to match how `src/` is loaded in this file: after ROOT
    lands on sys.path.
    """
    from ui.pipeline_3d import Stage, StageStatus
    from ui.pipeline_3d import render as render_pipeline

    log = {n: (s, d) for n, s, d in st.session_state["stage_log"]}
    stages = [
        Stage(
            num=num,
            name=name,
            # session_state is untyped; _set_stage only writes StageStatus values.
            status=cast(StageStatus, log.get(num, ("pending", ""))[0]),
            detail=log.get(num, ("pending", ""))[1],
        )
        for num, name in STAGE_DEFS
    ]
    with slot.container():
        render_pipeline(stages)
    return stages


def _gauge(label: str, value: str, sub: str = "", *, flag: bool = False) -> str:
    """One cell of the instrument readout: the number leads, the label follows.

    `flag` switches the cell to the risk pen — reserved for a measurement the
    reader should not trust, never used for emphasis.

    Values arrive at any length (a percentage, or a model name), so the type
    steps down rather than wrapping mid-word and pulling the strip's rules out
    of alignment.
    """
    fit = "" if len(value) <= 11 else " long" if len(value) <= 18 else " longer"
    sub_html = f'<div class="s">{sub}</div>' if sub else ""
    return (f'<div class="gauge{fit}{" flag" if flag else ""}">'
            f'<div class="v">{value}</div>'
            f'<div class="k">{label}</div>{sub_html}</div>')


def _gap_is_risky(gap: float) -> bool:
    """A train-test gap above 10 points means the model memorised the split."""
    return gap >= 0.10


def _section(title: str, note: str = "", level: str = "h3") -> None:
    """Section head: the title sits on its own rule, with an optional mono note.

    No tracked-caps label floats above it — the rule is the structure.
    """
    note_html = f'<div class="note">{note}</div>' if note else ""
    st.markdown(
        f'<div class="sect"><{level}>{title}</{level}>{note_html}</div>',
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
            out = r.get("output")
            return out if isinstance(out, dict) else {}
    return None


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<div class="side-brand">'
        '<div class="side-title">Agentic Data Analysis</div>'
        '<div class="side-sub">Set up the run here. Results are drawn on '
        'the sheet.</div></div>',
        unsafe_allow_html=True,
    )

    # ── Upload ────────────────────────────────────────────────────────────────
    st.markdown('<div class="side-head">Dataset</div>', unsafe_allow_html=True)
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
                st.error(f"Upload rejected. {_ve}")
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
    st.markdown('<div class="side-head">LLM Provider</div>', unsafe_allow_html=True)
    provider = st.selectbox(
        "Provider",
        ["openai", "anthropic", "gemini", "openrouter", "nvidia", "local"],
        format_func=lambda p: "Local / offline" if p == "local" else p,
    )

    NVIDIA_MODELS = [
        "openai/gpt-oss-120b",
        "meta/llama-3.1-70b-instruct",
        "meta/llama-3.3-70b-instruct",
        "mistralai/mistral-large-2-instruct",
        "microsoft/phi-3-medium-128k-instruct",
        "google/gemma-2-27b-it",
        "deepseek-ai/deepseek-r1",
    ]
    GEMINI_MODELS = ["gemini-flash-latest", "gemini-pro-latest", "gemini-2.5-flash", "gemini-2.5-pro"]
    LOCAL_MODELS = ["llama3.1", "llama3.2", "mistral", "qwen2.5", "deepseek-r1", "phi4"]

    if provider == "openai":
        model_list = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
        key_ph     = "sk-..."
    elif provider == "anthropic":
        model_list = ["claude-sonnet-4-6", "claude-opus-4-8",
                      "claude-haiku-4-5-20251001"]
        key_ph     = "sk-ant-..."
    elif provider == "gemini":
        model_list = GEMINI_MODELS
        key_ph     = "from aistudio.google.com/apikey"
    elif provider == "nvidia":
        model_list = NVIDIA_MODELS
        key_ph     = "nvapi-..."
    elif provider == "local":
        model_list = LOCAL_MODELS
        key_ph     = "usually not required"
    else:
        model_list = OR_MODELS
        key_ph     = "sk-or-..."

    model_sel = st.selectbox("Model", model_list)
    if provider in ("openrouter", "nvidia", "local"):
        custom_m = st.text_input(
            "Custom model string (overrides above)",
            placeholder={
                "openrouter": "e.g. cohere/command-r-plus",
                "nvidia": "e.g. nvidia/llama-3.1-nemotron-70b-instruct",
                "local": "e.g. the exact tag your server has pulled/loaded",
            }[provider],
        )
        final_model = custom_m.strip() if custom_m.strip() else model_sel
    else:
        final_model = model_sel

    local_base_url = ""
    if provider == "local":
        local_base_url = st.text_input(
            "Server URL (OpenAI-compatible)",
            value="http://localhost:11434/v1",
            help="Works with Ollama, LM Studio, vLLM, llama.cpp server, "
                 "text-generation-webui, etc. Must be reachable from this "
                 "machine — no data leaves it.",
        )

    _key_label = {
        "openai": "OpenAI",
        "anthropic": "Anthropic",
        "gemini": "Gemini",
        "openrouter": "OpenRouter",
        "nvidia": "NVIDIA",
        "local": "Local server",
    }.get(provider, provider)
    api_key = st.text_input(
        f"{_key_label} API Key" + (" (optional)" if provider == "local" else ""),
        type="password",
        placeholder=key_ph,
    )

    # ── Analysis Settings ─────────────────────────────────────────────────────
    st.markdown('<div class="side-head">Analysis Settings</div>', unsafe_allow_html=True)
    max_iter   = st.slider("Max iterations", 3, 25, 10)
    enable_rlm = st.toggle("Enable RLM decomposition (Stage 6)", value=True)

    st.markdown('<div class="side-head">Anti-Overfitting</div>', unsafe_allow_html=True)
    max_depth = st.slider("Max tree depth", 2, 15, 6,
                          help="Lower = less overfitting for tree-based models")
    test_pct  = st.slider("Test split %", 10, 40, 20, step=5)
    n_cv      = st.slider("CV folds (k)", 3, 10, 5)

    st.divider()

    # ── Buttons ───────────────────────────────────────────────────────────────
    has_file = st.session_state["preview_df"] is not None
    has_key  = bool(api_key.strip()) or provider == "local"
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
        st.caption("Upload a CSV or Excel file to enable the run.")
    elif not has_key:
        st.caption("Add your API key to enable the run.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA — header
# ══════════════════════════════════════════════════════════════════════════════
hero_text, hero_plate = st.columns([0.46, 0.54], gap="large",
                                   vertical_alignment="center")

with hero_text:
    st.markdown(
        '<div class="hero">'
        '<h1>Every finding, measured twice.</h1>'
        '<p class="hero-sub">Upload a dataset. The agent plans the analysis, '
        'runs the tests, trains the models, then reports what generalises — '
        'and what only fits.</p></div>',
        unsafe_allow_html=True,
    )

# The plate: a live technical drawing of the run, ruled off the headline and
# running past the container edge. The pipeline executes further down this same
# script pass, so the drawing is filled into this placeholder afterwards — that
# way it shows the state of the run that just happened.
with hero_plate.container(key="plate"):
    pipeline_slot = st.empty()

# The datum line under the hero carries the run's readings, filled at the same
# time as the plate.
datum_slot = st.empty()


# ══════════════════════════════════════════════════════════════════════════════
# DATASET PREVIEW — always visible once a file is loaded
# ══════════════════════════════════════════════════════════════════════════════
preview_df: pd.DataFrame | None = st.session_state["preview_df"]

if preview_df is not None and not st.session_state["analysis_done"]:
    _section("Dataset preview", st.session_state["preview_name"])
    _miss_cells = int(preview_df.isnull().sum().sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(_gauge("Rows", f"{len(preview_df):,}"), unsafe_allow_html=True)
    c2.markdown(_gauge("Columns", str(len(preview_df.columns))),
                unsafe_allow_html=True)
    c3.markdown(_gauge("Missing cells", f"{_miss_cells:,}",
                       flag=_miss_cells > 0), unsafe_allow_html=True)
    c4.markdown(
        _gauge("Numeric columns",
               str(len(preview_df.select_dtypes(include="number").columns))),
        unsafe_allow_html=True,
    )

    with st.expander("First 10 rows", expanded=True):
        st.dataframe(_safe_df(preview_df.head(10)), width='stretch')

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("#### Column types and missing values")
        dtype_df = pd.DataFrame(
            [(c, str(t), int(preview_df[c].isnull().sum()))
             for c, t in preview_df.dtypes.items()],
            columns=["Column", "Type", "Missing"],
        )
        st.dataframe(_safe_df(dtype_df), width='stretch', height=200)
    with col_r:
        st.markdown("#### Descriptive statistics")
        st.dataframe(_safe_df(preview_df.describe()), width='stretch', height=200)
    st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE — runs synchronously inside st.status() on button click
# ══════════════════════════════════════════════════════════════════════════════
if run_clicked:
    _reset_pipeline()
    for num, _ in STAGE_DEFS:
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
        "gemini":     lambda: os.environ.__setitem__("GEMINI_API_KEY",     api_key.strip()),
        "openrouter": lambda: os.environ.__setitem__("OPENROUTER_API_KEY", api_key.strip()),
        "nvidia":     lambda: os.environ.__setitem__("NVIDIA_API_KEY",     api_key.strip()),
        "local":      lambda: (
            os.environ.__setitem__("LOCAL_LLM_API_KEY", api_key.strip() or "not-needed"),
            os.environ.__setitem__("LOCAL_LLM_BASE_URL", local_base_url.strip() or "http://localhost:11434/v1"),
        ),
    }[provider]()

    # ── Spinner placeholder — replaced after run completes ───────────────
    _spinner_ph = st.empty()
    _spinner_ph.markdown(
        '<div class="run-banner">Running the analysis'
        '<span class="sub">Usually 1–3 minutes, depending on the dataset and '
        'the model.</span></div>',
        unsafe_allow_html=True,
    )

    # ── Collect progress lines into session state (no st.write during run) ─
    _progress_lines: list[str] = []

    def _upd(num: str, s: str, detail: str = "") -> None:
        _set_stage(num, s, detail)
        _ico = {"done": "[done]", "active": "[run ]", "error": "[fail]",
                "skipped": "[skip]"}.get(s, "[    ]")
        _nm  = next(n for no, n in STAGE_DEFS if no == num)
        _progress_lines.append(f"{_ico} Stage {num}: {_nm}" + (f"  {detail}" if detail else ""))

    # ── LLM preflight — fail fast with the REAL error instead of running
    #    the whole pipeline on the deterministic fallback ──────────────────
    from src.core.controller import AgentController, LLMClient

    _ok, _ping_err = LLMClient().ping()
    if not _ok:
        _spinner_ph.empty()
        _set_stage("2", "error", "LLM unreachable")
        st.session_state["analysis_error"] = _ping_err
        st.error(
            f"Could not reach the model, so the analysis did not start. "
            f"Provider `{provider}`, model `{final_model}`."
        )
        st.code(_ping_err, language=None)
        st.info(
            "Check that the model ID exists on this provider, that the API key "
            "is valid, and that the account has credits. Then run it again."
        )
        _draw_pipeline_rig(pipeline_slot)  # the hero slot must not stay empty
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
            _progress_lines.append(f"       {'ok  ' if status=='success' else '... '}{detail}")

        def _on_iter(iteration: int, stage: str) -> None:
            if "stage2" in stage:
                _set_stage("2", "active", f"iter {iteration} — LLM reasoning…")
                _progress_lines.append(f"[run ] Iteration {iteration}: model reasoning")
            elif "stage4" in stage or "stage5" in stage:
                _set_stage("4", "active", f"iter {iteration} — interpreting results…")
                _set_stage("5", "active", f"iter {iteration} — refining plan…")
                _progress_lines.append(f"[run ] Iteration {iteration}: interpreting and refining")

        agent.on_step_callback      = _on_step
        agent.on_iteration_callback = _on_iter

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
        st.error("The run stopped on an error. The traceback is below.")
        st.code(err, language="python")


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE RIG — 3D hero, plus the same stages as text
# ══════════════════════════════════════════════════════════════════════════════
stages_3d = _draw_pipeline_rig(pipeline_slot)

# The datum line: the same run state as the drawing, in numbers.
_done_n = sum(1 for st_ in stages_3d if st_.status == "done")
_errored = next((st_ for st_ in stages_3d if st_.status == "error"), None)
_running = next((st_ for st_ in stages_3d if st_.status == "active"), None)
if _errored is not None:
    _run_state = f"Stopped at stage {_errored.num}"
elif _running is not None:
    _run_state = f"Running stage {_running.num}"
elif st.session_state["analysis_done"]:
    _run_state = "Complete"
else:
    _run_state = "Not started"

datum_slot.markdown(
    _datum([
        ("Run", _run_state),
        ("Stages finished", f"{_done_n} of {len(stages_3d)}"),
        ("Dataset", st.session_state["preview_name"] or "none loaded"),
        ("Model", final_model),
    ]),
    unsafe_allow_html=True,
)

# Text mirror of the drawing — the accessible readout, and the fallback when a
# browser can't do WebGL.
if st.session_state["stage_log"]:
    with st.expander("Stage ledger", expanded=False):
        for stage in stages_3d:
            st.markdown(
                _stage_card(stage.num, stage.name, stage.status, stage.detail),
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
    _section("What the agent found", level="h2")

    if st.session_state.get("llm_warning"):
        st.warning(st.session_state["llm_warning"])

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
        m1.markdown(_gauge("Best model", best_model), unsafe_allow_html=True)
        m2.markdown(_gauge("Cross-validated score", best_cv,
                           "mean across folds"), unsafe_allow_html=True)
        m3.markdown(
            _gauge("Train–test gap", best_gap_str, "how much it memorised",
                   flag=gap_val is not None and _gap_is_risky(gap_val)),
            unsafe_allow_html=True,
        )
        m4.markdown(_gauge("Outliers", outlier_pct, "of all rows"),
                    unsafe_allow_html=True)
        m5.markdown(_gauge("Task", task_type), unsafe_allow_html=True)

        for _w in (train_out.get("overfit_warnings", []) if train_out else []):
            st.markdown(f'<div class="wc"><span class="mk">Risk</span>{_w}</div>',
                        unsafe_allow_html=True)

        # Model comparison — interactive Vega-Lite grouped bars
        if train_out:
            _mt_map2 = train_out.get("models_trained", {})
            if _mt_map2:
                st.markdown("#### How each model scored")
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
                        "mark": {"type": "bar"},
                        "height": 300,
                        "background": "transparent",
                        "config": VEGA_PLOT_CONFIG,
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
                                    "range": ["#8aa6c2", PEN_BLUE, PLOT_INK],
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
                st.markdown("#### Strongest correlations")
                _corr_df = pd.DataFrame(
                    [{"pair": f"{r['col_a']} ↔ {r['col_b']}",
                      "correlation": r["correlation"]} for r in _top]
                )
                st.vega_lite_chart(
                    _corr_df,
                    {
                        "mark": {"type": "bar"},
                        "height": max(160, len(_top) * 30),
                        "background": "transparent",
                        "config": VEGA_PLOT_CONFIG,
                        "encoding": {
                            "y": {"field": "pair", "type": "nominal",
                                  "sort": "-x", "title": None},
                            "x": {"field": "correlation", "type": "quantitative",
                                  "scale": {"domain": [-1.1, 1.1]},
                                  "title": "Correlation coefficient"},
                            "color": {
                                "condition": {"test": "datum.correlation >= 0",
                                              "value": PEN_BLUE},
                                "value": PLOT_INK,
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
                _spec.setdefault("config", VEGA_PLOT_CONFIG)
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
            _dupes = int(prof.get("duplicate_rows", 0))
            p1, p2, p3, p4 = st.columns(4)
            p1.markdown(_gauge("Quality score", f"{_q}/100", "out of 100",
                               flag=_q < 60), unsafe_allow_html=True)
            p2.markdown(_gauge("Duplicate rows", f"{_dupes:,}", flag=_dupes > 0),
                        unsafe_allow_html=True)
            p3.markdown(_gauge("Memory", f"{prof.get('memory_mb', 0)} MB"),
                        unsafe_allow_html=True)
            p4.markdown(_gauge("Columns profiled", str(prof.get("column_count", 0))),
                        unsafe_allow_html=True)

            for _w in prof.get("warnings", []):
                st.markdown(f'<div class="wc"><span class="mk">Risk</span>{_w}</div>',
                            unsafe_allow_html=True)

            st.markdown("#### What each column holds")
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
                "Note":    ("target" if col == meta.target_column else "")
                           + (" high cardinality"
                              if col in meta.high_cardinality_cols else ""),
            } for col, dtype in meta.columns.items()]
            st.dataframe(_safe_df(pd.DataFrame(_col_rows)), width='stretch')

            if meta.missing_values:
                st.markdown("#### Missing values by column")
                _miss = pd.DataFrame(
                    [(c, v) for c, v in meta.missing_values.items()],
                    columns=["Column", "Count"],
                ).sort_values("Count", ascending=False)
                st.bar_chart(_safe_df(_miss.set_index("Column")))

            if meta.class_balance:
                st.markdown("#### Class balance")
                _cb = pd.DataFrame(
                    [(str(k), v) for k, v in meta.class_balance.items()],
                    columns=["Class", "Count"],
                )
                st.bar_chart(_safe_df(_cb.set_index("Class")))

        if clean_out:
            st.markdown("#### What cleaning changed")
            _c1, _c2, _c3 = st.columns(3)
            _c1.metric("Strategy",       clean_out.get("strategy_used", "—"))
            _c2.metric("Missing before", clean_out.get("missing_before", "—"))
            _c3.metric("Missing after",  clean_out.get("missing_after", "—"))

        if outlier_out:
            st.markdown("#### Outliers found")
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
                f"**Task:** {_task2} &nbsp;|&nbsp; **Best:** `{_best2}` "
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
                    "Model": f"{_nm} (best)" if _nm == _best2 else _nm,
                    f"Train {_pk2}": f"{_tr2.get(_pk2,0)*100:.1f}%",
                    f"Test {_pk2}":  f"{_te2.get(_pk2,0)*100:.1f}%",
                    "CV mean":  f"{_m.get('cv_mean',0)*100:.1f}%",
                    "CV std":   f"±{_m.get('cv_std',0)*100:.1f}%",
                    "Gap": (f"{_g*100:.1f}%" + (" risk" if _g and _g > .10 else "")
                            if _g is not None else "—"),
                    _sk2.replace("_", " "): (
                        f"{_te2.get(_sk2,0)*100:.1f}%"
                        if _sk2 != "rmse" else f"{_te2.get(_sk2,0):.4f}"
                    ),
                })
            st.dataframe(_safe_df(pd.DataFrame(_rows)), width='stretch')

            _warn2 = train_out.get("overfit_warnings", [])
            if _warn2:
                st.markdown("#### Where these models may not hold")
                for _w2 in _warn2:
                    st.markdown(
                        f'<div class="wc"><span class="mk">Risk</span>{_w2}</div>',
                        unsafe_allow_html=True)
            else:
                st.success("Every model’s train–test gap stayed within range.")

        if eval_out:
            st.markdown("#### Per-class performance")
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
            st.markdown("#### Statistical test")
            _c1, _c2, _c3 = st.columns(3)
            _c1.metric("Test",        stat_out.get("test_name", "—"))
            _c2.metric("p-value",     f"{stat_out.get('p_value', 0):.4f}")
            _c3.metric("Significant",
                       "Yes" if stat_out.get("significant") else "No")
            st.info(stat_out.get("interpretation", ""))

    # ── Insights ──────────────────────────────────────────────────────────────
    with tab_ins:
        if report.get("llm_fallback"):
            st.warning(
                "The model became unreachable mid-run, so these findings were "
                "assembled from the tool outputs alone. The error is shown "
                "above. Fix it and run again to get the written interpretation."
            )
        if report.get("reasoning"):
            st.markdown("#### How the agent read the data")
            st.markdown(
                f'<div class="reason">{report["reasoning"]}</div>',
                unsafe_allow_html=True,
            )
        if report.get("insights"):
            st.markdown("#### What it found")
        for _i, _ins in enumerate(report.get("insights", []), start=1):
            st.markdown(
                f'<div class="ic"><span class="mk">{_i:02d}</span>{_ins}</div>',
                unsafe_allow_html=True)
        if report.get("recommendations"):
            st.markdown("#### What to do next")
        for _rec in report.get("recommendations", []):
            st.markdown(
                f'<div class="rc"><span class="mk">Do</span>{_rec}</div>',
                unsafe_allow_html=True)
        if report.get("key_metrics"):
            st.markdown("#### Key numbers")
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
                    "Download the shareable HTML report",
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
                for _rep_f in sorted(_rdir2.iterdir()):
                    st.download_button(
                        f"Download {_rep_f.name}", _rep_f.read_bytes(), _rep_f.name,
                        mime=_mimes.get(_rep_f.suffix, "application/octet-stream"),
                        key=f"dlr_{_rep_f.name}",
                    )

            _mdir = _out / "models"
            if _mdir.exists() and any(_mdir.iterdir()):
                st.markdown("**Trained Models (.pkl)**")
                for _mdl_f in sorted(_mdir.iterdir()):
                    st.download_button(
                        f"Download {_mdl_f.name}", _mdl_f.read_bytes(), _mdl_f.name,
                        mime="application/octet-stream",
                        key=f"dlm_{_mdl_f.name}",
                    )

            _vdir = _out / "visualizations"
            if _vdir.exists() and any(_vdir.iterdir()):
                st.markdown("**Visualizations**")
                _vcols = st.columns(2)
                for _i, _viz_f in enumerate(sorted(_vdir.glob("*.png"))):
                    with _vcols[_i % 2]:
                        st.image(str(_viz_f), caption=_viz_f.name,
                                 width='stretch')
                        st.download_button(
                            f"Download {_viz_f.name}", _viz_f.read_bytes(), _viz_f.name,
                            mime="image/png",
                            key=f"dlv_{_viz_f.name}",
                        )

            st.markdown("**Full JSON report**")
            st.download_button(
                "Download final_report.json",
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
        '<h2>Nothing on the table yet.</h2>'
        '<p>Drop a CSV or Excel file in the sidebar, say in plain English what '
        'you want to learn from it, and add your API key. The drawing above '
        'fills in as the run works through its seven stages.</p>'
        '<div class="steps">'
        '<div>01 Ingest</div><div>02 Reason</div><div>03 Execute</div>'
        '<div>04 Interpret</div><div>05 Refine</div><div>06 Decompose</div>'
        '<div>07 Report</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )
