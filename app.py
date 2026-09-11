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

import html
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

from src.core.security import ALLOWED_EXTENSIONS

# ── Project root on sys.path ─────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Agentic Data Analysis",
    page_icon="🧾",
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

    # Assign to the sub-module entries in sys.modules directly —
    # never traverse sys.modules["rich"].tree as an attribute chain.
    sys.modules["rich"].Console = _C                  # type: ignore[attr-defined]
    sys.modules["rich.console"].Console = _C          # type: ignore[attr-defined]
    sys.modules["rich.panel"].Panel = _P              # type: ignore[attr-defined]
    sys.modules["rich.tree"].Tree = _Tr               # type: ignore[attr-defined]
    sys.modules["rich.table"].Table = _T              # type: ignore[attr-defined]
    sys.modules["rich.progress"].Progress = _Pr       # type: ignore[attr-defined]
    sys.modules["rich.progress"].SpinnerColumn = _Sp  # type: ignore[attr-defined]
    sys.modules["rich.progress"].TextColumn = _Tx     # type: ignore[attr-defined]


_stub_rich()


def _inject_theme_css(theme: str = "day") -> None:
    """Inject dynamic Ledger CSS supporting Day and Night modes."""
    is_night = theme == "night"
    stock       = "#241c14" if is_night else "#f7eedd"
    sheet       = "#2f251a" if is_night else "#fffbf2"
    sheet_alt   = "#3a2e1f" if is_night else "#f1e4cb"
    ink         = "#f3e9d8" if is_night else "#3a2b1e"
    graphite    = "#b8a688" if is_night else "#8a7660"
    pen         = "#f0a24a" if is_night else "#a34f20"
    pen_hover   = "#ffb86b" if is_night else "#7e3d18"
    risk        = "#e2685a" if is_night else "#a33526"
    accent      = "#d99a4e" if is_night else "#e08a3e"
    positive    = "#7fb77e" if is_night else "#5b8c5a"
    rule        = "#4a3c28" if is_night else "#e4d4bc"
    rule_faint  = "#3a2e1f" if is_night else "#eee3cb"
    lift        = "0 4px 18px rgba(0,0,0,.35)" if is_night else "0 4px 14px rgba(58,43,30,.14)"
    lift_sm     = "0 2px 8px rgba(0,0,0,.3)" if is_night else "0 2px 8px rgba(58,43,30,.10)"
    code_bg     = "#2a2015" if is_night else "#f1e4cb"

    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Baloo+2:wght@500;600;700;800&family=Mukta:wght@400;500;600;700&display=swap');

:root {{
    --stock:       {stock};
    --sheet:       {sheet};
    --sheet-alt:   {sheet_alt};
    --ink:         {ink};
    --graphite:    {graphite};
    --pen:         {pen};
    --pen-hover:   {pen_hover};
    --risk:        {risk};
    --accent:      {accent};
    --positive:    {positive};
    --rule:        {rule};
    --rule-faint:  {rule_faint};
    --lift:        {lift};
    --lift-sm:     {lift_sm};
    --code-bg:     {code_bg};
    --radius:      14px;
    --radius-pill: 999px;

    --sans:    'Mukta', ui-sans-serif, 'Segoe UI', system-ui, sans-serif;
    --heading: 'Baloo 2', 'Mukta', ui-sans-serif, sans-serif;
    --mono:    'Cascadia Code', Consolas, ui-monospace, monospace;
}}

/* ── The page ── */
#MainMenu, footer, .stAppDeployButton {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.stApp {{
    background-color: var(--stock);
    background-image: none;
    transition: background-color 0.2s ease;
}}
.block-container {{ max-width: 1180px; padding-top: 2.2rem; }}
html, body, .stApp, [class*="css"] {{ font-family: var(--sans); color: var(--ink); }}
hr {{ border: none; border-top: 1px solid var(--rule) !important; }}
a {{ color: var(--pen) !important; text-underline-offset: 3px; font-weight: 600; }}

section[data-testid="stSidebar"] {{
    background: var(--sheet);
    border-right: 1px solid var(--rule);
    background-image: none;
}}
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] label p {{ font-size: 13.5px; color: var(--graphite); font-weight: 600; }}

::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-thumb {{ background: var(--rule); border-radius: 6px; border: 2px solid var(--stock); }}
::-webkit-scrollbar-track {{ background: transparent; }}

/* ── Type: warm & rounded ── */
h1, h2, h3, h4, h5, h6 {{
    font-family: var(--heading) !important;
    letter-spacing: -.01em;
    color: var(--ink);
}}
h1 {{ font-weight: 800 !important; }}
h2 {{ font-weight: 700 !important; font-size: 27px !important; line-height: 1.15; }}
h3 {{ font-weight: 700 !important; font-size: 19px !important; }}
h4 {{ font-weight: 700 !important; font-size: 15.5px !important; letter-spacing: 0; }}
.stMarkdown p, .stMarkdown li {{ font-size: 15.5px; line-height: 1.65; max-width: 72ch; }}
code, kbd, pre, .stCode {{ font-family: var(--mono) !important; }}
[data-testid="stMetricValue"] {{ font-family: var(--heading) !important; font-weight: 700; }}

/* ── Quick Facts bar ── */
.datum {{ display: flex; align-items: stretch; flex-wrap: wrap;
         background: var(--sheet); border: 1px solid var(--rule);
         border-radius: var(--radius); box-shadow: var(--lift-sm);
         margin: 0 0 1.5rem; overflow: hidden; }}
.datum .cell {{ padding: .7rem 1.2rem; margin-right: 0;
               border-right: 1px solid var(--rule-faint); }}
.datum .cell:last-child {{ border-right: none; }}
.datum .k {{ font-size: 12px; color: var(--graphite); font-weight: 600; }}
.datum .v {{ font-family: var(--sans); font-size: 14px; font-weight: 700; color: var(--ink); margin-top: 2px; }}

/* ── Section head ── */
.sect {{ margin: 2.2rem 0 1.1rem; border-bottom: 2px solid var(--rule);
        padding-bottom: .5rem; }}
.sect:first-child {{ margin-top: .4rem; }}
.sect h2, .sect h3 {{ margin: 0; padding: 0; }}
.sect .note {{ font-size: 13px; color: var(--graphite); margin-top: .3rem; }}

/* ── Hero ── */
.hero {{ padding: .2rem 0 1rem; }}
.hero h1 {{
    font-size: clamp(36px, 5.6vw, 64px);
    font-weight: 800;
    line-height: 1.04;
    letter-spacing: -.02em;
    margin: 0;
    max-width: 15ch;
    animation: riseIn 650ms cubic-bezier(.16,.84,.34,1) both;
}}
@keyframes riseIn {{
    from {{ opacity: 0; transform: translateY(10px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
.hero .hero-sub {{
    color: var(--graphite); font-size: 16px; line-height: 1.6;
    margin: 1rem 0 0; max-width: 54ch;
}}
@media (prefers-reduced-motion: reduce) {{ .hero h1 {{ animation: none; }} }}

.st-key-plate {{ padding-left: 16px; margin-right: -2.8rem; }}
@media (max-width: 900px) {{ .st-key-plate {{ margin-right: 0; padding-left: 0; }} }}

/* ── Sidebar masthead & Theme controls ── */
.side-brand {{ margin: .1rem 0 .8rem; }}
.side-title {{ font-family: var(--heading); font-weight: 800; font-size: 18px;
              line-height: 1.15; color: var(--ink); }}
.side-sub {{ font-size: 12.5px; color: var(--graphite); margin-top: 4px;
            max-width: 26ch; line-height: 1.45; }}
.side-head {{ font-family: var(--sans); font-weight: 700; font-size: 12.5px;
             color: var(--graphite); margin: 1.5rem 0 .6rem; }}
.side-head:first-of-type {{ margin-top: .5rem; }}

/* ── Buttons ── */
.stButton button, .stDownloadButton button {{
    font-family: var(--sans); font-weight: 700; font-size: 14.5px;
    border-radius: var(--radius-pill) !important; letter-spacing: 0;
    transition: transform .12s ease, box-shadow .12s ease, background .12s ease;
}}
.stButton button[kind="primary"], .stDownloadButton button[kind="primary"] {{
    background: var(--pen); color: var(--sheet); border: none;
    box-shadow: var(--lift-sm);
}}
.stButton button[kind="primary"]:hover:enabled,
.stDownloadButton button[kind="primary"]:hover:enabled {{
    background: var(--pen-hover); color: var(--sheet);
    transform: translateY(-1px); box-shadow: var(--lift);
}}
.stButton button[kind="primary"]:disabled {{
    background: var(--sheet-alt); color: var(--graphite);
    border: 1px dashed var(--rule); box-shadow: none;
}}
.stButton button[kind="secondary"], .stDownloadButton button[kind="secondary"] {{
    background: var(--sheet); border: 1px solid var(--rule); color: var(--ink);
    box-shadow: var(--lift-sm);
}}
.stButton button[kind="secondary"]:hover:enabled,
.stDownloadButton button[kind="secondary"]:hover:enabled {{
    background: var(--sheet-alt); color: var(--ink); border-color: var(--pen);
    transform: translateY(-1px);
}}
:focus-visible {{ outline: 2px solid var(--pen) !important; outline-offset: 2px; }}
.stButton button:focus-visible, .stDownloadButton button:focus-visible {{
    outline: 2px solid var(--pen) !important; outline-offset: 3px;
}}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 6px; background: var(--sheet-alt); border-radius: var(--radius-pill);
    padding: 5px; overflow-x: auto; border: none;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: var(--radius-pill); padding: 8px 18px; background: transparent;
    border: none; transition: all 0.14s ease;
}}
.stTabs [data-baseweb="tab"] p {{ font-size: 14px; font-weight: 700;
                                 color: var(--graphite); letter-spacing: 0; }}
.stTabs [data-baseweb="tab"]:hover p {{ color: var(--ink); }}
.stTabs [aria-selected="true"] {{ background: var(--pen) !important; }}
.stTabs [aria-selected="true"] p {{ color: var(--sheet) !important; }}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display: none; }}
.stTabs [data-baseweb="tab-panel"] {{ padding-top: 1.5rem; }}

/* ── Inputs ── */
[data-testid="stFileUploaderDropzone"] {{
    background: var(--sheet-alt); border: 2px dashed var(--rule); border-radius: var(--radius);
}}
[data-testid="stFileUploaderDropzone"]:hover {{ border-color: var(--pen);
                                               background: var(--sheet); }}
.stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {{
    border-radius: 10px !important; border-color: var(--rule) !important;
    background: var(--sheet) !important; color: var(--ink) !important;
}}
.stTextInput input:focus, .stTextArea textarea:focus {{ border-color: var(--pen) !important; }}

/* ── Stat tile ── */
.gauge {{ background: var(--sheet); border: 1px solid var(--rule);
         border-radius: var(--radius); box-shadow: var(--lift-sm);
         padding: 1rem 1.1rem; height: 100%; min-height: 96px; position: relative; }}
.gauge .v {{ font-family: var(--heading); font-size: 26px; font-weight: 700;
            line-height: 1.1; letter-spacing: -.01em; color: var(--ink);
            overflow-wrap: anywhere; }}
.gauge.long .v   {{ font-size: 19px; }}
.gauge.longer .v {{ font-size: 14.5px; line-height: 1.25; }}
.gauge .k {{ font-size: 12.5px; color: var(--graphite); margin-top: .4rem; font-weight: 600; }}
.gauge .s {{ font-size: 11.5px; color: var(--graphite); margin-top: 2px; }}
.gauge.flag {{ border-color: var(--risk); background: color-mix(in srgb, var(--risk) 8%, var(--sheet)); }}
.gauge.flag .v {{ color: var(--risk); }}

/* ── Callout cards ── */
.defect-stamp {{
    border: 1px solid var(--risk);
    background: color-mix(in srgb, var(--risk) 8%, var(--sheet));
    border-radius: var(--radius);
    padding: 1.1rem 1.4rem;
    margin: 1.2rem 0;
    box-shadow: var(--lift-sm);
}}
.defect-stamp .stamp-tag {{
    font-family: var(--sans); font-size: 12px; font-weight: 700;
    color: var(--risk); display: block; margin-bottom: 4px;
}}
.defect-stamp .stamp-title {{
    font-family: var(--heading); font-size: 18px; font-weight: 800; color: var(--risk);
    margin-bottom: 6px;
}}
.defect-stamp .stamp-desc {{
    font-size: 14.5px; line-height: 1.58; color: var(--ink); max-width: 68ch;
}}

.cert-stamp {{
    border: 1px solid var(--pen);
    background: color-mix(in srgb, var(--pen) 8%, var(--sheet));
    border-radius: var(--radius);
    padding: 1.1rem 1.4rem;
    margin: 1.2rem 0;
    box-shadow: var(--lift-sm);
}}
.cert-stamp .stamp-tag {{
    font-family: var(--sans); font-size: 12px; font-weight: 700;
    color: var(--pen); display: block; margin-bottom: 4px;
}}
.cert-stamp .stamp-title {{
    font-family: var(--heading); font-size: 18px; font-weight: 800; color: var(--pen);
    margin-bottom: 6px;
}}
.cert-stamp .stamp-desc {{
    font-size: 14.5px; line-height: 1.58; color: var(--ink); max-width: 68ch;
}}

/* ── Team grid & cards ── */
.agent-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 14px;
    margin: 1.2rem 0;
}}
.agent-card {{
    background: var(--sheet);
    border: 1px solid var(--rule);
    border-radius: var(--radius);
    box-shadow: var(--lift-sm);
    padding: 1rem 1.1rem;
    transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
}}
.agent-card:hover {{
    transform: translateY(-2px);
    box-shadow: var(--lift);
    border-color: var(--pen);
}}
.agent-card.agent-active {{
    border-color: var(--pen);
    background: color-mix(in srgb, var(--pen) 6%, var(--sheet));
}}
.agent-card.agent-flagged {{
    border-color: var(--risk);
}}
.agent-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 8px;
    padding-bottom: 7px;
    border-bottom: 1px solid var(--rule-faint);
}}
.agent-role {{
    font-family: var(--heading);
    font-weight: 700;
    font-size: 14.5px;
    color: var(--ink);
}}
.agent-badge {{
    font-family: var(--sans);
    font-size: 10.5px;
    font-weight: 700;
    padding: 3px 9px;
    border-radius: var(--radius-pill);
    background: var(--sheet-alt);
    color: var(--graphite);
}}
.agent-badge.running {{ background: color-mix(in srgb, var(--pen) 18%, var(--sheet)); color: var(--pen); }}
.agent-badge.done {{ background: var(--pen); color: var(--sheet); }}
.agent-badge.error {{ background: color-mix(in srgb, var(--risk) 18%, var(--sheet)); color: var(--risk); }}
.agent-desc {{
    font-size: 12.5px;
    line-height: 1.5;
    color: var(--graphite);
    margin: 5px 0 9px;
}}
.agent-metric {{
    font-family: var(--sans);
    font-size: 11.5px;
    font-weight: 600;
    color: var(--graphite);
    background: var(--sheet-alt);
    padding: 4px 8px;
    border-radius: 8px;
    display: inline-block;
}}

/* ── Handoff Stream Feed ── */
.handoff-stream {{
    margin: 1.5rem 0;
    border-left: 2px solid var(--rule);
    padding-left: 1.2rem;
}}
.handoff-item {{
    margin-bottom: 1rem;
    position: relative;
}}
.handoff-item::before {{
    content: "";
    position: absolute;
    left: calc(-1.2rem - 5px);
    top: 5px;
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: var(--pen);
}}
.handoff-meta {{
    font-family: var(--sans);
    font-size: 12px;
    color: var(--pen);
    font-weight: 700;
    margin-bottom: 2px;
}}
.handoff-text {{
    font-size: 14.5px;
    line-height: 1.55;
    color: var(--ink);
}}

/* ── Executive Directive ── */
.exec-directive {{
    background: var(--sheet);
    border: 1px solid var(--rule);
    border-left: 4px solid var(--pen);
    border-radius: var(--radius);
    box-shadow: var(--lift-sm);
    padding: 1.3rem 1.5rem;
    margin-bottom: 1.5rem;
}}
.exec-directive .dir-label {{
    font-family: var(--sans);
    font-size: 12px;
    color: var(--pen);
    font-weight: 700;
    margin-bottom: 5px;
}}
.exec-directive .dir-content {{
    font-size: 15.5px;
    line-height: 1.65;
    color: var(--ink);
}}

/* ── Cards ── */
[data-testid="stExpander"] details {{
    background: var(--sheet); border: 1px solid var(--rule) !important;
    border-radius: var(--radius); box-shadow: var(--lift-sm);
}}
[data-testid="stExpander"] summary {{ font-weight: 700; font-size: 14.5px; }}
[data-testid="stExpander"] summary:hover {{ color: var(--pen); }}
[data-testid="stCode"] pre, pre {{
    background: var(--code-bg) !important; border: 1px solid var(--rule);
    border-radius: 10px; font-size: 12.5px; color: var(--ink) !important;
}}
[data-testid="stAlert"] {{ border-radius: var(--radius); }}
[data-testid="stAlertContainer"] {{
    background: var(--sheet-alt) !important; border-radius: var(--radius);
    border-left: 4px solid var(--graphite);
    padding: .6rem .8rem .6rem 1.1rem; color: var(--ink) !important;
}}
[data-testid="stAlertContainer"] p {{ color: inherit !important; font-size: 14.5px; }}
[data-testid="stAlertContainer"] svg {{ fill: currentColor; }}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) {{
    border-left-color: var(--positive); color: var(--positive) !important;
}}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]),
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) {{
    border-left-color: var(--risk); color: var(--risk) !important;
}}
[data-testid="stDataFrame"], [data-testid="stTable"] {{
    border-radius: var(--radius); box-shadow: var(--lift-sm); overflow: hidden;
}}

/* ── Steps list ── */
.sc {{ display: flex; align-items: center; gap: 12px;
      padding: .6rem .2rem; border-bottom: 1px solid var(--rule-faint);
      font-size: 14.5px; color: var(--graphite); }}
.sc .sc-num {{ font-family: var(--sans); font-size: 12px; font-weight: 700; color: var(--sheet);
              flex: none; width: 1.8em; height: 1.8em; display: flex; align-items: center;
              justify-content: center; border-radius: 50%; background: var(--rule); }}
.sc .nm {{ color: var(--ink); font-weight: 600; }}
.sc .detail {{ margin-left: auto; font-size: 12px;
              color: var(--graphite); text-align: right; padding-left: 1rem; }}
.sc.done  .sc-num {{ background: var(--pen); }}
.sc.active .sc-num {{ background: var(--pen); }}
.sc.active {{ background: color-mix(in srgb, var(--pen) 6%, transparent); border-radius: 10px; }}
.sc.active .nm::after {{ content: " — working"; font-weight: 400;
                        color: var(--pen); font-size: 12.5px; }}
.sc.skip  .sc-num {{ background: var(--rule); color: var(--graphite); }}
.sc.skip .nm {{ color: var(--graphite); font-weight: 400; }}
.sc.err   .sc-num {{ background: var(--risk); }}
.sc.err .nm {{ color: var(--risk); }}

/* ── Annotations ── */
.ic, .rc, .wc {{
    border-left: 3px solid var(--rule); border-radius: 0 10px 10px 0;
    padding: .5rem .8rem .5rem 1rem;
    margin: 0 0 .75rem; font-size: 15px; line-height: 1.6; max-width: 74ch;
    color: var(--ink); background: var(--sheet-alt);
}}
.ic {{ border-left-color: var(--graphite); }}
.rc {{ border-left-color: var(--pen); }}
.wc {{ border-left-color: var(--risk); color: var(--risk); background: color-mix(in srgb, var(--risk) 6%, var(--sheet-alt)); }}
.ic .mk, .rc .mk, .wc .mk {{
    font-family: var(--sans); font-size: 11.5px; font-weight: 700; color: var(--graphite);
    display: block; margin-bottom: 2px;
}}
.rc .mk {{ color: var(--pen); }}
.wc .mk {{ color: var(--risk); }}

.reason {{ background: var(--sheet); border: 1px solid var(--rule);
          border-radius: var(--radius); box-shadow: var(--lift-sm); padding: 1.3rem 1.5rem;
          font-size: 15.5px; color: var(--ink); line-height: 1.72; max-width: 72ch; }}

.run-banner {{ border-radius: var(--radius);
              background: color-mix(in srgb, var(--pen) 10%, var(--sheet)); padding: .8rem 1.1rem;
              color: var(--pen); font-size: 14.5px; font-weight: 700;
              margin: .4rem 0 1.2rem; }}
.run-banner .sub {{ display: block; font-weight: 500; color: var(--graphite);
                   font-size: 13px; margin-top: 2px; }}

.empty {{ padding: 3rem 0 3.5rem; max-width: 58ch; }}
.empty h2 {{ font-size: clamp(28px, 4vw, 42px); font-family: var(--heading);
            font-weight: 800; line-height: 1.05;
            margin: 0 0 1rem; }}
.empty p {{ color: var(--graphite); font-size: 16px; line-height: 1.62; margin: 0; }}
.empty .steps {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 2rem; }}
.empty .steps div {{ background: var(--sheet); border: 1px solid var(--rule);
                     border-radius: var(--radius-pill); padding: .4rem 1rem;
                     font-size: 13px; font-weight: 600; color: var(--graphite); }}
</style>
""", unsafe_allow_html=True)


# ── Session-state initialisation ──────────────────────────────────────────────
_DEFAULTS: dict[str, Any] = {
    "theme":          "day",
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

# ── Inject theme-aware CSS immediately (must run after session_state is ready) ─
_inject_theme_css(st.session_state.get("theme", "day"))


# ── Constants ─────────────────────────────────────────────────────────────────
STAGE_DEFS = [
    ("1", "Reading Your File"),
    ("2", "Understanding Your Question"),
    ("3", "Running the Numbers"),
    ("4", "Making Sense of It"),
    ("5", "Double-Checking"),
    ("6", "Solving the Tricky Parts"),
    ("7", "Writing Your Report"),
]

# Shared Vega-Lite config — charts are plotted in the same warm ink palette
# as the rest of the console (DESIGN.md). The terracotta pen is the measured
# series; the warm red is reserved for the series that carries risk.
PLOT_INK = "#3a2b1e"
PLOT_GRAPHITE = "#8a7660"
PLOT_RULE = "#e4d4bc"
PEN_BLUE = "#a34f20"
PEN_RED = "#a33526"

def _get_vega_config(theme: str = "day") -> dict[str, Any]:
    is_night = theme == "night"
    p_ink = "#f3e9d8" if is_night else "#3a2b1e"
    p_graphite = "#b8a688" if is_night else "#8a7660"
    p_rule = "#4a3c28" if is_night else "#e4d4bc"
    p_pen = "#f0a24a" if is_night else "#a34f20"
    p_risk = "#e2685a" if is_night else "#a33526"

    return {
        "font": "Mukta, 'Segoe UI', sans-serif",
        "axis": {
            "labelColor": p_graphite,
            "titleColor": p_graphite,
            "gridColor": p_rule,
            "gridDash": [2, 3],
            "domainColor": p_ink,
            "tickColor": p_ink,
            "labelFont": "Mukta, sans-serif",
            "labelFontSize": 11,
            "titleFont": "Baloo 2, sans-serif",
            "titleFontWeight": 600,
        },
        "legend": {
            "labelColor": p_ink,
            "titleColor": p_graphite,
            "labelFont": "Mukta, sans-serif",
            "titleFont": "Baloo 2, sans-serif",
            "symbolType": "square",
        },
        "view": {"stroke": "transparent"},
        "range": {
            "category": [
                p_pen,
                p_risk,
                "#d99a4e" if is_night else "#c08a2e",
                "#7fb77e" if is_night else "#5b8c5a",
                "#e2c58a" if is_night else "#8a7660",
                "#c98a6b" if is_night else "#b5714a",
            ]
        },
    }

VEGA_PLOT_CONFIG = _get_vega_config("day")

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
    """A ruled measurement bar. Each reading gets its own cell and hairline.

    Values can come straight from user input (e.g. the free-text "Custom
    model string" field feeding the "Model" cell) — HTML-escape both key
    and value before interpolating into markup rendered with
    unsafe_allow_html=True, or a value like
    `<img src=x onerror=alert(1)>` executes as-is (IMPROVEMENTS.md #10).
    """
    body = "".join(
        f'<div class="cell"><div class="k">{html.escape(k)}</div>'
        f'<div class="v">{html.escape(v)}</div></div>'
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
    cur_theme = st.session_state.get("theme", "day")
    with slot.container():
        render_pipeline(stages, height=420, theme=cur_theme)
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
    """A train-test gap above 10 points means the model memorised the split.

    Uses the same threshold the training tool itself warns on
    (`src.tools.ml_pipeline.OVERFIT_THRESHOLD`, and the same strict `>`),
    so the KPI gauge and the comparison table's overfit_warnings can never
    disagree about the same model (IMPROVEMENTS.md #10).
    """
    from src.tools.ml_pipeline import OVERFIT_THRESHOLD
    return gap > OVERFIT_THRESHOLD


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


#: Tool names already given a bespoke section elsewhere in the UI — the
#: generic renderer below only covers what's left, so a successful tool
#: result is never reachable *only* via the raw "Full Technical Log" JSON.
_BESPOKE_RENDERED_TOOLS = frozenset({
    "ingest_dataset", "clean_data", "detect_outliers", "correlation_analysis",
    "select_statistical_test", "train_model", "evaluate_model",
    "generate_report", "generate_visualizations", "planner",
})


def _render_other_findings(tool_results: list[dict[str, Any]]) -> None:
    """
    Fallback card for any successful tool result without a bespoke section
    (cluster_data, time_series_analysis, text_analysis, geospatial_analysis,
    dimensionality_analysis, and any future tool). Each already writes a
    well-formed sentence into output.summary; this surfaces that plus a few
    headline numbers per known shape instead of leaving the finding
    reachable only via the raw JSON log at the bottom of Downloads.
    """
    seen: set[str] = set()
    shown_any = False
    for r in tool_results:
        name = r.get("tool_name", "")
        if name in _BESPOKE_RENDERED_TOOLS or name in seen or r.get("status") != "success":
            continue
        out = r.get("output")
        if not isinstance(out, dict):
            continue
        seen.add(name)
        shown_any = True
        st.markdown(f"#### {name.replace('_', ' ').title()}")
        summary = out.get("summary")
        if summary:
            st.info(str(summary))

        if name == "cluster_data":
            c1, c2, c3 = st.columns(3)
            c1.metric("Clusters found", out.get("n_clusters", "—"))
            c2.metric("Silhouette score", out.get("silhouette_score", "—"))
            c3.metric("Separation", out.get("separation_quality", "—"))
        elif name == "time_series_analysis":
            c1, c2, c3 = st.columns(3)
            c1.metric("Trend", str(out.get("trend_direction", "—")).title())
            c2.metric("Stationary?", "Yes" if out.get("is_stationary") else "No")
            lags = out.get("seasonal_lags_detected") or []
            c3.metric("Seasonal lag(s)", ", ".join(str(x) for x in lags) or "None found")
        elif name == "text_analysis":
            c1, c2, c3 = st.columns(3)
            c1.metric("Vocabulary size", out.get("vocab_size", "—"))
            c2.metric("Avg. words / row", out.get("avg_word_count", "—"))
            top = out.get("top_tokens") or []
            words = [t.get("token", t) if isinstance(t, dict) else t for t in top[:6]]
            if words:
                st.caption("Most frequent words: " + ", ".join(str(w) for w in words))
        elif name == "geospatial_analysis":
            c1, c2 = st.columns(2)
            c1.metric("Points mapped", out.get("n_points", "—"))
            centroid = out.get("centroid") or {}
            if centroid:
                c2.metric("Centroid", f"{centroid.get('lat', '—')}, {centroid.get('lon', '—')}")
        elif name == "dimensionality_analysis":
            c1, c2, c3 = st.columns(3)
            c1.metric("Numeric features", out.get("n_features", "—"))
            threshold = out.get("variance_threshold")
            c2.metric(
                f"Components for {threshold:.0%} variance" if threshold else "Components needed",
                out.get("n_components_for_threshold", "—"),
            )
            pairs = out.get("high_correlation_pairs") or []
            c3.metric("Highly correlated pairs", len(pairs))

    if not shown_any:
        st.caption("No additional analyses ran for this dataset.")


def _render_defect_stamp(gap_val: float | None) -> str:
    """Render authentic engineering defect stamp or certification seal for generalization."""
    if gap_val is None:
        return ""
    if _gap_is_risky(gap_val):
        return f"""
        <div class="defect-stamp">
            <span class="stamp-tag">⚠ Heads up — this might not hold up</span>
            <div class="stamp-title">THE MODEL MEMORISED THE EXAMPLES ({gap_val*100:.1f}% GAP)</div>
            <div class="stamp-desc">
                It did noticeably better on the data it trained on than on data it hadn't seen —
                a sign it memorised quirks rather than learning the real pattern.
                We've ranked it lower because of this.
            </div>
        </div>
        """
    else:
        return f"""
        <div class="cert-stamp">
            <span class="stamp-tag">✓ Good news — this should hold up</span>
            <div class="stamp-title">THE MODEL PERFORMED CONSISTENTLY ({gap_val*100:.1f}% GAP)</div>
            <div class="stamp-desc">
                It did about as well on new data as on the data it trained on —
                a good sign the pattern it found is real, not a fluke.
            </div>
        </div>
        """


def _render_agent_grid(stage_log: list[tuple[str, str, str]]) -> str:
    """Render visual architecture cards for the autonomous multi-agent teamwork roster."""
    log_map = {n: s for n, s, _ in stage_log}

    agents = [
        {
            "icon": "🧭",
            "name": "Planner",
            "role": "Plans the approach",
            "desc": "Reads your question and breaks it into a step-by-step plan, then decides when enough checking has been done.",
            "stage": "2",
            "tool": "Reasoning",
        },
        {
            "icon": "🛡️",
            "name": "File Checker",
            "role": "Checks your file is safe and healthy",
            "desc": "Makes sure your file is safe to open, figures out what each column means, spots anything unusual, and gives your data a health score out of 100.",
            "stage": "1",
            "tool": "Checks & cleans",
        },
        {
            "icon": "📐",
            "name": "Fact-Checker",
            "role": "Tests what's actually true",
            "desc": "Runs the right statistical tests to check whether a pattern is real or could just be chance, and finds which columns move together.",
            "stage": "3",
            "tool": "Statistical tests",
        },
        {
            "icon": "⚡",
            "name": "Model Builder",
            "role": "Builds and tests prediction models",
            "desc": "Trains several different prediction models and tests each one on different slices of your data, so a lucky guess doesn't get mistaken for a good model.",
            "stage": "3",
            "tool": "Model training",
        },
        {
            "icon": "🔍",
            "name": "Reality-Checker",
            "role": "Catches models that just memorised",
            "desc": "Compares how each model does on data it trained on versus data it's never seen. If a model only looks good because it memorised the examples, this agent flags it and marks it down.",
            "stage": "4",
            "tool": "Model checking",
        },
        {
            "icon": "🔁",
            "name": "Double-Checker",
            "role": "Goes back for another pass",
            "desc": "Looks at what's been found so far, and if there are loose ends or your question isn't fully answered yet, sends the work back for another round.",
            "stage": "5",
            "tool": "Another pass",
        },
        {
            "icon": "🌐",
            "name": "Detail Handler",
            "role": "Handles the tricky, many-part questions",
            "desc": "When a question has too many moving parts to answer in one go, this splits it into smaller pieces, solves each one separately, and brings the answers back together.",
            "stage": "6",
            "tool": "Splitting up work",
        },
        {
            "icon": "📊",
            "name": "Report Writer",
            "role": "Builds your charts and report",
            "desc": "Builds charts that fit your data, then puts everything together into the report you can download and share.",
            "stage": "7",
            "tool": "Charts & report",
        },
    ]

    cards_html = []
    for ag in agents:
        st_val = log_map.get(ag["stage"], "pending")
        if st_val == "done":
            badge_cls = "done"
            badge_txt = "Completed"
            card_cls = "agent-card"
        elif st_val == "active":
            badge_cls = "running"
            badge_txt = "Executing"
            card_cls = "agent-card agent-active"
        elif st_val == "error":
            badge_cls = "error"
            badge_txt = "Flagged"
            card_cls = "agent-card agent-flagged"
        else:
            badge_cls = ""
            badge_txt = "Standby"
            card_cls = "agent-card"

        cards_html.append(f"""
        <div class="{card_cls}">
            <div class="agent-header">
                <span class="agent-role">{ag['icon']} {ag['name']}</span>
                <span class="agent-badge {badge_cls}">[{badge_txt}]</span>
            </div>
            <div class="agent-desc">{ag['desc']}</div>
            <div class="agent-metric">Role: {ag['role']} · Tool: {ag['tool']}</div>
        </div>
        """.strip())

    return f'<div class="agent-grid">{"".join(cards_html)}</div>'


def _render_handoff_stream(progress_lines: list[str], tool_results: list[dict[str, Any]]) -> str:
    """Render timeline feed of inter-agent messages and handoffs."""
    items_html = []
    if progress_lines:
        for line in progress_lines[:20]:
            if not line.strip():
                continue
            meta = "Note"
            if "[done]" in line:
                meta = "Done"
            elif "[run ]" in line:
                meta = "Started"
            elif "ok" in line:
                meta = "Finished"
            clean_text = line.replace("[done]", "").replace("[run ]", "").replace("[    ]", "").replace("[fail]", "⚠ ").strip()
            items_html.append(f"""
            <div class="handoff-item">
                <div class="handoff-meta">{meta}</div>
                <div class="handoff-text">{clean_text}</div>
            </div>
            """.strip())
    elif tool_results:
        for r in tool_results:
            name = r.get("tool_name", "Step")
            status = r.get("status", "success")
            summary = r.get("output", {}).get("summary", "") or r.get("error", "")
            time_ms = r.get("execution_time_ms", 0)
            items_html.append(f"""
            <div class="handoff-item">
                <div class="handoff-meta">{name} · {status} · {time_ms:.0f}ms</div>
                <div class="handoff-text">{summary}</div>
            </div>
            """.strip())
    else:
        items_html.append("""
        <div class="handoff-item">
            <div class="handoff-meta">Waiting</div>
            <div class="handoff-text">Your helpers are ready. Upload a file to get started.</div>
        </div>
        """.strip())
    return f'<div class="handoff-stream">{"".join(items_html)}</div>'


def _render_agent_deep_dive(agent_name: str, tool_results: list[dict[str, Any]], report: dict[str, Any]) -> None:
    """Render structured details for an inspected agent persona."""
    details = {
        "🧭 Planner": {
            "mission": "Reads your question and turns it into a step-by-step plan — what to check first, what to try next, and when the plan needs adjusting.",
            "directive": "Only works from summaries and statistics, never your raw data rows — the way a manager works from a report rather than the raw ledger.",
            "tools": "Reasoning and planning",
            "output": report.get("reasoning", "Waiting for a plan."),
        },
        "🛡️ File Checker": {
            "mission": "Checks your file is safe to open, figures out what each column means, and gives your data a health score.",
            "directive": "Scores your data 0–100 based on missing values, duplicate rows, and anything that looks off.",
            "tools": "File safety checks, data profiling",
            "output": f"Health score {st.session_state.get('profile', {}).get('quality_score', '—')}/100. Cleaned up any issues found.",
        },
        "📐 Fact-Checker": {
            "mission": "Runs statistical tests to check whether a pattern in your data is real, or could just be chance.",
            "directive": "Checks how your data is shaped before picking which test is fair to use — the right test depends on the shape.",
            "tools": "Statistical tests, correlation checks",
            "output": "Tests complete: checked which columns move together and whether the differences are real.",
        },
        "⚡ Model Builder": {
            "mission": "Trains a few different prediction models on your data and scores each one.",
            "directive": "Tests every model on several different slices of the data, not just one, so a lucky split doesn't make a bad model look good.",
            "tools": "Model training (several approaches, tested against each other)",
            "output": f"Best model so far: {report.get('best_model', 'N/A')}. Tested multiple times on different slices of your data.",
        },
        "🔍 Reality-Checker": {
            "mission": "Compares how each model performs on data it trained on versus data it's never seen.",
            "directive": "If a model does noticeably better on familiar data than new data, it's flagged as having memorised rather than learned — and marked down.",
            "tools": "Model checking",
            "output": "Checked every model for memorisation. Applied a penalty to any that didn't hold up.",
        },
        "🔁 Double-Checker": {
            "mission": "Looks at what's been found so far and decides whether your question has really been answered.",
            "directive": "Sends the work back for another pass if things haven't settled down yet, up to a set limit of tries.",
            "tools": "Review and another pass",
            "output": "Finished reviewing — went back for more passes where needed.",
        },
        "🌐 Detail Handler": {
            "mission": "Splits a big, many-part question into smaller pieces, solves each on its own, then brings the answers back together.",
            "directive": "Keeps each piece small and separate, so a complicated question doesn't overwhelm any single step.",
            "tools": "Splitting up and recombining work",
            "output": f"{len(report.get('rlm_sub_results', []))} smaller questions solved separately and combined.",
        },
        "📊 Report Writer": {
            "mission": "Builds charts that fit your data and puts everything into a report you can download and share.",
            "directive": "Uses the same easy-to-read style for the charts and the report as the rest of the app, and gives you both a written version and a webpage version.",
            "tools": "Charts and report writing",
            "output": "Report finished, with charts, key findings, and what to do next.",
        },
    }
    info = details.get(agent_name, details["🧭 Planner"])
    c1, c2 = st.columns([0.6, 0.4])
    with c1:
        st.markdown(f"**What it does:** {info['mission']}")
        st.markdown(f"**Its rule:** {info['directive']}")
    with c2:
        st.markdown(f"**What it uses:** {info['tools']}")
        st.markdown(f"**What it found:** {info['output']}")


def _load_teamwork_preview() -> None:
    """Populate full autonomous multi-agent teamwork demo with sample customer churn data."""
    _reset_pipeline()
    sample_path = ROOT / "data" / "sample_customer_churn.csv"
    if sample_path.exists():
        raw_bytes = sample_path.read_bytes()
        df = pd.read_csv(sample_path)
    else:
        import numpy as np
        np.random.seed(42)
        df = pd.DataFrame({
            "tenure": np.random.randint(1, 72, 100),
            "monthly_charges": np.random.uniform(20, 120, 100).round(2),
            "total_charges": np.random.uniform(100, 8000, 100).round(2),
            "contract": np.random.choice(["Month-to-month", "One year", "Two year"], 100),
            "internet_service": np.random.choice(["DSL", "Fiber optic", "No"], 100),
            "payment_method": np.random.choice(["Electronic check", "Mailed check", "Bank transfer"], 100),
            "churn": np.random.choice([0, 1], 100, p=[0.73, 0.27]),
        })
        raw_bytes = df.to_csv(index=False).encode("utf-8")

    st.session_state["preview_df"] = df
    st.session_state["preview_name"] = "sample_customer_churn.csv"
    st.session_state["orig_name"] = "sample_customer_churn.csv"
    st.session_state["preview_bytes"] = raw_bytes

    tmp = tempfile.mkdtemp()
    st.session_state["tmp_dir"] = tmp
    out_dir = Path(tmp) / "output"
    rep_dir = out_dir / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)

    st.session_state["stage_log"] = [
        ("1", "done", "100 rows × 8 cols · task=classification · target=churn"),
        ("2", "done", "5 steps planned by the Planner"),
        ("3", "done", "6 tools executed: clean, outliers, corr, test, train, eval"),
        ("4", "done", "Anti-overfit audit passed: gap 4.2% < 10%"),
        ("5", "done", "Converged in 2 iterations (residual variance resolved)"),
        ("6", "done", "2 RLM sub-tasks offloaded via REPL context"),
        ("7", "done", "analysis_report.md & report.html compiled"),
    ]

    st.session_state["tool_results"] = [
        {
            "tool_name": "clean_data",
            "status": "success",
            "execution_time_ms": 42.0,
            "output": {
                "summary": "Cleaned dataset: 0 missing values found. Handled numeric types and standardized categorical levels.",
                "strategy_used": "median",
                "missing_before": 0,
                "missing_after": 0,
            },
        },
        {
            "tool_name": "detect_outliers",
            "status": "success",
            "execution_time_ms": 58.0,
            "output": {
                "summary": "Detected 4 outlier rows across total_charges using IQR method (3.0 threshold). Kept in dataset.",
                "total_outliers": 4,
                "outlier_percentage": 4.0,
                "per_column_outliers": {"total_charges": 4, "monthly_charges": 0, "tenure": 0},
            },
        },
        {
            "tool_name": "correlation_analysis",
            "status": "success",
            "execution_time_ms": 85.0,
            "output": {
                "summary": "Identified strongest correlation pairs with churn: tenure (-0.35) and monthly_charges (+0.28).",
                "top_correlations": [
                    {"col_a": "tenure", "col_b": "churn", "correlation": -0.352},
                    {"col_a": "monthly_charges", "col_b": "churn", "correlation": 0.284},
                    {"col_a": "monthly_charges", "col_b": "total_charges", "correlation": 0.651},
                    {"col_a": "tenure", "col_b": "total_charges", "correlation": 0.824},
                ],
            },
        },
        {
            "tool_name": "select_statistical_test",
            "status": "success",
            "execution_time_ms": 36.0,
            "output": {
                "summary": "Mann-Whitney U test confirmed statistically significant tenure difference between churners and retainers (p=0.0004).",
                "test_name": "Mann-Whitney U Test",
                "p_value": 0.00041,
                "significant": True,
                "interpretation": "Tenure of churned customers is significantly lower than retained customers (median 10 mos vs 38 mos, p < 0.001).",
            },
        },
        {
            "tool_name": "train_model",
            "status": "success",
            "execution_time_ms": 320.0,
            "output": {
                "summary": "Trained 3 stratified 5-fold models. Random Forest achieved highest CV accuracy (81.0% ± 3.2%).",
                "task_type": "classification",
                "best_model": "RandomForestClassifier",
                "n_cv_folds": 5,
                "test_size": 0.2,
                "overfit_warnings": [],
                "models_trained": {
                    "RandomForestClassifier": {
                        "cv_mean": 0.810,
                        "cv_std": 0.032,
                        "train_metrics": {"accuracy": 0.852, "f1_score": 0.840},
                        "test_metrics": {"accuracy": 0.810, "f1_score": 0.795},
                        "train_test_gap": 0.042,
                    },
                    "LogisticRegression": {
                        "cv_mean": 0.790,
                        "cv_std": 0.028,
                        "train_metrics": {"accuracy": 0.800, "f1_score": 0.772},
                        "test_metrics": {"accuracy": 0.780, "f1_score": 0.760},
                        "train_test_gap": 0.020,
                    },
                    "GradientBoostingClassifier": {
                        "cv_mean": 0.775,
                        "cv_std": 0.035,
                        "train_metrics": {"accuracy": 0.885, "f1_score": 0.871},
                        "test_metrics": {"accuracy": 0.760, "f1_score": 0.735},
                        "train_test_gap": 0.125,
                    },
                },
            },
        },
        {
            "tool_name": "evaluate_model",
            "status": "success",
            "execution_time_ms": 48.0,
            "output": {
                "summary": "Model evaluation complete. Precision 0.82, Recall 0.79 for Retained (0); Precision 0.78, Recall 0.74 for Churned (1).",
                "classification_report": {
                    "Retained (0)": {"precision": 0.824, "recall": 0.795, "f1-score": 0.809, "support": 15},
                    "Churned (1)": {"precision": 0.780, "recall": 0.740, "f1-score": 0.759, "support": 5},
                },
            },
        },
    ]

    st.session_state["profile"] = {
        "quality_score": 92,
        "duplicate_rows": 0,
        "memory_mb": 0.12,
        "column_count": len(df.columns),
        "columns": [
            {"name": c, "kind": "numeric" if pd.api.types.is_numeric_dtype(df[c]) else "categorical",
             "dtype": str(df[c].dtype), "missing_pct": 0.0, "nunique": int(df[c].nunique()), "flags": []}
            for c in df.columns
        ],
        "warnings": [],
    }

    st.session_state["dashboard"] = [
        {
            "chart_id": "model_comparison",
            "title": "Cross-Validation Accuracy vs Generalization Gap",
            "description": "Comparison of models ranking by 5-fold CV score and train-test gap to penalize memorization.",
            "spec": {
                "mark": "bar",
                "data": {"values": [
                    {"Model": "Random Forest", "Metric": "CV Accuracy", "Score": 81.0},
                    {"Model": "Random Forest", "Metric": "Generalization Gap", "Score": 4.2},
                    {"Model": "Logistic Regression", "Metric": "CV Accuracy", "Score": 79.0},
                    {"Model": "Logistic Regression", "Metric": "Generalization Gap", "Score": 2.0},
                    {"Model": "Gradient Boosting", "Metric": "CV Accuracy", "Score": 77.5},
                    {"Model": "Gradient Boosting", "Metric": "Generalization Gap", "Score": 12.5},
                ]},
                "encoding": {
                    "x": {"field": "Model", "type": "nominal", "axis": {"labelAngle": 0}},
                    "xOffset": {"field": "Metric"},
                    "y": {"field": "Score", "type": "quantitative", "title": "Percentage (%)"},
                    "color": {"field": "Metric", "type": "nominal"},
                },
            },
        },
        {
            "chart_id": "top_correlations",
            "title": "Key Drivers: Feature Correlation with Customer Churn",
            "description": "Tenure exhibits strong protective negative correlation (-0.35), while high monthly charges drive churn (+0.28).",
            "spec": {
                "mark": "bar",
                "data": {"values": [
                    {"Feature": "Tenure (Months)", "Correlation": -0.352},
                    {"Feature": "Monthly Charges", "Correlation": 0.284},
                    {"Feature": "Paperless Billing", "Correlation": 0.175},
                    {"Feature": "Total Charges", "Correlation": -0.198},
                ]},
                "encoding": {
                    "y": {"field": "Feature", "type": "nominal", "sort": "-x"},
                    "x": {"field": "Correlation", "type": "quantitative", "scale": {"domain": [-0.5, 0.5]}},
                    "color": {
                        "condition": {"test": "datum.Correlation >= 0", "value": "#a34f20"},
                        "value": "#8a7660",
                    },
                },
            },
        },
    ]

    st.session_state["final_report"] = {
        "best_model": "RandomForestClassifier",
        "reasoning": (
            "Customer churn is primarily driven by tenure length and high monthly billing tiers. "
            "Customers on month-to-month contracts with tenure < 12 months exhibit a 44% higher probability of churning. "
            "The Random Forest model demonstrated superior cross-validated generalization (81.0% accuracy, train-test gap 4.2%), "
            "comfortably satisfying the 10% anti-overfitting safety threshold."
        ),
        "insights": [
            "Early-tenure vulnerability: First 12 months account for 68% of all churn instances.",
            "Billing sensitivity: Accounts paying over $75/mo without fiber reliability churn at 2.3× baseline.",
            "Contractual resilience: Annual and two-year agreements reduce churn by 78% relative to monthly contracts.",
        ],
        "recommendations": [
            "Deploy targeted 90-day onboarding incentives for high-charge month-to-month cohorts.",
            "Offer contract term upgrades with bundled savings prior to the critical 6-month drop-off cliff.",
            "Route at-risk accounts identified by the Random Forest model to proactive retention concierges.",
        ],
        "rlm_sub_results": [
            {
                "task_name": "Tenure Stratification Analysis",
                "query": "Quantify churn hazard rate across 0-6mo, 6-12mo, and 12-24mo cohorts",
                "finding": "Hazard rate peaks at month 4 (31.2% hazard rate), dropping to 4.1% past month 24.",
            },
            {
                "task_name": "Billing Tier Elasticity",
                "query": "Evaluate elasticity between monthly charge increments and churn probability",
                "finding": "Every $10 increase above $65/mo produces an incremental 4.8% churn risk.",
            },
        ],
    }

    st.session_state["progress_lines"] = [
        "[done] Stage 1: Reading Your File  100 rows × 8 cols · task=classification · target=churn",
        "[run ] Iteration 1: model reasoning",
        "[done] Stage 2: Understanding Your Question  5 steps planned by the Planner",
        "       ok  clean_data: median imputation on missing numeric cells",
        "       ok  detect_outliers: 4 anomaly rows flagged via IQR",
        "       ok  correlation_analysis: mapped feature associations with churn",
        "       ok  select_statistical_test: Mann-Whitney U test (p=0.0004)",
        "       ok  train_model: 5-fold CV on RandomForest, LogisticRegression, GradientBoosting",
        "       ok  evaluate_model: Reality-Checker confirmed train-test gap 4.2% < 10% [Certified]",
        "[done] Stage 3: Running the Numbers  6 tools executed successfully",
        "[run ] Iteration 1: interpreting and refining",
        "[done] Stage 4: Making Sense of It  Key drivers tenure and charges synthesized",
        "[done] Stage 5: Double-Checking  Loop converged in 2 iterations",
        "[done] Stage 6: Solving the Tricky Parts  2 sub-tasks offloaded via REPL context",
        "[done] Stage 7: Writing Your Report  Certified markdown and HTML dossier published",
    ]

    md_content = f"# Executive Analytical Dossier: Customer Churn Analysis\\n\\n{st.session_state['final_report']['reasoning']}\\n\\n## Recommendations\\n- " + "\\n- ".join(st.session_state['final_report']['recommendations'])
    (rep_dir / "analysis_report.md").write_text(md_content, encoding="utf-8")
    (rep_dir / "report.html").write_text("<html><body>" + md_content + "</body></html>", encoding="utf-8")
    (rep_dir / "final_report.json").write_text(json.dumps(st.session_state["final_report"], indent=2), encoding="utf-8")

    st.session_state["analysis_done"] = True


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<div class="side-brand">'
        '<div class="side-title">Agentic Data Analysis</div>'
        '<div class="side-sub">Your friendly assistant for making sense of data.</div></div>',
        unsafe_allow_html=True,
    )

    # ── Theme Selector ────────────────────────────────────────────────────────
    _cur_theme = st.session_state.get("theme", "day")
    theme_sel = st.selectbox(
        "Theme",
        ["Day Mode", "Night Mode"],
        index=0 if _cur_theme == "day" else 1,
        help="Switch between a bright look for daytime and a cozy dark look for night.",
    )
    _new_theme = "night" if theme_sel == "Night Mode" else "day"
    if _new_theme != _cur_theme:
        st.session_state["theme"] = _new_theme
        st.rerun()

    # ── Upload ────────────────────────────────────────────────────────────────
    st.markdown('<div class="side-head">Dataset</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "CSV, TSV or Excel",
        type=sorted(ext.lstrip(".") for ext in ALLOWED_EXTENSIONS),
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
        placeholder="e.g. outcome, price, result  (blank = find groupings)",
    )

    objective = st.text_area(
        "What do you want to know? (optional, plain English)",
        placeholder="e.g. What's driving this result? Which rows are the "
                    "outliers, and why?",
        height=90,
        help="Your helpers will prioritise analyses that answer this question "
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

    demo_clicked = st.button(
        "⚡ See a Sample Report (Demo)",
        width='stretch',
        help="Instantly load sample data and see what a finished report looks like.",
    )
    if demo_clicked:
        _load_teamwork_preview()
        st.rerun()

    if not has_file:
        if st.button("📂 Load Sample Data", width='stretch'):
            sample_path = ROOT / "data" / "sample_customer_churn.csv"
            if sample_path.exists():
                st.session_state["preview_df"] = pd.read_csv(sample_path)
                st.session_state["preview_name"] = "sample_customer_churn.csv"
                st.session_state["orig_name"] = "sample_customer_churn.csv"
                st.session_state["preview_bytes"] = sample_path.read_bytes()
                st.rerun()
        st.caption("Upload a CSV/Excel file or click \"See a Sample Report\" above.")
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
        '<h1>We check every answer twice.</h1>'
        '<p class="hero-sub">Upload any spreadsheet — sales, survey, sports, science, '
        'whatever you\'ve got. Your assistant studies it, tests its own conclusions, '
        'and tells you which patterns are real — and which are just luck.</p></div>',
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
        st.session_state["read_report"] = agent.memory.get_context("read_report")
        st.session_state["coercions"] = agent.memory.get_context("coercions")
        st.session_state["profile_status"] = agent.memory.get_context("profile_status")
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
# HERO PLATE & DATUM REFRESH
# ══════════════════════════════════════════════════════════════════════════════
stages_3d = _draw_pipeline_rig(pipeline_slot)

# The datum line: the same run state as the drawing, in words and figures.
_done = sum(1 for _, s, _ in st.session_state["stage_log"] if s == "done")
_errored = any(s == "error" for _, s, _ in st.session_state["stage_log"])
_running = any(s == "active" for _, s, _ in st.session_state["stage_log"])
_status_str = (
    "FAILED" if _errored else
    "RUNNING" if _running else
    f"{_done}/7 COMPLETE" if _done > 0 else
    "IDLE"
)

_cur_theme_name = "Day Mode" if st.session_state.get("theme", "day") == "day" else "Night Mode"
_cells = [
    ("State", _status_str),
    ("Theme", _cur_theme_name),
    ("File", st.session_state.get("preview_name") or "None loaded"),
    ("Model", final_model if "final_model" in locals() else "N/A"),
]
datum_slot.markdown(_datum(_cells), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS — 5-TAB ARCHITECTURAL DOSSIER
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.get("analysis_done"):
    report: dict[str, Any] = st.session_state.get("final_report") or {}
    tool_results: list[dict[str, Any]] = st.session_state.get("tool_results") or []
    meta: Any = st.session_state.get("metadata")  # DatasetMetadata | None (lazy import)
    tmp_dir: str = st.session_state.get("tmp_dir") or ""
    outdir = str(Path(tmp_dir) / "output") if tmp_dir else ""
    dash: list[dict[str, Any]] | None = st.session_state.get("dashboard")
    profile: dict[str, Any] | None = st.session_state.get("profile")

    (tab_brief, tab_team, tab_dash, tab_lab, tab_vault) = st.tabs([
        "📋 Summary",
        "👥 Your Helpers",
        "📊 Charts",
        "🔬 Full Details",
        "📁 Downloads",
    ])

    train_out   = _find_tool(tool_results, "train_model")
    eval_out    = _find_tool(tool_results, "evaluate_model")
    corr_out    = _find_tool(tool_results, "correlation_analysis")
    outlier_out = _find_tool(tool_results, "detect_outliers")
    stat_out    = _find_tool(tool_results, "select_statistical_test")
    clean_out   = _find_tool(tool_results, "clean_data")
    vega_cfg    = _get_vega_config(st.session_state.get("theme", "day"))

    # ═════════════════════════════════════════════════════════════════════════
    # TIER 1: EXECUTIVE BRIEFING
    # ═════════════════════════════════════════════════════════════════════════
    with tab_brief:
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

        # Executive Objective Answer
        # _user_obj is straight from the sidebar's free-text box, and
        # report["reasoning"] is LLM output over (possibly hostile) dataset
        # content — neither is trusted HTML. Escape both before
        # interpolating into markup rendered with unsafe_allow_html=True
        # (IMPROVEMENTS.md #10).
        _user_obj = os.environ.get("USER_OBJECTIVE") or objective.strip()
        if _user_obj and report.get("reasoning"):
            st.markdown(
                f'<div class="exec-directive">'
                f'<div class="dir-label">Executive Directive · Answer to Objective</div>'
                f'<div style="font-weight:700;margin-bottom:6px;color:var(--pen);">You asked: {html.escape(_user_obj)}</div>'
                f'<div class="dir-content">{html.escape(report["reasoning"])}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        elif report.get("reasoning"):
            st.markdown(
                f'<div class="exec-directive">'
                f'<div class="dir-label">Executive Finding · Agent Synthesis</div>'
                f'<div class="dir-content">{html.escape(report["reasoning"])}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # When there's no ML target, the gauges below are mostly "—" —
        # promote whatever analysis actually ran (segments, trend, map,
        # text) to the top instead of leaving it findable only in Full
        # Details, since for a no-target dataset it usually *is* the story.
        if not train_out:
            _dominant = (
                _find_tool(tool_results, "cluster_data")
                or _find_tool(tool_results, "time_series_analysis")
                or _find_tool(tool_results, "geospatial_analysis")
                or _find_tool(tool_results, "text_analysis")
                or _find_tool(tool_results, "dimensionality_analysis")
            )
            if _dominant and _dominant.get("summary"):
                st.info(f"**What we found:** {_dominant['summary']}")

        # Instrument Gauges
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.markdown(_gauge("Best model", best_model), unsafe_allow_html=True)
        m2.markdown(_gauge("Cross-validated score", best_cv, "mean across folds"), unsafe_allow_html=True)
        m3.markdown(
            _gauge("Train–test gap", best_gap_str, "how much it memorised",
                   flag=gap_val is not None and _gap_is_risky(gap_val)),
            unsafe_allow_html=True,
        )
        m4.markdown(_gauge("Outliers", outlier_pct, "of all rows"), unsafe_allow_html=True)
        m5.markdown(_gauge("Task type", task_type), unsafe_allow_html=True)

        # Generalization Defect / Certification Stamp
        if gap_val is not None:
            st.markdown(_render_defect_stamp(gap_val), unsafe_allow_html=True)

        for _w in (train_out.get("overfit_warnings", []) if train_out else []):
            st.markdown(f'<div class="wc"><span class="mk">Risk</span>{_w}</div>', unsafe_allow_html=True)

        # Key Discoveries & Actions
        col_ins, col_rec = st.columns(2)
        with col_ins:
            st.markdown("#### Key Discoveries")
            _ins_list = report.get("insights", [])
            if _ins_list:
                for _i, _ins in enumerate(_ins_list, start=1):
                    st.markdown(f'<div class="ic"><span class="mk">{_i:02d}</span>{html.escape(str(_ins))}</div>', unsafe_allow_html=True)
            else:
                st.caption("No explicit statistical discoveries recorded.")

        with col_rec:
            st.markdown("#### Recommended Actions")
            _rec_list = report.get("recommendations", [])
            if _rec_list:
                for _rec in _rec_list:
                    st.markdown(f'<div class="rc"><span class="mk">Do</span>{html.escape(str(_rec))}</div>', unsafe_allow_html=True)
            else:
                st.caption("No operational recommendations generated.")

        # Quick Model Scoring Comparison Chart
        if train_out:
            _mt_map2 = train_out.get("models_trained", {})
            if _mt_map2:
                st.markdown("#### How each model scored (Train vs Test vs CV)")
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
                        "height": 280,
                        "background": "transparent",
                        "config": vega_cfg,
                        "encoding": {
                            "x": {"field": "model", "type": "nominal", "axis": {"labelAngle": 0, "title": None}},
                            "xOffset": {"field": "metric"},
                            "y": {"field": "score", "type": "quantitative", "title": f"{_pk} %", "scale": {"domain": [0, 110]}},
                            "color": {
                                "field": "metric",
                                "scale": {
                                    "domain": ["Train", "Test", "CV mean"],
                                    "range": ["#8aa6c2", PEN_BLUE, PLOT_INK],
                                },
                                "legend": {"orient": "top", "title": None},
                            },
                            "tooltip": [{"field": "model"}, {"field": "metric"}, {"field": "score", "title": f"{_pk} %"}],
                        },
                    },
                    use_container_width=True,
                )

        if corr_out:
            _top = corr_out.get("top_correlations", [])[:8]
            if _top:
                st.markdown("#### Strongest feature correlations")
                _corr_df = pd.DataFrame(
                    [{"pair": f"{r['col_a']} ↔ {r['col_b']}", "correlation": r["correlation"]} for r in _top]
                )
                st.vega_lite_chart(
                    _corr_df,
                    {
                        "mark": {"type": "bar"},
                        "height": max(150, len(_top) * 28),
                        "background": "transparent",
                        "config": vega_cfg,
                        "encoding": {
                            "y": {"field": "pair", "type": "nominal", "sort": "-x", "title": None},
                            "x": {"field": "correlation", "type": "quantitative", "scale": {"domain": [-1.1, 1.1]}, "title": "Correlation coefficient"},
                            "color": {
                                "condition": {"test": "datum.correlation >= 0", "value": PEN_BLUE},
                                "value": PLOT_INK,
                            },
                            "tooltip": [{"field": "pair"}, {"field": "correlation"}],
                        },
                    },
                    use_container_width=True,
                )

    # ═════════════════════════════════════════════════════════════════════════
    # TIER 2: MULTI-AGENT TEAMWORK CONSOLE
    # ═════════════════════════════════════════════════════════════════════════
    with tab_team:
        st.markdown(
            '<div class="datum">'
            '<div class="cell"><div class="k">How it works</div><div class="v">Reasoning, then doing, kept separate</div></div>'
            '<div class="cell"><div class="k">Team</div><div class="v">8 helpers, each with one job</div></div>'
            '<div class="cell"><div class="k">Status</div><div class="v">All ready</div></div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown("#### Meet Your Helpers")
        st.caption("Each helper does one job, and hands off to the next.")
        st.markdown(_render_agent_grid(st.session_state["stage_log"]), unsafe_allow_html=True)

        st.markdown("#### Look Inside a Helper")
        st.caption("Pick one to see exactly what it does, its rule of thumb, and what it found.")
        agent_names = [
            "🧭 Planner",
            "🛡️ File Checker",
            "📐 Fact-Checker",
            "⚡ Model Builder",
            "🔍 Reality-Checker",
            "🔁 Double-Checker",
            "🌐 Detail Handler",
            "📊 Report Writer",
        ]
        chosen_agent = st.selectbox("Choose a helper to look inside", agent_names, label_visibility="collapsed")
        _render_agent_deep_dive(chosen_agent, tool_results, report)

        st.markdown("#### What Was Said, Step by Step")
        st.caption("A record of what each helper passed to the next, and when.")
        st.markdown(
            _render_handoff_stream(st.session_state.get("progress_lines", []), tool_results),
            unsafe_allow_html=True,
        )

        # RLM Sub-task Decomposition Trace
        sub_results = report.get("rlm_sub_results")
        if sub_results:
            st.markdown("#### How the Tricky Parts Were Split Up")
            st.caption("Big questions got broken into smaller ones so nothing got lost.")
            for _s_idx, _sub in enumerate(sub_results, 1):
                with st.expander(f"Part {_s_idx:02d}: {_sub.get('task_name', 'Smaller Question')}", expanded=True):
                    st.json(_sub)

    # ═════════════════════════════════════════════════════════════════════════
    # TIER 3: DYNAMIC DASHBOARD
    # ═════════════════════════════════════════════════════════════════════════
    with tab_dash:
        dashboard: list[dict[str, Any]] | None = st.session_state.get("dashboard")
        if dashboard:
            st.caption("Built automatically to fit your data.")
            _full_width_ids = {"model_comparison", "top_correlations", "scatter_top_pair", "time_series"}
            _grid_charts: list[dict[str, Any]] = []

            def _render_chart(_ch: dict[str, Any]) -> None:
                st.markdown(f"**{_ch.get('title', '')}**")
                _spec = dict(_ch.get("spec", {}))
                _spec.setdefault("background", "transparent")
                _spec.setdefault("config", vega_cfg)
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

    # ═════════════════════════════════════════════════════════════════════════
    # TIER 4: STATISTICAL & ML LAB
    # ═════════════════════════════════════════════════════════════════════════
    with tab_lab:
        st.markdown("### Full Details")
        st.caption("The complete technical picture, for anyone who wants to check our work.")

        with st.expander("How the Models Were Tested", expanded=True):
            if train_out:
                _mt2   = train_out.get("models_trained", {})
                _best2 = train_out.get("best_model", "")
                _task2 = train_out.get("task_type", "classification")
                _pk2   = "accuracy" if _task2 == "classification" else "r2"
                _sk2   = "f1_score" if _task2 == "classification" else "rmse"

                st.markdown(
                    f"**Type of problem:** `{_task2}` &nbsp;|&nbsp; **Best model:** `{_best2}` "
                    f"&nbsp;|&nbsp; **Times each model was tested:** `{train_out.get('n_cv_folds', 5)}` "
                    f"&nbsp;|&nbsp; **Held back for testing:** `{int(train_out.get('test_size', 0.2)*100)}%`"
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
                        "Gap": (f"{_g*100:.1f}%" + (" [RISK]" if _g and _g >= .10 else "")
                                if _g is not None else "—"),
                        _sk2.replace("_", " "): (
                            f"{_te2.get(_sk2,0)*100:.1f}%"
                            if _sk2 != "rmse" else f"{_te2.get(_sk2,0):.4f}"
                        ),
                    })
                st.dataframe(_safe_df(pd.DataFrame(_rows)), width='stretch')

            if eval_out:
                st.markdown("#### Accuracy by Category")
                st.caption("Precision: of the times it guessed this category, how often it was right. Recall: of all the actual cases, how many it caught.")
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

        with st.expander("Data Health Check", expanded=False):
            _profile_status = st.session_state.get("profile_status")
            if isinstance(_profile_status, str) and _profile_status.startswith("failed"):
                st.warning(
                    "Running in degraded mode — profiling failed, so dataset-nature "
                    f"tools (time-series, text, geo...) were unavailable. Reason: {_profile_status[8:]}"
                )

            _read_report = st.session_state.get("read_report")
            _coercions = st.session_state.get("coercions")
            if _read_report or _coercions:
                st.markdown("#### Reading & Repairs")
                if _read_report:
                    _rr_bits = [f"format `{_read_report.get('format')}`", f"encoding `{_read_report.get('encoding')}`"]
                    if not _read_report.get("encoding_confident", True):
                        _rr_bits[-1] += " (guessed)"
                    if _read_report.get("delimiter"):
                        _rr_bits.append(f"delimiter `{_read_report.get('delimiter')!r}`" + ("" if _read_report.get("delimiter_sniffed") else " (from extension)"))
                    st.caption("Detected at read time: " + ", ".join(_rr_bits) + ".")
                    for _note in _read_report.get("notes", []):
                        st.caption(f"⚠ {_note}")
                if _coercions:
                    st.caption(f"{len(_coercions)} column(s) repaired:")
                    st.dataframe(_safe_df(pd.DataFrame([
                        {"Column": c["column"], "Rule": c["rule"], "Converted": c["n_converted"], "Failed": c["n_failed"]}
                        for c in _coercions
                    ])), width='stretch')

            prof: dict[str, Any] | None = st.session_state.get("profile")
            if prof:
                _q = int(prof.get("quality_score", 0))
                _dupes = int(prof.get("duplicate_rows", 0))
                p1, p2, p3, p4 = st.columns(4)
                p1.markdown(_gauge("Quality score", f"{_q}/100", "out of 100", flag=_q < 60), unsafe_allow_html=True)
                p2.markdown(_gauge("Duplicate rows", f"{_dupes:,}", flag=_dupes > 0), unsafe_allow_html=True)
                p3.markdown(_gauge("Memory footprint", f"{prof.get('memory_mb', 0)} MB"), unsafe_allow_html=True)
                p4.markdown(_gauge("Profiled columns", str(prof.get("column_count", 0))), unsafe_allow_html=True)

                st.markdown("#### What's in Each Column")
                _prows = [{
                    "Column":    c.get("name"),
                    "Kind":      c.get("kind"),
                    "Dtype":     c.get("dtype"),
                    "Missing %": c.get("missing_pct"),
                    "Unique":    c.get("nunique"),
                    "Flags":     ", ".join(c.get("flags", [])),
                } for c in prof.get("columns", [])]
                st.dataframe(_safe_df(pd.DataFrame(_prows)), width='stretch')

            if clean_out:
                st.markdown("#### What Got Cleaned Up")
                _c1, _c2, _c3 = st.columns(3)
                _c1.metric("How", clean_out.get("strategy_used", "—"))
                _c2.metric("Missing values before", clean_out.get("missing_before", "—"))
                _c3.metric("Missing values after",  clean_out.get("missing_after", "—"))

        with st.expander("Unusual Rows & How Columns Relate", expanded=False):
            if outlier_out:
                st.markdown("#### Rows That Don't Fit the Pattern")
                _c1, _c2 = st.columns(2)
                _c1.metric("Unusual rows found", outlier_out.get("total_outliers", "—"))
                _c2.metric("Share of all rows", f"{outlier_out.get('outlier_percentage','—')}%")
                _pc = outlier_out.get("per_column_outliers", {})
                if _pc:
                    _pc_df = pd.DataFrame(
                        [(c, v) for c, v in _pc.items() if v > 0],
                        columns=["Column", "Unusual values"],
                    ).sort_values("Unusual values", ascending=False)
                    if not _pc_df.empty:
                        st.dataframe(_safe_df(_pc_df), width='stretch')

        with st.expander("Is the Pattern Real?", expanded=False):
            if stat_out:
                _c1, _c2, _c3 = st.columns(3)
                _c1.metric("Test used", stat_out.get("test_name", "—"))
                _c2.metric("p-value", f"{stat_out.get('p_value', 0):.4f}")
                _c3.metric("Likely real, not chance", "Yes" if stat_out.get("significant") else "No")
                st.info(stat_out.get("interpretation", "No interpretation recorded."))
            else:
                st.caption("No statistical test was needed for this run.")

        with st.expander("Other Analyses", expanded=False):
            st.caption("Segmentation, trends, text, and geography — run when your data called for them.")
            _render_other_findings(tool_results)

    # ═════════════════════════════════════════════════════════════════════════
    # TIER 5: ARTIFACT VAULT & EXPORTS
    # ═════════════════════════════════════════════════════════════════════════
    with tab_vault:
        st.markdown("### Downloads")
        st.caption("Everything from this run, ready to keep or share.")

        if tmp_dir:
            _out = Path(tmp_dir) / "output"
            _rdir = _out / "reports"

            col_dl1, col_dl2 = st.columns(2)

            with col_dl1:
                st.markdown("#### Reports")
                _html = _rdir / "report.html"
                if _html.exists():
                    st.download_button(
                        "📄 Download Shareable HTML Report",
                        _html.read_bytes(), "report.html", mime="text/html",
                        key="dl_html_vault", type="primary",
                    )
                _mds = sorted(_rdir.glob("*.md")) if _rdir.exists() else []
                if _mds:
                    st.download_button(
                        f"📝 Download Markdown Report ({_mds[0].name})",
                        _mds[0].read_bytes(), _mds[0].name, mime="text/markdown",
                        key="dl_md_vault",
                    )
                st.download_button(
                    "💾 Download Raw Data (final_report.json)",
                    json.dumps(report, indent=2, default=str),
                    "final_report.json", mime="application/json",
                    key="dl_json_vault",
                )

            with col_dl2:
                st.markdown("#### Trained Models")
                _mdir = _out / "models"
                if _mdir.exists() and any(_mdir.iterdir()):
                    for _mdl_f in sorted(_mdir.iterdir()):
                        st.download_button(
                            f"📦 Download Model: {_mdl_f.name}",
                            _mdl_f.read_bytes(), _mdl_f.name,
                            mime="application/octet-stream",
                            key=f"dlm_vault_{_mdl_f.name}",
                        )
                else:
                    st.caption("No models were saved for this run.")

            st.divider()
            st.markdown("#### Report Preview")
            if _mds:
                st.markdown(_mds[0].read_text(encoding="utf-8"))
            else:
                st.json(report)

            with st.expander("Full Technical Log", expanded=False):
                st.json(tool_results)


# ══════════════════════════════════════════════════════════════════════════════
# EMPTY STATE
# ══════════════════════════════════════════════════════════════════════════════
if (preview_df is None
        and not st.session_state["analysis_done"]
        and not st.session_state["stage_log"]):
    st.markdown(
        '<div class="empty">'
        '<h2>Let\'s see what your data shows.</h2>'
        '<p>Add a CSV or Excel file in the sidebar, tell us what you\'d like to know, '
        'and enter your API key. Your helpers will study it, test their answers, '
        'and double-check everything before showing you the results.</p>'
        '<div class="steps">'
        '<div>1. Read</div><div>2. Understand</div><div>3. Run</div>'
        '<div>4. Explain</div><div>5. Check</div><div>6. Solve</div>'
        '<div>7. Report</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("#### Meet Your Helpers")
    st.markdown(_render_agent_grid([]), unsafe_allow_html=True)
    st.write("")
    if st.button("▶ See a Sample Report (Demo)", type="primary"):
        _load_teamwork_preview()
        st.rerun()
