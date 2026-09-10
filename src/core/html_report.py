"""
HTML Report Builder — a single shareable artifact for the whole analysis.

Produces a self-contained `report.html`: executive summary driven by the
user's objective, key insights, model drivers, metrics, the full interactive
dashboard (Vega-Lite via the vega-embed CDN), and the tool execution log.

Rules:
  - Pure string building, no I/O — the controller writes the file.
  - Every dataset- or LLM-derived string is HTML-escaped.
  - Charts render client-side from embedded JSON; viewing needs internet
    access for the CDN scripts (acceptable for a shareable artifact).
"""
from __future__ import annotations

import html
import json
import time
from typing import Any

# "Drafting Table" (DESIGN.md): mineral stock, ink linework, two plotter pens.
# The shared report is the same sheet as the console, printed.
_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body {
  background: #dcdbd3;
  background-image:
    repeating-linear-gradient(to right,  rgba(23,28,31,.045) 0 1px, transparent 1px 28px),
    repeating-linear-gradient(to bottom, rgba(23,28,31,.045) 0 1px, transparent 1px 28px);
  color: #171c1f;
  font-family: 'Archivo', 'Segoe UI', system-ui, sans-serif;
  margin: 0; padding: 3.5rem 1.5rem; line-height: 1.62;
}
.wrap { max-width: 940px; margin: 0 auto; }
h1 { font-variation-settings: 'wdth' 118; font-weight: 800;
     font-size: clamp(34px, 5.4vw, 58px); line-height: .95; letter-spacing: -.038em;
     margin: .3rem 0 .2rem; max-width: 18ch; }
h2 { font-variation-settings: 'wdth' 112; font-weight: 700; font-size: 27px;
     letter-spacing: -.028em; line-height: 1.05; color: #171c1f;
     border-bottom: 1px solid #171c1f; padding-bottom: .35rem; margin: 3rem 0 1rem; }
.sub { color: #54585b; font-family: 'IBM Plex Mono', monospace; font-size: .78rem;
       margin-bottom: 2rem; }
.objective { border-left: 3px solid #12467e; padding: .3rem 0 .3rem 1rem;
             margin: 1.4rem 0; max-width: 72ch; }
.card { padding: .35rem 0 .35rem 1rem; margin: .1rem 0 .8rem;
        border-left: 3px solid #c8c6bc; max-width: 74ch; }
.card.exec { background: #efeee8; border: 1px solid #171c1f;
             box-shadow: 3px 3px 0 rgba(23,28,31,.09);
             padding: 1.4rem 1.6rem; max-width: 72ch; }
.insight { border-left-color: #54585b; }
.rec { border-left-color: #12467e; }
.driver { border-left-color: #54585b; font-size: .93rem; }
.treat { border-left-color: #c8c6bc; font-size: .88rem; color: #54585b; }
.warn { border-left-color: #b5271a; color: #b5271a; }
table { border-collapse: collapse; width: 100%; font-size: .86rem;
        background: #efeee8; box-shadow: 3px 3px 0 rgba(23,28,31,.09); }
th, td { border: 1px solid #c8c6bc; padding: .45rem .7rem; text-align: left; }
td { font-family: 'IBM Plex Mono', monospace; font-size: .8rem; }
th { background: #e4e3dc; color: #171c1f; font-weight: 700; font-size: .78rem;
     border-color: #171c1f; }
.chart { background: #efeee8; border: 1px solid #171c1f;
         box-shadow: 3px 3px 0 rgba(23,28,31,.09);
         padding: 1.1rem 1.2rem 1rem; margin: 1.4rem 0; }
.chart h3 { margin: .1rem 0 .2rem; font-weight: 700; font-size: 1rem;
            letter-spacing: -.02em; }
.chart p { margin: .15rem 0 .9rem; color: #54585b; font-size: .82rem; max-width: 68ch; }
.vega-holder { width: 100%; }
.badge { display: inline-block; border: 1px solid #12467e; color: #12467e;
         font-family: 'IBM Plex Mono', monospace; padding: .18rem .7rem;
         font-size: .74rem; margin: 0 .35rem .35rem 0; }
.footer { margin-top: 4rem; border-top: 1px solid #171c1f; padding-top: .7rem;
          color: #54585b; font-family: 'IBM Plex Mono', monospace; font-size: .74rem; }
"""

_FONTS_CDN = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@75..125,400..800'
    '&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">'
)

_VEGA_CDN = (
    '<script src="https://cdn.jsdelivr.net/npm/vega@5"></script>'
    '<script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>'
    '<script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>'
)

_VEGA_PLOT_CONFIG: dict[str, Any] = {
    "background": "#efeee8",
    "font": "Archivo, 'Segoe UI', sans-serif",
    "axis": {"labelColor": "#54585b",
             "titleColor": "#54585b",
             "gridColor": "#c8c6bc",
             "gridDash": [2, 3],
             "domainColor": "#171c1f",
             "tickColor": "#171c1f",
             "labelFont": "IBM Plex Mono, monospace",
             "labelFontSize": 11,
             "titleFont": "Archivo, sans-serif",
             "titleFontWeight": 600},
    "legend": {"labelColor": "#171c1f",
               "titleColor": "#54585b",
               "labelFont": "Archivo, sans-serif",
               "titleFont": "Archivo, sans-serif",
               "symbolType": "square"},
    "view": {"stroke": "transparent"},
    "range": {"category": ["#12467e", "#b5271a", "#7a8b99",
                           "#c08a2e", "#3f6f5b", "#8e6e9e"]},
}


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _cards(items: list[Any], css_class: str, prefix: str = "") -> str:
    return "\n".join(
        f'<div class="card {css_class}">{prefix}{_esc(item)}</div>' for item in items
    )


def _find_tool_output(tool_results: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for r in reversed(tool_results):
        if r.get("tool_name") == name and r.get("status") == "success":
            out = r.get("output")
            if isinstance(out, dict):
                return out
    return {}


def build_html_report(
    dataset_name: str,
    llm_insights: dict[str, Any],
    tool_results: list[dict[str, Any]],
    charts: list[dict[str, Any]],
    objective: str = "",
    profile: dict[str, Any] | None = None,
) -> str:
    """
    Assemble the full self-contained HTML report.

    Args:
        dataset_name: Display name for the header.
        llm_insights: Final report dict (reasoning/insights/recommendations/...).
        tool_results: Serialised ToolResult dicts.
        charts:       Dashboard ChartSpec dicts ({chart_id,title,description,spec}).
        objective:    The user's natural-language goal, if any.
        profile:      DatasetProfile.to_dict(), if available.
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    sections: list[str] = []

    # ---- header + badges ----
    badges = ""
    if profile:
        badges += f'<span class="badge">quality {_esc(profile.get("quality_score", "?"))}/100</span>'
        badges += f'<span class="badge">{_esc(profile.get("row_count", "?"))} rows</span>'
        badges += f'<span class="badge">{_esc(profile.get("column_count", "?"))} columns</span>'
    best_model = llm_insights.get("best_model")
    if best_model:
        badges += f'<span class="badge">best model: {_esc(best_model)}</span>'

    sections.append(
        f"<h1>What we found in {_esc(dataset_name)}</h1>"
        f'<div class="sub">Drawn up {timestamp} by the Agentic Data Analysis '
        f"System</div>"
        f"<div>{badges}</div>"
    )

    # ---- objective + executive summary ----
    if objective:
        sections.append(
            f'<div class="objective">You asked: {_esc(objective)}</div>'
        )
    reasoning = llm_insights.get("reasoning", "")
    if reasoning:
        sections.append(
            f"<h2>The short version</h2><div class='card exec'>{_esc(reasoning)}</div>"
        )

    # ---- insights & recommendations ----
    insights = llm_insights.get("insights") or []
    if insights:
        sections.append("<h2>What the data shows</h2>" + _cards(insights, "insight"))
    recs = llm_insights.get("recommendations") or []
    if recs:
        sections.append("<h2>What to do next</h2>" + _cards(recs, "rec"))

    # ---- drivers (explainability) ----
    eval_out = _find_tool_output(tool_results, "evaluate_model")
    narrative = eval_out.get("driver_narrative") or []
    if narrative:
        sections.append("<h2>What drives the predictions</h2>" + _cards(narrative, "driver"))

    # ---- automatic treatments ----
    train_out = _find_tool_output(tool_results, "train_model")
    treatments = train_out.get("treatments_applied") or []
    if treatments:
        sections.append(
            "<h2>What the agent changed before modelling</h2>"
            + _cards(treatments, "treat")
        )

    # ---- key metrics ----
    key_metrics = llm_insights.get("key_metrics") or {}
    if key_metrics:
        rows = "".join(
            f"<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>" for k, v in key_metrics.items()
        )
        sections.append(
            "<h2>Key numbers</h2><table><tr><th>Metric</th><th>Value</th></tr>"
            + rows + "</table>"
        )

    # ---- interactive dashboard ----
    if charts:
        chart_divs = "".join(
            f'<div class="chart"><h3>{_esc(c.get("title", ""))}</h3>'
            f'<p>{_esc(c.get("description", ""))}</p>'
            f'<div class="vega-holder" id="chart_{i}"></div></div>'
            for i, c in enumerate(charts)
        )
        specs = [dict(c.get("spec", {}), config=_VEGA_PLOT_CONFIG, width="container")
                 for c in charts]
        specs_json = json.dumps(specs, default=str).replace("</", "<\\/")
        sections.append(
            "<h2>The charts</h2>"
            + chart_divs
            + f"<script>const SPECS = {specs_json};"
            + "SPECS.forEach((s, i) => vegaEmbed('#chart_' + i, s, {actions: false}));"
            + "</script>"
        )

    # ---- tool log ----
    if tool_results:
        rows = "".join(
            f"<tr><td>{_esc(r.get('tool_name', '?'))}</td>"
            f"<td>{_esc(r.get('status', '?'))}</td>"
            f"<td>{_esc(str(r.get('output', {}).get('summary', r.get('error', '')) or '')[:140])}</td></tr>"
            for r in tool_results
        )
        sections.append(
            "<h2>Every step it ran</h2><table><tr><th>Tool</th><th>Status</th>"
            "<th>Summary</th></tr>" + rows + "</table>"
        )

    sections.append(
        '<div class="footer">Agentic Data Analysis System · reasoning ↔ execution '
        "separation with RLM context offloading</div>"
    )

    body = "\n".join(sections)
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>Analysis Report — {_esc(dataset_name)}</title>"
        f"{_FONTS_CDN}{_VEGA_CDN}<style>{_CSS}</style></head>"
        f"<body><div class='wrap'>{body}</div></body></html>"
    )
