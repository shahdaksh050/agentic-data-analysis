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

_CSS = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { background:#0f1117; color:#d6d9e0; font-family:'Segoe UI',system-ui,sans-serif;
       margin:0; padding:2.5rem 1.5rem; line-height:1.65; }
.wrap { max-width: 980px; margin: 0 auto; }
h1 { color:#fff; font-size:1.7rem; margin:0 0 .2rem; }
h2 { color:#93b4f7; font-size:1.15rem; margin:2.2rem 0 .8rem;
     border-bottom:1px solid #2a2d3e; padding-bottom:.4rem; }
.sub { color:#777; font-size:.85rem; margin-bottom:1.6rem; }
.objective { background:#0c1228; border:1px solid #3b5bdb; border-radius:8px;
             padding:.9rem 1.1rem; margin:1rem 0; color:#b9ccf5; }
.card { background:#1a1d27; border:1px solid #2a2d3e; border-radius:8px;
        padding:1rem 1.2rem; margin:.5rem 0; }
.insight { border-left:3px solid #3b5bdb; }
.rec { border-left:3px solid #1a7a4a; }
.driver { border-left:3px solid #e67e22; font-size:.92rem; }
.treat { border-left:3px solid #8e6fd8; font-size:.88rem; color:#b9a8e8; }
table { border-collapse:collapse; width:100%; font-size:.88rem; }
th, td { border:1px solid #2a2d3e; padding:.45rem .7rem; text-align:left; }
th { background:#161924; color:#9aa3b5; }
.chart { background:#12141d; border:1px solid #2a2d3e; border-radius:8px;
         padding:1rem; margin:1.2rem 0; }
.chart h3 { margin:.1rem 0 .2rem; color:#e0e0e0; font-size:1rem; }
.chart p { margin:.1rem 0 .8rem; color:#888; font-size:.82rem; }
.vega-holder { width:100%; }
.badge { display:inline-block; background:#0c1c14; border:1px solid #1a7a4a;
         color:#7ed9a5; border-radius:99px; padding:.15rem .8rem;
         font-size:.78rem; margin-right:.4rem; }
.footer { margin-top:3rem; color:#555; font-size:.78rem; text-align:center; }
"""

_VEGA_CDN = (
    '<script src="https://cdn.jsdelivr.net/npm/vega@5"></script>'
    '<script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>'
    '<script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>'
)

_VEGA_DARK_CONFIG: dict[str, Any] = {
    "background": "#12141d",
    "axis": {"labelColor": "#bbb", "titleColor": "#bbb", "gridColor": "#2a2d3e",
             "domainColor": "#333", "tickColor": "#333"},
    "legend": {"labelColor": "#bbb", "titleColor": "#bbb"},
    "view": {"stroke": "#2a2d3e"},
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
        f"<h1>Analysis Report — {_esc(dataset_name)}</h1>"
        f'<div class="sub">Generated {timestamp} · Agentic Data Analysis System</div>'
        f"<div>{badges}</div>"
    )

    # ---- objective + executive summary ----
    if objective:
        sections.append(
            f'<div class="objective"><b>Your question:</b> {_esc(objective)}</div>'
        )
    reasoning = llm_insights.get("reasoning", "")
    if reasoning:
        sections.append(f"<h2>Executive Summary</h2><div class='card'>{_esc(reasoning)}</div>")

    # ---- insights & recommendations ----
    insights = llm_insights.get("insights") or []
    if insights:
        sections.append("<h2>Key Insights</h2>" + _cards(insights, "insight", "💡 "))
    recs = llm_insights.get("recommendations") or []
    if recs:
        sections.append("<h2>Recommendations</h2>" + _cards(recs, "rec", "→ "))

    # ---- drivers (explainability) ----
    eval_out = _find_tool_output(tool_results, "evaluate_model")
    narrative = eval_out.get("driver_narrative") or []
    if narrative:
        sections.append("<h2>What Drives the Predictions</h2>" + _cards(narrative, "driver"))

    # ---- automatic treatments ----
    train_out = _find_tool_output(tool_results, "train_model")
    treatments = train_out.get("treatments_applied") or []
    if treatments:
        sections.append(
            "<h2>Automatic Data Treatments</h2>" + _cards(treatments, "treat", "🛠 ")
        )

    # ---- key metrics ----
    key_metrics = llm_insights.get("key_metrics") or {}
    if key_metrics:
        rows = "".join(
            f"<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>" for k, v in key_metrics.items()
        )
        sections.append(
            "<h2>Key Metrics</h2><table><tr><th>Metric</th><th>Value</th></tr>"
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
        specs = [dict(c.get("spec", {}), config=_VEGA_DARK_CONFIG, width="container")
                 for c in charts]
        specs_json = json.dumps(specs, default=str).replace("</", "<\\/")
        sections.append(
            "<h2>Interactive Dashboard</h2>"
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
            "<h2>Analysis Steps</h2><table><tr><th>Tool</th><th>Status</th>"
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
        f"{_VEGA_CDN}<style>{_CSS}</style></head>"
        f"<body><div class='wrap'>{body}</div></body></html>"
    )
