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

# Sauce Labs style reference (DESIGN.md): obsidian canvas, single neon-green
# accent, mint-frost light card as the executive-summary counterpoint.
_CSS = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { background:#132322; color:rgba(255,255,255,.86);
       font-family:'Inter','Segoe UI',system-ui,sans-serif;
       margin:0; padding:3rem 1.5rem; line-height:1.65; letter-spacing:-.08px; }
.wrap { max-width: 980px; margin: 0 auto; }
h1 { color:#fff; font-family:'Inter Tight','Inter',sans-serif; font-weight:500;
     font-size:2.1rem; line-height:1.12; margin:.55rem 0 .25rem; }
h2 { font-family:'Inter Tight','Inter',sans-serif; font-weight:500;
     font-size:11px; letter-spacing:1.6px; text-transform:uppercase;
     color:#3ddc91; margin:2.8rem 0 .9rem; }
.eyebrow { font-family:'Inter Tight','Inter',sans-serif; font-weight:500;
           font-size:10px; letter-spacing:1.6px; text-transform:uppercase;
           color:#3ddc91; }
.sub { color:rgba(255,255,255,.45); font-size:.85rem; margin-bottom:1.5rem; }
.objective { background:rgba(151,221,188,.10); border:1px solid rgba(151,221,188,.35);
             border-radius:16px; padding:1rem 1.2rem; margin:1.2rem 0; color:#d9efe6; }
.card { background:#0e1a19; border:1px solid rgba(255,255,255,.07);
        border-radius:16px; padding:1rem 1.2rem; margin:.55rem 0; }
.card.exec { background:#edf7f5; color:#132322; border:none; border-radius:20px;
             padding:1.6rem 1.8rem; box-shadow:rgba(0,0,0,.04) 1px 0 9px 2px; }
.insight { border-left:3px solid #97ddbc; }
.rec { border-left:3px solid #3ddc91; }
.driver { border-left:3px solid #ffcd48; font-size:.92rem; }
.treat { border-left:3px solid rgba(255,255,255,.25); font-size:.88rem;
         color:rgba(255,255,255,.6); }
table { border-collapse:collapse; width:100%; font-size:.88rem; }
th, td { border:1px solid rgba(255,255,255,.10); padding:.5rem .75rem; text-align:left; }
th { background:#0e1a19; color:rgba(255,255,255,.55);
     font-family:'Inter Tight','Inter',sans-serif; font-weight:500;
     font-size:10px; letter-spacing:1.2px; text-transform:uppercase; }
.chart { background:#0e1a19; border:1px solid rgba(255,255,255,.07);
         border-radius:20px; padding:1.2rem 1.3rem 1rem; margin:1.3rem 0; }
.chart h3 { margin:.1rem 0 .2rem; color:#fff;
            font-family:'Inter Tight','Inter',sans-serif; font-weight:500;
            font-size:1.02rem; }
.chart p { margin:.15rem 0 .9rem; color:rgba(255,255,255,.45); font-size:.82rem; }
.vega-holder { width:100%; }
.badge { display:inline-block; background:rgba(61,220,145,.10);
         border:1px solid rgba(61,220,145,.45); color:#3ddc91;
         border-radius:99px; padding:.2rem .9rem; font-size:.78rem;
         margin:0 .4rem .4rem 0; }
.footer { margin-top:3.5rem; color:rgba(255,255,255,.3); font-size:.78rem;
          text-align:center; }
"""

_FONTS_CDN = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500'
    '&family=Inter+Tight:wght@400;500&display=swap" rel="stylesheet">'
)

_VEGA_CDN = (
    '<script src="https://cdn.jsdelivr.net/npm/vega@5"></script>'
    '<script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>'
    '<script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>'
)

_VEGA_DARK_CONFIG: dict[str, Any] = {
    "background": "#0e1a19",
    "font": "Inter, sans-serif",
    "axis": {"labelColor": "rgba(255,255,255,0.62)",
             "titleColor": "rgba(255,255,255,0.62)",
             "gridColor": "rgba(255,255,255,0.07)",
             "domainColor": "rgba(255,255,255,0.18)",
             "tickColor": "rgba(255,255,255,0.18)",
             "labelFont": "Inter, sans-serif",
             "titleFont": "Inter, sans-serif"},
    "legend": {"labelColor": "rgba(255,255,255,0.72)",
               "titleColor": "rgba(255,255,255,0.72)",
               "labelFont": "Inter, sans-serif",
               "titleFont": "Inter, sans-serif"},
    "view": {"stroke": "transparent"},
    "range": {"category": ["#3ddc91", "#ffcd48", "#97ddbc",
                           "#1c8f5c", "#d6f0b2", "#62b5a4"]},
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
        '<div class="eyebrow">Agentic Data Analysis · Report</div>'
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
        sections.append(
            f"<h2>Executive Summary</h2><div class='card exec'>{_esc(reasoning)}</div>"
        )

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
        f"{_FONTS_CDN}{_VEGA_CDN}<style>{_CSS}</style></head>"
        f"<body><div class='wrap'>{body}</div></body></html>"
    )
