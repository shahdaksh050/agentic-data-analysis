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

from src.core.multiple_testing import apply_benjamini_hochberg

# "Ledger" (DESIGN.md): warm paper, friendly ink, one terracotta pen.
# The shared report is the same warm sheet as the console, printed.
_CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body {
  background: #f7eedd;
  color: #3a2b1e;
  font-family: 'Mukta', 'Segoe UI', system-ui, sans-serif;
  margin: 0; padding: 3.5rem 1.5rem; line-height: 1.65;
}
.wrap { max-width: 900px; margin: 0 auto; }
h1 { font-family: 'Baloo 2', 'Mukta', sans-serif; font-weight: 800;
     font-size: clamp(32px, 5vw, 52px); line-height: 1.06;
     margin: .3rem 0 .2rem; max-width: 18ch; color: #3a2b1e; }
h2 { font-family: 'Baloo 2', 'Mukta', sans-serif; font-weight: 700; font-size: 25px;
     line-height: 1.15; color: #3a2b1e;
     border-bottom: 2px solid #e4d4bc; padding-bottom: .4rem; margin: 3rem 0 1rem; }
.sub { color: #8a7660; font-size: .85rem; margin-bottom: 2rem; }
.objective { border-left: 4px solid #a34f20; border-radius: 0 10px 10px 0;
             background: #fffbf2; padding: .6rem 0 .6rem 1rem;
             margin: 1.4rem 0; max-width: 72ch; }
.card { padding: .5rem 0 .5rem 1rem; margin: .1rem 0 .8rem;
        border-left: 3px solid #e4d4bc; border-radius: 0 10px 10px 0;
        background: #fffbf2; max-width: 74ch; }
.card.exec { background: #fffbf2; border: 1px solid #e4d4bc; border-left: 4px solid #a34f20;
             border-radius: 14px; box-shadow: 0 4px 14px rgba(58,43,30,.12);
             padding: 1.5rem 1.7rem; max-width: 72ch; }
.insight { border-left-color: #8a7660; }
.rec { border-left-color: #a34f20; }
.driver { border-left-color: #8a7660; font-size: .95rem; }
.treat { border-left-color: #e4d4bc; font-size: .9rem; color: #8a7660; }
.warn { border-left-color: #a33526; color: #a33526; }
table { border-collapse: separate; border-spacing: 0; width: 100%; font-size: .88rem;
        background: #fffbf2; border: 1px solid #e4d4bc; border-radius: 12px; overflow: hidden;
        box-shadow: 0 4px 14px rgba(58,43,30,.10); }
th, td { border-bottom: 1px solid #eee3cb; padding: .55rem .8rem; text-align: left; }
th { background: #f1e4cb; color: #3a2b1e; font-weight: 700; font-size: .8rem; }
.chart { background: #fffbf2; border: 1px solid #e4d4bc; border-radius: 14px;
         box-shadow: 0 4px 14px rgba(58,43,30,.10);
         padding: 1.2rem 1.3rem 1.1rem; margin: 1.4rem 0; }
.chart h3 { margin: .1rem 0 .2rem; font-weight: 700; font-size: 1.05rem;
            font-family: 'Baloo 2', 'Mukta', sans-serif; }
.chart p { margin: .15rem 0 .9rem; color: #8a7660; font-size: .85rem; max-width: 68ch; }
.vega-holder { width: 100%; }
.badge { display: inline-block; border: 1px solid #a34f20; color: #a34f20;
         background: #fffbf2; border-radius: 999px; font-weight: 600;
         padding: .25rem .85rem; font-size: .78rem; margin: 0 .4rem .4rem 0; }
.footer { margin-top: 4rem; border-top: 1px solid #e4d4bc; padding-top: .8rem;
          color: #8a7660; font-size: .8rem; }
"""

_FONTS_CDN = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Baloo+2:wght@500;600;700;800'
    '&family=Mukta:wght@400;500;600;700&display=swap" rel="stylesheet">'
)

_VEGA_CDN = (
    '<script src="https://cdn.jsdelivr.net/npm/vega@5"></script>'
    '<script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>'
    '<script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>'
)

_VEGA_PLOT_CONFIG: dict[str, Any] = {
    "background": "#fffbf2",
    "font": "Mukta, 'Segoe UI', sans-serif",
    "axis": {"labelColor": "#8a7660",
             "titleColor": "#8a7660",
             "gridColor": "#e4d4bc",
             "gridDash": [2, 3],
             "domainColor": "#3a2b1e",
             "tickColor": "#3a2b1e",
             "labelFont": "Mukta, sans-serif",
             "labelFontSize": 11,
             "titleFont": "Baloo 2, sans-serif",
             "titleFontWeight": 600},
    "legend": {"labelColor": "#3a2b1e",
               "titleColor": "#8a7660",
               "labelFont": "Mukta, sans-serif",
               "titleFont": "Baloo 2, sans-serif",
               "symbolType": "square"},
    "view": {"stroke": "transparent"},
    "range": {"category": ["#a34f20", "#a33526", "#c08a2e",
                           "#5b8c5a", "#8a7660", "#b5714a"]},
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
    read_report: dict[str, Any] | None = None,
    coercions: list[dict[str, Any]] | None = None,
    plan_rationales: list[dict[str, Any]] | None = None,
    statistical_test_pvalues: list[dict[str, Any]] | None = None,
    unverified_claims: list[str] | None = None,
    profile_status: str | None = None,
    degradations: list[str] | None = None,
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
        read_report:  src.core.io.ReadReport as a dict, if available (item 2).
        coercions:    src.core.coercion.Coercion dicts, if any (item 3).
        plan_rationales: [{step_number, tool_name, rationale}, ...] (item 6).
        statistical_test_pvalues: p-values accumulated this run, for the
            Benjamini-Hochberg correction (item 4).
        unverified_claims: Numeric literals P0.7 couldn't trace to a tool result.
        profile_status: "ok" or "failed: <reason>" (item 7).
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    sections: list[str] = []

    # ---- header + badges ----
    badges = ""
    if profile:
        badges += f'<span class="badge">quality {_esc(profile.get("quality_score", "?"))}/100</span>'
        badges += f'<span class="badge">{_esc(profile.get("row_count", "?"))} rows</span>'
        badges += f'<span class="badge">{_esc(profile.get("column_count", "?"))} columns</span>'
        # Mirrors the Markdown report's "Recognised as ..." line so the two
        # reports do not diverge on how the data was classified.
        for match in profile.get("domains") or []:
            badges += (
                f'<span class="badge">{_esc(match.get("domain"))} data '
                f'({float(match.get("confidence", 0)):.2f})</span>'
            )
    best_model = llm_insights.get("best_model")
    if best_model:
        badges += f'<span class="badge">best model: {_esc(best_model)}</span>'

    sections.append(
        f"<h1>What we found in {_esc(dataset_name)}</h1>"
        f'<div class="sub">Prepared {timestamp} by your data assistant</div>'
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

    # ---- methodology (item 6) — why each step ran, from the planner ----
    if plan_rationales:
        rows = "".join(
            f"<tr><td>{_esc(r.get('step_number', '—'))}</td>"
            f"<td>{_esc(r.get('tool_name', '—'))}</td>"
            f"<td>{_esc(r.get('rationale', ''))}</td></tr>"
            for r in plan_rationales
        )
        sections.append(
            "<h2>Why these analyses</h2><table><tr><th>Step</th><th>Tool</th>"
            "<th>Rationale</th></tr>" + rows + "</table>"
        )

    # ---- limitations & caveats (item 4 / 5 / 7 / 10 / P0.7) ----
    if degradations is None:
        # Direct callers without an accumulated degradations log (tests,
        # scripts) still get the same information, derived on the spot.
        from src.core.degradations import collect_degradations

        degradations = collect_degradations(read_report, coercions, profile, profile_status)
    limitation_cards: list[str] = list(degradations)
    if unverified_claims:
        limitation_cards.extend(str(c) for c in unverified_claims)

    bh = apply_benjamini_hochberg(statistical_test_pvalues or [])
    if limitation_cards or bh:
        sections.append("<h2>Limitations &amp; caveats</h2>" + _cards(limitation_cards, "warn"))
        if bh:
            bh_rows = []
            for t in bh:
                p_val = f"{t.get('p_value', 0):.4f}"
                p_adj = f"{t.get('p_adjusted', 0):.4f}"
                sig = "Yes" if t.get("significant_after_correction") else "No"
                bh_rows.append(
                    f"<tr><td>{_esc(t.get('feature_column', '—'))}</td>"
                    f"<td>{_esc(t.get('test_name', '—'))}</td>"
                    f"<td>{_esc(p_val)}</td><td>{_esc(p_adj)}</td><td>{sig}</td></tr>"
                )
            rows = "".join(bh_rows)
            sections.append(
                f"<p>{len(bh)} statistical test(s) ran this session — "
                "Benjamini-Hochberg-corrected significance (FDR, α=0.05):</p>"
                "<table><tr><th>Feature</th><th>Test</th><th>p-value</th>"
                "<th>BH-adjusted p</th><th>Significant after correction</th></tr>"
                + rows + "</table>"
            )

    sections.append(
        '<div class="footer">Made for you by your data assistant.</div>'
    )

    body = "\n".join(sections)
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>Analysis Report — {_esc(dataset_name)}</title>"
        f"{_FONTS_CDN}{_VEGA_CDN}<style>{_CSS}</style></head>"
        f"<body><div class='wrap'>{body}</div></body></html>"
    )
