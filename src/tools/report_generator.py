"""
Report Generator Tool — Execution Layer.

Stage 7: Report Generation.

Compiles all tool results, LLM insights, and metric summaries into:
  - A structured Markdown report  (output/reports/analysis_report.md)
  - A raw JSON dump               (output/reports/final_report.json)

The Markdown report is human-readable and suitable for conversion to PDF
via pandoc or any Markdown renderer.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.core.multiple_testing import DEFAULT_ALPHA as _BH_ALPHA
from src.core.multiple_testing import apply_benjamini_hochberg as _apply_benjamini_hochberg
from src.tools.base import BaseTool

if TYPE_CHECKING:
    from src.core.memory import MemorySystem


def _format_data_overview(
    data_profile: dict[str, Any] | None,
    read_report: dict[str, Any] | None,
    coercions: list[dict[str, Any]] | None,
) -> list[str]:
    """Shape, quality, and — critically — what was *detected or assumed* at
    read time (item 2) and *repaired* before profiling (item 3), so the
    report explains what it's actually looking at, not just what it found."""
    if not (data_profile or read_report or coercions):
        return []
    lines = ["## Data Overview", ""]
    if data_profile:
        kind_counts: dict[str, int] = {}
        for col in data_profile.get("columns", []):
            kind = col.get("kind", "?")
            kind_counts[kind] = kind_counts.get(kind, 0) + 1
        kinds_str = ", ".join(f"{v} {k}" for k, v in sorted(kind_counts.items()))
        lines.append(
            f"- **Shape**: {data_profile.get('row_count', '—')} rows × "
            f"{data_profile.get('column_count', '—')} columns"
        )
        lines.append(f"- **Column kinds**: {kinds_str or '—'}")
        lines.append(f"- **Quality score**: {data_profile.get('quality_score', '—')}/100")
        lines.append(f"- **Duplicate rows**: {data_profile.get('duplicate_rows', '—')}")
        if not data_profile.get("is_sufficient", True):
            lines.append(
                f"- **⚠ Data sufficiency**: {data_profile.get('sufficiency_reason') or 'Insufficient data.'}"
            )
        lines.append("")
    if read_report:
        bits = [
            f"format `{read_report.get('format')}`",
            f"encoding `{read_report.get('encoding')}`"
            + ("" if read_report.get("encoding_confident", True) else " (guessed)"),
        ]
        if read_report.get("delimiter"):
            bits.append(
                f"delimiter `{read_report.get('delimiter')!r}`"
                + ("" if read_report.get("delimiter_sniffed") else " (assumed from extension)")
            )
        lines.append(f"**Detected at read time**: {', '.join(bits)}.")
        for note in read_report.get("notes") or []:
            lines.append(f"  - ⚠ {note}")
        lines.append("")
    if coercions:
        lines.append(f"**Repaired before analysis** ({len(coercions)} column(s)):")
        lines.append("")
        lines.append("| Column | Rule | Converted | Failed |")
        lines.append("|--------|------|-----------|--------|")
        for c in coercions:
            lines.append(f"| {c.get('column')} | {c.get('rule')} | {c.get('n_converted')} | {c.get('n_failed')} |")
        lines.append("")
    return lines


def _format_methodology(plan_rationales: list[dict[str, Any]] | None) -> list[str]:
    """The planner is required to justify every step (prompt_manager.py's
    SYSTEM_PROMPT_CORE), but that rationale used to be truncated to 60 chars
    in a console panel and then discarded — this is the narrative the brief
    asks for, generated all along and just never surfaced."""
    if not plan_rationales:
        return []
    lines = [
        "## Methodology",
        "",
        "Why each analysis was chosen, in the planner's own words:",
        "",
        "| Step | Tool | Rationale |",
        "|------|------|-----------|",
    ]
    for r in plan_rationales:
        rationale = str(r.get("rationale", "")).replace("|", "\\|")
        lines.append(f"| {r.get('step_number', '—')} | {r.get('tool_name', '—')} | {rationale} |")
    lines.append("")
    return lines


def _format_limitations(
    data_profile: dict[str, Any] | None,
    statistical_test_pvalues: list[dict[str, Any]] | None,
    unverified_claims: list[str] | None,
    profile_status: str | None,
    degradations: list[str] | None = None,
) -> list[str]:
    """Everything a careful reader needs to know before trusting a number in
    this report: every recorded fallback/repair (item 10's degradations
    log, when the caller has it — the controller always does), the
    multiple-comparisons correction (item 4), and anything the
    verbatim-metric guard (P0.7) couldn't verify."""
    if degradations is None:
        # Direct callers that don't accumulate a degradations log (tests,
        # scripts) still get the same information, derived on the spot.
        from src.core.degradations import collect_degradations

        degradations = collect_degradations(None, None, data_profile, profile_status)
    bh = _apply_benjamini_hochberg(statistical_test_pvalues or [])
    if not (degradations or bh or unverified_claims):
        return []

    lines = ["## Limitations & Caveats", ""]
    for item in degradations:
        lines.append(f"- {item}")
    if degradations:
        lines.append("")

    if bh:
        lines.append(
            f"**Multiple-comparison correction**: {len(bh)} statistical test(s) ran this session. "
            f"Benjamini-Hochberg-corrected significance (FDR, α={_BH_ALPHA}):"
        )
        lines.append("")
        lines.append("| Feature | Test | p-value | BH-adjusted p | Significant after correction |")
        lines.append("|---------|------|---------|----------------|-------------------------------|")
        for t in bh:
            lines.append(
                f"| {t.get('feature_column', '—')} | {t.get('test_name', '—')} | "
                f"{t.get('p_value', 0):.4f} | {t.get('p_adjusted', 0):.4f} | "
                f"{'Yes' if t.get('significant_after_correction') else 'No'} |"
            )
        lines.append("")

    if unverified_claims:
        lines.append("**Unverified claims** (numeric literals in the synthesis not traceable to a tool result):")
        for c in unverified_claims:
            lines.append(f"- {c}")
        lines.append("")

    return lines

#: Tools already covered by a dedicated report section (Model Performance)
#: or not an analytical finding at all — everything else gets a rich
#: subsection below instead of surviving only as one truncated line in the
#: flat Tool Execution Log table.
_BESPOKE_REPORTED_TOOLS = frozenset({
    "ingest_dataset", "clean_data", "detect_outliers", "correlation_analysis",
    "select_statistical_test", "train_model", "evaluate_model",
    "generate_report", "generate_visualizations", "planner",
})


def _format_additional_analyses(tool_results: list[dict[str, Any]]) -> list[str]:
    """
    Markdown subsection per successful tool result without a bespoke
    section above — cluster_data, time_series_analysis, text_analysis,
    geospatial_analysis, dimensionality_analysis, and any future tool.
    Mirrors app.py's `_render_other_findings` so the file-based report and
    the Streamlit UI agree on what "the full picture" contains.
    """
    lines: list[str] = []
    seen: set[str] = set()
    for r in tool_results:
        name = r.get("tool_name", "")
        if name in _BESPOKE_REPORTED_TOOLS or name in seen or r.get("status") != "success":
            continue
        out = r.get("output")
        if not isinstance(out, dict):
            continue
        seen.add(name)
        lines.append(f"### {name.replace('_', ' ').title()}")
        lines.append("")
        summary = out.get("summary")
        if summary:
            lines.append(str(summary))
            lines.append("")

        if name == "cluster_data":
            lines.append(f"- Clusters found: {out.get('n_clusters', '—')}")
            lines.append(f"- Silhouette score: {out.get('silhouette_score', '—')}")
            lines.append(f"- Separation: {out.get('separation_quality', '—')}")
        elif name == "time_series_analysis":
            lines.append(f"- Trend: {out.get('trend_direction', '—')}")
            lines.append(f"- Stationary: {'Yes' if out.get('is_stationary') else 'No'}")
            lags = out.get("seasonal_lags_detected") or []
            lines.append(f"- Seasonal lag(s): {', '.join(str(x) for x in lags) or 'None found'}")
        elif name == "text_analysis":
            lines.append(f"- Vocabulary size: {out.get('vocab_size', '—')}")
            lines.append(f"- Avg. words/row: {out.get('avg_word_count', '—')}")
            top = out.get("top_tokens") or []
            words = [t.get("token", t) if isinstance(t, dict) else t for t in top[:6]]
            if words:
                lines.append(f"- Most frequent words: {', '.join(str(w) for w in words)}")
        elif name == "geospatial_analysis":
            lines.append(f"- Points mapped: {out.get('n_points', '—')}")
            centroid = out.get("centroid") or {}
            if centroid:
                lines.append(f"- Centroid: {centroid.get('lat', '—')}, {centroid.get('lon', '—')}")
        elif name == "dimensionality_analysis":
            lines.append(f"- Numeric features: {out.get('n_features', '—')}")
            lines.append(f"- Components for target variance: {out.get('n_components_for_threshold', '—')}")
            pairs = out.get("high_correlation_pairs") or []
            lines.append(f"- Highly correlated pairs: {len(pairs)}")

        lines.append("")
    return lines


class GenerateReportTool(BaseTool):
    """
    Compile all analysis results into a structured Markdown report.

    Inputs come from the MemorySystem via the `tool_results_json` parameter
    (a JSON string of the serialised tool results list) and the `llm_insights`
    parameter (the LLM's final interpretation dict).
    """

    name = "generate_report"
    description = (
        "Assemble the final analysis report from all tool results and LLM insights. "
        "Produces a Markdown report and a JSON data file. "
        "Returns output file paths."
    )
    output_subdir = "reports"
    uses_cleaned_file = False  # takes no file_path param

    def applies_to(self, profile: Any, metadata: Any) -> float:
        # AgentController._generate_final_report calls this tool directly
        # and unconditionally after the reasoning loop ends (Stage 7) — it
        # is never something the planner itself needs to schedule.
        return 0.0

    def prepare_params(
        self, params: dict[str, Any], memory: MemorySystem, output_root: str
    ) -> dict[str, Any]:
        # The accumulated tool results live in memory, not in anything the
        # LLM plan can supply — always inject them fresh. Note: in practice
        # AgentController._generate_final_report calls this tool directly
        # via .run(), bypassing prepare_params entirely (applies_to() is
        # 0.0, so the planner never schedules it) — that call site injects
        # the same context explicitly. This still fills in the same
        # defaults for any other caller (tests, a future planned use).
        params = super().prepare_params(params, memory, output_root)
        meta = memory.dataset_metadata
        params.setdefault("dataset_name", Path(meta.file_path).stem if meta else "dataset")
        params["tool_results_json"] = json.dumps(
            [r.to_dict() for r in memory.tool_results], default=str
        )
        params.setdefault("llm_insights", {})
        params.setdefault("data_profile", memory.get_context("data_profile"))
        params.setdefault("read_report", memory.get_context("read_report"))
        params.setdefault("coercions", memory.get_context("coercions"))
        params.setdefault("plan_rationales", memory.get_context("plan_rationales"))
        params.setdefault("statistical_test_pvalues", memory.get_context("statistical_test_pvalues"))
        params.setdefault("unverified_claims", memory.get_context("unverified_claims"))
        params.setdefault("profile_status", memory.get_context("profile_status"))
        params.setdefault("degradations", memory.get_context("degradations"))
        return params

    def execute(
        self,
        dataset_name: str = "dataset",
        tool_results_json: str = "[]",
        llm_insights: dict[str, Any] | None = None,
        output_dir: str = "output/reports",
        data_profile: dict[str, Any] | None = None,
        read_report: dict[str, Any] | None = None,
        coercions: list[dict[str, Any]] | None = None,
        plan_rationales: list[dict[str, Any]] | None = None,
        statistical_test_pvalues: list[dict[str, Any]] | None = None,
        unverified_claims: list[str] | None = None,
        profile_status: str | None = None,
        degradations: list[str] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        if llm_insights is None:
            llm_insights = {}
        # LLM sometimes double-serializes llm_insights as a JSON string
        if isinstance(llm_insights, str):
            try:
                llm_insights = json.loads(llm_insights)
            except json.JSONDecodeError:
                llm_insights = {}
        if not isinstance(llm_insights, dict):
            llm_insights = {}

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        if not tool_results_json or not tool_results_json.strip():
            tool_results_json = "[]"
        try:
            _parsed = json.loads(tool_results_json)
        except json.JSONDecodeError:
            try:
                _parsed, _offset = json.JSONDecoder().raw_decode(tool_results_json.strip())
            except json.JSONDecodeError:
                _parsed = []
        tool_results: list[dict[str, Any]] = _parsed if isinstance(_parsed, list) else []

        # Build Markdown
        md_lines: list[str] = [
            "# Agentic Data Analysis Report",
            "",
            f"**Dataset**: {dataset_name}  ",
            f"**Generated**: {timestamp}  ",
            "**Powered by**: Recursive Language Model Inference (Zhang et al., 2024)",
            "",
            "---",
            "",
        ]

        # Data Overview — shape/quality plus what was detected-or-assumed at
        # read time (item 2) and repaired before analysis (item 3).
        md_lines += _format_data_overview(data_profile, read_report, coercions)

        # Executive summary from LLM
        reasoning = llm_insights.get("reasoning", "")
        if reasoning:
            md_lines += ["## Executive Summary", "", reasoning, ""]

        insights = llm_insights.get("insights") or []
        recs = llm_insights.get("recommendations") or []

        if insights:
            md_lines += ["## Key Insights", ""]
            for i, insight in enumerate(insights, 1):
                md_lines.append(f"{i}. {insight}")
            md_lines.append("")

        # Recommendations
        if recs:
            md_lines += ["## Recommendations", ""]
            for rec in recs:
                md_lines.append(f"- {rec}")
            md_lines.append("")

        # Model performance
        best_model = llm_insights.get("best_model")
        key_metrics = llm_insights.get("key_metrics", {})
        if best_model or key_metrics:
            md_lines += ["## Model Performance", ""]
            if best_model:
                md_lines.append(f"**Best model**: {best_model}")
            if key_metrics:
                md_lines.append("")
                md_lines.append("| Metric | Value |")
                md_lines.append("|--------|-------|")
                for k, v in key_metrics.items():
                    md_lines.append(f"| {k} | {v} |")
            md_lines.append("")

        # Additional analyses (cluster/time-series/text/geo/dimensionality —
        # anything without a bespoke section above)
        additional = _format_additional_analyses(tool_results)
        if additional:
            md_lines += ["## Additional Analyses", "", *additional]

        # Methodology — why each analysis was chosen, from the planner's own
        # rationale (previously generated every run and then discarded).
        md_lines += _format_methodology(plan_rationales)

        # Tool execution log
        if tool_results:
            md_lines += ["## Tool Execution Log", "", "| Tool | Status | Summary |", "|------|--------|---------|"]
            for r in tool_results:
                name = r.get("tool_name", "?")
                status = r.get("status", "?")
                summary = r.get("output", {}).get("summary", r.get("error", ""))[:120]
                md_lines.append(f"| {name} | {status} | {summary} |")
            md_lines.append("")

        # Limitations & Caveats — data-quality warnings, multiple-comparison
        # correction (item 4), degraded-profiling notice (item 7), and any
        # unverified metric claims (P0.7).
        md_lines += _format_limitations(
            data_profile, statistical_test_pvalues, unverified_claims, profile_status, degradations
        )

        md_lines += [
            "---",
            "",
            "*Report generated by the Agentic Data Analysis System.*",
            "*Architecture: Reasoning ↔ Execution separation with RLM context offloading.*",
        ]

        md_content = "\n".join(md_lines)
        md_path = Path(output_dir) / f"{dataset_name}_report.md"
        md_path.write_text(md_content, encoding="utf-8")

        json_path = Path(output_dir) / f"{dataset_name}_raw.json"
        json_path.write_text(
            json.dumps(
                {
                    "timestamp": timestamp,
                    "dataset": dataset_name,
                    "llm_insights": llm_insights,
                    "tool_results": tool_results,
                    "data_profile": data_profile,
                    "read_report": read_report,
                    "coercions": coercions,
                    "plan_rationales": plan_rationales,
                    "statistical_test_pvalues_bh_corrected": _apply_benjamini_hochberg(
                        statistical_test_pvalues or []
                    ),
                    "unverified_claims": unverified_claims,
                    "profile_status": profile_status,
                    "degradations": degradations,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        return {
            "summary": f"Report generated: {md_path} and {json_path}.",
            "markdown_path": str(md_path),
            "json_path": str(json_path),
            "n_insights": len(insights),
            "n_recommendations": len(recs),
        }

    def get_schema(self) -> dict[str, Any]:
        return {
            "dataset_name": {
                "type": "string",
                "description": "Name used in the report filename and header.",
                "required": False,
            },
            "tool_results_json": {
                "type": "string",
                "description": "JSON-serialised list of tool result dicts.",
                "required": False,
            },
            "llm_insights": {
                "type": "dict",
                "description": "Final LLM interpretation dict (insights, recommendations, metrics).",
                "required": False,
            },
            "output_dir": {
                "type": "string",
                "description": "Output directory. Default: output/reports.",
                "required": False,
            },
        }
