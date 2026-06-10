"""
Dashboard Agent — builds a dashboard that fits the data, not a template.

Given the dataset, its profile, and the accumulated tool results, this
agent decides *which* charts are worth showing (the way a data scientist
would) and emits self-contained Vega-Lite specs:

  - class balance        when a classification target exists
  - histograms           for the most informative numeric features
  - category counts      for low-cardinality categoricals
  - scatter              for the strongest numeric relationship
  - box plots            for the feature that best separates the classes
  - time series          when a datetime column is present
  - model comparison     when training results exist
  - correlation bars     when correlation results exist

Rules:
  - Pure computation: no Streamlit, no file I/O, no LLM. Deterministic.
  - Identifiers and constant columns are never charted.
  - Data is inlined into each spec, sampled/aggregated to stay small.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.core.profiler import DatasetProfile

#: Hard cap on inline rows per chart (keeps specs lightweight).
MAX_POINTS = 1_000

#: How many numeric histograms / categorical bars to show at most.
MAX_HISTOGRAMS = 4
MAX_CATEGORY_CHARTS = 3

#: Categorical columns with more classes than this get truncated to top-N.
MAX_CATEGORIES_SHOWN = 12

_SAMPLE_SEED = 42


@dataclass
class ChartSpec:
    """One renderable chart: metadata plus a complete Vega-Lite spec."""

    chart_id: str
    title: str
    description: str
    spec: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "chart_id": self.chart_id,
            "title": self.title,
            "description": self.description,
            "spec": self.spec,
        }


def dashboard_to_json(charts: list[ChartSpec]) -> str:
    """Serialise a dashboard for saving to output/dashboard.json."""
    return json.dumps([c.to_dict() for c in charts], indent=2, default=str)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_tool_output(
    tool_results: list[dict[str, Any]], name: str
) -> dict[str, Any] | None:
    """Most recent successful output dict for a tool, or None."""
    for r in reversed(tool_results):
        if r.get("tool_name") == name and r.get("status") == "success":
            out = r.get("output")
            return out if isinstance(out, dict) else None
    return None


def _to_primitive(value: Any) -> Any:
    """Coerce numpy/pandas scalars into JSON-safe Python primitives."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _records(df: pd.DataFrame, cols: list[str], max_rows: int = MAX_POINTS) -> list[dict[str, Any]]:
    """Sampled, primitive-typed records for inlining into a Vega-Lite spec."""
    subset = df[cols].dropna()
    if len(subset) > max_rows:
        subset = subset.sample(n=max_rows, random_state=_SAMPLE_SEED)
    return [
        {col: _to_primitive(val) for col, val in row.items()}
        for row in subset.to_dict(orient="records")
    ]


def _chartable(profile: DatasetProfile, *kinds: str) -> list[str]:
    """Column names of the given kinds, excluding identifiers/constants."""
    return [c.name for c in profile.columns_of_kind(*kinds)]


def _rank_numeric_features(
    df: pd.DataFrame, numeric_cols: list[str], target: str | None
) -> list[str]:
    """
    Order numeric features by usefulness: |correlation with target| when a
    usable target exists, otherwise by variance (normalised).
    """
    usable = [c for c in numeric_cols if c != target and c in df.columns]
    if not usable:
        return []

    if target and target in df.columns:
        target_series = df[target]
        if not pd.api.types.is_numeric_dtype(target_series):
            codes, _ = pd.factorize(target_series)
            target_series = pd.Series(codes, index=df.index)
        scores: dict[str, float] = {}
        for col in usable:
            corr = df[col].corr(target_series)
            scores[col] = abs(float(corr)) if pd.notna(corr) else 0.0
        return sorted(usable, key=lambda c: scores[c], reverse=True)

    variances: dict[str, float] = {}
    for col in usable:
        clean = df[col].dropna()
        if clean.empty or float(clean.abs().max()) == 0.0:
            variances[col] = 0.0
        else:
            scale = float(clean.abs().max())
            variances[col] = float((clean / scale).var()) if len(clean) > 1 else 0.0
    return sorted(usable, key=lambda c: variances[c], reverse=True)


# ---------------------------------------------------------------------------
# Chart builders — each returns a ChartSpec or None when not applicable
# ---------------------------------------------------------------------------

def _class_balance_chart(
    df: pd.DataFrame, target: str | None, task_type: str | None
) -> ChartSpec | None:
    if not target or target not in df.columns or task_type != "classification":
        return None
    counts = df[target].dropna().astype(str).value_counts().head(MAX_CATEGORIES_SHOWN)
    if counts.empty:
        return None
    values = [{"class": str(k), "count": int(v)} for k, v in counts.items()]
    return ChartSpec(
        chart_id="class_balance",
        title=f"Class Balance — {target}",
        description="Distribution of the target classes. Heavy imbalance means accuracy is misleading.",
        spec={
            "data": {"values": values},
            "mark": {"type": "bar", "cornerRadiusEnd": 3},
            "height": 220,
            "encoding": {
                "x": {"field": "class", "type": "nominal", "axis": {"labelAngle": 0, "title": None}},
                "y": {"field": "count", "type": "quantitative", "title": "rows"},
                "color": {"field": "class", "type": "nominal", "legend": None},
                "tooltip": [{"field": "class"}, {"field": "count"}],
            },
        },
    )


def _histogram_charts(
    df: pd.DataFrame, ranked_numeric: list[str]
) -> list[ChartSpec]:
    charts: list[ChartSpec] = []
    for col in ranked_numeric[:MAX_HISTOGRAMS]:
        values = _records(df, [col])
        if not values:
            continue
        charts.append(ChartSpec(
            chart_id=f"hist_{col}",
            title=f"Distribution — {col}",
            description=f"Histogram of '{col}'. Watch for skew, gaps, and outlier tails.",
            spec={
                "data": {"values": values},
                "mark": {"type": "bar", "cornerRadiusEnd": 2},
                "height": 200,
                "encoding": {
                    "x": {"field": col, "type": "quantitative", "bin": {"maxbins": 30}},
                    "y": {"aggregate": "count", "title": "rows"},
                    "tooltip": [{"aggregate": "count", "title": "rows"}],
                },
            },
        ))
    return charts


def _category_charts(
    df: pd.DataFrame, profile: DatasetProfile, target: str | None
) -> list[ChartSpec]:
    charts: list[ChartSpec] = []
    cat_cols = [
        c for c in profile.columns_of_kind("categorical", "boolean")
        if c.name != target and "high_cardinality" not in c.flags
    ]
    cat_cols.sort(key=lambda c: c.nunique)
    for col in cat_cols[:MAX_CATEGORY_CHARTS]:
        counts = df[col.name].dropna().astype(str).value_counts().head(MAX_CATEGORIES_SHOWN)
        if counts.empty:
            continue
        values = [{"category": str(k), "count": int(v)} for k, v in counts.items()]
        charts.append(ChartSpec(
            chart_id=f"cat_{col.name}",
            title=f"Category Counts — {col.name}",
            description=f"Frequency of each '{col.name}' value (top {MAX_CATEGORIES_SHOWN}).",
            spec={
                "data": {"values": values},
                "mark": {"type": "bar", "cornerRadiusEnd": 2},
                "height": max(120, 24 * len(values)),
                "encoding": {
                    "y": {"field": "category", "type": "nominal", "sort": "-x", "title": None},
                    "x": {"field": "count", "type": "quantitative", "title": "rows"},
                    "tooltip": [{"field": "category"}, {"field": "count"}],
                },
            },
        ))
    return charts


def _scatter_chart(
    df: pd.DataFrame,
    ranked_numeric: list[str],
    target: str | None,
    task_type: str | None,
    corr_output: dict[str, Any] | None,
) -> ChartSpec | None:
    pair: tuple[str, str] | None = None
    if corr_output:
        for entry in corr_output.get("top_correlations", []):
            a, b = str(entry.get("col_a", "")), str(entry.get("col_b", ""))
            if a in df.columns and b in df.columns and a != target and b != target:
                pair = (a, b)
                break
    if pair is None and len(ranked_numeric) >= 2:
        pair = (ranked_numeric[0], ranked_numeric[1])
    if pair is None:
        return None

    cols = list(pair)
    color_field: str | None = None
    if (
        task_type == "classification"
        and target
        and target in df.columns
        and df[target].nunique(dropna=True) <= 10
    ):
        color_field = target
        cols.append(target)

    values = _records(df, cols)
    if not values:
        return None
    if color_field:
        for row in values:
            row[color_field] = str(row[color_field])

    encoding: dict[str, Any] = {
        "x": {"field": pair[0], "type": "quantitative", "scale": {"zero": False}},
        "y": {"field": pair[1], "type": "quantitative", "scale": {"zero": False}},
        "tooltip": [{"field": c} for c in cols],
    }
    if color_field:
        encoding["color"] = {"field": color_field, "type": "nominal"}

    return ChartSpec(
        chart_id="scatter_top_pair",
        title=f"Relationship — {pair[0]} vs {pair[1]}",
        description="The strongest numeric relationship in the data"
                    + (" — colored by target class." if color_field else "."),
        spec={
            "data": {"values": values},
            "mark": {"type": "circle", "opacity": 0.55, "size": 36},
            "height": 280,
            "encoding": encoding,
        },
    )


def _box_plot_chart(
    df: pd.DataFrame, ranked_numeric: list[str], target: str | None, task_type: str | None
) -> ChartSpec | None:
    if (
        task_type != "classification"
        or not target
        or target not in df.columns
        or not ranked_numeric
        or df[target].nunique(dropna=True) > 10
    ):
        return None
    feature = ranked_numeric[0]
    values = _records(df, [feature, target])
    if not values:
        return None
    for row in values:
        row[target] = str(row[target])
    return ChartSpec(
        chart_id=f"box_{feature}",
        title=f"Separation — {feature} by {target}",
        description=f"How '{feature}' (the most target-linked feature) differs across classes.",
        spec={
            "data": {"values": values},
            "mark": {"type": "boxplot", "extent": 1.5},
            "height": 240,
            "encoding": {
                "x": {"field": target, "type": "nominal", "axis": {"labelAngle": 0}},
                "y": {"field": feature, "type": "quantitative", "scale": {"zero": False}},
                "color": {"field": target, "type": "nominal", "legend": None},
            },
        },
    )


def _time_series_chart(
    df: pd.DataFrame, profile: DatasetProfile, ranked_numeric: list[str]
) -> ChartSpec | None:
    datetime_cols = _chartable(profile, "datetime")
    if not datetime_cols or not ranked_numeric:
        return None
    time_col, value_col = datetime_cols[0], ranked_numeric[0]

    frame = df[[time_col, value_col]].dropna()
    if frame.empty:
        return None
    parsed = pd.to_datetime(frame[time_col], errors="coerce", format="mixed")
    frame = frame.assign(**{time_col: parsed}).dropna()
    if frame.empty:
        return None

    monthly = (
        frame.set_index(time_col)[value_col]
        .resample("MS")
        .mean()
        .dropna()
        .reset_index()
    )
    values = [
        {"period": ts.strftime("%Y-%m-%d"), "value": round(float(v), 4)}
        for ts, v in zip(monthly[time_col], monthly[value_col], strict=True)
    ]
    if len(values) < 2:
        return None
    return ChartSpec(
        chart_id="time_series",
        title=f"Trend — monthly mean {value_col}",
        description=f"'{value_col}' aggregated by month over '{time_col}'.",
        spec={
            "data": {"values": values},
            "mark": {"type": "line", "point": True},
            "height": 240,
            "encoding": {
                "x": {"field": "period", "type": "temporal", "title": None},
                "y": {"field": "value", "type": "quantitative", "title": value_col,
                      "scale": {"zero": False}},
                "tooltip": [{"field": "period", "type": "temporal"}, {"field": "value"}],
            },
        },
    )


def _model_comparison_chart(train_output: dict[str, Any] | None) -> ChartSpec | None:
    if not train_output:
        return None
    models = train_output.get("models_trained", {})
    if not isinstance(models, dict) or not models:
        return None
    task = str(train_output.get("task_type", "classification"))
    metric = "accuracy" if task == "classification" else "r2"

    rows: list[dict[str, Any]] = []
    for name, m in models.items():
        if not isinstance(m, dict):
            continue
        rows += [
            {"model": str(name), "metric": "Train",
             "score": round(float(m.get("train_metrics", {}).get(metric, 0)) * 100, 2)},
            {"model": str(name), "metric": "Test",
             "score": round(float(m.get("test_metrics", {}).get(metric, 0)) * 100, 2)},
            {"model": str(name), "metric": "CV mean",
             "score": round(float(m.get("cv_mean", 0)) * 100, 2)},
        ]
    if not rows:
        return None
    return ChartSpec(
        chart_id="model_comparison",
        title="Model Comparison",
        description=f"Train vs held-out test vs cross-validated {metric}. "
                    "A large train-test gap signals overfitting.",
        spec={
            "data": {"values": rows},
            "mark": {"type": "bar", "cornerRadiusEnd": 2},
            "height": 280,
            "encoding": {
                "x": {"field": "model", "type": "nominal", "axis": {"labelAngle": 0, "title": None}},
                "xOffset": {"field": "metric"},
                "y": {"field": "score", "type": "quantitative",
                      "title": f"{metric} %", "scale": {"domain": [0, 110]}},
                "color": {
                    "field": "metric",
                    "scale": {"domain": ["Train", "Test", "CV mean"],
                              "range": ["#5b8dee", "#2ecc71", "#e67e22"]},
                    "legend": {"orient": "top", "title": None},
                },
                "tooltip": [{"field": "model"}, {"field": "metric"},
                            {"field": "score", "title": f"{metric} %"}],
            },
        },
    )


def _correlation_chart(corr_output: dict[str, Any] | None) -> ChartSpec | None:
    if not corr_output:
        return None
    top = corr_output.get("top_correlations", [])[:10]
    if not top:
        return None
    values = [
        {"pair": f"{e.get('col_a')} ↔ {e.get('col_b')}",
         "correlation": round(float(e.get("correlation", 0)), 4)}
        for e in top if isinstance(e, dict)
    ]
    return ChartSpec(
        chart_id="top_correlations",
        title="Top Feature Correlations",
        description="Strongest pairwise relationships. Green = positive, red = negative.",
        spec={
            "data": {"values": values},
            "mark": {"type": "bar", "cornerRadiusEnd": 2},
            "height": max(160, len(values) * 30),
            "encoding": {
                "y": {"field": "pair", "type": "nominal", "sort": "-x", "title": None},
                "x": {"field": "correlation", "type": "quantitative",
                      "scale": {"domain": [-1.1, 1.1]}, "title": "correlation coefficient"},
                "color": {
                    "condition": {"test": "datum.correlation >= 0", "value": "#2ecc71"},
                    "value": "#e74c3c",
                },
                "tooltip": [{"field": "pair"}, {"field": "correlation"}],
            },
        },
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_dashboard(
    df: pd.DataFrame,
    profile: DatasetProfile,
    target_column: str | None = None,
    task_type: str | None = None,
    tool_results: list[dict[str, Any]] | None = None,
) -> list[ChartSpec]:
    """
    Select and build the charts that best fit this dataset and its results.

    Args:
        df:            The dataset (cleaned version preferred when available).
        profile:       DatasetProfile from profile_dataframe().
        target_column: ML target, if any.
        task_type:     "classification" | "regression" | "clustering" | "eda".
        tool_results:  Serialised ToolResult dicts from the memory system.

    Returns:
        Ordered list of ChartSpec — results charts first, then EDA charts.
    """
    results = tool_results or []
    numeric_cols = _chartable(profile, "numeric")
    ranked = _rank_numeric_features(df, numeric_cols, target_column)

    train_out = _find_tool_output(results, "train_model")
    corr_out = _find_tool_output(results, "correlation_analysis")

    candidates: list[ChartSpec | None] = [
        _model_comparison_chart(train_out),
        _correlation_chart(corr_out),
        _class_balance_chart(df, target_column, task_type),
        _box_plot_chart(df, ranked, target_column, task_type),
        _scatter_chart(df, ranked, target_column, task_type, corr_out),
        _time_series_chart(df, profile, ranked),
    ]
    charts = [c for c in candidates if c is not None]
    charts.extend(_histogram_charts(df, ranked))
    charts.extend(_category_charts(df, profile, target_column))
    return charts
