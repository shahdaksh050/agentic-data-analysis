"""
Time-Series Analysis Tool — Execution Layer.

Stage 3: Trend, stationarity, and autocorrelation diagnostics for datasets
with a genuine time axis (gated on DatasetProfile.is_time_series).

Kept to statistics that generalise across irregular/coarse-grained data
rather than a full seasonal decomposition, which needs a reliable
inferred frequency that real-world timestamps rarely provide cleanly.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.tools.base import BaseTool, ToolExecutionError
from src.tools.data_processing import _read_df

if TYPE_CHECKING:
    from src.core.memory import DatasetMetadata
    from src.core.profiler import DatasetProfile

#: Candidate seasonal lags checked via autocorrelation (weekly/monthly/yearly-ish).
_SEASONAL_LAGS = (7, 12, 30, 365)

#: |autocorrelation| at or above this is reported as a seasonal signal.
_SEASONALITY_THRESHOLD = 0.3

#: Two-sided ADF p-value at/below this rejects the unit-root (non-stationary) null.
_ADF_ALPHA = 0.05


def _autodetect_datetime_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            return str(col)
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            continue
        sample = series.dropna().head(20)
        if sample.empty:
            continue
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        if parsed.notna().mean() >= 0.9:
            return str(col)
    return None


def _autodetect_value_column(df: pd.DataFrame, date_column: str) -> str | None:
    numeric = [c for c in df.select_dtypes(include="number").columns if c != date_column]
    return str(numeric[0]) if numeric else None


class TimeSeriesAnalysisTool(BaseTool):
    """Trend direction, stationarity, and seasonality diagnostics for a time-indexed value."""

    name = "time_series_analysis"
    description = (
        "Analyse a time-indexed numeric column: trend direction/slope, stationarity "
        "(Augmented Dickey-Fuller test), lag-1 autocorrelation, and seasonal signal at "
        "weekly/monthly/yearly-ish lags. Use when the data profile shows a datetime column "
        "(is_time_series). Auto-detects date_column and value_column when omitted."
    )

    def applies_to(self, profile: DatasetProfile | None, metadata: DatasetMetadata | None) -> float:
        return 1.0 if profile is not None and profile.is_time_series else 0.0

    def execute(  # type: ignore[override]
        self,
        file_path: str,
        date_column: str | None = None,
        value_column: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        df = _read_df(file_path)

        if date_column is None:
            date_column = _autodetect_datetime_column(df)
        if date_column is None or date_column not in df.columns:
            raise ToolExecutionError(
                "No usable datetime column found. Pass date_column explicitly."
            )

        if value_column is None:
            value_column = _autodetect_value_column(df, date_column)
        if value_column is None or value_column not in df.columns:
            raise ToolExecutionError(
                "No numeric value_column found to analyse. Pass value_column explicitly."
            )

        dates = pd.to_datetime(df[date_column], errors="coerce", format="mixed")
        working = pd.DataFrame({"_date": dates, "_value": df[value_column]}).dropna()
        working = working.sort_values("_date")

        if len(working) < 10:
            raise ToolExecutionError(
                f"Only {len(working)} usable (date, value) rows after dropping missing "
                f"values — need at least 10 for time-series diagnostics."
            )

        values = working["_value"].to_numpy(dtype=float)
        t = np.arange(len(values), dtype=float)

        # ---- Trend: linear fit against row order (works for irregular spacing) ----
        slope, intercept = np.polyfit(t, values, 1)
        fitted = slope * t + intercept
        ss_res = float(np.sum((values - fitted) ** 2))
        ss_tot = float(np.sum((values - values.mean()) ** 2))
        r_squared = round(1 - ss_res / ss_tot, 4) if ss_tot > 0 else 0.0
        direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat"

        # ---- Stationarity (Augmented Dickey-Fuller) ----
        try:
            from statsmodels.tsa.stattools import adfuller

            _adf_stat, adf_p, *_rest = adfuller(values, autolag="AIC")
            is_stationary = bool(adf_p <= _ADF_ALPHA)
            adf_p_value: float | None = round(float(adf_p), 4)
        except Exception:
            # statsmodels unavailable or the series is degenerate for ADF
            # (e.g. constant) — fall back rather than failing the whole tool.
            adf_p_value = None
            is_stationary = bool(abs(slope) < 1e-9)

        # ---- Autocorrelation ----
        series = pd.Series(values)
        lag1_autocorr = round(float(series.autocorr(lag=1)), 4) if len(series) > 1 else 0.0

        seasonality: dict[str, float] = {}
        for lag in _SEASONAL_LAGS:
            if len(series) > lag * 2:
                corr = series.autocorr(lag=lag)
                if corr is not None and not np.isnan(corr):
                    seasonality[str(lag)] = round(float(corr), 4)
        seasonal_lags_detected = [
            lag for lag, corr in seasonality.items() if abs(corr) >= _SEASONALITY_THRESHOLD
        ]

        return {
            "summary": (
                f"Trend is {direction} (slope={slope:.4g}, R²={r_squared}) over "
                f"{len(working)} points. "
                + (
                    f"Series is {'stationary' if is_stationary else 'non-stationary'} "
                    f"(ADF p={adf_p_value})."
                    if adf_p_value is not None
                    else f"Series is {'likely stationary' if is_stationary else 'likely non-stationary'} (ADF unavailable)."
                )
                + (
                    f" Seasonal signal at lag(s) {', '.join(seasonal_lags_detected)}."
                    if seasonal_lags_detected
                    else " No strong seasonal signal at checked lags."
                )
            ),
            "date_column": date_column,
            "value_column": value_column,
            "rows_used": len(working),
            "trend_direction": direction,
            "trend_slope": round(float(slope), 6),
            "trend_r_squared": r_squared,
            "is_stationary": is_stationary,
            "adf_p_value": adf_p_value,
            "autocorrelation_lag1": lag1_autocorr,
            "seasonality_by_lag": seasonality,
            "seasonal_lags_detected": seasonal_lags_detected,
        }

    def get_schema(self) -> dict[str, Any]:
        return {
            "file_path": {"type": "string", "description": "Path to the (cleaned) dataset.", "required": True},
            "date_column": {
                "type": "string",
                "description": "Datetime column to sort/index by. Auto-detected if omitted.",
                "required": False,
            },
            "value_column": {
                "type": "string",
                "description": "Numeric column to analyse over time. Auto-detected if omitted.",
                "required": False,
            },
        }
