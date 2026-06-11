"""
Data Profiler — the 'first look' a data scientist takes at a dataset.

Runs automatically at ingestion (Stage 1) and produces a structured profile:
  - Per-column semantics: numeric / categorical / datetime / boolean /
    identifier / constant, with the statistics that matter for each kind.
  - Dataset-level health: duplicates, missingness, memory footprint.
  - A 0-100 quality score plus human-readable warnings.
  - A compact prompt summary so the planning LLM reasons from the same
    profile the user sees in the UI.

Pure computation — no file I/O, no LLM calls, deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

#: A categorical column with more unique values than this is "high cardinality".
HIGH_CARDINALITY_THRESHOLD = 50

#: Missing-fraction above which a column is flagged as a data-quality problem.
HIGH_MISSING_FRACTION = 0.20

#: Absolute skewness above which a numeric column is flagged as skewed.
SEVERE_SKEW_THRESHOLD = 2.0

#: Identifier-style column name fragments.
_ID_NAME_HINTS = ("id", "uuid", "guid", "index", "key", "code", "number", "no")


@dataclass
class ColumnProfile:
    """Profile of a single column."""

    name: str
    dtype: str
    kind: str                 # numeric | categorical | datetime | boolean | identifier | constant
    missing_count: int
    missing_pct: float
    nunique: int
    stats: dict[str, float] = field(default_factory=dict)      # numeric columns
    top_values: dict[str, int] = field(default_factory=dict)   # categorical columns
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "kind": self.kind,
            "missing_count": self.missing_count,
            "missing_pct": self.missing_pct,
            "nunique": self.nunique,
            "stats": self.stats,
            "top_values": self.top_values,
            "flags": self.flags,
        }


@dataclass
class DatasetProfile:
    """Full structured profile of a dataset."""

    row_count: int
    column_count: int
    duplicate_rows: int
    memory_mb: float
    columns: list[ColumnProfile]
    warnings: list[str]
    quality_score: int        # 0-100

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_count": self.row_count,
            "column_count": self.column_count,
            "duplicate_rows": self.duplicate_rows,
            "memory_mb": self.memory_mb,
            "columns": [c.to_dict() for c in self.columns],
            "warnings": self.warnings,
            "quality_score": self.quality_score,
        }

    def columns_of_kind(self, *kinds: str) -> list[ColumnProfile]:
        return [c for c in self.columns if c.kind in kinds]

    def to_prompt_string(self, max_warnings: int = 8) -> str:
        """
        Compact profile summary (~150 tokens) for LLM context injection.

        Column names are dataset-derived (untrusted) and are sanitised before
        they reach a prompt. Warnings embed column names too, so they pass
        through the same sanitiser.
        """
        from src.core.security import sanitize_for_prompt as _sp

        kind_counts: dict[str, int] = {}
        for col in self.columns:
            kind_counts[col.kind] = kind_counts.get(col.kind, 0) + 1
        kinds = ", ".join(f"{v} {k}" for k, v in sorted(kind_counts.items()))
        lines = [
            f"Data profile: quality score {self.quality_score}/100; "
            f"{self.duplicate_rows} duplicate rows; column kinds: {kinds}.",
        ]
        skewed = [_sp(c.name) for c in self.columns if "severe_skew" in c.flags]
        if skewed:
            lines.append(f"Severely skewed numerics: {', '.join(skewed[:6])}.")
        ids = [_sp(c.name) for c in self.columns if c.kind == "identifier"]
        if ids:
            lines.append(f"Identifier columns (exclude from modelling): {', '.join(ids[:6])}.")
        if self.warnings:
            lines.append(
                "Warnings: " + " | ".join(_sp(w, max_len=160) for w in self.warnings[:max_warnings])
            )
        return "\n".join(lines)


def _is_datetime_like(series: pd.Series) -> bool:
    """True for datetime dtypes or string columns that parse as dates."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return False
    sample = series.dropna().head(20)
    if sample.empty:
        return False
    try:
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    except (ValueError, TypeError):
        return False
    return bool(parsed.notna().mean() >= 0.9)


def _is_identifier_like(name: str, series: pd.Series, row_count: int) -> bool:
    """Heuristic: near-unique column whose name hints at an identifier."""
    if row_count == 0:
        return False
    nunique = int(series.nunique(dropna=True))
    uniqueness = nunique / row_count
    name_l = name.lower()
    name_hit = any(
        name_l == h or name_l.endswith(f"_{h}") or name_l.endswith(h)
        for h in _ID_NAME_HINTS
    )
    return (uniqueness >= 0.98 and name_hit) or (
        uniqueness == 1.0 and not pd.api.types.is_float_dtype(series)
    )


def _profile_column(name: str, series: pd.Series, row_count: int) -> ColumnProfile:
    missing = int(series.isna().sum())
    missing_pct = round(100.0 * missing / row_count, 2) if row_count else 0.0
    nunique = int(series.nunique(dropna=True))
    flags: list[str] = []
    stats: dict[str, float] = {}
    top_values: dict[str, int] = {}

    if nunique <= 1:
        kind = "constant"
        flags.append("constant")
    elif pd.api.types.is_bool_dtype(series):
        kind = "boolean"
    # Datetime check must precede the identifier check: a daily time index is
    # 100% unique but is a time axis, not an ID.
    elif _is_datetime_like(series):
        kind = "datetime"
    elif _is_identifier_like(name, series, row_count):
        kind = "identifier"
        flags.append("id_like")
    elif pd.api.types.is_numeric_dtype(series):
        kind = "numeric"
        clean = series.dropna()
        if not clean.empty:
            stats = {
                "mean": round(float(clean.mean()), 4),
                "std": round(float(clean.std()), 4) if len(clean) > 1 else 0.0,
                "min": round(float(clean.min()), 4),
                "max": round(float(clean.max()), 4),
                "median": round(float(clean.median()), 4),
            }
            if len(clean) > 2:
                skew = float(clean.skew())
                stats["skew"] = round(skew, 4)
                if abs(skew) >= SEVERE_SKEW_THRESHOLD:
                    flags.append("severe_skew")
    else:
        kind = "categorical"
        counts = series.dropna().astype(str).value_counts().head(5)
        top_values = {str(k): int(v) for k, v in counts.items()}
        if nunique > HIGH_CARDINALITY_THRESHOLD:
            flags.append("high_cardinality")

    if missing_pct > HIGH_MISSING_FRACTION * 100:
        flags.append("high_missing")

    return ColumnProfile(
        name=name,
        dtype=str(series.dtype),
        kind=kind,
        missing_count=missing,
        missing_pct=missing_pct,
        nunique=nunique,
        stats=stats,
        top_values=top_values,
        flags=flags,
    )


def profile_dataframe(df: pd.DataFrame, target_column: str | None = None) -> DatasetProfile:
    """
    Build a full DatasetProfile from a DataFrame.

    Args:
        df:            The raw (uncleaned) dataset.
        target_column: Optional target — enables class-imbalance checks.
    """
    row_count = len(df)
    duplicate_rows = int(df.duplicated().sum())
    memory_mb = round(float(df.memory_usage(deep=True).sum()) / 1_048_576, 2)

    columns = [_profile_column(str(c), df[c], row_count) for c in df.columns]

    warnings: list[str] = []
    penalty = 0

    total_cells = max(1, row_count * max(1, len(df.columns)))
    missing_frac = float(df.isna().sum().sum()) / total_cells
    if missing_frac > 0:
        penalty += min(25, int(missing_frac * 100))
        if missing_frac > 0.05:
            warnings.append(f"{missing_frac:.0%} of all cells are missing.")

    if duplicate_rows:
        dup_frac = duplicate_rows / max(1, row_count)
        penalty += min(15, int(dup_frac * 100))
        warnings.append(f"{duplicate_rows} duplicate rows ({dup_frac:.1%}).")

    for col in columns:
        if col.kind == "constant":
            penalty += 3
            warnings.append(f"Column '{col.name}' is constant — carries no signal.")
        if "high_missing" in col.flags:
            penalty += 4
            warnings.append(f"Column '{col.name}' is {col.missing_pct:.0f}% missing.")
        if "high_cardinality" in col.flags:
            penalty += 2
            warnings.append(
                f"Column '{col.name}' has {col.nunique} categories — "
                "one-hot encoding would explode; consider dropping or target-encoding."
            )

    if target_column and target_column in df.columns:
        tgt = df[target_column].dropna()
        if not tgt.empty and not pd.api.types.is_float_dtype(tgt) and tgt.nunique() <= 20:
            counts = tgt.value_counts()
            if len(counts) >= 2:
                minority_frac = float(counts.iloc[-1]) / float(counts.sum())
                if minority_frac < 0.10:
                    penalty += 5
                    warnings.append(
                        f"Target '{target_column}' is imbalanced — minority class is "
                        f"{minority_frac:.1%}. Prefer F1/recall over accuracy."
                    )

    if row_count < 100:
        penalty += 10
        warnings.append(f"Only {row_count} rows — results will have high variance.")

    quality_score = max(0, 100 - penalty)

    return DatasetProfile(
        row_count=row_count,
        column_count=len(df.columns),
        duplicate_rows=duplicate_rows,
        memory_mb=memory_mb,
        columns=columns,
        warnings=warnings,
        quality_score=quality_score,
    )
