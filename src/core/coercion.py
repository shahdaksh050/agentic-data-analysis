"""
Type coercion / repair pass — numerics trapped in strings (U0.7).

Real-world exports routinely put numeric values inside strings: "$123.45",
"45.3%", "1,234,567". Left alone these become `identifier` or
high-cardinality `categorical` columns and are silently dropped from every
numeric analysis while also being counted against the quality score.

Runs once, at ingestion, before profiling — coerce_types() never runs
against already-profiled dtypes, so profile_dataframe() always sees the
repaired frame.

Every coercion is recorded and reported, never silent — the same
discipline ml_pipeline.py applies to `treatments_applied`.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.core.profiler import has_identifier_name_hint

#: Fraction of non-null values that must match a rule before a column is coerced.
COERCE_MATCH_THRESHOLD = 0.95

_CURRENCY_RE = re.compile(r"^[\$€£¥]\s*-?[\d,]+(\.\d+)?$|^-?[\d,]+(\.\d+)?\s*[\$€£¥]$")
_PERCENT_RE = re.compile(r"^-?[\d,]+(\.\d+)?\s*%$")
_THOUSANDS_RE = re.compile(r"^-?\d{1,3}(,\d{3})+$")
_BOOL_TRUE = {"y", "yes", "true", "t"}
_BOOL_FALSE = {"n", "no", "false", "f"}


@dataclass
class Coercion:
    """One column's type repair — what changed, and what didn't parse."""

    column: str
    from_kind: str          # "string"
    to_kind: str             # "numeric" | "boolean"
    rule: str                 # "currency" | "percent" | "thousands" | "yes_no" | "numeric"
    n_converted: int
    n_failed: int
    failed_examples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "from_kind": self.from_kind,
            "to_kind": self.to_kind,
            "rule": self.rule,
            "n_converted": self.n_converted,
            "n_failed": self.n_failed,
            "failed_examples": self.failed_examples,
        }


def _parse_currency(s: str) -> float | None:
    if not _CURRENCY_RE.match(s):
        return None
    cleaned = re.sub(r"[\$€£¥,]", "", s)
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_percent(s: str) -> float | None:
    if not _PERCENT_RE.match(s):
        return None
    cleaned = s.rstrip("%").replace(",", "")
    try:
        return float(cleaned) / 100.0
    except ValueError:
        return None


def _parse_thousands(s: str) -> float | None:
    if not _THOUSANDS_RE.match(s):
        return None
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def _parse_bool(s: str) -> bool | None:
    t = s.lower()
    if t in _BOOL_TRUE:
        return True
    if t in _BOOL_FALSE:
        return False
    return None


def _make_numeric_generic_parser(decimal_comma: bool) -> Callable[[str], float | None]:
    def _parse(s: str) -> float | None:
        # European decimals ("4,5" meaning 4.5) collide with US thousands
        # grouping ("4,500"). Only read ',' as a decimal point when the
        # file's sniffed delimiter was ';' (src.core.io.ReadReport) — the
        # signal that this file is EU-locale in the first place.
        t = s.replace(",", ".") if decimal_comma else s.replace(",", "")
        try:
            return float(t)
        except ValueError:
            return None

    return _parse


#: (rule name, target kind, parser) — checked in this order per column.
_RULES: list[tuple[str, str, Callable[[str], Any]]] = [
    ("currency", "numeric", _parse_currency),
    ("percent", "numeric", _parse_percent),
    ("thousands", "numeric", _parse_thousands),
    ("yes_no", "boolean", _parse_bool),
]


def _try_coerce_column(
    values: pd.Series, decimal_comma: bool
) -> tuple[str, str, pd.Series] | None:
    total = len(values)
    if total == 0:
        return None
    for rule, to_kind, parser in _RULES:
        parsed = values.map(parser)
        if parsed.notna().sum() / total >= COERCE_MATCH_THRESHOLD:
            return rule, to_kind, parsed
    generic_parsed = values.map(_make_numeric_generic_parser(decimal_comma))
    if generic_parsed.notna().sum() / total >= COERCE_MATCH_THRESHOLD:
        return "numeric", "numeric", generic_parsed
    return None


def coerce_types(df: pd.DataFrame, delimiter: str | None = None) -> tuple[pd.DataFrame, list[Coercion]]:
    """
    Repair numeric/boolean values trapped in string columns.

    Args:
        df:        Raw (uncoerced) dataset, straight from src.core.io.read_any.
        delimiter: The delimiter src.core.io.ReadReport sniffed/assumed for
                   this file — ';' signals a European-locale export, which
                   changes how a lone ',' inside a number is interpreted.

    Returns:
        (repaired_df, coercions) — coercions is empty when nothing changed.
        Every entry is a real, reportable change; nothing here is silent.
    """
    out = df.copy()
    coercions: list[Coercion] = []
    decimal_comma = delimiter == ";"

    for col in out.columns:
        series = out[col]
        if (
            pd.api.types.is_numeric_dtype(series)
            or pd.api.types.is_bool_dtype(series)
            or pd.api.types.is_datetime64_any_dtype(series)
        ):
            continue
        if has_identifier_name_hint(str(col)):
            continue

        non_null = series.dropna().astype(str).str.strip()
        non_null = non_null[non_null != ""]
        if non_null.empty:
            continue

        result = _try_coerce_column(non_null, decimal_comma)
        if result is None:
            continue
        rule, to_kind, parsed = result

        matched_mask = parsed.notna()
        n_converted = int(matched_mask.sum())
        n_failed = int((~matched_mask).sum())
        if n_converted == 0:
            continue

        failed_examples = list(dict.fromkeys(non_null[~matched_mask].tolist()))[:5]

        new_col = pd.Series(pd.NA, index=series.index, dtype=object)
        new_col.loc[non_null.index] = parsed.to_numpy()

        if to_kind == "boolean":
            out[col] = new_col.astype("boolean")
        else:
            out[col] = pd.to_numeric(new_col, errors="coerce")

        coercions.append(
            Coercion(
                column=str(col),
                from_kind="string",
                to_kind=to_kind,
                rule=rule,
                n_converted=n_converted,
                n_failed=n_failed,
                failed_examples=[str(x) for x in failed_examples],
            )
        )

    return out, coercions
