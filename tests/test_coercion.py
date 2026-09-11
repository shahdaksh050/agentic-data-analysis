"""Tests for src.core.coercion — numerics trapped in strings (U0.7)."""
from __future__ import annotations

import pandas as pd
import pytest

from src.core.coercion import coerce_types


class TestCurrency:
    def test_dollar_with_thousands_separator(self) -> None:
        df = pd.DataFrame({"amount": ["$1,234.56"] * 20})
        out, coercions = coerce_types(df)
        assert out["amount"].iloc[0] == 1234.56
        assert pd.api.types.is_numeric_dtype(out["amount"])
        assert coercions[0].rule == "currency"
        assert coercions[0].column == "amount"


class TestPercent:
    def test_percent_stored_as_fraction(self) -> None:
        df = pd.DataFrame({"rate": ["45.3%"] * 20})
        out, coercions = coerce_types(df)
        assert out["rate"].iloc[0] == pytest.approx(0.453)
        assert coercions[0].rule == "percent"


class TestBoolean:
    def test_yes_no_becomes_bool(self) -> None:
        df = pd.DataFrame({"active": (["Y", "N"] * 10)})
        out, coercions = coerce_types(df)
        assert out["active"].iloc[0] == True  # noqa: E712
        assert out["active"].iloc[1] == False  # noqa: E712
        assert coercions[0].rule == "yes_no"
        assert coercions[0].to_kind == "boolean"


class TestIdentifierGuard:
    def test_zipcode_stays_string_not_coerced(self) -> None:
        """A numeric-looking but zero-padded zipcode must never be coerced —
        it would silently strip the leading zero."""
        df = pd.DataFrame({"zipcode": ["04521", "04522", "04523"] * 10})
        out, coercions = coerce_types(df)
        assert out["zipcode"].tolist() == df["zipcode"].tolist()
        assert coercions == []


class TestMixedTypeThreshold:
    def test_below_95_percent_not_coerced(self) -> None:
        """A 50/50 mixed column must not be silently coerced."""
        values = (["1", "2", "not_a_number", "also_bad"] * 10)
        df = pd.DataFrame({"value": values})
        out, coercions = coerce_types(df)
        assert not pd.api.types.is_numeric_dtype(out["value"])
        assert coercions == []

    def test_above_95_percent_coerced_with_failures_recorded(self) -> None:
        values = ["1.5"] * 95 + ["garbage"] * 5
        df = pd.DataFrame({"value": values})
        out, coercions = coerce_types(df)
        assert pd.api.types.is_numeric_dtype(out["value"])
        assert coercions[0].rule == "numeric"
        assert coercions[0].n_converted == 95
        assert coercions[0].n_failed == 5
        assert "garbage" in coercions[0].failed_examples


class TestEveryCoercionReported:
    def test_multiple_columns_all_appear_in_returned_list(self) -> None:
        df = pd.DataFrame({
            "amount": ["$10.00"] * 20,
            "rate": ["5.0%"] * 20,
            "active": ["yes", "no"] * 10,
            "plain_text": ["hello world"] * 20,
        })
        out, coercions = coerce_types(df)
        coerced_cols = {c.column for c in coercions}
        assert coerced_cols == {"amount", "rate", "active"}
        assert not pd.api.types.is_numeric_dtype(out["plain_text"])


class TestEuropeanDecimal:
    def test_comma_as_decimal_only_when_semicolon_delimited(self) -> None:
        df = pd.DataFrame({"value": ["4,5", "10,25", "3,0"] * 10})
        out, _coercions = coerce_types(df, delimiter=";")
        assert out["value"].iloc[0] == 4.5
        assert out["value"].iloc[1] == 10.25

    def test_comma_as_thousands_when_not_semicolon_delimited(self) -> None:
        df = pd.DataFrame({"value": ["4,500", "10,250"] * 10})
        out, coercions = coerce_types(df, delimiter=",")
        assert out["value"].iloc[0] == 4500
        assert coercions[0].rule == "thousands"
