"""Tests for src.core.degradations — item 10 observability."""
from __future__ import annotations

from src.core.degradations import collect_degradations


class TestCollectDegradations:
    def test_empty_when_everything_clean(self) -> None:
        read_report = {
            "format": "csv", "encoding": "utf-8", "encoding_confident": True,
            "delimiter": ",", "delimiter_sniffed": True, "duplicate_headers": [], "notes": [],
        }
        profile = {"is_sufficient": True, "warnings": []}
        assert collect_degradations(read_report, [], profile, "ok") == []

    def test_profiling_failure_surfaced(self) -> None:
        log = collect_degradations(None, None, None, "failed: boom")
        assert any("degraded mode" in entry for entry in log)
        assert any("boom" in entry for entry in log)

    def test_unconfident_encoding_surfaced(self) -> None:
        read_report = {"format": "csv", "encoding": "cp1252", "encoding_confident": False,
                        "delimiter": ",", "delimiter_sniffed": True, "duplicate_headers": [], "notes": []}
        log = collect_degradations(read_report, [], {"is_sufficient": True, "warnings": []}, "ok")
        assert any("not confidently detected" in entry for entry in log)

    def test_coercions_surfaced(self) -> None:
        coercions = [{"column": "amount", "to_kind": "numeric", "rule": "currency",
                      "n_converted": 9, "n_failed": 1}]
        log = collect_degradations(None, coercions, None, "ok")
        assert any("amount" in entry and "currency" in entry for entry in log)

    def test_insufficient_data_surfaced(self) -> None:
        profile = {"is_sufficient": False, "sufficiency_reason": "Only 1 row(s).", "warnings": []}
        log = collect_degradations(None, [], profile, "ok")
        assert "Only 1 row(s)." in log

    def test_duplicate_headers_surfaced(self) -> None:
        read_report = {"format": "csv", "encoding": "utf-8", "encoding_confident": True,
                        "delimiter": ",", "delimiter_sniffed": True, "duplicate_headers": ["a"], "notes": []}
        log = collect_degradations(read_report, [], {"is_sufficient": True, "warnings": []}, "ok")
        assert any("Duplicate column header 'a'" in entry for entry in log)
