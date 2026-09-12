"""Tests for src/core/sandbox.py."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.core import sandbox
from tests.fixtures import single_column


class TestStaticCheck:
    def test_valid_code_passes(self) -> None:
        code = "RESULT = 1\n"
        assert sandbox._static_check(code) is None

    def test_syntax_error_caught(self) -> None:
        code = "RESULT = (\n"
        result = sandbox._static_check(code)
        assert result is not None
        error_type, hint = result
        assert error_type == "syntax"
        assert "syntax error" in hint.lower()

    def test_missing_result_assignment_caught(self) -> None:
        code = "x = 1\n"
        result = sandbox._static_check(code)
        assert result is not None
        error_type, hint = result
        assert error_type == "static_check"
        assert "RESULT" in hint

    def test_result_assignment_nested_in_if_not_recognized(self) -> None:
        # Only a module-level assignment counts — matches the spec's "at
        # the top level" requirement; the worker's own runtime check
        # (Task 2) is what catches an unreachable nested assignment.
        code = "if False:\n    RESULT = 1\n"
        result = sandbox._static_check(code)
        assert result is not None
        assert result[0] == "static_check"

    def test_disallowed_import_caught(self) -> None:
        code = "import os\nRESULT = 1\n"
        result = sandbox._static_check(code)
        assert result is not None
        error_type, hint = result
        assert error_type == "static_check"
        assert "os" in hint

    def test_disallowed_import_from_caught(self) -> None:
        code = "from socket import socket\nRESULT = 1\n"
        result = sandbox._static_check(code)
        assert result is not None
        assert result[0] == "static_check"

    def test_allowed_import_passes(self) -> None:
        code = "import numpy as np\nRESULT = 1\n"
        assert sandbox._static_check(code) is None

    def test_blocked_builtin_name_caught(self) -> None:
        code = "f = open('x.txt')\nRESULT = 1\n"
        result = sandbox._static_check(code)
        assert result is not None
        error_type, hint = result
        assert error_type == "static_check"
        assert "open" in hint


class TestRunSandboxed:
    def test_success_end_to_end(self, tmp_path: Path) -> None:
        dataset = single_column(tmp_path)
        result = sandbox.run_sandboxed(
            "RESULT = float(df['value'].mean())\n", str(dataset)
        )
        assert result.status == "ok"
        assert isinstance(result.result, float)
        assert result.error_type is None
        assert result.duration_ms > 0

    def test_static_check_failure_short_circuits_before_subprocess(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dataset = single_column(tmp_path)

        def _fail_if_called(*args: object, **kwargs: object) -> None:
            raise AssertionError("subprocess.Popen should not be called")

        monkeypatch.setattr(sandbox.subprocess, "Popen", _fail_if_called)
        result = sandbox.run_sandboxed("x = 1\n", str(dataset))
        assert result.status == "error"
        assert result.error_type == "static_check"

    def test_timeout_kills_process(self, tmp_path: Path) -> None:
        # timeout_s comfortably exceeds worker startup (which imports pandas),
        # so the kill provably interrupts the busy loop rather than the
        # interpreter's own import phase.
        dataset = single_column(tmp_path)
        code = "RESULT = 0\nwhile True:\n    pass\n"
        result = sandbox.run_sandboxed(code, str(dataset), timeout_s=8.0)
        assert result.status == "error"
        assert result.error_type == "timeout"

    def test_memory_limit_kills_process(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _FakeMemInfo:
            rss = 999 * 1024 * 1024

        class _FakePsProcess:
            def __init__(self, pid: int) -> None:
                pass

            def memory_info(self) -> _FakeMemInfo:
                return _FakeMemInfo()

        monkeypatch.setattr(sandbox.psutil, "Process", _FakePsProcess)
        dataset = single_column(tmp_path)
        # Long-running so the parent's poll fires before natural completion.
        code = "RESULT = 0\ntotal = 0\nfor i in range(10**9):\n    total += i\nRESULT = total\n"
        result = sandbox.run_sandboxed(code, str(dataset), timeout_s=30.0, memory_limit_mb=50)
        assert result.status == "error"
        assert result.error_type == "memory"

    def test_worker_crash_without_result_file_reports_runtime_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        crashing_worker = tmp_path / "crashing_worker.py"
        crashing_worker.write_text("import sys\nsys.exit(3)\n", encoding="utf-8")
        monkeypatch.setattr(sandbox, "_WORKER_SCRIPT", crashing_worker)

        dataset = single_column(tmp_path)
        result = sandbox.run_sandboxed("RESULT = 1\n", str(dataset))
        assert result.status == "error"
        assert result.error_type == "runtime"
