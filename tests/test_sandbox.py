"""Tests for src/core/sandbox.py."""
from __future__ import annotations

from src.core import sandbox


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
