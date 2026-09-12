"""
Tests for src/core/_sandbox_worker.py's execution logic, called directly
(no subprocess) for speed. Some cases here deliberately bypass
src.core.sandbox._static_check to verify the worker enforces its own
restrictions independently — defense in depth, not redundancy: a
RESULT assignment can be present in source but never actually execute
(e.g. inside `if False:`), which only a runtime check catches.
"""
from __future__ import annotations

from pathlib import Path

from src.core import _sandbox_worker as worker
from tests.fixtures import single_column


class TestExecute:
    def test_scalar_result(self, tmp_path: Path) -> None:
        dataset = single_column(tmp_path)
        payload = worker._execute("RESULT = float(df['value'].mean())\n", str(dataset))
        assert payload["status"] == "ok"
        assert isinstance(payload["result"], float)
        assert payload["error_type"] is None

    def test_dataframe_result_converted_to_records(self, tmp_path: Path) -> None:
        dataset = single_column(tmp_path)
        payload = worker._execute("RESULT = df.head(3)\n", str(dataset))
        assert payload["status"] == "ok"
        assert isinstance(payload["result"], list)
        assert len(payload["result"]) == 3
        assert "value" in payload["result"][0]

    def test_series_result_converted(self, tmp_path: Path) -> None:
        dataset = single_column(tmp_path)
        payload = worker._execute("RESULT = df['value'].head(2)\n", str(dataset))
        assert payload["status"] == "ok"
        assert isinstance(payload["result"], list)
        assert len(payload["result"]) == 2

    def test_ndarray_result_converted(self, tmp_path: Path) -> None:
        dataset = single_column(tmp_path)
        code = "import numpy as np\nRESULT = np.array([1, 2, 3]).tolist()\n"
        payload = worker._execute(code, str(dataset))
        assert payload["status"] == "ok"
        assert payload["result"] == [1, 2, 3]

    def test_schema_is_prebuilt_and_usable(self, tmp_path: Path) -> None:
        dataset = single_column(tmp_path)
        payload = worker._execute("RESULT = SCHEMA['value']\n", str(dataset))
        assert payload["status"] == "ok"
        assert payload["result"] == "numeric"

    def test_print_output_captured(self, tmp_path: Path) -> None:
        dataset = single_column(tmp_path)
        payload = worker._execute("print('hello sandbox')\nRESULT = 1\n", str(dataset))
        assert payload["status"] == "ok"
        assert "hello sandbox" in payload["stdout"]

    def test_stdout_capped(self, tmp_path: Path) -> None:
        dataset = single_column(tmp_path)
        code = "print('x' * 20000)\nRESULT = 1\n"
        payload = worker._execute(code, str(dataset))
        assert payload["status"] == "ok"
        assert len(payload["stdout"]) < 20000
        assert "truncated" in payload["stdout"]

    def test_result_never_assigned_at_runtime(self, tmp_path: Path) -> None:
        # Static-check-shaped source, but the assignment never runs.
        dataset = single_column(tmp_path)
        payload = worker._execute("if False:\n    RESULT = 1\n", str(dataset))
        assert payload["status"] == "error"
        assert payload["error_type"] == "static_check"

    def test_import_blocked_at_runtime(self, tmp_path: Path) -> None:
        # Bypasses src.core.sandbox._static_check on purpose — this test
        # is specifically about the worker's own import hook.
        dataset = single_column(tmp_path)
        payload = worker._execute("import os\nRESULT = 1\n", str(dataset))
        assert payload["status"] == "error"
        assert payload["error_type"] == "import_blocked"

    def test_syntax_error_reported(self, tmp_path: Path) -> None:
        # Bypasses src.core.sandbox._static_check on purpose (that gate
        # rejects this before a worker would ever see it in production) —
        # this test is specifically about the worker's own fallback.
        dataset = single_column(tmp_path)
        payload = worker._execute("RESULT = (\n", str(dataset))
        assert payload["status"] == "error"
        assert payload["error_type"] == "syntax"
        assert payload["hint"] is not None

    def test_runtime_exception_reported(self, tmp_path: Path) -> None:
        dataset = single_column(tmp_path)
        payload = worker._execute("RESULT = 1 / 0\n", str(dataset))
        assert payload["status"] == "error"
        assert payload["error_type"] == "runtime"
        assert "ZeroDivisionError" in payload["traceback"]
        assert payload["hint"] is not None  # extracted from the exception message

    def test_output_invalid_for_non_serializable_result(self, tmp_path: Path) -> None:
        dataset = single_column(tmp_path)
        code = "class Thing:\n    pass\nRESULT = Thing()\n"
        payload = worker._execute(code, str(dataset))
        assert payload["status"] == "error"
        assert payload["error_type"] == "output_invalid"

    def test_dataset_load_failure_reported_as_runtime(self, tmp_path: Path) -> None:
        payload = worker._execute("RESULT = 1\n", str(tmp_path / "does_not_exist.csv"))
        assert payload["status"] == "error"
        assert payload["error_type"] == "runtime"
