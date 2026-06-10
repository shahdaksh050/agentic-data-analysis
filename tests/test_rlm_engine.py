"""Unit tests for the RLM Engine — recursive invocation and REPL state."""
from __future__ import annotations

from typing import Any

import pytest

from src.rlm.engine import REPLEnvironment, RLMEngine, RLMSubTask


def _echo_llm(system_prompt: str, user_prompt: str) -> dict[str, Any]:
    return {"status": "complete", "echo": user_prompt[:40]}


@pytest.fixture
def engine() -> RLMEngine:
    return RLMEngine(llm_callable=_echo_llm, system_prompt="system", max_depth=2)


class TestREPLEnvironment:
    def test_set_and_get(self) -> None:
        env = REPLEnvironment()
        env.set("k", [1, 2, 3])
        assert env.get("k") == [1, 2, 3]

    def test_get_default(self) -> None:
        assert REPLEnvironment().get("missing", "fallback") == "fallback"

    def test_summary_empty(self) -> None:
        assert REPLEnvironment().summary() == "(empty)"


class TestRLMEngine:
    def test_invoke_returns_llm_dict(self, engine: RLMEngine) -> None:
        resp = engine.invoke("hello", depth=0, stage="test")
        assert resp["status"] == "complete"

    def test_invoke_records_trace(self, engine: RLMEngine) -> None:
        engine.set_iteration(3)
        engine.invoke("prompt-1", depth=0, stage="s1")
        engine.invoke("prompt-2", depth=1, stage="s2")
        trace = engine.trace
        assert len(trace) == 2
        assert trace[0].iteration == 3
        assert trace[1].depth == 1
        assert trace[1].stage == "s2"

    def test_max_depth_exceeded_raises(self, engine: RLMEngine) -> None:
        with pytest.raises(RecursionError):
            engine.invoke("too deep", depth=3, stage="overflow")

    def test_trace_property_returns_copy(self, engine: RLMEngine) -> None:
        engine.invoke("x", depth=0, stage="s")
        trace = engine.trace
        trace.clear()
        assert len(engine.trace) == 1

    def test_decompose_and_invoke_returns_per_task_results(self, engine: RLMEngine) -> None:
        tasks = [
            RLMSubTask("t1", "first", {"cols": ["a"]}),
            RLMSubTask("t2", "second", {"cols": ["b"]}),
        ]
        results = engine.decompose_and_invoke(tasks, lambda t: f"analyse {t.task_id}", depth=1)
        assert set(results.keys()) == {"t1", "t2"}
        assert all(r["status"] == "complete" for r in results.values())

    def test_decompose_stores_repl_context_and_results(self, engine: RLMEngine) -> None:
        tasks = [RLMSubTask("grp", "group", {"cols": ["a", "b"]})]
        engine.decompose_and_invoke(tasks, lambda t: t.description, depth=1)
        assert engine.repl_env.get("subtask_ctx_grp") == {"cols": ["a", "b"]}
        assert engine.repl_env.get("subtask_result_grp") is not None
