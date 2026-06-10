"""Integration tests for the Agent Controller — full 7-stage workflow with a mock LLM."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.core.controller import AgentController, LLMClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_csv(tmp_path: pytest.TempPathFactory) -> str:
    rng = np.random.default_rng(42)
    n = 80
    fa = rng.normal(0, 1, n)
    df = pd.DataFrame(
        {
            "feature_a": fa,
            "feature_b": rng.normal(5, 2, n),
            "feature_c": rng.uniform(0, 10, n),
            "label": (fa + rng.normal(0, 0.5, n) > 0).astype(int),
        }
    )
    df.loc[::15, "feature_a"] = np.nan
    p = tmp_path / "sample.csv"
    df.to_csv(p, index=False)
    return str(p)


@pytest.fixture
def agent(
    sample_csv: str, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> AgentController:
    monkeypatch.delenv("MAX_ITERATIONS", raising=False)
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
    return AgentController(max_iterations=5, enable_rlm=False)


class _ScriptedLLM:
    """Stands in for LLMClient — returns scripted responses per call."""

    def __init__(self, responses: list[dict[str, Any] | Exception]) -> None:
        self._responses = responses
        self.calls = 0

    def call(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        idx = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        item = self._responses[idx]
        if isinstance(item, Exception):
            raise item
        return item


FINAL_RESPONSE: dict[str, Any] = {
    "status": "complete",
    "reasoning": "Done.",
    "insights": ["feature_a drives the label."],
    "recommendations": ["Ship it."],
    "best_model": "logistic_regression",
    "key_metrics": {"cv_mean": "0.8"},
}


# ---------------------------------------------------------------------------
# Step parsing
# ---------------------------------------------------------------------------

class TestParseSteps:
    def test_valid_steps_parsed(self) -> None:
        steps = AgentController._parse_steps(
            {"steps": [{"step_number": 1, "tool_name": "clean_data", "parameters": {"a": 1}}]}
        )
        assert len(steps) == 1
        assert steps[0].tool_name == "clean_data"

    def test_malformed_entries_skipped(self) -> None:
        steps = AgentController._parse_steps(
            {
                "steps": [
                    "not a dict",
                    {"parameters": {}},  # missing tool_name
                    {"tool_name": "", "parameters": {}},  # empty tool_name
                    {"tool_name": "detect_outliers", "parameters": "oops"},  # bad params
                ]
            }
        )
        assert len(steps) == 1
        assert steps[0].tool_name == "detect_outliers"
        assert steps[0].parameters == {}

    def test_missing_step_number_defaults_to_index(self) -> None:
        steps = AgentController._parse_steps(
            {"steps": [{"tool_name": "clean_data"}, {"tool_name": "detect_outliers"}]}
        )
        assert [s.step_number for s in steps] == [1, 2]

    def test_non_list_steps_returns_empty(self) -> None:
        assert AgentController._parse_steps({"steps": "garbage"}) == []


# ---------------------------------------------------------------------------
# Full workflow (mock LLM)
# ---------------------------------------------------------------------------

class TestAnalyzeWorkflow:
    def test_happy_path_runs_all_stages(self, agent: AgentController, sample_csv: str) -> None:
        plan = {
            "status": "in_progress",
            "reasoning": "Clean then explore.",
            "steps": [
                {
                    "step_number": 1,
                    "tool_name": "clean_data",
                    "parameters": {"file_path": sample_csv, "strategy": "median"},
                    "rationale": "Handle missing values.",
                },
                {
                    "step_number": 2,
                    "tool_name": "correlation_analysis",
                    "parameters": {"file_path": sample_csv, "target_column": "label"},
                    "rationale": "Explore relationships.",
                },
            ],
        }
        agent.llm_client = _ScriptedLLM([plan, FINAL_RESPONSE])  # type: ignore[assignment]

        meta = agent.load_dataset(sample_csv, target_hint="label", interactive=False)
        assert meta.task_type == "classification"

        final = agent.analyze()
        assert final["status"] == "complete"
        assert agent.memory.get_context("cleaned_file_path")

        report_dir = Path(agent._output_dir) / "reports"
        assert list(report_dir.glob("*.md")), "Stage 7 must write a Markdown report"

    def test_llm_failure_triggers_fallback_and_deterministic_final(
        self, agent: AgentController, sample_csv: str
    ) -> None:
        agent.llm_client = _ScriptedLLM([RuntimeError("api down")])  # type: ignore[assignment]
        agent.load_dataset(sample_csv, target_hint="label", interactive=False)

        final = agent.analyze()

        assert final["status"] == "complete"
        assert agent.memory.get_context("llm_error")
        # Fallback plan must have actually executed tools
        executed = {r.tool_name for r in agent.memory.tool_results}
        assert {"clean_data", "detect_outliers", "correlation_analysis"} <= executed
        # Deterministic synthesis must reference real results
        assert final["insights"]

    def test_repeatedly_failing_tool_is_skipped(
        self, agent: AgentController, sample_csv: str
    ) -> None:
        bad_step = {
            "status": "in_progress",
            "reasoning": "Try a bad tool call.",
            "steps": [
                {
                    "step_number": 1,
                    "tool_name": "select_statistical_test",
                    "parameters": {
                        "file_path": sample_csv,
                        "feature_column": "does_not_exist",
                        "group_column": "label",
                    },
                    "rationale": "This will fail.",
                }
            ],
        }
        agent.llm_client = _ScriptedLLM(  # type: ignore[assignment]
            [bad_step, bad_step, bad_step, FINAL_RESPONSE]
        )
        agent.load_dataset(sample_csv, target_hint="label", interactive=False)
        agent.analyze()

        statuses = [
            r.status for r in agent.memory.tool_results
            if r.tool_name == "select_statistical_test"
        ]
        assert statuses.count("error") == 2, "two real attempts allowed"
        assert "skipped" in statuses, "third attempt must be skipped, not executed"

    def test_analyze_without_dataset_raises(self, agent: AgentController) -> None:
        with pytest.raises(RuntimeError, match="No dataset loaded"):
            agent.analyze()


# ---------------------------------------------------------------------------
# Objective, profiling, and dashboard integration
# ---------------------------------------------------------------------------

class TestObjectiveAndProfile:
    def test_objective_env_stored_in_memory(
        self, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("USER_OBJECTIVE", "what drives the label?")
        monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
        agent = AgentController(max_iterations=3, enable_rlm=False)
        assert agent.objective == "what drives the label?"
        assert agent.memory.get_context("user_objective") == "what drives the label?"

    def test_no_objective_leaves_context_empty(self, agent: AgentController) -> None:
        assert agent.memory.get_context("user_objective") is None

    def test_load_dataset_produces_profile(
        self, agent: AgentController, sample_csv: str
    ) -> None:
        agent.load_dataset(sample_csv, target_hint="label", interactive=False)
        assert agent.last_profile is not None
        profile_dict = agent.memory.get_context("data_profile")
        assert profile_dict["row_count"] == 80
        assert agent.memory.get_context("data_profile_summary")

    def test_objective_echoed_in_deterministic_final(
        self, sample_csv: str, tmp_path: pytest.TempPathFactory,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("USER_OBJECTIVE", "find churn drivers")
        monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
        agent = AgentController(max_iterations=3, enable_rlm=False)
        agent.llm_client = _ScriptedLLM([RuntimeError("api down")])  # type: ignore[assignment]
        agent.load_dataset(sample_csv, target_hint="label", interactive=False)
        final = agent.analyze()
        assert "find churn drivers" in final["reasoning"]

    def test_dashboard_json_written_after_analyze(
        self, agent: AgentController, sample_csv: str
    ) -> None:
        agent.llm_client = _ScriptedLLM([RuntimeError("api down")])  # type: ignore[assignment]
        agent.load_dataset(sample_csv, target_hint="label", interactive=False)
        agent.analyze()
        dash_path = agent.memory.get_context("dashboard_path")
        assert dash_path and Path(dash_path).exists()
        import json
        charts = json.loads(Path(dash_path).read_text(encoding="utf-8"))
        assert charts, "dashboard must contain at least one chart"
        assert all("spec" in c for c in charts)


# ---------------------------------------------------------------------------
# LLM client JSON parsing
# ---------------------------------------------------------------------------

class TestLLMClientParseJson:
    def test_plain_json(self) -> None:
        assert LLMClient._parse_json('{"a": 1}') == {"a": 1}

    def test_fenced_json(self) -> None:
        assert LLMClient._parse_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_truncated_json_repaired(self) -> None:
        parsed = LLMClient._parse_json('{"status": "in_progress", "steps": [{"tool_name": "clean')
        assert parsed["status"] == "in_progress"

    def test_garbage_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            LLMClient._parse_json("complete nonsense !!!")
