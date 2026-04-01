"""
Agent Controller — Orchestration and Reasoning Layer.

This is the 'brain' of the system. It:
  1. Accepts user datasets and routes them to the MemorySystem.
  2. Calls the LLM (via RLMEngine) for reasoning and planning.
  3. Dispatches tool calls to the Execution Layer.
  4. Manages the iterative reasoning-execution cycle.
  5. Produces the final structured report.

ARCHITECTURAL BOUNDARY:
  - This file handles PLANNING and ORCHESTRATION only.
  - Never call tools directly from here; always go through the ToolRegistry.
  - Never put raw data in LLM prompts; always go through PromptManager.
"""
from __future__ import annotations

import json
import os
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.core.memory import AnalysisStep, DatasetMetadata, MemorySystem
from src.core.prompt_manager import PromptManager
from src.rlm.engine import RLMEngine

console = Console()


class LLMClient:
    """
    Thin wrapper around LLM provider APIs.

    Supports OpenAI and Anthropic. Uses env vars for configuration.
    Never hardcodes credentials.
    """

    def __init__(self) -> None:
        self.provider: str = os.getenv("LLM_PROVIDER", "openai").lower()
        self.model: str = os.getenv("LLM_MODEL", "gpt-4o")
        self.temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
        self.max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))

    def call(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """
        Call the configured LLM and parse the JSON response.

        Args:
            system_prompt: The system-level instructions.
            user_prompt: The user-level task prompt.

        Returns:
            Parsed JSON dict from the LLM.

        Raises:
            ValueError: If the response cannot be parsed as JSON.
        """
        raw_text = self._dispatch(system_prompt, user_prompt)
        return self._parse_json_response(raw_text)

    def _dispatch(self, system_prompt: str, user_prompt: str) -> str:
        """Route to the correct LLM SDK."""
        if self.provider == "openai":
            return self._call_openai(system_prompt, user_prompt)
        elif self.provider == "anthropic":
            return self._call_anthropic(system_prompt, user_prompt)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        """Call OpenAI Chat Completions API."""
        from openai import OpenAI  # type: ignore[import-untyped]

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or "{}"

    def _call_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        """Call Anthropic Messages API."""
        import anthropic  # type: ignore[import-untyped]

        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        message = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return message.content[0].text

    @staticmethod
    def _parse_json_response(raw: str) -> dict[str, Any]:
        """Extract and parse the JSON block from an LLM response."""
        # Strip markdown code fences if present
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]
        try:
            return json.loads(raw.strip())
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM returned non-JSON response: {raw[:300]}") from exc


class ToolRegistry:
    """
    Registry of all available tools.

    Automatically discovers and maps tool names to BaseTool instances.
    The LLM references tools by name; this class resolves those to callables.
    """

    def __init__(self) -> None:
        from src.tools.data_processing import (
            CleanDataTool,
            CorrelationAnalysisTool,
            DetectOutliersTool,
            IngestDatasetTool,
        )
        from src.tools.statistical_analysis import SelectStatisticalTestTool
        from src.tools.ml_pipeline import TrainModelTool, EvaluateModelTool
        from src.tools.visualization import GenerateVisualizationsTool
        from src.tools.report_generator import GenerateReportTool

        _tools = [
            IngestDatasetTool(),
            CleanDataTool(),
            DetectOutliersTool(),
            CorrelationAnalysisTool(),
            SelectStatisticalTestTool(),
            TrainModelTool(),
            EvaluateModelTool(),
            GenerateVisualizationsTool(),
            GenerateReportTool(),
        ]
        self._registry = {t.name: t for t in _tools}

    def get(self, name: str) -> Any:
        """Look up a tool by name. Raises KeyError if not found."""
        if name not in self._registry:
            raise KeyError(
                f"Unknown tool '{name}'. Available: {list(self._registry.keys())}"
            )
        return self._registry[name]

    def get_all_descriptions(self) -> str:
        """Return concatenated tool descriptions for LLM prompt injection."""
        return "\n\n".join(t.to_prompt_description() for t in self._registry.values())


class AgentController:
    """
    Top-level orchestrator for the Agentic Data Analysis System.

    Implements the reasoning-execution loop with RLM-based context management.
    """

    def __init__(
        self,
        max_iterations: int | None = None,
        enable_rlm: bool | None = None,
        memory_persist_path: str | None = None,
    ) -> None:
        self.max_iterations = int(os.getenv("MAX_ITERATIONS", str(max_iterations or 15)))
        self.enable_rlm = (
            enable_rlm
            if enable_rlm is not None
            else os.getenv("ENABLE_RLM_INFERENCE", "true").lower() == "true"
        )
        self.memory = MemorySystem(persist_path=memory_persist_path)
        self.llm_client = LLMClient()
        self.tool_registry = ToolRegistry()
        self._rlm_engine: RLMEngine | None = None
        self._prompt_manager: PromptManager | None = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def load_dataset(self, file_path: str) -> DatasetMetadata:
        """
        Ingest a dataset and store metadata in the Memory System.

        Args:
            file_path: Path to a CSV or Excel file.

        Returns:
            The extracted DatasetMetadata.
        """
        console.print(Panel(
            f"[bold]Loading dataset:[/] {file_path}",
            title="[bold green]Agent Controller",
            border_style="green",
        ))

        ingest_tool = self.tool_registry.get("ingest_dataset")
        result = ingest_tool.run(file_path=file_path)

        if result.status == "error":
            raise RuntimeError(f"Dataset ingestion failed: {result.error_message}")

        metadata = DatasetMetadata(**result.output["metadata"])
        self.memory.store_dataset_metadata(metadata)
        self.memory.append_tool_result(result)
        return metadata

    def analyze(self) -> dict[str, Any]:
        """
        Run the full autonomous analysis pipeline.

        Returns:
            Final analysis report as a structured dict.
        """
        if not self.memory.dataset_metadata:
            raise RuntimeError("No dataset loaded. Call load_dataset() first.")

        # Initialize PromptManager and RLM Engine
        tool_descriptions = self.tool_registry.get_all_descriptions()
        self._prompt_manager = PromptManager(self.memory, tool_descriptions)
        self._rlm_engine = RLMEngine(
            llm_callable=self.llm_client.call,
            system_prompt=self._prompt_manager.get_system_prompt(),
            max_depth=int(os.getenv("RLM_MAX_DEPTH", "5")),
        )

        console.print("\n[bold magenta]🚀 Starting Autonomous Analysis[/]\n")
        final_result: dict[str, Any] = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("Reasoning...", total=None)

            for iteration in range(1, self.max_iterations + 1):
                self.memory.iteration_count = iteration
                progress.update(task, description=f"Reasoning cycle {iteration}/{self.max_iterations}")

                # --- REASONING PHASE ---
                if iteration == 1:
                    user_prompt = self._prompt_manager.get_initial_user_prompt()
                else:
                    user_prompt = self._prompt_manager.get_iteration_user_prompt()

                llm_response = self._rlm_engine.invoke(user_prompt, depth=0)

                # --- CHECK COMPLETION ---
                if llm_response.get("status") == "complete":
                    console.print("\n[bold green]✅ Analysis Complete![/]")
                    final_result = llm_response
                    break

                # --- PARSE & STORE PLAN ---
                steps_data = llm_response.get("steps", [])
                if not steps_data:
                    console.print("[yellow]⚠ LLM returned no steps. Requesting final synthesis.[/]")
                    final_prompt = self._prompt_manager.get_final_interpretation_prompt()
                    final_result = self._rlm_engine.invoke(final_prompt, depth=0)
                    break

                steps = [
                    AnalysisStep(
                        step_number=s["step_number"],
                        tool_name=s["tool_name"],
                        parameters=s.get("parameters", {}),
                        rationale=s.get("rationale", ""),
                    )
                    for s in steps_data
                ]
                self.memory.store_analysis_plan(steps)

                # --- EXECUTION PHASE ---
                for step in steps:
                    try:
                        tool = self.tool_registry.get(step.tool_name)
                        result = tool.run(**step.parameters)
                        self.memory.append_tool_result(result)
                        self.memory.mark_step_complete(step.step_number, result)
                    except KeyError as exc:
                        console.print(f"[red]❌ Tool not found: {exc}[/]")

                self.memory.save()

            else:
                # Max iterations reached
                console.print("[yellow]⚠ Max iterations reached. Generating final report.[/]")
                final_prompt = self._prompt_manager.get_final_interpretation_prompt()
                final_result = self._rlm_engine.invoke(final_prompt, depth=0)

        # Print reasoning trace
        if self.enable_rlm and self._rlm_engine:
            console.print()
            self._rlm_engine.print_reasoning_trace()

        return final_result
