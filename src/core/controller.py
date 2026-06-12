"""
Agent Controller — Orchestration and Reasoning Layer.

Implements the full 7-stage agent workflow:

  Stage 1 — Dataset Ingestion          load_dataset()
  Stage 2 — Initial Reasoning Phase    analyze() → first RLM invoke
  Stage 3 — Tool Selection & Execution analyze() → execution loop
  Stage 4 — Result Interpretation      analyze() → iteration prompt
  Stage 5 — Iterative Refinement       analyze() → loop until complete
  Stage 6 — RLM Workflow Management    _run_rlm_decomposition()
  Stage 7 — Report Generation          _generate_final_report()

ARCHITECTURAL BOUNDARY:
  - This file handles PLANNING and ORCHESTRATION only.
  - Raw data never enters LLM prompts — only summaries via PromptManager.
  - Tools are always called through ToolRegistry, never directly.
  - The LLM is always called through RLMEngine, never directly.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.core.dashboard import build_dashboard, dashboard_to_json
from src.core.memory import AnalysisStep, DatasetMetadata, MemorySystem, ToolResult
from src.core.profiler import DatasetProfile, profile_dataframe
from src.core.prompt_manager import PromptManager
from src.rlm.engine import RLMEngine, RLMSubTask

console = Console()


def _read_dataframe(file_path: str) -> pd.DataFrame:
    """Load a CSV/Excel dataset for profiling and dashboard generation."""
    lower = file_path.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(file_path)
    if lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(file_path)
    raise ValueError(f"Unsupported dataset format: {file_path}")

# Max retries before abandoning a failed step
MAX_STEP_RETRIES = 2

# Target auto-detection confidence thresholds
_AUTODETECT_HIGH = 0.75   # proceed autonomously above this
_AUTODETECT_LOW  = 0.40   # prompt user (CLI) or best-guess (UI) above this


# ---------------------------------------------------------------------------
# LLM Client — thin, provider-agnostic wrapper
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Thin wrapper around LLM provider APIs.

    Supports OpenAI and Anthropic. Credentials come from environment
    variables only — never hardcoded.
    """

    def __init__(self) -> None:
        self.provider: str = os.getenv("LLM_PROVIDER", "openai").lower()
        self.model: str = os.getenv("LLM_MODEL", "gpt-4o")
        self.temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
        self.max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))
        self.timeout: float = float(os.getenv("LLM_TIMEOUT", "120"))

    def ping(self) -> tuple[bool, str]:
        """
        Cheap connectivity + model-validity check (a few tokens).

        Returns (True, "") on success, (False, "<ExceptionType>: <detail>")
        on any failure — so callers can fail fast with the real reason
        instead of running a whole analysis on the deterministic fallback.
        """
        saved = self.max_tokens
        self.max_tokens = 16
        try:
            self._dispatch("You are a connectivity check. Reply with OK.", "Reply with OK.")
            return True, ""
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"
        finally:
            self.max_tokens = saved

    def call(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """
        Call the configured LLM and return the parsed JSON response.

        Raises:
            ValueError: If the response cannot be parsed as JSON.
        """
        raw = self._dispatch(system_prompt, user_prompt)
        return self._parse_json(raw)

    def _dispatch(self, system_prompt: str, user_prompt: str) -> str:
        if self.provider == "anthropic":
            return self._call_anthropic(system_prompt, user_prompt)
        return self._call_openai_compat(system_prompt, user_prompt)

    def _call_openai_compat(self, system_prompt: str, user_prompt: str) -> str:
        """OpenAI, OpenRouter, and NVIDIA NIM all use the OpenAI-compatible SDK."""
        from openai import OpenAI
        if self.provider == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY", "")
            base_url: str | None = "https://openrouter.ai/api/v1"
            extra_headers: dict[str, str] = {
                "HTTP-Referer": os.getenv("OPENROUTER_REFERER", "https://github.com/agentic-data-analysis"),
                "X-Title": "Agentic Data Analysis",
            }
        elif self.provider == "nvidia":
            api_key = os.getenv("NVIDIA_API_KEY", "")
            base_url = "https://integrate.api.nvidia.com/v1"
            extra_headers = {}
        else:
            api_key = os.getenv("OPENAI_API_KEY", "")
            base_url = None
            extra_headers = {}
        if not api_key:
            _key_names = {
                "openrouter": "OPENROUTER_API_KEY",
                "nvidia": "NVIDIA_API_KEY",
            }
            raise ValueError(
                f"No API key set for provider '{self.provider}'. "
                f"Set {_key_names.get(self.provider, 'OPENAI_API_KEY')}."
            )
        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": self.timeout,
            "max_retries": 2,
        }
        if base_url:
            client_kwargs["base_url"] = base_url
        if extra_headers:
            client_kwargs["default_headers"] = extra_headers
        client = OpenAI(**client_kwargs)
        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if self.provider == "openai":
            create_kwargs["response_format"] = {"type": "json_object"}
        if self.provider == "nvidia":
            # NVIDIA NIM requires streaming; gpt-oss-120b also emits
            # reasoning_content chunks (chain-of-thought) before the answer.
            create_kwargs["stream"] = True
            create_kwargs["top_p"] = 1
            stream = client.chat.completions.create(**create_kwargs)
            content_parts: list[str] = []
            reasoning_parts: list[str] = []
            for chunk in stream:
                if not getattr(chunk, "choices", None):
                    err = getattr(chunk, "error", None)
                    if err:
                        msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                        raise ValueError(f"nvidia error for model '{self.model}': {msg}")
                    continue
                delta = chunk.choices[0].delta
                # gpt-oss-120b is a reasoning model: the chain-of-thought
                # arrives in reasoning_content; the final answer arrives in
                # content. Collect both — content is preferred; if it ends up
                # empty (some reasoning-only models), fall back to reasoning.
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning is not None:
                    reasoning_parts.append(reasoning)
                text = getattr(delta, "content", None)
                if text is not None:
                    content_parts.append(text)
            content = "".join(content_parts).strip() or "".join(reasoning_parts).strip()
            if not content:
                raise ValueError(
                    f"NVIDIA returned empty content for model '{self.model}'. "
                    "Check that the model ID is correct and your account has access."
                )
            return content

        resp = client.chat.completions.create(**create_kwargs)
        # OpenRouter can return HTTP 200 with an error body instead of raising
        # (invalid model slug, moderation, no credits). The SDK then yields
        # choices=None — surface the real message instead of a TypeError.
        err = getattr(resp, "error", None)
        if err:
            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            raise ValueError(
                f"{self.provider} error for model '{self.model}': {msg}"
            )
        if not getattr(resp, "choices", None):
            raise ValueError(
                f"{self.provider} returned no completion for model '{self.model}' — "
                f"the model ID may be invalid, unavailable, or blocked by your "
                f"account's data policy."
            )
        content = resp.choices[0].message.content
        if content is None:
            raise ValueError("LLM returned None content")
        return str(content)

    def _call_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        import anthropic
        from anthropic.types import TextBlock

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("No API key set for provider 'anthropic'. Set ANTHROPIC_API_KEY.")
        client = anthropic.Anthropic(api_key=api_key, timeout=self.timeout, max_retries=2)
        msg = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        for block in msg.content:
            if isinstance(block, TextBlock):
                return block.text
        raise ValueError("Anthropic response contained no text block.")

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        # Strip markdown fences if present
        for fence in ("```json", "```"):
            if fence in raw:
                raw = raw.split(fence)[1].split("```")[0]
                break

        cleaned = raw.strip()

        try:
            return cast(dict[str, Any], json.loads(cleaned))
        except json.JSONDecodeError:
            pass

        # Try json-repair library if installed (handles all edge cases)
        try:
            from json_repair import repair_json
            candidate = repair_json(cleaned, return_objects=False)
            if candidate:
                return cast(dict[str, Any], json.loads(candidate))
        except (ImportError, json.JSONDecodeError):
            pass

        # Manual repair: close open strings/structures and fix trailing : or ,
        repaired = LLMClient._repair_truncated_json(cleaned)
        try:
            return cast(dict[str, Any], json.loads(repaired))
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM returned non-JSON: {raw[:300]}") from exc

    @staticmethod
    def _repair_truncated_json(s: str) -> str:
        """Close any open strings and bracket structures left by a truncated LLM response."""
        stack: list[str] = []
        in_string = False
        escape_next = False

        for ch in s:
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch in ("{", "["):
                stack.append("}" if ch == "{" else "]")
            elif ch in ("}", "]"):
                if stack and stack[-1] == ch:
                    stack.pop()

        result = s

        # 1. Close any open string literal
        if in_string:
            result += '"'

        if not stack:
            return result

        # 2. Examine the last meaningful (non-whitespace) character to decide
        #    what padding is needed before the closing brackets.
        tail = result.rstrip()
        last_ch = tail[-1] if tail else ""

        if last_ch == ":":
            # Truncated right after a colon — value never started
            result = tail + " null"
        elif last_ch == ",":
            # Trailing comma — remove it so the structure closes cleanly
            result = tail[:-1]

        # 3. Close all open brackets/braces in innermost-first order
        result += "".join(reversed(stack))
        return result


# ---------------------------------------------------------------------------
# Tool Registry — discovers and maps all tools
# ---------------------------------------------------------------------------

class ToolRegistry:
    """
    Registry of all available execution-layer tools.

    The LLM references tools by name; this class resolves them to
    callable BaseTool instances.
    """

    def __init__(self) -> None:
        from src.tools.clustering import ClusterDataTool
        from src.tools.data_processing import (
            CleanDataTool,
            CorrelationAnalysisTool,
            DetectOutliersTool,
            IngestDatasetTool,
        )
        from src.tools.ml_pipeline import EvaluateModelTool, TrainModelTool
        from src.tools.report_generator import GenerateReportTool
        from src.tools.statistical_analysis import SelectStatisticalTestTool
        from src.tools.visualization import GenerateVisualizationsTool

        _tools = [
            IngestDatasetTool(),
            CleanDataTool(),
            DetectOutliersTool(),
            CorrelationAnalysisTool(),
            SelectStatisticalTestTool(),
            TrainModelTool(),
            EvaluateModelTool(),
            ClusterDataTool(),
            GenerateVisualizationsTool(),
            GenerateReportTool(),
        ]
        self._registry = {t.name: t for t in _tools}

    def get(self, name: str) -> Any:
        if name not in self._registry:
            raise KeyError(
                f"Unknown tool '{name}'. Available: {list(self._registry.keys())}"
            )
        return self._registry[name]

    def has(self, name: str) -> bool:
        return name in self._registry

    def names(self) -> list[str]:
        return list(self._registry.keys())

    def get_all_descriptions(self) -> str:
        return "\n\n".join(t.to_prompt_description() for t in self._registry.values())


# ---------------------------------------------------------------------------
# Agent Controller — the top-level orchestrator
# ---------------------------------------------------------------------------

class AgentController:
    """
    Top-level orchestrator implementing the full 7-stage workflow.

    Reasoning and execution are strictly separated:
      - Reasoning: LLMClient → RLMEngine → PromptManager
      - Execution: ToolRegistry → BaseTool.run()
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
        self._output_dir: str = os.getenv("OUTPUT_DIR", "output")
        # Natural-language analysis objective supplied by the user (optional).
        self.objective: str = os.getenv("USER_OBJECTIVE", "").strip()
        if self.objective:
            self.memory.set_context("user_objective", self.objective)
        # Most recent dataset profile (set during load_dataset).
        self.last_profile: DatasetProfile | None = None
        # Charts from the most recent dashboard build (for the HTML report).
        self._last_charts: list[dict[str, Any]] = []
        self._rlm_decomposed: bool = False   # run decomposition at most once per session
        # Failures per tool across iterations — LLM-replanned steps are new
        # objects each cycle, so retry budgets must be tracked here.
        self._tool_failure_counts: dict[str, int] = {}
        # Optional callback fired after each tool: (tool_name, status, detail) -> None
        self.on_step_callback: Any = None
        # Optional callback fired after each LLM iteration: (iteration, stage) -> None
        self.on_iteration_callback: Any = None

    # ------------------------------------------------------------------
    # Stage 1 — Dataset Ingestion
    # ------------------------------------------------------------------

    def load_dataset(
        self,
        file_path: str,
        target_hint: str | None = None,
        interactive: bool = True,
    ) -> DatasetMetadata:
        """
        Stage 1: Ingest the dataset and store metadata in Memory.

        The LLM never sees raw data — only the compact metadata string.

        Args:
            file_path:   Path to the CSV or Excel file.
            target_hint: Optional column name to use as the ML target.
            interactive: When True and auto-detection confidence is low,
                         prompt the user via stdin. Set to False in
                         non-interactive environments (Streamlit, API).

        Returns:
            DatasetMetadata stored in the Memory System.
        """
        console.print(
            Panel(
                f"[bold]Stage 1 — Dataset Ingestion[/]\nFile: {file_path}",
                title="[bold green]Agent Controller",
                border_style="green",
            )
        )

        # Override target from CLI hint or environment
        target = target_hint or os.getenv("TARGET_COLUMN_HINT")

        ingest_tool = self.tool_registry.get("ingest_dataset")
        result = ingest_tool.run(file_path=file_path, target_column=target)

        if result.status == "error":
            raise RuntimeError(f"Dataset ingestion failed: {result.error_message}")

        raw_meta = result.output["metadata"]
        metadata = DatasetMetadata(**raw_meta)

        # Auto-detect target column when user hasn't provided one
        if not metadata.target_column:
            col, confidence = metadata.detect_target_with_confidence()

            if col and confidence >= _AUTODETECT_HIGH:
                # High confidence — proceed autonomously
                metadata.target_column = col
                console.print(
                    f"  [green]Auto-detected '{col}' as the target "
                    f"(confidence {confidence:.0%}). Proceeding with analysis...[/]"
                )
                metadata.task_type = metadata.infer_task_type()

            elif col and confidence >= _AUTODETECT_LOW:
                # Medium confidence — prompt if interactive, else use best guess
                if interactive:
                    console.print(
                        f"\n[yellow]Low-confidence target detection: '{col}' "
                        f"(confidence {confidence:.0%}).[/]"
                    )
                    user_input = input(
                        f"  Press ENTER to accept '{col}', or type another column name: "
                    ).strip()
                    chosen = user_input if user_input else col
                    if chosen in metadata.columns:
                        metadata.target_column = chosen
                        metadata.task_type = metadata.infer_task_type()
                    else:
                        console.print(f"  [yellow]Column '{chosen}' not found. Using EDA mode.[/]")
                        metadata.task_type = "eda"
                else:
                    console.print(
                        f"  [yellow]Using best-guess target '{col}' "
                        f"(confidence {confidence:.0%}). Pass target_hint to override.[/]"
                    )
                    metadata.target_column = col
                    metadata.task_type = metadata.infer_task_type()

            else:
                # Very low confidence / no viable candidate
                if interactive:
                    console.print("\n[yellow]Could not confidently determine the target column.[/]")
                    avail = ", ".join(list(metadata.columns.keys())[:15])
                    console.print(f"  Available columns: {avail}")
                    user_input = input(
                        "  Please input the target column name"
                        " (or press ENTER for EDA mode): "
                    ).strip()
                    if user_input and user_input in metadata.columns:
                        metadata.target_column = user_input
                        metadata.task_type = metadata.infer_task_type()
                    else:
                        metadata.task_type = "eda"
                else:
                    console.print("  [dim]No target column detected. Defaulting to EDA mode.[/]")
                    metadata.task_type = "eda"

        elif not metadata.task_type:
            metadata.task_type = metadata.infer_task_type()

        self.memory.store_dataset_metadata(metadata)
        self.memory.append_tool_result(result)

        # ---- Data profiling (the data scientist's "first look") ----
        # Failure here must never block the pipeline — it only enriches it.
        try:
            df = _read_dataframe(file_path)
            profile = profile_dataframe(df, target_column=metadata.target_column)
            self.last_profile = profile
            self.memory.set_context("data_profile", profile.to_dict())
            self.memory.set_context("data_profile_summary", profile.to_prompt_string())
            console.print(
                f"  [cyan]🔬 Profile: quality {profile.quality_score}/100, "
                f"{len(profile.warnings)} warning(s).[/]"
            )
        except Exception as exc:
            console.print(f"  [yellow]⚠ Data profiling failed (non-fatal): {exc}[/]")

        return metadata

    # ------------------------------------------------------------------
    # Stages 2-7 — Full autonomous analysis pipeline
    # ------------------------------------------------------------------

    def analyze(self) -> dict[str, Any]:
        """
        Run the complete autonomous analysis pipeline (Stages 2-7).

        Returns:
            Final analysis report as a structured dict.
        """
        if not self.memory.dataset_metadata:
            raise RuntimeError("No dataset loaded. Call load_dataset() first.")

        # Initialise PromptManager and RLMEngine
        tool_desc = self.tool_registry.get_all_descriptions()
        self._prompt_manager = PromptManager(self.memory, tool_desc)
        self._rlm_engine = RLMEngine(
            llm_callable=self.llm_client.call,
            system_prompt=self._prompt_manager.get_system_prompt(),
            max_depth=int(os.getenv("RLM_MAX_DEPTH", "5")),
        )

        console.print("\n[bold magenta]🚀 Starting Autonomous Analysis — Stages 2-7[/]\n")
        final_result: dict[str, Any] = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task_id = progress.add_task("Reasoning…", total=None)

            for iteration in range(1, self.max_iterations + 1):
                self.memory.iteration_count = iteration
                self._rlm_engine.set_iteration(iteration)
                progress.update(
                    task_id,
                    description=f"Stage 2/4/5 — Reasoning cycle {iteration}/{self.max_iterations}",
                )

                # ---- Stage 2 / 4+5: Reasoning Phase ----
                if iteration == 1:
                    user_prompt = self._prompt_manager.get_initial_user_prompt()
                    stage_label = "stage2:initial_reasoning"
                else:
                    user_prompt = self._prompt_manager.get_iteration_user_prompt()
                    stage_label = f"stage4-5:iteration_{iteration}"

                if self.on_iteration_callback:
                    self.on_iteration_callback(iteration, stage_label)

                # ---- Reasoning with graceful degradation ----
                # An LLM/API failure must never abort a running analysis:
                # iteration 1 falls back to a deterministic plan, later
                # iterations synthesise a final answer from existing results.
                try:
                    llm_response = self._rlm_engine.invoke(
                        user_prompt, depth=0, stage=stage_label
                    )
                    if llm_response.get("status") == "error":
                        raise RuntimeError(
                            str(llm_response.get("error", "Unknown LLM error"))
                        )
                except Exception as exc:
                    self.memory.set_context("llm_error", f"{type(exc).__name__}: {exc}")
                    console.print(
                        f"[yellow]⚠ LLM failure on iteration {iteration}: {exc}[/]"
                    )
                    if iteration == 1:
                        console.print("[yellow]  → Using deterministic fallback plan.[/]")
                        llm_response = self._build_fallback_plan()
                    else:
                        console.print("[yellow]  → Synthesising final report from results.[/]")
                        final_result = self._deterministic_final()
                        break

                # ---- Check for completion (Stage 7 trigger) ----
                if llm_response.get("status") == "complete":
                    console.print("\n[bold green]✅ LLM signalled analysis complete.[/]")
                    final_result = llm_response
                    break

                # ---- Parse plan steps (tolerant of malformed entries) ----
                steps = self._parse_steps(llm_response)
                if not steps:
                    console.print(
                        f"[yellow]⚠ No valid steps on iteration {iteration} — synthesising final report.[/]"
                    )
                    final_result = self._deterministic_final()
                    break
                self.memory.store_analysis_plan(steps)

                # ---- Stage 3: Tool Selection & Execution ----
                progress.update(task_id, description=f"Stage 3 — Executing {len(steps)} tool(s)…")
                self._execute_steps(steps)

                # ---- Stage 6: RLM Decomposition (if enabled & many features, once only) ----
                if self.enable_rlm and not self._rlm_decomposed and self._should_decompose():
                    progress.update(task_id, description="Stage 6 — RLM task decomposition…")
                    self._run_rlm_decomposition()
                    self._rlm_decomposed = True

                self.memory.save()

            else:
                # Max iterations reached
                console.print("[yellow]⚠ Max iterations reached — generating final report.[/]")
                final_prompt = self._prompt_manager.get_final_interpretation_prompt()
                try:
                    final_result = self._rlm_engine.invoke(
                        final_prompt, depth=0, stage="stage7:max_iter_synthesis"
                    )
                    if final_result.get("status") == "error":
                        raise RuntimeError(str(final_result.get("error", "Unknown LLM error")))
                except Exception as exc:
                    self.memory.set_context("llm_error", f"{type(exc).__name__}: {exc}")
                    final_result = self._deterministic_final()

        # ---- Stage 7: Report Generation ----
        self._generate_final_report(final_result)
        self._generate_dashboard()
        self._generate_html_report(final_result)

        # Print reasoning trace
        if self._rlm_engine:
            console.print()
            self._rlm_engine.print_reasoning_trace()

        return final_result

    # ------------------------------------------------------------------
    # Resilience helpers — plan parsing and LLM-failure fallbacks
    # ------------------------------------------------------------------

    def _parse_steps(self, llm_response: dict[str, Any]) -> list[AnalysisStep]:
        """
        Parse LLM plan steps, skipping malformed entries instead of crashing.

        Anti-hallucination guard: steps naming tools that do not exist in the
        ToolRegistry are rejected here (never executed), and a planner note is
        recorded so the next reasoning cycle sees the correction.
        """
        steps: list[AnalysisStep] = []
        rejected: list[str] = []
        raw_steps = llm_response.get("steps", [])
        if not isinstance(raw_steps, list):
            return steps
        for idx, s in enumerate(raw_steps, 1):
            if not isinstance(s, dict):
                continue
            tool_name = s.get("tool_name")
            if not isinstance(tool_name, str) or not tool_name:
                continue
            if not self.tool_registry.has(tool_name):
                rejected.append(tool_name)
                continue
            parameters = s.get("parameters", {})
            if not isinstance(parameters, dict):
                parameters = {}
            step_number = s.get("step_number")
            steps.append(
                AnalysisStep(
                    step_number=step_number if isinstance(step_number, int) else idx,
                    tool_name=tool_name,
                    parameters=parameters,
                    rationale=str(s.get("rationale", "")),
                )
            )
        if rejected:
            console.print(
                f"  [yellow]⚠ Rejected {len(rejected)} hallucinated tool name(s): "
                f"{', '.join(rejected)}[/]"
            )
            self.memory.append_tool_result(
                ToolResult(
                    tool_name="planner",
                    status="skipped",
                    output={
                        "summary": (
                            f"Rejected unknown tool name(s): {', '.join(sorted(set(rejected)))}. "
                            f"Only these tools exist: {', '.join(self.tool_registry.names())}."
                        )
                    },
                )
            )
        return steps

    def _build_fallback_plan(self) -> dict[str, Any]:
        """
        Deterministic analysis plan used when the LLM is unreachable on the
        first reasoning cycle. Mirrors the mandatory plan structure from the
        initial prompt: clean → outliers → correlation → (train + evaluate
        when a target exists, otherwise EDA visualisations).
        """
        meta = self.memory.dataset_metadata
        if meta is None:
            raise RuntimeError("No dataset loaded — cannot build a fallback plan.")
        fp = meta.file_path
        steps: list[dict[str, Any]] = [
            {
                "step_number": 1,
                "tool_name": "clean_data",
                "parameters": {
                    "file_path": fp,
                    "strategy": "median",
                    "target_column": meta.target_column,
                },
                "rationale": "Fallback plan: impute missing values before analysis.",
            },
            {
                "step_number": 2,
                "tool_name": "detect_outliers",
                "parameters": {"file_path": fp, "method": "iqr"},
                "rationale": "Fallback plan: flag anomalous rows.",
            },
            {
                "step_number": 3,
                "tool_name": "correlation_analysis",
                "parameters": {"file_path": fp, "target_column": meta.target_column},
                "rationale": "Fallback plan: quantify feature relationships.",
            },
        ]
        if meta.target_column and meta.task_type in ("classification", "regression"):
            steps += [
                {
                    "step_number": 4,
                    "tool_name": "train_model",
                    "parameters": {
                        "file_path": fp,
                        "target_column": meta.target_column,
                        "task_type": meta.task_type,
                    },
                    "rationale": "Fallback plan: train baseline models with CV.",
                },
                {
                    "step_number": 5,
                    "tool_name": "evaluate_model",
                    "parameters": {
                        "file_path": fp,
                        "target_column": meta.target_column,
                        "task_type": meta.task_type,
                    },
                    "rationale": "Fallback plan: evaluate the best model on held-out data.",
                },
            ]
        else:
            steps += [
                {
                    "step_number": 4,
                    "tool_name": "cluster_data",
                    "parameters": {"file_path": fp},
                    "rationale": "Fallback plan: no target — discover natural segments.",
                },
                {
                    "step_number": 5,
                    "tool_name": "generate_visualizations",
                    "parameters": {"file_path": fp, "chart_type": "correlation_heatmap"},
                    "rationale": "Fallback plan: EDA visualisation without a target.",
                },
            ]
        return {
            "status": "in_progress",
            "reasoning": "LLM unavailable — executing deterministic fallback plan.",
            "steps": steps,
        }

    def _generate_dashboard(self) -> None:
        """
        Build the dynamic dashboard from the final state of the analysis and
        save it as JSON next to the reports. Non-fatal on any failure.
        """
        meta = self.memory.dataset_metadata
        if meta is None:
            return
        source = str(self.memory.get_context("cleaned_file_path") or meta.file_path)
        try:
            df = _read_dataframe(source)
            profile = profile_dataframe(df, target_column=meta.target_column)
            charts = build_dashboard(
                df,
                profile,
                target_column=meta.target_column,
                task_type=meta.task_type,
                tool_results=[r.to_dict() for r in self.memory.tool_results],
            )
            out_path = Path(self._output_dir) / "reports" / "dashboard.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(dashboard_to_json(charts), encoding="utf-8")
            self._last_charts = [c.to_dict() for c in charts]
            self.memory.set_context("dashboard_path", str(out_path))
            console.print(
                f"  [green]📊 Dashboard generated: {len(charts)} chart(s) → {out_path}[/]"
            )
        except Exception as exc:
            console.print(f"  [yellow]⚠ Dashboard generation failed (non-fatal): {exc}[/]")

    def _generate_html_report(self, llm_final: dict[str, Any]) -> None:
        """
        Write the self-contained HTML report (summary + interactive dashboard).
        Non-fatal on any failure.
        """
        meta = self.memory.dataset_metadata
        if meta is None:
            return
        try:
            from src.core.html_report import build_html_report

            html_doc = build_html_report(
                dataset_name=Path(meta.file_path).stem,
                llm_insights=llm_final,
                tool_results=[r.to_dict() for r in self.memory.tool_results],
                charts=self._last_charts,
                objective=self.objective,
                profile=self.memory.get_context("data_profile"),
            )
            out_path = Path(self._output_dir) / "reports" / "report.html"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(html_doc, encoding="utf-8")
            self.memory.set_context("html_report_path", str(out_path))
            console.print(f"  [green]🌐 HTML report generated → {out_path}[/]")
        except Exception as exc:
            console.print(f"  [yellow]⚠ HTML report generation failed (non-fatal): {exc}[/]")

    def _deterministic_final(self) -> dict[str, Any]:
        """Synthesise a final report dict from accumulated tool results, no LLM needed."""
        insights: list[str] = []
        recommendations: list[str] = []
        key_metrics: dict[str, Any] = {}
        best_model = self.memory.get_context("best_model_name")

        train = self.memory.get_last_result_for("train_model")
        if train and train.status == "success":
            models_trained = train.output.get("models_trained", {})
            best = train.output.get("best_model")
            if best and best in models_trained:
                best_model = best
                metrics = models_trained[best]
                key_metrics["cv_mean"] = metrics.get("cv_mean")
                key_metrics["cv_std"] = metrics.get("cv_std")
                key_metrics["train_test_gap"] = metrics.get("train_test_gap")
                insights.append(
                    f"Best model '{best}' reached cross-validated score "
                    f"{metrics.get('cv_mean')} ± {metrics.get('cv_std')} "
                    f"with train-test gap {metrics.get('train_test_gap')}."
                )
            warnings = train.output.get("overfit_warnings", [])
            insights.extend(f"Overfitting warning: {w}" for w in warnings)
            if warnings:
                recommendations.append(
                    "Reduce model complexity (lower max_depth) or add regularisation "
                    "to close the train-test gap."
                )

        corr = self.memory.get_last_result_for("correlation_analysis")
        if corr and corr.status == "success":
            top = corr.output.get("top_correlations", [])
            if top:
                pair = top[0]
                insights.append(
                    f"Strongest feature correlation: {pair.get('col_a')} ↔ "
                    f"{pair.get('col_b')} (r={pair.get('correlation')})."
                )

        outliers = self.memory.get_last_result_for("detect_outliers")
        if outliers and outliers.status == "success":
            insights.append(
                f"Outlier scan ({outliers.output.get('method')}): "
                f"{outliers.output.get('total_outliers')} rows flagged "
                f"({outliers.output.get('outlier_percentage')}% of data)."
            )

        stat = self.memory.get_last_result_for("select_statistical_test")
        if stat and stat.status == "success":
            insights.append(str(stat.output.get("summary", "")))

        cluster = self.memory.get_last_result_for("cluster_data")
        if cluster and cluster.status == "success":
            sizes = cluster.output.get("cluster_sizes", {})
            insights.append(
                f"Segmentation: {cluster.output.get('n_clusters')} natural clusters "
                f"(silhouette={cluster.output.get('silhouette_score')}, "
                f"{cluster.output.get('separation_quality')} separation); "
                f"sizes: {sizes}."
            )

        if not insights:
            insights.append("Analysis produced no tool results to synthesise.")
        if not recommendations:
            recommendations.append(
                "Re-run with a reachable LLM provider for narrative interpretation "
                "of these deterministic findings."
            )

        llm_error = self.memory.get_context("llm_error")
        reasoning = (
            "Deterministic synthesis: the LLM became unreachable mid-run, so "
            "findings were compiled directly from tool outputs."
        )
        if llm_error:
            reasoning += f" (LLM error: {llm_error})"
        if self.objective:
            reasoning = f"User objective: {self.objective}\n{reasoning}"

        return {
            "status": "complete",
            "llm_fallback": True,
            "reasoning": reasoning,
            "insights": insights,
            "recommendations": recommendations,
            "best_model": best_model,
            "key_metrics": key_metrics,
        }

    # ------------------------------------------------------------------
    # Stage 3 execution helper
    # ------------------------------------------------------------------

    def _execute_steps(self, steps: list[AnalysisStep]) -> None:
        """
        Stage 3 — execute each tool in the plan with retry budgets.

        Retry logic: failures are counted per tool across iterations
        (the LLM re-plans with fresh step objects each cycle). Once a
        tool has failed MAX_STEP_RETRIES times it is skipped instead of
        executed again, so one broken tool can never stall the pipeline.
        """
        total_steps = len(steps)
        for idx, step in enumerate(steps, 1):
            if self._tool_failure_counts.get(step.tool_name, 0) >= MAX_STEP_RETRIES:
                console.print(
                    f"  [yellow]⏭ Step {step.step_number}: {step.tool_name} skipped "
                    f"(exceeded {MAX_STEP_RETRIES} retries).[/]"
                )
                skip_result = ToolResult(
                    tool_name=step.tool_name,
                    status="skipped",
                    output={
                        "summary": (
                            f"Skipped: '{step.tool_name}' already failed "
                            f"{MAX_STEP_RETRIES} times. Do not plan it again."
                        )
                    },
                )
                self.memory.append_tool_result(skip_result)
                self.memory.mark_step_complete(step.step_number, skip_result)
                continue

            console.print(
                f"  [cyan]→ Step {step.step_number}: {step.tool_name}[/] "
                f"[dim]{step.rationale[:60]}[/]"
            )
            # Fire pre-execution callback
            if self.on_step_callback:
                self.on_step_callback(step.tool_name, "running",
                                      f"{idx}/{total_steps} — {step.tool_name}…")
            # ── Auto-substitute cleaned_file_path & output_dir into params ──
            params = dict(step.parameters)
            cleaned = self.memory.get_context("cleaned_file_path")
            if cleaned:
                # Replace any placeholder or original path reference
                if "file_path" in params:
                    raw = params["file_path"]
                    # Substitute if it looks like a placeholder or the original file
                    if (raw in ("cleaned_file_path", "<cleaned_file_path>",
                                "path/to/cleaned", "")
                            or not raw.endswith((".csv", ".xlsx", ".xls"))):
                        params["file_path"] = cleaned
                    # Also substitute if it's the original (non-cleaned) path
                    # and a cleaned version now exists
                    elif step.tool_name not in ("clean_data", "ingest_dataset"):
                        params["file_path"] = cleaned

            # Inject output_dir for tools that write files
            if step.tool_name in ("train_model", "evaluate_model",
                                   "generate_visualizations", "generate_report"):
                if "output_dir" not in params or not params.get("output_dir"):
                    base = self._output_dir
                    subdir = {
                        "train_model":           "models",
                        "evaluate_model":        "models",
                        "generate_visualizations": "visualizations",
                        "generate_report":       "reports",
                    }.get(step.tool_name, "output")
                    params["output_dir"] = str(Path(base) / subdir)

            # ── Auto-substitute best_model_path for tools that need a trained model ──
            if step.tool_name in ("evaluate_model", "generate_visualizations"):
                best_path = self.memory.get_context("best_model_path")
                if best_path:
                    raw_mp = params.get("model_path", "")
                    if not raw_mp or not Path(raw_mp).exists():
                        params["model_path"] = best_path

            # ── generate_report needs the accumulated results, which the LLM
            #    cannot supply — inject them from memory ──
            if step.tool_name == "generate_report":
                meta = self.memory.dataset_metadata
                params.setdefault(
                    "dataset_name", Path(meta.file_path).stem if meta else "dataset"
                )
                params["tool_results_json"] = json.dumps(
                    [r.to_dict() for r in self.memory.tool_results], default=str
                )
                params.setdefault("llm_insights", {})

            # Defense in depth: _parse_steps filters unknown tools, but never
            # let a registry miss crash the whole pipeline.
            try:
                tool = self.tool_registry.get(step.tool_name)
            except KeyError as exc:
                result = ToolResult(
                    tool_name=step.tool_name,
                    status="error",
                    output={},
                    error_message=str(exc),
                )
                self.memory.append_tool_result(result)
                self.memory.mark_step_complete(step.step_number, result)
                continue
            result = tool.run(**params)
            self.memory.append_tool_result(result)
            self.memory.mark_step_complete(step.step_number, result)

            if result.status == "error":
                self._tool_failure_counts[step.tool_name] = (
                    self._tool_failure_counts.get(step.tool_name, 0) + 1
                )
                self.memory.increment_retry(step.step_number)

            # Store important outputs in memory context for downstream tools
            if step.tool_name == "train_model" and result.status == "success":
                best = result.output.get("best_model", "")
                mt = result.output.get("models_trained", {})
                if best and best in mt:
                    model_path = mt[best].get("model_path", "")
                    if model_path:
                        self.memory.set_context("best_model_path", model_path)
                        self.memory.set_context("best_model_name", best)

            # If cleaning produced a cleaned file, store it for downstream tools
            if step.tool_name == "clean_data" and result.status == "success":
                cleaned_path = result.output.get("cleaned_file_path")
                if cleaned_path:
                    self.memory.set_context("cleaned_file_path", cleaned_path)
                    console.print(
                        f"  [dim]Cleaned file stored → {cleaned_path}[/]"
                    )
            # Fire post-execution callback
            if self.on_step_callback:
                summary = result.output.get("summary", "")[:80] if result.status == "success" else result.error_message
                self.on_step_callback(step.tool_name, result.status, f"{idx}/{total_steps} done — {summary}")

    # ------------------------------------------------------------------
    # Stage 6 — RLM decomposition
    # ------------------------------------------------------------------

    def _should_decompose(self) -> bool:
        """Trigger decomposition when the dataset has many features (>15)."""
        meta = self.memory.dataset_metadata
        return meta is not None and meta.column_count > 15

    def _run_rlm_decomposition(self) -> None:
        """
        Stage 6: Decompose the feature space into sub-groups and run
        targeted LLM sub-calls on each group.

        This is the core RLM innovation: instead of one monolithic context,
        each feature group gets its own focused reasoning call.
        """
        if self._rlm_engine is None or self._prompt_manager is None:
            return

        meta = self.memory.dataset_metadata
        if meta is None:
            return

        num_cols = meta.numerical_cols
        cat_cols = meta.categorical_cols

        # Partition numerical columns into groups of ≤8
        groups: dict[str, list[str]] = {}
        chunk_size = 8
        for i in range(0, len(num_cols), chunk_size):
            groups[f"numerical_group_{i // chunk_size + 1}"] = num_cols[i: i + chunk_size]
        if cat_cols:
            groups["categorical_group"] = cat_cols[:10]

        sub_tasks = [
            RLMSubTask(
                task_id=gid,
                description=f"Analyse {len(cols)}-feature group: {', '.join(cols[:5])}…",
                context={"columns": cols, "group_id": gid},
            )
            for gid, cols in groups.items()
        ]

        if not sub_tasks:
            return

        def build_prompt(task: RLMSubTask) -> str:
            assert self._prompt_manager is not None
            ctx_summary = json.dumps(task.context)
            return self._prompt_manager.get_rlm_subtask_prompt(
                task_id=task.task_id,
                description=task.description,
                context_summary=ctx_summary,
            )

        sub_results = self._rlm_engine.decompose_and_invoke(
            sub_tasks=sub_tasks,
            prompt_builder=build_prompt,
            depth=1,
        )

        # Store sub-results in memory context for final synthesis
        self.memory.set_context("rlm_sub_results", sub_results)
        console.print(
            f"  [green]✓ RLM decomposition complete: "
            f"{len(sub_results)} sub-task(s) resolved.[/]"
        )

    # ------------------------------------------------------------------
    # Stage 7 — Report generation
    # ------------------------------------------------------------------

    def _generate_final_report(self, llm_final: dict[str, Any]) -> None:
        """
        Stage 7: Invoke GenerateReportTool to produce the Markdown/JSON report.

        The report tool is called with the serialised tool results and the
        LLM's final interpretation so it can produce a complete document.
        """
        meta = self.memory.dataset_metadata
        dataset_name = Path(meta.file_path).stem if meta else "dataset"

        tool_results_json = json.dumps(
            [r.to_dict() for r in self.memory.tool_results], default=str
        )

        report_tool = self.tool_registry.get("generate_report")
        result = report_tool.run(
            dataset_name=dataset_name,
            tool_results_json=tool_results_json,
            llm_insights=llm_final,
            output_dir=str(Path(self._output_dir) / "reports"),
        )

        self.memory.append_tool_result(result)

        if result.status == "success":
            console.print(
                Panel(
                    f"[bold green]Stage 7 — Report Generated[/]\n"
                    f"Markdown: {result.output.get('markdown_path')}\n"
                    f"JSON:     {result.output.get('json_path')}",
                    border_style="green",
                )
            )
        else:
            console.print(
                f"[yellow]⚠ Report generation failed: {result.error_message}[/]"
            )
