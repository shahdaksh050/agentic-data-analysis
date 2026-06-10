"""
Prompt Manager — centralised LLM prompt engineering.

All prompt templates live here. The Agent Controller never constructs
raw strings; it always delegates to PromptManager. This makes the
system's reasoning strategy version-controlled and easy to iterate on.

Workflow coverage:
  Stage 2 — Initial Reasoning Phase (get_initial_user_prompt)
  Stage 4 — Result Interpretation   (get_iteration_user_prompt)
  Stage 5 — Iterative Refinement    (get_iteration_user_prompt)
  Stage 7 — Report Generation       (get_final_interpretation_prompt)
"""
from __future__ import annotations

from src.core.memory import MemorySystem

# ---------------------------------------------------------------------------
# System prompt — injected once per session
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_CORE = """\
You are an expert autonomous data scientist operating inside a multi-step \
agentic pipeline. Your role is PLANNING and REASONING ONLY — you do not write \
or execute Python code directly.

The Agent Controller will execute every tool you specify. \
You must produce valid, parseable JSON every time.

## Available Tools
{tool_descriptions}

## Strict Response Contract

### Form 1 — Action Plan  (return when more analysis is needed)
```json
{{
  "status": "in_progress",
  "reasoning": "2-3 sentences: what you observed and why you chose these steps.",
  "steps": [
    {{
      "step_number": 1,
      "tool_name": "<exact tool name>",
      "parameters": {{ "<param>": "<value>" }},
      "rationale": "One sentence: why this tool with these parameters."
    }}
  ]
}}
```

### Form 2 — Final Answer  (return when analysis is genuinely complete)
```json
{{
  "status": "complete",
  "reasoning": "Synthesis of all findings.",
  "insights": [
    "Concrete finding 1.",
    "Concrete finding 2."
  ],
  "recommendations": [
    "Actionable recommendation 1."
  ],
  "best_model": "<model_name or null>",
  "key_metrics": {{ "<metric>": "<value>" }}
}}
```

## Hard Rules
- NEVER include raw data rows, arrays, or full DataFrames in your response.
- Reference tools by their EXACT `tool_name`.
- Provide a `rationale` for EVERY step — this is a research-grade system.
- If a tool failed in the previous iteration, adapt the plan. \
  Do not repeat the identical call that already errored.
- When a `cleaned_file_path` appears in results, use it as the input \
  `file_path` for all subsequent tools that read data.
- Respond with ONLY the JSON object — no markdown fences, no preamble.
"""


# ---------------------------------------------------------------------------
# Stage 2 — Initial reasoning (first LLM call)
# ---------------------------------------------------------------------------

INITIAL_ANALYSIS_PROMPT = """## Dataset Overview
{dataset_metadata}

## Your Task
You are in Stage 2 of the analysis pipeline (Initial Reasoning Phase).
Produce an ordered analysis plan using EXACT tool_names from the tool list above.

MANDATORY RULES — follow these precisely:
1. ALWAYS begin with clean_data (strategy: "median") passing the original file_path.
2. ALWAYS follow with detect_outliers on the cleaned file.
3. ALWAYS run correlation_analysis on the cleaned file.
4. If task type is "eda" (no target column):
   - Run generate_visualizations (chart_type: "correlation_heatmap") and
     generate_visualizations (chart_type: "distributions").
   - Do NOT run train_model, evaluate_model, or select_statistical_test.
5. If a target column IS identified (classification or regression):
   - Run select_statistical_test with an appropriate feature_column and the target as group_column.
   - Run train_model with the cleaned file, the target_column, and the correct task_type.
   - Run evaluate_model using the best model path from train_model output.
   - Run generate_visualizations (chart_type: "feature_importance") and "roc_curve".
6. ALWAYS end with generate_report.
7. For every tool that reads data after clean_data, use the cleaned_file_path
   returned by clean_data — NOT the original file path.
8. Include file_path in every tool's parameters that requires it.
9. Produce 5–8 steps total. Never exceed 8 steps per iteration.

Respond in Form 1 (Action Plan).
"""


# ---------------------------------------------------------------------------
# Stage 4+5 — Result interpretation & iterative refinement
# ---------------------------------------------------------------------------

ITERATION_PROMPT = """\
## Dataset Overview
{dataset_metadata}

## Analysis Results So Far (Stage 4 — Result Interpretation)
{results_summary}

## Pending Steps
{pending_steps}

## Failed Steps (if any)
{failed_steps}

## Your Task
You are in Stage 4/5 of the pipeline (Result Interpretation / Iterative Refinement).

1. Review every result carefully.
2. If a tool failed, diagnose why and produce a corrected step.
3. If new findings change the analysis direction, update the plan.
4. If ALL key analyses are complete, return Form 2 (Final Answer).
5. Otherwise return Form 1 with only the remaining/corrected steps.

Remember: always use `cleaned_file_path` as the source for downstream tools \
when cleaning has already been performed.

Respond in the correct JSON form.
"""


# ---------------------------------------------------------------------------
# Stage 7 — Final report synthesis
# ---------------------------------------------------------------------------

FINAL_INTERPRETATION_PROMPT = """\
## Dataset Overview
{dataset_metadata}

## Complete Analysis Results (Stage 7 — Report Generation)
{results_summary}

## Your Task
Synthesise ALL results into the final report. Be specific — reference actual \
metric values, column names, and model names from the results above.

Your insights must be data-driven and actionable. \
Recommendations must be concrete and implementable.

Respond in Form 2 (Final Answer). This is your last call.
"""


# ---------------------------------------------------------------------------
# Stage 6 — RLM sub-task prompt (used by RLMEngine.decompose_and_invoke)
# ---------------------------------------------------------------------------

RLM_SUBTASK_PROMPT = """\
## Sub-Task: {task_id}
{description}

## Sub-Context (from REPL environment)
{context_summary}

## Dataset Overview
{dataset_metadata}

## Results So Far
{results_summary}

## Your Task
Analyse ONLY the sub-task above. Return Form 1 with a focused 1-3 step plan \
for this sub-task, or Form 2 if the sub-task is already resolved by existing results.
"""


# ---------------------------------------------------------------------------
# PromptManager
# ---------------------------------------------------------------------------

class PromptManager:
    """Assembles structured LLM prompts from memory state and tool descriptions."""

    def __init__(self, memory: MemorySystem, tool_descriptions: str) -> None:
        self.memory = memory
        self.tool_descriptions = tool_descriptions

    def get_system_prompt(self) -> str:
        return SYSTEM_PROMPT_CORE.format(tool_descriptions=self.tool_descriptions)

    def get_initial_user_prompt(self) -> str:
        meta = self.memory.dataset_metadata
        file_path  = meta.file_path  if meta else "UNKNOWN_PATH"
        target_col = meta.target_column if meta else None
        task_type  = meta.task_type if meta else "eda"

        concrete_instructions = (
            f"\n## Concrete Values for Your Plan\n"
            f"- original_file_path  = \"{file_path}\"\n"
            f"- target_column       = \"{target_col}\"\n"
            f"- task_type           = \"{task_type}\"\n"
            f"- Use these EXACT string values in your JSON parameters.\n"
            f"- After clean_data runs, the controller will automatically "
            f"substitute cleaned_file_path for file_path in subsequent tools.\n"
        )
        return INITIAL_ANALYSIS_PROMPT.format(
            dataset_metadata=self.memory.get_metadata_prompt(),
        ) + concrete_instructions

    def get_iteration_user_prompt(self) -> str:
        meta = self.memory.dataset_metadata
        pending = self.memory.get_pending_steps()
        failed = self.memory.get_failed_steps()
        # Inject cleaned path if available
        cleaned = self.memory.get_context("cleaned_file_path")
        best_model_path = self.memory.get_context("best_model_path")
        best_model_name = self.memory.get_context("best_model_name")

        pending_str = (
            "\n".join(
                f"Step {s.step_number}: {s.tool_name}({json_compact(s.parameters)})"
                for s in pending
            )
            or "None — re-evaluate whether more analysis is needed."
        )
        failed_str = (
            "\n".join(
                f"Step {s.step_number}: {s.tool_name} → "
                f"{s.result.error_message if s.result else 'unknown error'} "
                f"(retries: {s.retry_count})"
                for s in failed
            )
            or "None."
        )

        concrete = ""
        if cleaned:
            concrete += f"\n## Current File Paths\n- cleaned_file_path = \"{cleaned}\"\n"
        if best_model_path:
            concrete += f"- best_model_path = \"{best_model_path}\"\n"
            concrete += f"- best_model_name = \"{best_model_name}\"\n"
        if meta and meta.target_column:
            concrete += f"- target_column = \"{meta.target_column}\"\n"

        return ITERATION_PROMPT.format(
            dataset_metadata=self.memory.get_metadata_prompt(),
            results_summary=self.memory.get_results_summary(),
            pending_steps=pending_str,
            failed_steps=failed_str,
        ) + concrete

    def get_final_interpretation_prompt(self) -> str:
        return FINAL_INTERPRETATION_PROMPT.format(
            dataset_metadata=self.memory.get_metadata_prompt(),
            results_summary=self.memory.get_results_summary(),
        )

    def get_rlm_subtask_prompt(
        self,
        task_id: str,
        description: str,
        context_summary: str,
    ) -> str:
        return RLM_SUBTASK_PROMPT.format(
            task_id=task_id,
            description=description,
            context_summary=context_summary,
            dataset_metadata=self.memory.get_metadata_prompt(),
            results_summary=self.memory.get_results_summary(),
        )


def json_compact(d: dict) -> str:  # type: ignore[type-arg]
    import json
    return json.dumps(d, default=str)
