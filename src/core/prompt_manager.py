"""
Prompt Manager for the Agentic Data Analysis System.

Assembles structured prompts for each reasoning phase.
Keeps all prompt engineering centralized and version-controlled.
"""
from __future__ import annotations

from src.core.memory import MemorySystem


SYSTEM_PROMPT_CORE = """\
You are an expert autonomous data scientist. You operate within a multi-step \
agentic system that separates reasoning (your role) from execution (Python tools).

Your role is PLANNING ONLY — you do not write or execute Python code directly.
You produce structured JSON plans that the Agent Controller will execute.

## Available Tools
{tool_descriptions}

## Response Format
Always respond with a valid JSON object in one of these two forms:

### Form 1: Action Plan (when more analysis is needed)
```json
{{
  "status": "in_progress",
  "reasoning": "Your brief explanation of what you observed and why you chose these steps.",
  "steps": [
    {{
      "step_number": 1,
      "tool_name": "<tool_name>",
      "parameters": {{ "<param>": "<value>" }},
      "rationale": "Why this tool with these parameters."
    }}
  ]
}}
```

### Form 2: Final Answer (when analysis is complete)
```json
{{
  "status": "complete",
  "reasoning": "Your synthesis of all findings.",
  "insights": ["Insight 1", "Insight 2"],
  "recommendations": ["Recommendation 1"],
  "best_model": "<model_name or null>",
  "key_metrics": {{ "<metric>": "<value>" }}
}}
```

## Rules
- NEVER include data rows or full arrays in your response.
- ONLY reference tools by their exact `tool_name`.
- Provide `rationale` for every step — this is a research-grade system.
- If a tool failed, adapt your plan. Do not repeat the exact same call.
"""

INITIAL_ANALYSIS_PROMPT = """\
## Dataset Overview
{dataset_metadata}

## Task
Analyze the dataset above and produce a comprehensive analysis plan.
Focus on: data quality, statistical characterization, ML task identification,
model selection, and evaluation strategy.

Respond in the JSON action plan format.
"""

ITERATION_PROMPT = """\
## Dataset Overview
{dataset_metadata}

## Analysis Results So Far
{results_summary}

## Pending Steps
{pending_steps}

## Task
Review the results above and decide the next set of actions.
- If the pending steps are still valid, confirm executing them.
- If results reveal new requirements, adjust the plan.
- If analysis is complete, return the Final Answer JSON.

Respond in the specified JSON format.
"""

FINAL_INTERPRETATION_PROMPT = """\
## Dataset Overview
{dataset_metadata}

## Complete Analysis Results
{results_summary}

## Task
Synthesize all results into a final, comprehensive analysis.
Provide actionable insights and concrete recommendations.
Respond in the Final Answer JSON format.
"""


class PromptManager:
    """Assembles LLM prompts from memory state and tool descriptions."""

    def __init__(self, memory: MemorySystem, tool_descriptions: str) -> None:
        self.memory = memory
        self.tool_descriptions = tool_descriptions

    def get_system_prompt(self) -> str:
        """Build the LLM system prompt with tool descriptions injected."""
        return SYSTEM_PROMPT_CORE.format(tool_descriptions=self.tool_descriptions)

    def get_initial_user_prompt(self) -> str:
        """User prompt for the very first reasoning call."""
        return INITIAL_ANALYSIS_PROMPT.format(
            dataset_metadata=self.memory.get_metadata_prompt(),
        )

    def get_iteration_user_prompt(self) -> str:
        """User prompt for subsequent reasoning cycles."""
        pending = self.memory.get_pending_steps()
        pending_str = (
            "\n".join(
                f"Step {s.step_number}: {s.tool_name}({s.parameters})" for s in pending
            )
            if pending
            else "No pending steps — re-evaluate if more analysis is needed."
        )
        return ITERATION_PROMPT.format(
            dataset_metadata=self.memory.get_metadata_prompt(),
            results_summary=self.memory.get_results_summary(),
            pending_steps=pending_str,
        )

    def get_final_interpretation_prompt(self) -> str:
        """User prompt for the final synthesis call."""
        return FINAL_INTERPRETATION_PROMPT.format(
            dataset_metadata=self.memory.get_metadata_prompt(),
            results_summary=self.memory.get_results_summary(),
        )
