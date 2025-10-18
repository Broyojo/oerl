import json
import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
from .openrouter import OpenRouterClient
from .types import Trajectory, GradingResult


class TrajectoryGrader:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "openai/gpt-oss-20b:free",
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ):
        self.model = model
        self.openrouter = OpenRouterClient(api_key=api_key)
        self.temperature = temperature
        self.max_tokens = max_tokens

    def grade_trajectory(self, trajectory: Trajectory) -> GradingResult:
        """Grade a single trajectory for novelty, creativity, and open-endedness."""
        prompt = self._build_grading_prompt(trajectory)

        class ScoreResponse(BaseModel):
            score: float

        result = self.openrouter.chat_completion_structured(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            response_structure=ScoreResponse,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return GradingResult(
            trajectory_id=getattr(trajectory, "id", "unknown") or "unknown",
            grader_model=self.model,
            score=float(result.score),
        )

    def _build_grading_prompt(self, trajectory: Trajectory) -> str:
        """Build the grading prompt for the LLM."""
        trajectory_text = self._format_trajectory(trajectory)

        # Load the single output prompt template
        prompt_path = Path(__file__).parent / "prompts" / "single_output.txt"

        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt_template = f.read()

            # Replace the trajectory placeholder
            prompt = prompt_template.replace("{trajectory}", trajectory_text)
            return prompt

        except FileNotFoundError:
            raise Exception(f"Prompt template not found at {prompt_path}")

    def _format_trajectory(self, trajectory: Trajectory) -> str:
        """Format trajectory for the grading prompt."""
        lines = []

        # Include ID if it exists
        if hasattr(trajectory, "id") and trajectory.id:
            lines.append(f"ID: {trajectory.id}")

        lines.append("\nMESSAGES:")
        for i, msg in enumerate(trajectory.messages, 1):
            lines.append(f"\n{i}. {msg.role.upper()}: {msg.content}")
            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    lines.append(
                        f"   TOOL_CALL: {tool_call.function['name']}({tool_call.function.get('arguments', '')})"
                    )

        return "\n".join(lines)
