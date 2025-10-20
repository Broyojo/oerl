import json
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from .openrouter import OpenRouterClient
from .types import Trajectory, GradingResult
from .db import VectorDatabase


class TrajectoryGrader:
    def __init__(
        self,
        vector_store: Optional[VectorDatabase],
        api_key: Optional[str] = None,
        model: str = "openai/gpt-oss-20b:free",
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ):
        self.vector_store = vector_store
        self.model = model
        self.openrouter = OpenRouterClient(api_key=api_key)
        self.temperature = temperature
        self.max_tokens = max_tokens

    def grade_trajectory(
        self, trajectory: Trajectory, n_results: int = 3
    ) -> GradingResult:
        """
        Performs the full read-augment-grade-write cycle for a single trajectory.

        1.  READ: Searches the vector store for similar, previously graded trajectories.
        2.  AUGMENT: Injects these examples into a new, more powerful prompt.
        3.  GRADE: Asks the LLM to score the new trajectory based on this context.
        4.  WRITE: Stores the new trajectory's text, embedding, and score back
            into the vector store for future comparisons.
        """

        trajectory_text = self._format_trajectory(trajectory)

        similar_trajectories = self.vector_store.search(
            trajectory=trajectory_text,
            n_results=n_results,
        )

        prompt = self._build_grading_prompt(trajectory_text, similar_trajectories)

        class ScoreResponse(BaseModel):
            score: float
            reasoning: str

        result = self.openrouter.chat_completion_structured(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            response_structure=ScoreResponse,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        self.vector_store.add(
            trajectory_id=getattr(trajectory, "id", "unknown") or "unknown",
            trajectory=trajectory_text,
            metadata={},  # not using imo cause score could bias future searches
        )

        return GradingResult(
            trajectory_id=getattr(trajectory, "id", "unknown") or "unknown",
            grader_model=self.model,
            score=float(result.score),
            reasoning=result.reasoning,
        )

    def _build_grading_prompt(
        self, trajectory_text: str, similar_trajectories: List[Dict[str, Any]]
    ) -> str:
        """Build the grading prompt for the LLM."""
        similar_trajectory_text = self._format_similar_results(similar_trajectories)

        # Load the single output prompt template
        prompt_path = Path(__file__).parent / "prompts" / "single_output.txt"

        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt_template = f.read()

            # Replace the trajectory placeholder
            prompt = prompt_template.replace("{trajectory}", trajectory_text)
            prompt = prompt.replace("{similar_trajectories}", similar_trajectory_text)

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

    def _format_similar_results(self, results: List[Dict[str, Any]]) -> str:
        """Format similar trajectories for inclusion in the prompt."""

        if not results.get("ids") or not results["ids"][0]:
            return "No similar results found."

        lines = []

        for i in range(len(results["ids"][0])):
            traj_id = results["ids"][0][i]
            distance = results["distances"][0][i]
            metadata = results["metadatas"][0][i]
            traj_text = metadata.get("trajectory_text", "No result text available.")

            lines.append(f"\n--- EXAMPLE {i+1} (ID: {traj_id}) ---")
            lines.append(f"Similarity (Distance): {distance:.4f}")
            lines.append(f"--- Trajectory Content ---")
            lines.append(traj_text)
            lines.append(f"--- END EXAMPLE {i+1} ---")

        return "\n".join(lines)
