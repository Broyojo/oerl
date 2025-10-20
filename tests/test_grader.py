"""
Comprehensive tests for the TrajectoryGrader class.
Tests all functionality including trajectory formatting, grading, and the full RAG cycle.
Uses real OpenRouter API calls and example trajectories from fixtures.
"""

import json
import os
import tempfile
import shutil
import pytest
from pathlib import Path
from grader.grader import TrajectoryGrader
from grader.types import Message, Trajectory, ToolCall, GradingResult
from grader.db import VectorDatabase


@pytest.fixture
def temp_db_path():
    """Create a temporary directory for the test database."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def vector_db(temp_db_path):
    """Create a clean VectorDatabase instance for each test."""
    return VectorDatabase(
        db_path=temp_db_path,
        collection_name="test_grading",
    )


@pytest.fixture
def grader(openrouter_api_key, openrouter_model, vector_db):
    """Create a TrajectoryGrader instance."""
    return TrajectoryGrader(
        vector_store=vector_db,
        api_key=openrouter_api_key,
        model=openrouter_model,
        temperature=0.3,
        max_tokens=2000,
    )


@pytest.fixture
def load_fixture_trajectory():
    """Factory fixture to load trajectory from JSON fixture files."""

    def _load(filename: str) -> Trajectory:
        fixture_path = Path(__file__).parent / "fixtures" / filename
        with open(fixture_path, "r") as f:
            data = json.load(f)
        return Trajectory(**data)

    return _load


class TestTrajectoryGraderInitialization:
    """Test grader initialization and configuration."""

    def test_init_with_all_parameters(self, openrouter_api_key, vector_db):
        """Test initialization with all parameters specified."""
        grader = TrajectoryGrader(
            vector_store=vector_db,
            api_key=openrouter_api_key,
            model="openai/gpt-4o-mini",
            temperature=0.5,
            max_tokens=1500,
        )

        assert grader.vector_store == vector_db
        assert grader.model == "openai/gpt-4o-mini"
        assert grader.temperature == 0.5
        assert grader.max_tokens == 1500
        assert grader.openrouter is not None

    def test_init_with_defaults(self, vector_db):
        """Test initialization with default parameters."""
        grader = TrajectoryGrader(vector_store=vector_db)

        assert grader.model == "openai/gpt-oss-20b:free"
        assert grader.temperature == 0.7
        assert grader.max_tokens == 2000

    def test_init_without_api_key_uses_env(self, vector_db, openrouter_api_key):
        """Test that initialization without api_key uses environment variable."""
        # Ensure the env var is set
        os.environ["OPENROUTER_API_KEY"] = openrouter_api_key

        grader = TrajectoryGrader(vector_store=vector_db)

        assert grader.openrouter.api_key == openrouter_api_key


class TestTrajectoryFormatting:
    """Test trajectory formatting for prompts."""

    def test_format_simple_trajectory(self, grader):
        """Test formatting a simple trajectory with basic messages."""
        trajectory = Trajectory(
            id="test-001",
            messages=[
                Message(role="user", content="Hello, how are you?"),
                Message(role="assistant", content="I'm doing well, thank you!"),
            ],
        )

        formatted = grader._format_trajectory(trajectory)

        assert "ID: test-001" in formatted
        assert "MESSAGES:" in formatted
        assert "USER: Hello, how are you?" in formatted
        assert "ASSISTANT: I'm doing well, thank you!" in formatted

    def test_format_trajectory_with_tool_calls(self, grader):
        """Test formatting trajectory with tool calls."""
        trajectory = Trajectory(
            id="test-002",
            messages=[
                Message(role="user", content="Calculate 5 + 3"),
                Message(
                    role="assistant",
                    content="I'll calculate that for you.",
                    tool_calls=[
                        ToolCall(
                            id="call_123",
                            type="function",
                            function={
                                "name": "add",
                                "arguments": '{"a": 5, "b": 3}',
                            },
                        )
                    ],
                ),
                Message(role="assistant", content="The result is 8."),
            ],
        )

        formatted = grader._format_trajectory(trajectory)

        assert "TOOL_CALL: add" in formatted
        assert '{"a": 5, "b": 3}' in formatted

    def test_format_trajectory_multiple_tool_calls(self, grader):
        """Test formatting trajectory with multiple tool calls in one message."""
        trajectory = Trajectory(
            id="test-003",
            messages=[
                Message(
                    role="assistant",
                    content="Processing multiple operations",
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            type="function",
                            function={"name": "func1", "arguments": "{}"},
                        ),
                        ToolCall(
                            id="call_2",
                            type="function",
                            function={"name": "func2", "arguments": '{"x": 10}'},
                        ),
                    ],
                ),
            ],
        )

        formatted = grader._format_trajectory(trajectory)

        assert "TOOL_CALL: func1" in formatted
        assert "TOOL_CALL: func2" in formatted
        assert '"x": 10' in formatted

    def test_format_trajectory_without_id(self, grader):
        """Test formatting trajectory without an ID."""
        trajectory = Trajectory(
            id="",
            messages=[
                Message(role="user", content="Test message"),
            ],
        )

        formatted = grader._format_trajectory(trajectory)

        # Should not include ID line if empty
        assert "MESSAGES:" in formatted
        assert "USER: Test message" in formatted

    def test_format_trajectory_preserves_message_order(self, grader):
        """Test that message order is preserved in formatting."""
        trajectory = Trajectory(
            id="test-order",
            messages=[
                Message(role="user", content="First"),
                Message(role="assistant", content="Second"),
                Message(role="user", content="Third"),
                Message(role="assistant", content="Fourth"),
            ],
        )

        formatted = grader._format_trajectory(trajectory)

        # Check that messages appear in order
        idx_first = formatted.index("First")
        idx_second = formatted.index("Second")
        idx_third = formatted.index("Third")
        idx_fourth = formatted.index("Fourth")

        assert idx_first < idx_second < idx_third < idx_fourth

    def test_format_complex_fixture_trajectory(self, grader, load_fixture_trajectory):
        """Test formatting with a complex fixture trajectory."""
        trajectory = load_fixture_trajectory("trajectory_creative_coding.json")
        formatted = grader._format_trajectory(trajectory)

        assert "traj-001-creative-coding" in formatted
        assert "visual art with functional programming" in formatted
        assert "TOOL_CALL: create_visual_lambda" in formatted
        assert "color_space" in formatted


class TestSimilarResultsFormatting:
    """Test formatting of similar trajectory results."""

    def test_format_empty_results(self, grader):
        """Test formatting when no similar results are found."""
        empty_results = {"ids": [[]], "distances": [[]], "metadatas": [[]]}
        formatted = grader._format_similar_results(empty_results)

        assert "No similar results found" in formatted

    def test_format_single_result(self, grader):
        """Test formatting a single similar result."""
        results = {
            "ids": [["traj-001"]],
            "distances": [[0.25]],
            "metadatas": [[{"trajectory_text": "Sample trajectory content"}]],
        }

        formatted = grader._format_similar_results(results)

        assert "EXAMPLE 1" in formatted
        assert "traj-001" in formatted
        assert "0.2500" in formatted
        assert "Sample trajectory content" in formatted

    def test_format_multiple_results(self, grader):
        """Test formatting multiple similar results."""
        results = {
            "ids": [["traj-001", "traj-002", "traj-003"]],
            "distances": [[0.1, 0.3, 0.5]],
            "metadatas": [
                [
                    {"trajectory_text": "First trajectory"},
                    {"trajectory_text": "Second trajectory"},
                    {"trajectory_text": "Third trajectory"},
                ]
            ],
        }

        formatted = grader._format_similar_results(results)

        assert "EXAMPLE 1" in formatted
        assert "EXAMPLE 2" in formatted
        assert "EXAMPLE 3" in formatted
        assert "traj-001" in formatted
        assert "traj-002" in formatted
        assert "traj-003" in formatted
        assert "First trajectory" in formatted
        assert "Second trajectory" in formatted
        assert "Third trajectory" in formatted

    def test_format_includes_distance_scores(self, grader):
        """Test that distance scores are formatted correctly."""
        results = {
            "ids": [["traj-001"]],
            "distances": [[0.123456]],
            "metadatas": [[{"trajectory_text": "Content"}]],
        }

        formatted = grader._format_similar_results(results)

        # Should format to 4 decimal places
        assert "0.1235" in formatted


class TestPromptBuilding:
    """Test grading prompt construction."""

    def test_build_prompt_includes_trajectory(self, grader):
        """Test that the built prompt includes the trajectory content."""
        trajectory_text = "USER: Test query\nASSISTANT: Test response"
        similar = {
            "ids": [[]],
            "distances": [[]],
            "metadatas": [[]],
        }

        prompt = grader._build_grading_prompt(trajectory_text, similar)

        assert trajectory_text in prompt

    def test_build_prompt_includes_similar_trajectories(self, grader):
        """Test that the prompt includes similar trajectory context."""
        trajectory_text = "Current trajectory"
        similar = {
            "ids": [["similar-001"]],
            "distances": [[0.2]],
            "metadatas": [[{"trajectory_text": "Similar trajectory content"}]],
        }

        prompt = grader._build_grading_prompt(trajectory_text, similar)

        assert "Similar trajectory content" in prompt
        assert "similar-001" in prompt

    def test_build_prompt_includes_scoring_instructions(self, grader):
        """Test that the prompt includes scoring criteria."""
        trajectory_text = "Test"
        similar = {"ids": [[]], "distances": [[]], "metadatas": [[]]}

        prompt = grader._build_grading_prompt(trajectory_text, similar)

        # Check for key scoring metrics
        assert "NOVELTY" in prompt
        assert "INTERESTINGNESS" in prompt
        assert "UNIQUENESS" in prompt
        assert "CREATIVITY" in prompt
        assert "EXPLORATION_DEPTH" in prompt
        assert "COHERENCE" in prompt

    def test_build_prompt_template_exists(self, grader):
        """Test that the prompt template file exists and is readable."""
        prompt_path = (
            Path(__file__).parent.parent / "grader" / "prompts" / "single_output.txt"
        )

        assert prompt_path.exists()
        assert prompt_path.is_file()

        with open(prompt_path, "r") as f:
            content = f.read()
            assert len(content) > 0
            assert "{trajectory}" in content
            assert "{similar_trajectories}" in content


class TestGradeTrajectory:
    """Test the complete grade_trajectory functionality with real API calls."""

    def test_grade_simple_trajectory(self, grader, openrouter_model):
        """Test grading a simple trajectory."""
        trajectory = Trajectory(
            id="grade-test-001",
            messages=[
                Message(role="user", content="What is 2 + 2?"),
                Message(role="assistant", content="The answer is 4."),
            ],
        )

        result = grader.grade_trajectory(trajectory, n_results=2)

        assert isinstance(result, GradingResult)
        assert result.trajectory_id == "grade-test-001"
        assert result.grader_model == openrouter_model
        assert isinstance(result.score, float)
        assert 0.0 <= result.score <= 100.0
        assert isinstance(result.reasoning, str)
        assert len(result.reasoning) > 0

    def test_grade_creative_trajectory(self, grader, load_fixture_trajectory):
        """Test grading a creative trajectory from fixtures."""
        trajectory = load_fixture_trajectory("trajectory_creative_coding.json")

        result = grader.grade_trajectory(trajectory, n_results=3)

        assert isinstance(result, GradingResult)
        assert result.trajectory_id == "traj-001-creative-coding"
        assert isinstance(result.score, float)
        assert 0.0 <= result.score <= 100.0
        # Creative trajectory should score reasonably high
        assert result.score > 40.0

    def test_grade_mundane_trajectory(self, grader, load_fixture_trajectory):
        """Test grading a mundane trajectory from fixtures."""
        trajectory = load_fixture_trajectory("trajectory_mundane_task.json")

        result = grader.grade_trajectory(trajectory, n_results=3)

        assert isinstance(result, GradingResult)
        assert result.trajectory_id == "traj-002-mundane-task"
        assert isinstance(result.score, float)
        assert 0.0 <= result.score <= 100.0
        # Mundane trajectory should score lower
        assert result.score < 70.0

    def test_grade_deep_exploration_trajectory(self, grader, load_fixture_trajectory):
        """Test grading a deep exploration trajectory."""
        trajectory = load_fixture_trajectory("trajectory_deep_exploration.json")

        result = grader.grade_trajectory(trajectory, n_results=3)

        assert isinstance(result, GradingResult)
        assert result.trajectory_id == "traj-003-deep-exploration"
        assert isinstance(result.score, float)
        assert 0.0 <= result.score <= 100.0
        # Deep exploration should score high
        assert result.score > 50.0

    def test_grade_trajectory_with_tool_calls(self, grader, load_fixture_trajectory):
        """Test grading trajectory with tool calls."""
        trajectory = load_fixture_trajectory("trajectory_artistic_algorithm.json")

        result = grader.grade_trajectory(trajectory, n_results=2)

        assert isinstance(result, GradingResult)
        assert result.trajectory_id == "traj-004-artistic-algorithm"
        assert isinstance(result.score, float)
        assert 0.0 <= result.score <= 100.0

    def test_grade_trajectory_stores_in_database(self, grader, vector_db):
        """Test that grading stores the trajectory in the database."""
        trajectory = Trajectory(
            id="store-test-001",
            messages=[
                Message(role="user", content="Test query"),
                Message(role="assistant", content="Test response"),
            ],
        )

        # Grade the trajectory
        result = grader.grade_trajectory(trajectory, n_results=1)

        # Verify it was stored
        search_results = vector_db.search(
            trajectory="Test query",
            n_results=1,
        )

        assert len(search_results["ids"][0]) > 0
        assert "store-test-001" in search_results["ids"][0]

    def test_grade_trajectory_uses_similar_context(
        self, grader, vector_db, load_fixture_trajectory
    ):
        """Test that grading uses similar trajectories for context."""
        # First, add a high-scoring creative trajectory
        creative_traj = load_fixture_trajectory("trajectory_creative_coding.json")
        grader.grade_trajectory(creative_traj, n_results=1)

        # Now grade a similar creative trajectory
        similar_traj = Trajectory(
            id="similar-creative",
            messages=[
                Message(
                    role="user",
                    content="Explore novel ways to visualize algorithms using art.",
                ),
                Message(
                    role="assistant",
                    content="I'll create a system where sorting algorithms generate unique paintings.",
                ),
            ],
        )

        result = grader.grade_trajectory(similar_traj, n_results=3)

        # The similar context should influence the score
        assert isinstance(result.score, float)
        assert result.score > 0.0

    def test_grade_multiple_trajectories_sequentially(
        self, grader, load_fixture_trajectory
    ):
        """Test grading multiple trajectories in sequence."""
        trajectories = [
            load_fixture_trajectory("trajectory_creative_coding.json"),
            load_fixture_trajectory("trajectory_mundane_task.json"),
            load_fixture_trajectory("trajectory_artistic_algorithm.json"),
        ]

        results = []
        for traj in trajectories:
            result = grader.grade_trajectory(traj, n_results=2)
            results.append(result)

        # All should be valid results
        assert len(results) == 3
        for result in results:
            assert isinstance(result, GradingResult)
            assert 0.0 <= result.score <= 100.0

    def test_grade_trajectory_reasoning_is_meaningful(self, grader):
        """Test that the reasoning provided is meaningful and not empty."""
        trajectory = Trajectory(
            id="reasoning-test",
            messages=[
                Message(
                    role="user",
                    content="Invent a new mathematical concept that bridges topology and music theory.",
                ),
                Message(
                    role="assistant",
                    content="I propose 'harmonic manifolds' where musical scales map onto topological surfaces, with chord progressions as continuous deformations.",
                ),
            ],
        )

        result = grader.grade_trajectory(trajectory, n_results=1)

        # Reasoning should be substantial
        assert len(result.reasoning) > 20
        # Should contain some scoring-related terms (may vary by model)
        reasoning_lower = result.reasoning.lower()
        scoring_terms = [
            "novel",
            "creative",
            "interest",
            "unique",
            "score",
            "exploration",
        ]
        assert any(term in reasoning_lower for term in scoring_terms)


class TestGraderWithDifferentParameters:
    """Test grader behavior with different configuration parameters."""

    def test_grade_with_low_temperature(
        self, openrouter_api_key, openrouter_model, vector_db
    ):
        """Test grading with low temperature (more deterministic)."""
        grader = TrajectoryGrader(
            vector_store=vector_db,
            api_key=openrouter_api_key,
            model=openrouter_model,
            temperature=0.0,
            max_tokens=1000,
        )

        trajectory = Trajectory(
            id="low-temp-test",
            messages=[
                Message(role="user", content="Simple test"),
                Message(role="assistant", content="Simple response"),
            ],
        )

        result = grader.grade_trajectory(trajectory, n_results=1)
        assert isinstance(result.score, float)

    def test_grade_with_high_temperature(
        self, openrouter_api_key, openrouter_model, vector_db
    ):
        """Test grading with high temperature (more creative)."""
        grader = TrajectoryGrader(
            vector_store=vector_db,
            api_key=openrouter_api_key,
            model=openrouter_model,
            temperature=0.9,
            max_tokens=1000,
        )

        trajectory = Trajectory(
            id="high-temp-test",
            messages=[
                Message(role="user", content="Test query"),
                Message(role="assistant", content="Test response"),
            ],
        )

        result = grader.grade_trajectory(trajectory, n_results=1)
        assert isinstance(result.score, float)

    def test_grade_with_different_n_results(self, grader):
        """Test grading with different numbers of similar results."""
        trajectory = Trajectory(
            id="n-results-test",
            messages=[
                Message(role="user", content="Query"),
                Message(role="assistant", content="Response"),
            ],
        )

        # Test with 1, 3, and 5 results
        for n in [1, 3, 5]:
            result = grader.grade_trajectory(trajectory, n_results=n)
            assert isinstance(result.score, float)
            assert 0.0 <= result.score <= 100.0


class TestGraderEdgeCases:
    """Test edge cases and error handling."""

    def test_grade_trajectory_with_empty_messages_list(self, grader):
        """Test grading a trajectory with no messages."""
        trajectory = Trajectory(
            id="empty-messages",
            messages=[],
        )

        # Should handle gracefully (though may score low)
        result = grader.grade_trajectory(trajectory, n_results=1)
        assert isinstance(result, GradingResult)

    def test_grade_trajectory_with_very_long_content(self, grader):
        """Test grading trajectory with very long message content."""
        long_content = "A" * 5000
        trajectory = Trajectory(
            id="long-content-test",
            messages=[
                Message(role="user", content=long_content),
                Message(role="assistant", content="Response to long content"),
            ],
        )

        result = grader.grade_trajectory(trajectory, n_results=1)
        assert isinstance(result, GradingResult)
        assert isinstance(result.score, float)

    def test_grade_trajectory_with_special_characters(self, grader):
        """Test grading trajectory with special characters."""
        trajectory = Trajectory(
            id="special-chars-test",
            messages=[
                Message(role="user", content="Test with @#$%^&*() and emojis 🚀🌟"),
                Message(
                    role="assistant", content="Response with unicode: 你好 Здравствуйте"
                ),
            ],
        )

        result = grader.grade_trajectory(trajectory, n_results=1)
        assert isinstance(result, GradingResult)
        assert isinstance(result.score, float)

    def test_grade_trajectory_without_id(self, grader):
        """Test grading trajectory with missing or empty ID."""
        trajectory = Trajectory(
            id="",
            messages=[
                Message(role="user", content="Test"),
                Message(role="assistant", content="Response"),
            ],
        )

        result = grader.grade_trajectory(trajectory, n_results=1)
        assert result.trajectory_id == "unknown"


class TestFullRAGCycle:
    """Test the complete Read-Augment-Grade-Write cycle."""

    def test_complete_rag_cycle(self, grader, vector_db, load_fixture_trajectory):
        """Test the complete RAG cycle with multiple trajectories."""
        # Load all fixture trajectories
        fixture_files = [
            "trajectory_creative_coding.json",
            "trajectory_mundane_task.json",
            "trajectory_deep_exploration.json",
            "trajectory_artistic_algorithm.json",
            "trajectory_standard_query.json",
        ]

        results = []
        for filename in fixture_files:
            trajectory = load_fixture_trajectory(filename)
            result = grader.grade_trajectory(trajectory, n_results=3)
            results.append(result)

        # All should be graded successfully
        assert len(results) == 5

        # Scores should vary based on creativity/exploration
        scores = [r.score for r in results]
        assert min(scores) < max(scores)  # Should have variation

        # Database should contain all trajectories
        for i, filename in enumerate(fixture_files):
            trajectory = load_fixture_trajectory(filename)
            search_results = vector_db.search(
                trajectory=trajectory.messages[0].content,
                n_results=1,
            )
            assert len(search_results["ids"][0]) > 0

    def test_rag_cycle_improves_with_context(self, grader, vector_db):
        """Test that having similar examples in the database affects grading."""
        # Grade first trajectory (no context)
        traj1 = Trajectory(
            id="rag-1",
            messages=[
                Message(role="user", content="Standard question"),
                Message(role="assistant", content="Standard answer"),
            ],
        )
        result1 = grader.grade_trajectory(traj1, n_results=3)

        # Add several creative trajectories to database
        for i in range(5):
            creative_traj = Trajectory(
                id=f"creative-{i}",
                messages=[
                    Message(
                        role="user",
                        content=f"Creative exploration {i} with novel ideas",
                    ),
                    Message(
                        role="assistant",
                        content=f"Innovative response {i} with unique approach",
                    ),
                ],
            )
            grader.grade_trajectory(creative_traj, n_results=3)

        # Grade another standard trajectory (with creative context)
        traj2 = Trajectory(
            id="rag-2",
            messages=[
                Message(role="user", content="Another standard question"),
                Message(role="assistant", content="Another standard answer"),
            ],
        )
        result2 = grader.grade_trajectory(traj2, n_results=3)

        # Both should be valid scores
        assert isinstance(result1.score, float)
        assert isinstance(result2.score, float)
        # The database now has context for comparison
        assert len(vector_db.search(trajectory="test", n_results=10)["ids"][0]) >= 6
