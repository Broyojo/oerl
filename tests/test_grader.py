from grader.grader import TrajectoryGrader
from grader.types import Message, Trajectory, GradingResult


def test_grade_trajectory_returns_scored_result(openrouter_api_key, openrouter_model):
    grader = TrajectoryGrader(
        api_key=openrouter_api_key,
        model=openrouter_model,
        temperature=0.0,
        max_tokens=256,
    )

    trajectory = Trajectory(
        id="trajectory-test",
        messages=[
            Message(
                role="user",
                content=(
                    "Consider a series of creative experiments an autonomous agent could run to "
                    "discover interesting alien lifeforms on a distant planet."
                ),
            ),
            Message(
                role="assistant",
                content=(
                    "The agent designs exploratory rover swarms, bio-sensor kites, and dream-sharing "
                    "protocols with the local species, documenting each surprising discovery."
                ),
            ),
        ],
    )

    result = grader.grade_trajectory(trajectory)

    assert isinstance(result, GradingResult)
    assert result.trajectory_id == "trajectory-test"
    assert result.grader_model == openrouter_model
    assert isinstance(result.score, float)
    assert 0.0 <= result.score <= 100.0
