# Test Fixtures

This directory contains example trajectories used for testing the grading system.

## Fixture Files

### trajectory_creative_coding.json

- **ID**: traj-001-creative-coding
- **Description**: A creative exploration of visual programming paradigms combining art with functional programming
- **Expected Score Range**: High (70-95)
- **Key Features**: Novel concepts, tool calls, creative exploration

### trajectory_mundane_task.json

- **ID**: traj-002-mundane-task
- **Description**: A simple arithmetic task with basic add operation
- **Expected Score Range**: Low (10-40)
- **Key Features**: Straightforward, no creativity, minimal exploration

### trajectory_deep_exploration.json

- **ID**: traj-003-deep-exploration
- **Description**: Deep philosophical investigation of quantum entanglement and distributed consciousness
- **Expected Score Range**: High (75-95)
- **Key Features**: Multiple turns, deep reasoning, theoretical exploration, high coherence

### trajectory_artistic_algorithm.json

- **ID**: traj-004-artistic-algorithm
- **Description**: Creating music from mathematical chaos theory using algorithms
- **Expected Score Range**: High (70-90)
- **Key Features**: Creative synthesis of domains, multiple tool calls, artistic innovation

### trajectory_standard_query.json

- **ID**: traj-005-standard-query
- **Description**: Basic weather query with standard response
- **Expected Score Range**: Very Low (5-30)
- **Key Features**: No creativity, standard interaction, minimal content

## Usage

Load fixtures in tests using the helper functions:

```python
from tests.fixture_loader import load_trajectory_fixture, load_all_trajectory_fixtures

# Load a single fixture
trajectory = load_trajectory_fixture("trajectory_creative_coding.json")

# Load all fixtures
all_trajectories = load_all_trajectory_fixtures()
```

Or use the pytest fixture:

```python
def test_something(load_fixture_trajectory):
    trajectory = load_fixture_trajectory("trajectory_creative_coding.json")
    # ... test code
```

## Trajectory Structure

Each trajectory JSON file follows this schema:

```json
{
  "id": "unique-trajectory-id",
  "messages": [
    {
      "role": "user|assistant",
      "content": "message content",
      "tool_calls": [
        // optional
        {
          "id": "call_id",
          "type": "function",
          "function": {
            "name": "function_name",
            "arguments": "{\"arg\": \"value\"}"
          }
        }
      ]
    }
  ]
}
```

## Scoring Criteria Reference

Trajectories are evaluated on:

- **Novelty (25%)**: Originality and unprecedented approaches
- **Interestingness (25%)**: Engaging and surprising content
- **Uniqueness (20%)**: Distinctiveness from typical patterns
- **Creativity (15%)**: Imaginative and inventive ideas
- **Exploration Depth (10%)**: Thoroughness of investigation
- **Coherence (5%)**: Logical flow and connections
