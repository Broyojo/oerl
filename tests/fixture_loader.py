"""
Helper utilities for loading and working with test fixtures.
"""

import json
from pathlib import Path
from typing import Dict, Any
from grader.types import Trajectory


def load_trajectory_fixture(filename: str) -> Trajectory:
    """
    Load a trajectory from a JSON fixture file.

    Args:
        filename: Name of the fixture file (e.g., 'trajectory_creative_coding.json')

    Returns:
        Trajectory object loaded from the fixture
    """
    fixture_path = Path(__file__).parent / "fixtures" / filename
    with open(fixture_path, "r") as f:
        data = json.load(f)
    return Trajectory(**data)


def load_all_trajectory_fixtures() -> Dict[str, Trajectory]:
    """
    Load all trajectory fixtures from the fixtures directory.

    Returns:
        Dictionary mapping fixture names to Trajectory objects
    """
    fixtures_dir = Path(__file__).parent / "fixtures"
    trajectories = {}

    for fixture_file in fixtures_dir.glob("trajectory_*.json"):
        with open(fixture_file, "r") as f:
            data = json.load(f)
        trajectories[fixture_file.stem] = Trajectory(**data)

    return trajectories


def get_fixture_data(filename: str) -> Dict[str, Any]:
    """
    Load raw fixture data without parsing into Trajectory.

    Args:
        filename: Name of the fixture file

    Returns:
        Raw dictionary data from the fixture file
    """
    fixture_path = Path(__file__).parent / "fixtures" / filename
    with open(fixture_path, "r") as f:
        return json.load(f)
