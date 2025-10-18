import os
import pytest


@pytest.fixture(scope="session")
def openrouter_api_key():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip(
            "OPENROUTER_API_KEY not set; skipping OpenRouter integration tests."
        )
    return api_key


@pytest.fixture(scope="session")
def openrouter_model():
    return os.getenv("OPENROUTER_TEST_MODEL", "openai/gpt-4o-mini")
