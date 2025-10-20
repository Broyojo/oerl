"""
Comprehensive tests for the OpenRouterClient class.
Tests all functionality including chat completion and structured outputs.
Uses real OpenRouter API calls.
"""

import os
import pytest
from pydantic import BaseModel, Field
from typing import Optional, List
from grader.openrouter import OpenRouterClient


class TestOpenRouterClientInitialization:
    """Test client initialization and configuration."""

    def test_init_with_api_key(self, openrouter_api_key):
        """Test initialization with explicit API key."""
        client = OpenRouterClient(api_key=openrouter_api_key)

        assert client.api_key == openrouter_api_key
        assert client.base_url == "https://openrouter.ai/api/v1"
        assert client.client is not None

    def test_init_with_env_api_key(self, openrouter_api_key):
        """Test initialization using environment variable."""
        os.environ["OPENROUTER_API_KEY"] = openrouter_api_key
        client = OpenRouterClient()

        assert client.api_key == openrouter_api_key

    def test_init_custom_base_url(self, openrouter_api_key):
        """Test initialization with custom base URL."""
        custom_url = "https://custom.api.url/v1"
        client = OpenRouterClient(
            api_key=openrouter_api_key,
            base_url=custom_url,
        )

        assert client.base_url == custom_url


class TestChatCompletion:
    """Test basic chat completion functionality."""

    def test_chat_completion_returns_string(self, openrouter_api_key, openrouter_model):
        """Test that chat completion returns a string response."""
        client = OpenRouterClient(api_key=openrouter_api_key)
        response = client.chat_completion(
            messages=[
                {"role": "user", "content": "Say hello"},
            ],
            model=openrouter_model,
            temperature=0.0,
            max_tokens=20,
        )

        assert isinstance(response, str)
        assert len(response) > 0

    def test_chat_completion_expected_phrase(
        self, openrouter_api_key, openrouter_model
    ):
        """Test that chat completion follows instructions correctly."""
        client = OpenRouterClient(api_key=openrouter_api_key)
        response = client.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "Follow the user's instructions exactly.",
                },
                {"role": "user", "content": "Reply only with the single word 'pong'."},
            ],
            model=openrouter_model,
            temperature=0.0,
            max_tokens=8,
        )

        assert isinstance(response, str)
        assert "pong" in response.lower()

    def test_chat_completion_with_system_message(
        self, openrouter_api_key, openrouter_model
    ):
        """Test chat completion with system message."""
        client = OpenRouterClient(api_key=openrouter_api_key)
        response = client.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that always responds in a friendly manner.",
                },
                {"role": "user", "content": "Hi!"},
            ],
            model=openrouter_model,
            temperature=0.3,
            max_tokens=50,
        )

        assert isinstance(response, str)
        assert len(response) > 0

    def test_chat_completion_multi_turn_conversation(
        self, openrouter_api_key, openrouter_model
    ):
        """Test chat completion with multi-turn conversation."""
        client = OpenRouterClient(api_key=openrouter_api_key)
        response = client.chat_completion(
            messages=[
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
                {"role": "user", "content": "What about 3+3?"},
            ],
            model=openrouter_model,
            temperature=0.0,
            max_tokens=10,
        )

        assert isinstance(response, str)
        assert "6" in response

    def test_chat_completion_temperature_variation(
        self, openrouter_api_key, openrouter_model
    ):
        """Test chat completion with different temperature settings."""
        client = OpenRouterClient(api_key=openrouter_api_key)

        # Low temperature (deterministic)
        response_low = client.chat_completion(
            messages=[{"role": "user", "content": "Say 'test'"}],
            model=openrouter_model,
            temperature=0.0,
            max_tokens=10,
        )

        # High temperature (more creative)
        response_high = client.chat_completion(
            messages=[{"role": "user", "content": "Say 'test'"}],
            model=openrouter_model,
            temperature=1.0,
            max_tokens=10,
        )

        assert isinstance(response_low, str)
        assert isinstance(response_high, str)

    def test_chat_completion_max_tokens_limit(
        self, openrouter_api_key, openrouter_model
    ):
        """Test that max_tokens parameter limits output length."""
        client = OpenRouterClient(api_key=openrouter_api_key)
        response = client.chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": "Write a long story about space exploration.",
                },
            ],
            model=openrouter_model,
            temperature=0.5,
            max_tokens=50,  # Very limited
        )

        assert isinstance(response, str)
        # Response should be limited by max_tokens
        assert len(response.split()) < 100


class TestChatCompletionStructured:
    """Test structured output functionality."""

    def test_structured_output_simple_model(self, openrouter_api_key, openrouter_model):
        """Test structured output with a simple Pydantic model."""
        client = OpenRouterClient(api_key=openrouter_api_key)

        class WordResponse(BaseModel):
            word: str

        result = client.chat_completion_structured(
            messages=[
                {
                    "role": "system",
                    "content": "You must comply with the response schema.",
                },
                {
                    "role": "user",
                    "content": "Provide the JSON field `word` with the exact value 'ping'.",
                },
            ],
            model=openrouter_model,
            response_structure=WordResponse,
            temperature=0.0,
            max_tokens=16,
        )

        assert isinstance(result, WordResponse)
        assert result.word.strip().lower() == "ping"

    def test_structured_output_complex_model(
        self, openrouter_api_key, openrouter_model
    ):
        """Test structured output with a complex Pydantic model."""
        client = OpenRouterClient(api_key=openrouter_api_key)

        class ComplexResponse(BaseModel):
            score: float
            reasoning: str
            category: str

        result = client.chat_completion_structured(
            messages=[
                {
                    "role": "user",
                    "content": "Rate the creativity of 'inventing a new color' from 0-100. Provide score, reasoning, and category.",
                },
            ],
            model=openrouter_model,
            response_structure=ComplexResponse,
            temperature=0.3,
            max_tokens=200,
        )

        assert isinstance(result, ComplexResponse)
        assert isinstance(result.score, float)
        assert 0.0 <= result.score <= 100.0
        assert isinstance(result.reasoning, str)
        assert len(result.reasoning) > 0
        assert isinstance(result.category, str)

    def test_structured_output_with_optional_fields(
        self, openrouter_api_key, openrouter_model
    ):
        """Test structured output with optional fields."""
        client = OpenRouterClient(api_key=openrouter_api_key)

        class ResponseWithOptional(BaseModel):
            required_field: str
            optional_field: Optional[str] = None

        result = client.chat_completion_structured(
            messages=[
                {
                    "role": "user",
                    "content": "Provide a response with required_field='test' and no optional_field.",
                },
            ],
            model=openrouter_model,
            response_structure=ResponseWithOptional,
            temperature=0.0,
        )

        assert isinstance(result, ResponseWithOptional)
        assert result.required_field is not None

    def test_structured_output_with_field_constraints(
        self, openrouter_api_key, openrouter_model
    ):
        """Test structured output with field constraints."""
        client = OpenRouterClient(api_key=openrouter_api_key)

        class ConstrainedResponse(BaseModel):
            score: float = Field(
                ge=0.0, le=100.0, description="Score between 0 and 100"
            )
            tags: List[str] = Field(description="List of tags")

        result = client.chat_completion_structured(
            messages=[
                {
                    "role": "user",
                    "content": "Provide a score of 75.5 and tags ['creative', 'innovative'].",
                },
            ],
            model=openrouter_model,
            response_structure=ConstrainedResponse,
            temperature=0.0,
        )

        assert isinstance(result, ConstrainedResponse)
        assert 0.0 <= result.score <= 100.0
        assert isinstance(result.tags, list)
        assert all(isinstance(tag, str) for tag in result.tags)

    def test_structured_output_with_nested_models(
        self, openrouter_api_key, openrouter_model
    ):
        """Test structured output with nested Pydantic models."""
        client = OpenRouterClient(api_key=openrouter_api_key)

        class InnerModel(BaseModel):
            value: int

        class OuterModel(BaseModel):
            name: str
            inner: InnerModel

        result = client.chat_completion_structured(
            messages=[
                {
                    "role": "user",
                    "content": "Provide name='test' and inner.value=42.",
                },
            ],
            model=openrouter_model,
            response_structure=OuterModel,
            temperature=0.0,
        )

        assert isinstance(result, OuterModel)
        assert isinstance(result.name, str)
        assert isinstance(result.inner, InnerModel)
        assert isinstance(result.inner.value, int)

    def test_structured_output_scoring_use_case(
        self, openrouter_api_key, openrouter_model
    ):
        """Test structured output for the actual grading use case."""
        client = OpenRouterClient(api_key=openrouter_api_key)

        class ScoreResponse(BaseModel):
            score: float
            reasoning: str

        result = client.chat_completion_structured(
            messages=[
                {
                    "role": "user",
                    "content": """Rate this trajectory on creativity (0-100):
                    User: Create art from code
                    Assistant: I'll generate fractal patterns using recursive algorithms
                    
                    Provide score and reasoning.""",
                },
            ],
            model=openrouter_model,
            response_structure=ScoreResponse,
            temperature=0.3,
            max_tokens=300,
        )

        assert isinstance(result, ScoreResponse)
        assert isinstance(result.score, float)
        assert 0.0 <= result.score <= 100.0
        assert isinstance(result.reasoning, str)
        assert len(result.reasoning) > 10

    def test_structured_output_with_max_tokens(
        self, openrouter_api_key, openrouter_model
    ):
        """Test structured output with max_tokens parameter."""
        client = OpenRouterClient(api_key=openrouter_api_key)

        class SimpleResponse(BaseModel):
            answer: str

        result = client.chat_completion_structured(
            messages=[
                {"role": "user", "content": "What is AI? Provide a brief answer."},
            ],
            model=openrouter_model,
            response_structure=SimpleResponse,
            temperature=0.5,
            max_tokens=100,
        )

        assert isinstance(result, SimpleResponse)
        assert isinstance(result.answer, str)

    def test_structured_output_without_max_tokens(
        self, openrouter_api_key, openrouter_model
    ):
        """Test structured output without specifying max_tokens."""
        client = OpenRouterClient(api_key=openrouter_api_key)

        class Response(BaseModel):
            text: str

        result = client.chat_completion_structured(
            messages=[
                {"role": "user", "content": "Say hello."},
            ],
            model=openrouter_model,
            response_structure=Response,
            temperature=0.0,
        )

        assert isinstance(result, Response)
        assert isinstance(result.text, str)


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_structured_output_returns_parsable_content(
        self, openrouter_api_key, openrouter_model
    ):
        """Test that structured output handles parsing correctly."""
        client = OpenRouterClient(api_key=openrouter_api_key)

        class TestResponse(BaseModel):
            status: str

        # Should complete successfully
        result = client.chat_completion_structured(
            messages=[
                {"role": "user", "content": "Respond with status='success'"},
            ],
            model=openrouter_model,
            response_structure=TestResponse,
            temperature=0.0,
        )

        assert isinstance(result, TestResponse)
        assert hasattr(result, "status")

    def test_chat_completion_empty_response_handling(
        self, openrouter_api_key, openrouter_model
    ):
        """Test handling of edge case responses."""
        client = OpenRouterClient(api_key=openrouter_api_key)

        # Even with max_tokens=1, should return some response
        response = client.chat_completion(
            messages=[{"role": "user", "content": "Hi"}],
            model=openrouter_model,
            temperature=0.0,
            max_tokens=5,
        )

        assert isinstance(response, str)


class TestIntegrationScenarios:
    """Test real-world integration scenarios."""

    def test_grading_workflow_simulation(self, openrouter_api_key, openrouter_model):
        """Test a complete grading workflow simulation."""
        client = OpenRouterClient(api_key=openrouter_api_key)

        class GradingResult(BaseModel):
            score: float
            reasoning: str

        # Simulate grading a trajectory
        trajectory_text = """
        ID: test-traj-001
        MESSAGES:
        1. USER: Explore creative uses of machine learning
        2. ASSISTANT: I'll investigate using neural networks to compose poetry
        """

        result = client.chat_completion_structured(
            messages=[
                {
                    "role": "user",
                    "content": f"""Rate this trajectory on creativity, novelty, and exploration depth.
                    Provide a score from 0-100 and detailed reasoning.
                    
                    Trajectory:
                    {trajectory_text}
                    """,
                },
            ],
            model=openrouter_model,
            response_structure=GradingResult,
            temperature=0.3,
            max_tokens=500,
        )

        assert isinstance(result, GradingResult)
        assert 0.0 <= result.score <= 100.0
        assert len(result.reasoning) > 20
