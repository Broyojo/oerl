from pydantic import BaseModel

from grader.openrouter import OpenRouterClient


def test_chat_completion_returns_expected_phrase(openrouter_api_key, openrouter_model):
    client = OpenRouterClient(api_key=openrouter_api_key)
    response = client.chat_completion(
        messages=[
            {"role": "system", "content": "Follow the user's instructions exactly."},
            {"role": "user", "content": "Reply only with the single word 'pong'."},
        ],
        model=openrouter_model,
        temperature=0.0,
        max_tokens=8,
    )

    assert isinstance(response, str)
    assert "pong" in response.lower()


def test_chat_completion_structured_parses_response(
    openrouter_api_key, openrouter_model
):
    client = OpenRouterClient(api_key=openrouter_api_key)

    class WordResponse(BaseModel):
        word: str

    result = client.chat_completion_structured(
        messages=[
            {"role": "system", "content": "You must comply with the response schema."},
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
