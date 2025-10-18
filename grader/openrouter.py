import os
from typing import Optional, List, Dict, Any, Type, Union
from pydantic import BaseModel
from openai import OpenAI


class OpenRouterClient:
    """Simple client for OpenRouter API calls."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = base_url
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> str:
        """Make a chat completion request and return the response content."""
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content

    def chat_completion_structured(
        self,
        messages: List[Dict[str, str]],
        model: str,
        response_structure: Union[Type[BaseModel], BaseModel],
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
    ) -> BaseModel:
        """Make a chat completion request with structured output and return the parsed response."""
        request_kwargs: Dict[str, Any] = {}
        if max_tokens is not None:
            request_kwargs["max_output_tokens"] = max_tokens

        text_format = (
            response_structure.__class__
            if isinstance(response_structure, BaseModel)
            else response_structure
        )

        response = self.client.responses.parse(
            model=model,
            input=messages,
            temperature=temperature,
            text_format=text_format,
            **request_kwargs,
        )

        parsed = response.output_parsed
        if parsed is None:
            raise ValueError("Structured response did not return parsable content.")

        return parsed
