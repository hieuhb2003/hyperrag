"""
LLM API client for HyP-DLM.

Used for:
  1. Query decomposition (DAG generation)
  2. Intermediate answer generation
  3. Final answer generation
"""

import json
import os
from typing import Any, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class LLMClient:
    """Unified LLM client supporting OpenAI-compatible APIs."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        from openai import OpenAI

        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
        )
        logger.step("LLM Client", f"Initialized with model='{model}'")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate a text completion."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        logger.debug(f"LLM call: model={self.model}, prompt_len={len(prompt)}")
        logger.start_timer("llm_call")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens or self.max_tokens,
        )

        result = response.choices[0].message.content or ""
        elapsed = logger.stop_timer("llm_call")

        tokens_in = response.usage.prompt_tokens if response.usage else 0
        tokens_out = response.usage.completion_tokens if response.usage else 0
        logger.debug(
            f"LLM response: {len(result)} chars, "
            f"tokens_in={tokens_in}, tokens_out={tokens_out} ({elapsed})"
        )
        return result

    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> Any:
        """Generate and parse JSON response."""
        raw = self.generate(prompt, system_prompt=system_prompt)

        # Try to extract JSON from the response
        # Handle cases where LLM wraps JSON in markdown code blocks
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last ``` lines
            start = 1
            end = len(lines)
            for i in range(len(lines) - 1, 0, -1):
                if lines[i].strip().startswith("```"):
                    end = i
                    break
            text = "\n".join(lines[start:end])

        try:
            result = json.loads(text)
            logger.debug(f"Parsed JSON successfully: {type(result)}")
            return result
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM JSON output: {e}")
            logger.debug(f"Raw output: {raw[:500]}")
            raise
