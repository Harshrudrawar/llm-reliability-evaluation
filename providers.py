"""Generation-provider abstraction.

The experiment runner targets one provider through a single `generate()`
interface, selected at runtime via the PROVIDER environment variable:

    PROVIDER=openai      (default)  -> OpenAI Responses API
    PROVIDER=anthropic              -> Anthropic Messages API

API keys are read from the environment only:
    OPENAI_API_KEY      (and optional OPENAI_MODEL)
    ANTHROPIC_API_KEY   (and optional ANTHROPIC_MODEL / ANTHROPIC_MAX_TOKENS)

The OpenAI path preserves the original behavior exactly. The SDKs are imported
lazily inside each provider so that running one provider does not require the
other provider's package to be installed.

Note on scoring: this module only affects *generation*. Semantic scoring in
score_results.py uses OpenAI embeddings regardless of the generation provider
(Anthropic has no first-party embeddings endpoint), so OPENAI_API_KEY is still
required for the scoring stage.
"""

import os
from typing import Dict, List, Optional, Tuple


class Provider:
    """Common interface. `generate` returns (output_text, response_id)."""

    name = "base"
    model = ""

    def generate(
        self,
        prompt: str,
        temperature: float,
        previous_response_id: Optional[str] = None,
    ) -> Tuple[str, str]:
        raise NotImplementedError


class OpenAIProvider(Provider):
    """OpenAI Responses API — unchanged from the original implementation.

    Multi-turn state (the challenge phases) is held server-side via
    `previous_response_id`.
    """

    name = "openai"

    def __init__(self, model: str, timeout: float = 60.0) -> None:
        from openai import OpenAI

        self.model = model
        # `timeout` bounds any single request so one slow call can't hang the run.
        # `max_retries=0` leaves retrying to the runner's retry loop (avoids
        # compounding SDK-level + runner-level retries). Reads OPENAI_API_KEY from env.
        self._client = OpenAI(timeout=timeout, max_retries=0)

    def generate(
        self,
        prompt: str,
        temperature: float,
        previous_response_id: Optional[str] = None,
    ) -> Tuple[str, str]:
        kwargs = {
            "model": self.model,
            "input": prompt,
            "temperature": temperature,
        }
        if previous_response_id:
            kwargs["previous_response_id"] = previous_response_id

        response = self._client.responses.create(**kwargs)
        output_text = (response.output_text or "").strip()
        response_id = getattr(response, "id", "")
        return output_text, response_id


class AnthropicProvider(Provider):
    """Anthropic Messages API.

    The Messages API is stateless, so multi-turn follow-ups are reconstructed
    client-side: each returned response_id maps to the full message history that
    produced it. A follow-up call that passes a previous_response_id continues
    that exact conversation, mirroring the semantics of OpenAI's
    `previous_response_id` without changing benchmark behavior.

    `temperature` is forwarded on every call, so the configured model must be one
    that accepts it (e.g. claude-haiku-4-5, claude-sonnet-4-6, claude-opus-4-6).
    Opus 4.8 / 4.7 and Fable 5 reject `temperature` with a 400 and cannot be used
    for this temperature-sweeping benchmark.
    """

    name = "anthropic"

    def __init__(self, model: str, max_tokens: int = 1024, timeout: float = 60.0) -> None:
        import anthropic

        self.model = model
        self._max_tokens = max_tokens
        # `timeout` bounds any single request; `max_retries=0` leaves retrying to
        # the runner's retry loop. Reads ANTHROPIC_API_KEY from the environment.
        self._client = anthropic.Anthropic(timeout=timeout, max_retries=0)
        self._history: Dict[str, List[Dict[str, str]]] = {}

    def generate(
        self,
        prompt: str,
        temperature: float,
        previous_response_id: Optional[str] = None,
    ) -> Tuple[str, str]:
        if previous_response_id and previous_response_id in self._history:
            messages = self._history[previous_response_id] + [
                {"role": "user", "content": prompt}
            ]
        else:
            messages = [{"role": "user", "content": prompt}]

        response = self._client.messages.create(
            model=self.model,
            max_tokens=self._max_tokens,
            temperature=temperature,
            messages=messages,
        )
        output_text = "".join(
            block.text
            for block in response.content
            if getattr(block, "type", "") == "text"
        ).strip()
        response_id = getattr(response, "id", "")
        if response_id:
            self._history[response_id] = messages + [
                {"role": "assistant", "content": output_text}
            ]
        return output_text, response_id


def get_provider() -> Provider:
    """Build the provider selected by the PROVIDER env var (default: openai)."""
    provider = os.getenv("PROVIDER", "openai").strip().lower()
    timeout = float(os.getenv("REQUEST_TIMEOUT", "60"))
    if provider == "openai":
        return OpenAIProvider(os.getenv("OPENAI_MODEL", "gpt-4o-mini"), timeout=timeout)
    if provider == "anthropic":
        return AnthropicProvider(
            os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5"),
            int(os.getenv("ANTHROPIC_MAX_TOKENS", "1024")),
            timeout=timeout,
        )
    raise ValueError(
        f"Unknown PROVIDER {provider!r}. Use PROVIDER=openai or PROVIDER=anthropic."
    )
