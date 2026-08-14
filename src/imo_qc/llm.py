"""OpenAI-compatible client with the two retry layers this tool depends on.

The layers are not interchangeable:

* **transport** -- 429, 5xx, connection errors, truncated streams. The request
  never produced an answer.
* **semantic** -- an answer arrived and is unusable: empty content, no JSON
  object, no ``<points>`` block. In practice this is the layer that keeps error
  rates low; judges return HTTP 200 with empty content often enough to matter.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Optional

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

from .config import ModelConfig, RetryConfig
from .models import MAX_POINTS, TokenUsage

#: Mirrors the grader prompt's required output. The scale is baked into the
#: prompt text, so it is baked in here too.
_POINTS_RE = re.compile(
    r"<points>\s*(10|[0-9])\s*out\s*of\s*" + str(MAX_POINTS) + r"\s*</points>", re.IGNORECASE
)
_FENCE_RE = re.compile(r"```[a-zA-Z]*\s*(.*?)\s*```", re.DOTALL)


class TransportError(Exception):
    """Retryable at the transport layer."""


class SemanticError(Exception):
    """The call succeeded but the answer is unusable."""


class Completion:
    __slots__ = ("text", "usage")

    def __init__(self, text: str, usage: TokenUsage):
        self.text = text
        self.usage = usage


def _strip_fence(text: str) -> Optional[str]:
    m = _FENCE_RE.search(text)
    return m.group(1) if m else None


def _first_json_object(text: str) -> Optional[dict[str, Any]]:
    """Pull the first balanced ``{...}`` out of prose.

    Judges wrap their JSON in explanations or code fences often enough that
    plain ``json.loads`` on the whole reply is not usable. Brace counting is
    string-aware so a ``}`` inside a reason does not end the object early.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
                return obj if isinstance(obj, dict) else None
    return None


def parse_points(text: str) -> int:
    """Read ``<points>N out of 10</points>``.

    Case-insensitive, first match wins: the prompt asks for exactly one block,
    and taking the first is the defined behaviour when a model writes several.
    """
    m = _POINTS_RE.search(text)
    if m is None:
        fenced = _strip_fence(text)
        if fenced:
            m = _POINTS_RE.search(fenced)
    if m is None:
        raise SemanticError("no <points> block in grader output")
    points = int(m.group(1))
    if not 0 <= points <= MAX_POINTS:
        raise SemanticError(f"score {points} out of range 0..{MAX_POINTS}")
    return points


def parse_verdict_json(text: str) -> tuple[str, str]:
    obj = _first_json_object(text)
    if obj is None:
        raise SemanticError("no JSON object in reviewer output")
    verdict = str(obj.get("verdict", "")).strip().lower()
    if verdict not in ("pass", "fail"):
        raise SemanticError(f"verdict must be pass or fail, got {verdict!r}")
    return verdict, str(obj.get("reason", "")).strip()


class LLMClient:
    """One configured endpoint.

    ``max_retries=0`` on the SDK is deliberate -- the SDK retries twice by
    default, which would silently multiply the transport budget.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        *,
        http_timeout_sec: float,
        retry: RetryConfig,
        gate: Optional[asyncio.Semaphore] = None,
    ):
        self.cfg = cfg
        self.retry = retry
        self._gate = gate
        self._client = AsyncOpenAI(
            base_url=cfg.base_url,
            api_key=cfg.api_key or "not-needed",
            max_retries=0,
            timeout=httpx.Timeout(
                connect=30.0, read=http_timeout_sec, write=60.0, pool=60.0
            ),
        )

    async def aclose(self) -> None:
        await self._client.close()

    # ------------------------------------------------------------------ requests

    def _request_kwargs(self, prompt: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.cfg.max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = self.cfg.max_completion_tokens
        elif self.cfg.max_tokens is not None:
            kwargs["max_tokens"] = self.cfg.max_tokens
        for field in ("temperature", "top_p", "seed"):
            value = getattr(self.cfg, field)
            if value is not None:
                kwargs[field] = value
        if self.cfg.extra_body:
            kwargs["extra_body"] = dict(self.cfg.extra_body)
        if self.cfg.stream:
            kwargs["stream"] = True
            kwargs["stream_options"] = {"include_usage": True}
        return kwargs

    async def _call_once(self, prompt: str) -> Completion:
        kwargs = self._request_kwargs(prompt)
        try:
            if self.cfg.stream:
                text, usage = await self._stream(kwargs)
            else:
                resp = await self._client.chat.completions.create(**kwargs)
                choice = resp.choices[0] if resp.choices else None
                text = (choice.message.content if choice and choice.message else "") or ""
                usage = resp.usage
        except APIStatusError as e:
            if e.status_code == 429 or e.status_code >= 500:
                raise TransportError(f"http {e.status_code}") from e
            raise
        except (APIConnectionError, APITimeoutError) as e:
            raise TransportError(str(e) or type(e).__name__) from e
        return Completion(text, _token_usage(usage))

    async def _stream(self, kwargs: dict[str, Any]) -> tuple[str, Any]:
        parts: list[str] = []
        usage = None
        stream = await self._client.chat.completions.create(**kwargs)
        async for chunk in stream:
            if getattr(chunk, "usage", None):
                usage = chunk.usage
            for choice in chunk.choices or []:
                delta = getattr(choice, "delta", None)
                if delta is not None and delta.content:
                    parts.append(delta.content)
        if usage is None:
            # No final usage frame means the stream ended early. Returning the
            # partial text would send a truncated solution to the grader, which
            # scores it low -- indistinguishable from a problem that genuinely
            # resisted the solver.
            raise TransportError("stream ended without a usage frame (truncated)")
        return "".join(parts), usage

    async def complete(self, prompt: str) -> Completion:
        """One completion, with transport-layer retries."""
        policy = self.retry.transport
        last: Exception | None = None
        for attempt in range(1, policy.max_attempts + 1):
            try:
                if self._gate is not None:
                    # Acquired per call, never held across solver->grader, so
                    # the two layers of concurrency cannot deadlock each other.
                    async with self._gate:
                        return await asyncio.wait_for(
                            self._call_once(prompt), timeout=self.cfg.timeout_sec
                        )
                return await asyncio.wait_for(
                    self._call_once(prompt), timeout=self.cfg.timeout_sec
                )
            except (TransportError, asyncio.TimeoutError) as e:
                last = e
                if attempt == policy.max_attempts:
                    break
                await asyncio.sleep(policy.base_backoff_ms / 1000 * (2 ** (attempt - 1)))
        raise TransportError(f"transport retries exhausted: {last}")

    # ------------------------------------------------------- semantic-layer wrappers

    async def _with_semantic_retry(self, prompt: str, parse):
        policy = self.retry.semantic
        last: Exception | None = None
        usage = TokenUsage()
        for attempt in range(1, policy.max_attempts + 1):
            completion = await self.complete(prompt)
            usage.add(completion.usage)
            try:
                if not completion.text.strip():
                    raise SemanticError("empty completion")
                return parse(completion.text), completion.text, usage
            except SemanticError as e:
                last = e
                if attempt == policy.max_attempts:
                    break
                await asyncio.sleep(policy.base_backoff_ms / 1000 * (2 ** (attempt - 1)))
        raise SemanticError(f"{last} (after {policy.max_attempts} attempts)")

    async def complete_verdict(self, prompt: str) -> tuple[tuple[str, str], str, TokenUsage]:
        """A ``{"verdict":..., "reason":...}`` answer."""
        return await self._with_semantic_retry(prompt, parse_verdict_json)

    async def complete_points(self, prompt: str) -> tuple[int, str, TokenUsage]:
        """A graded score plus the grader's raw text."""
        points, raw, usage = await self._with_semantic_retry(prompt, parse_points)
        return points, raw, usage

    async def complete_text(self, prompt: str) -> tuple[str, TokenUsage]:
        """Free-form output (the solver), with empty replies retried."""
        _, raw, usage = await self._with_semantic_retry(prompt, lambda t: None)
        return raw, usage


def _token_usage(usage: Any) -> TokenUsage:
    if usage is None:
        return TokenUsage(calls=1)
    prompt = getattr(usage, "prompt_tokens", 0) or 0
    completion = getattr(usage, "completion_tokens", 0) or 0
    total = getattr(usage, "total_tokens", 0) or (prompt + completion)
    return TokenUsage(
        calls=1, prompt_tokens=prompt, completion_tokens=completion, total_tokens=total
    )
