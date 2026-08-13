import asyncio
from types import SimpleNamespace

import httpx
import pytest
from conftest import fast_retry
from openai import APIStatusError

from imo_qc.config import ModelConfig
from imo_qc.llm import (
    Completion,
    LLMClient,
    SemanticError,
    TransportError,
    parse_points,
    parse_verdict_json,
)
from imo_qc.models import TokenUsage


def make_client(**cfg_kw) -> LLMClient:
    cfg = ModelConfig(base_url="http://localhost/v1", api_key="k", model="m", timeout_sec=5, **cfg_kw)
    return LLMClient(cfg, http_timeout_sec=60, retry=fast_retry())


def script(client: LLMClient, answers: list):
    """Replace the single-call layer with a scripted sequence."""
    calls: list[str] = []

    async def fake(prompt: str) -> Completion:
        calls.append(prompt)
        answer = answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return Completion(answer, TokenUsage(calls=1, total_tokens=7))

    client._call_once = fake  # type: ignore[method-assign]
    return calls


# ------------------------------------------------------------------- parsing


def test_parse_points_plain():
    assert parse_points("Reasoning...\n<points>7 out of 10</points>") == 7


def test_parse_points_is_case_insensitive_and_takes_the_first_block():
    text = "<POINTS>3 OUT OF 10</POINTS> then later <points>9 out of 10</points>"
    assert parse_points(text) == 3


def test_parse_points_strips_code_fence():
    assert parse_points("```\n<points>10 out of 10</points>\n```") == 10


def test_parse_points_without_block_is_semantic():
    with pytest.raises(SemanticError, match="no <points> block"):
        parse_points("I award full marks.")


def test_parse_points_rejects_out_of_scale_score():
    with pytest.raises(SemanticError, match="no <points> block"):
        parse_points("<points>12 out of 10</points>")


def test_parse_verdict_from_prose_and_fence():
    verdict, reason = parse_verdict_json('Sure!\n```json\n{"verdict":"fail","reason":"记号写反"}\n```')
    assert (verdict, reason) == ("fail", "记号写反")


def test_parse_verdict_handles_braces_inside_reason():
    verdict, reason = parse_verdict_json('{"verdict":"pass","reason":"集合 {1,2} 无歧义"}')
    assert verdict == "pass"
    assert reason == "集合 {1,2} 无歧义"


def test_parse_verdict_without_object_is_semantic():
    with pytest.raises(SemanticError, match="no JSON object"):
        parse_verdict_json("looks fine to me")


def test_parse_verdict_rejects_other_values():
    # A judge answering anything but pass/fail is an error, never a fail.
    with pytest.raises(SemanticError, match="pass or fail"):
        parse_verdict_json('{"verdict":"questionable","reason":"x"}')


# ------------------------------------------------------- transport-layer retries


async def test_retries_after_429_then_succeeds():
    client = make_client()
    calls = script(client, [TransportError("http 429"), "ok"])
    assert (await client.complete("p")).text == "ok"
    assert len(calls) == 2


async def test_transport_retries_are_exhausted():
    client = make_client()
    calls = script(client, [TransportError("http 500")] * 3)
    with pytest.raises(TransportError, match="exhausted"):
        await client.complete("p")
    assert len(calls) == 3  # max_attempts counts the first call


async def test_non_transport_errors_are_not_retried():
    client = make_client()
    calls = script(client, [ValueError("http 400 bad request"), "unused"])
    with pytest.raises(ValueError):
        await client.complete("p")
    assert len(calls) == 1


async def test_timeout_counts_as_transport_failure():
    client = make_client()
    calls = script(client, [asyncio.TimeoutError(), "ok"])
    assert (await client.complete("p")).text == "ok"
    assert len(calls) == 2


async def test_client_status_errors_are_classified(monkeypatch):
    """429 and 5xx become retryable; other 4xx do not."""
    client = make_client()

    def raiser(status: int):
        async def create(**kwargs):
            response = httpx.Response(status, request=httpx.Request("POST", "http://localhost/v1"))
            raise APIStatusError("boom", response=response, body=None)

        return create

    client._client.chat.completions.create = raiser(429)  # type: ignore[assignment]
    with pytest.raises(TransportError):
        await client._call_once("p")

    client._client.chat.completions.create = raiser(400)  # type: ignore[assignment]
    with pytest.raises(APIStatusError):
        await client._call_once("p")


# -------------------------------------------------------- semantic-layer retries


async def test_empty_completion_is_retried():
    """The most common judge failure: HTTP 200 with no content."""
    client = make_client()
    calls = script(client, ["", '{"verdict":"pass","reason":"ok"}'])
    (verdict, _), _, usage = await client.complete_verdict("p")
    assert verdict == "pass"
    assert len(calls) == 2
    assert usage.calls == 2  # tokens from the wasted call still count


async def test_missing_points_block_is_retried():
    client = make_client()
    calls = script(client, ["no score here", "<points>5 out of 10</points>"])
    points, _, _ = await client.complete_points("p")
    assert points == 5
    assert len(calls) == 2


async def test_semantic_retries_are_exhausted():
    client = make_client()
    calls = script(client, ["nope"] * 3)
    with pytest.raises(SemanticError, match="after 3 attempts"):
        await client.complete_points("p")
    assert len(calls) == 3


# ------------------------------------------------------------ streaming / usage


def _chunk(content=None, usage=None):
    choices = []
    if content is not None:
        choices = [SimpleNamespace(delta=SimpleNamespace(content=content))]
    return SimpleNamespace(choices=choices, usage=usage)


async def _stream_of(chunks):
    async def create(**kwargs):
        async def gen():
            for chunk in chunks:
                yield chunk

        return gen()

    return create


async def test_stream_collects_text_and_usage():
    client = make_client(stream=True)
    usage = SimpleNamespace(prompt_tokens=3, completion_tokens=4, total_tokens=7)
    client._client.chat.completions.create = await _stream_of(  # type: ignore[assignment]
        [_chunk("part "), _chunk("two"), _chunk(usage=usage)]
    )
    completion = await client._call_once("p")
    assert completion.text == "part two"
    assert completion.usage.total_tokens == 7


async def test_stream_without_usage_frame_is_treated_as_truncated():
    """Returning the partial text would send a half-written solution to the
    grader, which scores it low -- a false "the problem resisted the AI"."""
    client = make_client(stream=True)
    client._client.chat.completions.create = await _stream_of(  # type: ignore[assignment]
        [_chunk("half a solu")]
    )
    with pytest.raises(TransportError, match="truncated"):
        await client._call_once("p")


async def test_usage_falls_back_to_prompt_plus_completion():
    client = make_client()
    usage = SimpleNamespace(prompt_tokens=5, completion_tokens=6, total_tokens=None)
    message = SimpleNamespace(content="hi")

    async def create(**kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)

    client._client.chat.completions.create = create  # type: ignore[assignment]
    completion = await client._call_once("p")
    assert completion.usage.total_tokens == 11


async def test_request_kwargs_use_max_completion_tokens_and_stream_options():
    client = make_client(stream=True, max_completion_tokens=1024, extra_body={"reasoning_effort": "xhigh"})
    kwargs = client._request_kwargs("p")
    assert kwargs["max_completion_tokens"] == 1024
    assert "max_tokens" not in kwargs
    assert kwargs["stream"] is True
    assert kwargs["stream_options"] == {"include_usage": True}
    assert kwargs["extra_body"] == {"reasoning_effort": "xhigh"}


def test_sdk_retries_are_disabled():
    # The SDK retries twice by default, which would silently multiply the
    # transport budget configured here.
    assert make_client()._client.max_retries == 0


async def test_inflight_gate_limits_concurrent_calls():
    client = make_client()
    client._gate = asyncio.Semaphore(2)
    live = 0
    peak = 0

    async def fake(prompt: str) -> Completion:
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.01)
        live -= 1
        return Completion("ok", TokenUsage(calls=1))

    client._call_once = fake  # type: ignore[method-assign]
    await asyncio.gather(*(client.complete("p") for _ in range(6)))
    assert peak <= 2
