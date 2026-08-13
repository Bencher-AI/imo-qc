"""Shared fixtures. No test in this suite touches a network."""

from __future__ import annotations

from typing import Any, Callable, Optional

import pytest

from imo_qc.config import Config, ModelConfig, QualityChecksConfig, ResistanceConfig, RetryConfig, RetryPolicy
from imo_qc.models import Problem, Rubric, Solution, TokenUsage


def _endpoint(**kw: Any) -> ModelConfig:
    return ModelConfig(base_url="http://localhost/v1", api_key="k", model="test-model", **kw)


def fast_retry(transport: int = 3, semantic: int = 3) -> RetryConfig:
    """Same retry counts, no waiting."""
    return RetryConfig(
        transport=RetryPolicy(max_attempts=transport, base_backoff_ms=0),
        semantic=RetryPolicy(max_attempts=semantic, base_backoff_ms=0),
    )


@pytest.fixture
def config() -> Config:
    return Config(
        resistance=ResistanceConfig(attempts=3, solver=_endpoint(), grader=_endpoint()),
        quality_checks=QualityChecksConfig(model=_endpoint()),
        http_timeout_sec=3600,
        retry=fast_retry(),
    )


class FakeClient:
    """Stands in for :class:`imo_qc.llm.LLMClient`.

    Each answer may be a plain value, a list consumed in order, a callable taking
    the prompt, or an exception instance to raise.
    """

    def __init__(
        self,
        *,
        text: Any = "a solution",
        points: Any = 0,
        verdict: Any = ("pass", "looks fine"),
        tokens: int = 10,
    ):
        self.text = text
        self.points = points
        self.verdict = verdict
        self.tokens = tokens
        self.text_prompts: list[str] = []
        self.points_prompts: list[str] = []
        self.verdict_prompts: list[str] = []
        self.closed = False

    def _resolve(self, spec: Any, prompt: str) -> Any:
        if isinstance(spec, list):
            value = spec.pop(0)
        elif callable(spec):
            value = spec(prompt)
        else:
            value = spec
        if isinstance(value, BaseException):
            raise value
        return value

    def _usage(self) -> TokenUsage:
        return TokenUsage(calls=1, prompt_tokens=1, completion_tokens=self.tokens - 1, total_tokens=self.tokens)

    async def complete_text(self, prompt: str):
        self.text_prompts.append(prompt)
        return self._resolve(self.text, prompt), self._usage()

    async def complete_points(self, prompt: str):
        self.points_prompts.append(prompt)
        return self._resolve(self.points, prompt), "grader said so", self._usage()

    async def complete_verdict(self, prompt: str):
        self.verdict_prompts.append(prompt)
        return self._resolve(self.verdict, prompt), "raw", self._usage()

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture
def fake_solver() -> FakeClient:
    return FakeClient(text="here is my attempt")


@pytest.fixture
def fake_grader() -> FakeClient:
    return FakeClient(points=4)


@pytest.fixture
def fake_reviewer() -> FakeClient:
    return FakeClient()


def make_problem(
    *,
    statement: str = "设 n 为正整数，证明 n < n + 1。",
    statement_en: Optional[str] = None,
    short_answer: str = "",
    subjects: Optional[list[str]] = None,
    solutions: Optional[list[Solution]] = None,
) -> Problem:
    if solutions is None:
        solutions = [
            Solution(
                text="显然成立。",
                rubric=[Rubric(score=4, criterion="给出关键不等式"), Rubric(score=6, criterion="完成论证")],
            )
        ]
    return Problem(
        id="p-001",
        statement=statement,
        statement_en=statement_en,
        short_answer=short_answer,
        subjects=subjects or [],
        solutions=solutions,
    )


def full_rubric() -> list[Rubric]:
    return [Rubric(score=4, criterion="步骤一"), Rubric(score=6, criterion="步骤二")]


@pytest.fixture
def problem() -> Problem:
    return make_problem()
