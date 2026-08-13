"""The public entry point."""

from __future__ import annotations

import asyncio
import time
from typing import Optional

from .config import Config
from .llm import LLMClient
from .models import CheckResult, Problem, Report, ResistanceResult, TokenUsage, Usage
from .quality_checks import run_quality_checks
from .registry import resolve_checks
from .resistance import run_resistance

_IN_LOOP_HINT = (
    "{name}() is synchronous and cannot run inside an existing event loop; "
    "await a{name}() instead"
)


def _run_sync(coro, name: str):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    coro.close()
    raise RuntimeError(_IN_LOOP_HINT.format(name=name))


class QC:
    """Evaluates problems against the configured endpoints.

    Clients may be injected, which is how the tests run without touching a
    network.
    """

    def __init__(
        self,
        config: Config,
        *,
        solver: Optional[LLMClient] = None,
        grader: Optional[LLMClient] = None,
        reviewer: Optional[LLMClient] = None,
    ):
        self.config = config
        self._solver = solver
        self._grader = grader
        self._reviewer = reviewer
        self._gate: Optional[asyncio.Semaphore] = None

    # ------------------------------------------------------------------- clients

    def _gate_for_loop(self) -> asyncio.Semaphore:
        if self._gate is None:
            self._gate = asyncio.Semaphore(self.config.max_inflight_calls)
        return self._gate

    def _resistance_clients(self) -> tuple[LLMClient, LLMClient]:
        if self.config.resistance is None:
            raise ValueError("config has no `resistance` section")
        gate = self._gate_for_loop()
        if self._solver is None:
            self._solver = LLMClient(
                self.config.resistance.solver,
                http_timeout_sec=self.config.http_timeout_sec,
                retry=self.config.retry,
                gate=gate,
            )
        if self._grader is None:
            self._grader = LLMClient(
                self.config.resistance.grader,
                http_timeout_sec=self.config.http_timeout_sec,
                retry=self.config.retry,
                gate=gate,
            )
        return self._solver, self._grader

    def _reviewer_client(self) -> LLMClient:
        if self.config.quality_checks is None:
            raise ValueError("config has no `quality_checks` section")
        if self._reviewer is None:
            self._reviewer = LLMClient(
                self.config.quality_checks.model,
                http_timeout_sec=self.config.http_timeout_sec,
                retry=self.config.retry,
                gate=self._gate_for_loop(),
            )
        return self._reviewer

    async def aclose(self) -> None:
        for client in (self._solver, self._grader, self._reviewer):
            if client is not None:
                await client.aclose()

    # --------------------------------------------------------------------- async

    async def aresistance(self, problem: Problem, *, usage: Optional[Usage] = None) -> ResistanceResult:
        solver, grader = self._resistance_clients()
        usage = usage if usage is not None else Usage()
        assert self.config.resistance is not None
        return await run_resistance(
            problem,
            self.config.resistance,
            solver,
            grader,
            solver_usage=usage.solver,
            grader_usage=usage.grader,
        )

    async def aquality_checks(
        self,
        problem: Problem,
        checks: Optional[list[str]] = None,
        *,
        usage: Optional[Usage] = None,
    ) -> dict[str, CheckResult]:
        client = self._reviewer_client()
        usage = usage if usage is not None else Usage()
        return await run_quality_checks(
            problem, resolve_checks(checks), client, usage=usage.checks
        )

    async def aevaluate(
        self, problem: Problem, *, checks: Optional[list[str]] = None
    ) -> Report:
        """Run everything the config and the problem allow, concurrently."""
        started = time.monotonic()
        usage = Usage()
        tasks = []
        if self.config.resistance is not None:
            tasks.append(("resistance", self.aresistance(problem, usage=usage)))
        if self.config.quality_checks is not None:
            tasks.append(
                ("quality_checks", self.aquality_checks(problem, checks, usage=usage))
            )
        if not tasks:
            raise ValueError("config has neither `resistance` nor `quality_checks`")

        outcomes = await asyncio.gather(*(coro for _, coro in tasks))
        report = Report(problem_id=problem.id, usage=usage)
        for (kind, _), outcome in zip(tasks, outcomes):
            if kind == "resistance":
                report.resistance = outcome
            else:
                report.quality_checks = outcome
        usage.latency_ms = int((time.monotonic() - started) * 1000)
        return report

    # ---------------------------------------------------------------------- sync

    def evaluate(self, problem: Problem, *, checks: Optional[list[str]] = None) -> Report:
        return _run_sync(self.aevaluate(problem, checks=checks), "evaluate")

    def resistance(self, problem: Problem) -> ResistanceResult:
        return _run_sync(self.aresistance(problem), "resistance")

    def quality_checks(
        self, problem: Problem, checks: Optional[list[str]] = None
    ) -> dict[str, CheckResult]:
        return _run_sync(self.aquality_checks(problem, checks), "quality_checks")
