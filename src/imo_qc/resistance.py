"""AI resistance: let a solver try the problem, then grade it against the
reference solution and rubric.

Statement-level by design: grading always uses the main solution group, so a
problem has one resistance result no matter how many alternative solutions it
carries.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from . import prompts
from .config import ResistanceConfig
from .llm import LLMClient, SemanticError, TransportError
from .models import (
    MAX_POINTS,
    Attempt,
    Problem,
    ResistanceResult,
    Solution,
    TokenUsage,
)


def _skip(reason: str) -> ResistanceResult:
    return ResistanceResult(status="skipped", skip_reason=reason)


def precheck(problem: Problem) -> Optional[ResistanceResult]:
    """Why resistance cannot run, if it cannot."""
    main = problem.main_solution()
    if main is None or not main.text.strip():
        return _skip("missing solution")
    if not main.rubric:
        return _skip("missing rubric")
    total = main.rubric_total
    if abs(total - MAX_POINTS) > 1e-6:
        # The grader prompt and the score parser are both fixed at 10 points, so
        # a rubric summing to anything else produces scores that cannot be
        # compared with anything.
        pretty = int(total) if float(total).is_integer() else total
        return _skip(f"rubric sum={pretty} != {MAX_POINTS}")
    return None


async def _one_attempt(
    index: int,
    problem: Problem,
    main: Solution,
    solver: LLMClient,
    grader: LLMClient,
    solver_usage: TokenUsage,
    grader_usage: TokenUsage,
    prompts_dir: Optional[Path],
) -> Attempt:
    try:
        solution, su = await solver.complete_text(
            prompts.render_solver(problem, prompts_dir)
        )
        solver_usage.add(su)
        points, raw, gu = await grader.complete_points(
            prompts.render_grader(problem, main, solution, prompts_dir)
        )
        grader_usage.add(gu)
        return Attempt(
            attempt=index, status="ok", points=points, solution=solution, grader_raw=raw
        )
    except (TransportError, SemanticError) as e:
        return Attempt(attempt=index, status="error", error=str(e))


async def run_resistance(
    problem: Problem,
    cfg: ResistanceConfig,
    solver: LLMClient,
    grader: LLMClient,
    *,
    solver_usage: TokenUsage,
    grader_usage: TokenUsage,
    prompts_dir: Optional[Path] = None,
) -> ResistanceResult:
    skipped = precheck(problem)
    if skipped is not None:
        return skipped

    main = problem.main_solution()
    assert main is not None  # guaranteed by precheck

    tasks: dict[asyncio.Task[Attempt], int] = {}
    for i in range(1, cfg.attempts + 1):
        task = asyncio.create_task(
            _one_attempt(
                i, problem, main, solver, grader, solver_usage, grader_usage, prompts_dir
            )
        )
        tasks[task] = i

    done_attempts: dict[int, Attempt] = {}
    pending = set(tasks)
    stopped_early = False
    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            index = tasks[task]
            try:
                done_attempts[index] = task.result()
            except asyncio.CancelledError:  # pragma: no cover - defensive
                done_attempts[index] = Attempt(attempt=index, status="cancelled")
        if cfg.early_stop_at is not None and any(
            a.points is not None and a.points >= cfg.early_stop_at
            for a in done_attempts.values()
        ):
            stopped_early = True
            break

    if stopped_early:
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        for task in pending:
            done_attempts.setdefault(tasks[task], Attempt(attempt=tasks[task], status="cancelled"))

    # Every configured attempt is represented, so `len(attempts)` is always the
    # configured count and statistics over it stay honest.
    attempts = [
        done_attempts.get(i, Attempt(attempt=i, status="cancelled"))
        for i in range(1, cfg.attempts + 1)
    ]
    status = "ok" if any(a.status == "ok" for a in attempts) else "error"
    error = None
    if status == "error":
        error = next((a.error for a in attempts if a.error), "all attempts failed")
    return ResistanceResult(status=status, error=error, attempts=attempts)
