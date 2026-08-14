"""The nine quality dimensions.

Three of them (competition_scope, solvability, discrimination) judge one
solution group at a time; the rest judge the statement, with elegance also
reading the main solution and consistency reading every group at once. All of
them return the same shape -- a list of group results -- so callers never have
to know which is which.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from . import prompts
from .llm import LLMClient, SemanticError, TransportError
from .models import MAIN_GROUP_UID, CheckResult, GroupResult, Problem, Solution, TokenUsage
from .registry import CHECKS, CheckDecl


def _targets(decl: CheckDecl, problem: Problem) -> tuple[list[tuple[str, Optional[Solution]]], Optional[str]]:
    """The (uid, group) pairs to judge, or why the check cannot run."""
    if decl.per_group:
        pairs = [
            (uid, sol)
            for uid, sol in zip(problem.group_uids(), problem.solutions)
            if sol.text.strip()
        ]
        if not pairs:
            return [], "missing solution"
        if decl.needs_rubric:
            pairs = [(uid, sol) for uid, sol in pairs if sol.rubric]
            if not pairs:
                return [], "missing rubric"
        return pairs, None

    main = problem.main_solution()
    if decl.needs_solution and (main is None or not main.text.strip()):
        return [], "missing solution"
    return [(MAIN_GROUP_UID, main)], None


async def _judge(
    decl: CheckDecl,
    problem: Problem,
    group: Optional[Solution],
    uid: str,
    nonce: str,
    client: LLMClient,
    usage: TokenUsage,
    prompts_dir: Optional[Path],
) -> GroupResult:
    prompt = prompts.render_check(decl, problem, group, uid, nonce, prompts_dir)
    try:
        (verdict, reason), _, u = await client.complete_verdict(prompt)
        usage.add(u)
        return GroupResult(uid=uid, status="ok", verdict=verdict, reason=reason)
    except (TransportError, SemanticError) as e:
        return GroupResult(uid=uid, status="error", error=str(e))


async def run_quality_checks(
    problem: Problem,
    check_names: list[str],
    client: LLMClient,
    *,
    usage: dict[str, TokenUsage],
    nonce: Optional[str] = None,
    prompts_dir: Optional[Path] = None,
) -> dict[str, CheckResult]:
    nonce = nonce or prompts.make_nonce()
    results: dict[str, CheckResult] = {}
    jobs: list[tuple[str, asyncio.Task[GroupResult]]] = []

    for name in check_names:
        decl = CHECKS[name]
        targets, skip_reason = _targets(decl, problem)
        if skip_reason is not None:
            results[name] = CheckResult(
                name=name, per_group=decl.per_group, status="skipped", skip_reason=skip_reason
            )
            continue
        bucket = usage.setdefault(name, TokenUsage())
        for uid, group in targets:
            jobs.append(
                (
                    name,
                    asyncio.create_task(
                        _judge(decl, problem, group, uid, nonce, client, bucket, prompts_dir)
                    ),
                )
            )

    gathered = await asyncio.gather(*(task for _, task in jobs), return_exceptions=False)

    by_check: dict[str, list[GroupResult]] = {}
    for (name, _), group_result in zip(jobs, gathered):
        by_check.setdefault(name, []).append(group_result)

    for name, groups in by_check.items():
        decl = CHECKS[name]
        status = "ok" if any(g.status == "ok" for g in groups) else "error"
        results[name] = CheckResult(
            name=name, per_group=decl.per_group, status=status, groups=groups
        )

    # Preserve the requested order.
    return {name: results[name] for name in check_names if name in results}
