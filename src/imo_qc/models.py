"""Input and output models.

Everything crossing the public API boundary is a pydantic model, so
``report.model_dump()`` is always JSON-serialisable.
"""

from __future__ import annotations

import json
from typing import Literal, Optional

from pydantic import BaseModel, Field

# The grader prompt says "the 10-point rubric" and the score parser only accepts
# "N out of 10", so the scale is fixed rather than configurable. Changing it means
# editing grader.txt and the parser together.
MAX_POINTS = 10

MAIN_GROUP_UID = "main"

Verdict = Literal["pass", "fail"]
CheckStatus = Literal["ok", "skipped", "error"]


# --------------------------------------------------------------------------- input


class Rubric(BaseModel):
    """One point-bearing grading criterion."""

    score: float
    criterion: str


class Solution(BaseModel):
    """One solution group: a reference solution plus its own rubric."""

    text: str = ""
    rubric: list[Rubric] = Field(default_factory=list)

    @property
    def rubric_total(self) -> float:
        return sum(r.score for r in self.rubric)


class Problem(BaseModel):
    """A problem to evaluate.

    Only ``statement`` is always required. What each check additionally needs is
    declared in ``registry.CHECKS``; a check whose inputs are missing is reported
    as ``skipped`` rather than raising.
    """

    id: str
    statement: str
    statement_en: Optional[str] = None
    short_answer: str = ""
    subjects: list[str] = Field(default_factory=list)
    solutions: list[Solution] = Field(default_factory=list)

    def solver_statement(self) -> str:
        """The statement fed to the solver, which is prompted to answer in English."""
        return self.statement_en or self.statement

    def group_uids(self) -> list[str]:
        """Stable ids in input order: the first solution is the main group.

        Ordering is positional on purpose -- sorting these strings would put
        ``g10`` before ``g2`` and ``main`` last.
        """
        return [MAIN_GROUP_UID if i == 0 else f"g{i + 1}" for i in range(len(self.solutions))]

    def main_solution(self) -> Optional[Solution]:
        return self.solutions[0] if self.solutions else None


def serialize_rubric(rubric: list[Rubric]) -> str:
    """Render a rubric as the JSON text shown to the model.

    ``ensure_ascii=False`` matters: with the default the Chinese criteria would
    reach the model as ``\\uXXXX`` escapes.
    """
    items = [
        {
            "score": int(r.score) if float(r.score).is_integer() else r.score,
            "criterion": r.criterion,
        }
        for r in rubric
    ]
    return json.dumps(items, indent=2, ensure_ascii=False)


# -------------------------------------------------------------------------- output


class Attempt(BaseModel):
    """One solver->grader round. ``attempt`` is 1-based; the list index is not."""

    attempt: int
    status: Literal["ok", "error", "cancelled"]
    points: Optional[int] = None
    solution: str = ""
    grader_raw: str = ""
    error: Optional[str] = None


class ResistanceResult(BaseModel):
    """AI-resistance outcome.

    Resistance is statement-level: it always grades against the main solution
    group, so there is one result per problem rather than one per group.
    ``attempts`` always has ``config.resistance.attempts`` entries -- failed and
    cancelled rounds are kept so counts stay meaningful.
    """

    status: CheckStatus
    skip_reason: Optional[str] = None
    error: Optional[str] = None
    max_points: int = MAX_POINTS
    attempts: list[Attempt] = Field(default_factory=list)


class GroupResult(BaseModel):
    """A verdict for one solution group (uid ``main`` for statement-level checks)."""

    uid: str
    status: Literal["ok", "error"] = "ok"
    verdict: Optional[Verdict] = None
    reason: str = ""
    error: Optional[str] = None


class CheckResult(BaseModel):
    """One quality dimension.

    ``verdict`` lives on the group, never on the check, and is ``None`` whenever
    the call did not produce a usable answer -- callers must be able to tell
    "the model said fail" from "the call failed". Statement-level checks return a
    single group so the shape never varies.
    """

    name: str
    per_group: bool
    status: CheckStatus
    skip_reason: Optional[str] = None
    groups: list[GroupResult] = Field(default_factory=list)


class TokenUsage(BaseModel):
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, other: "TokenUsage") -> None:
        self.calls += other.calls
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens


class Usage(BaseModel):
    """Token accounting split by role -- solver and grader can be different
    models with very different prices, so a single total is not enough to cost a
    run."""

    solver: TokenUsage = Field(default_factory=TokenUsage)
    grader: TokenUsage = Field(default_factory=TokenUsage)
    checks: dict[str, TokenUsage] = Field(default_factory=dict)
    latency_ms: int = 0

    @property
    def total_tokens(self) -> int:
        return (
            self.solver.total_tokens
            + self.grader.total_tokens
            + sum(u.total_tokens for u in self.checks.values())
        )


class Report(BaseModel):
    problem_id: str
    resistance: Optional[ResistanceResult] = None
    quality_checks: dict[str, CheckResult] = Field(default_factory=dict)
    usage: Usage = Field(default_factory=Usage)
