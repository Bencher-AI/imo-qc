"""Which quality dimensions exist, and what each one needs to run."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CheckDecl:
    """Static declaration of one quality dimension.

    ``needs_*`` drives the ``skipped`` outcome: a dimension that cannot see the
    inputs its prompt asks about would otherwise judge a blank, and a blank is
    not a defect.
    """

    name: str
    per_group: bool = False
    needs_solution: bool = False
    needs_rubric: bool = False
    # consistency is the only dimension that compares groups against each other,
    # so it receives every group at once instead of being run per group.
    aggregate_groups: bool = False
    include_short_answer: bool = False
    include_subjects: bool = False


CHECKS: dict[str, CheckDecl] = {
    "self_contained": CheckDecl("self_contained"),
    "consistency": CheckDecl(
        "consistency",
        aggregate_groups=True,
        include_short_answer=True,
        include_subjects=True,
    ),
    "competition_scope": CheckDecl("competition_scope", per_group=True, needs_solution=True),
    "solvability": CheckDecl("solvability", per_group=True, needs_solution=True),
    "discrimination": CheckDecl(
        "discrimination", per_group=True, needs_solution=True, needs_rubric=True
    ),
    "anti_trick": CheckDecl("anti_trick"),
    "expression": CheckDecl("expression"),
    # Statement-level, but reads the main solution. Subjective by design -- the
    # prompt itself says to pass when in doubt.
    "elegance": CheckDecl("elegance", needs_solution=True),
    "novelty": CheckDecl("novelty"),
}

#: Default run order.
ALL_CHECKS: list[str] = list(CHECKS)

#: The seven objective dimensions, excluding the two subjective ones.
SEVEN_CHECKS: list[str] = [
    "self_contained",
    "consistency",
    "competition_scope",
    "solvability",
    "discrimination",
    "anti_trick",
    "expression",
]


def resolve_checks(names: list[str] | None) -> list[str]:
    if names is None:
        return list(ALL_CHECKS)
    unknown = [n for n in names if n not in CHECKS]
    if unknown:
        raise ValueError(f"unknown checks: {', '.join(unknown)}; known: {', '.join(ALL_CHECKS)}")
    return list(names)
