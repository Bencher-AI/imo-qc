"""Prompt loading and assembly.

The prompt files are the source of truth for wording *and* ordering: each
dimension's text ends with "the data below is, in order, X / Y / Z", so the data
block must be assembled in exactly that order with exactly those section
headings. Changing either without changing the prompt silently misleads the model.
"""

from __future__ import annotations

import secrets
from functools import lru_cache
from pathlib import Path
from typing import Optional

from .models import Problem, Solution, serialize_rubric
from .registry import CheckDecl

_DIR = Path(__file__).parent / "prompts"

# Section headings, reproduced verbatim from the prompts that reference them.
H_STATEMENT_ZH = "【题面 statement（中文）】"
H_STATEMENT_EN = "【题面 statement（英文）】"
H_SHORT_ANSWER = "【简答 short_answer】"
H_SUBJECT = "【学科 subject】"
H_SOLUTION = "【solution】"
H_RUBRIC = "【rubric】"
# consistency sees every group at once and uses its own headings; the two sets
# are not interchangeable.
H_SOLUTIONS_AGG = "【各组 solution（聚合）】"
H_RUBRICS_AGG = "【各组 rubric（聚合）】"


@lru_cache(maxsize=None)
def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").rstrip("\n")


def load(name: str, prompts_dir: Optional[Path] = None) -> str:
    """Read a prompt, preferring ``prompts_dir`` when it holds a file of that name.

    Overriding by directory rather than by editing the installed package keeps a
    translated or reworded prompt set outside site-packages, where an upgrade
    cannot silently revert it.
    """
    if prompts_dir is not None:
        override = Path(prompts_dir) / f"{name}.txt"
        if override.exists():
            return _read(override)
    return _read(_DIR / f"{name}.txt")


def make_nonce() -> str:
    """A fresh boundary token per evaluation.

    The literal string ``NONCE`` inside the guard texts is prose explaining the
    scheme -- it is deliberately not substituted. Only the data tags carry the
    real value.
    """
    return secrets.token_hex(4)


def wrap(nonce: str, data: str) -> str:
    return f"<data-{nonce}>\n{data}\n</data-{nonce}>"


def _section(heading: str, value: str) -> str | None:
    """A section, or nothing at all when the value is empty.

    Empty inputs drop their whole section rather than showing an empty one, so
    the model is never asked to judge a blank field.
    """
    value = (value or "").strip()
    return f"{heading}\n{value}" if value else None


def _aggregate(heading: str, uids: list[str], values: list[str]) -> str | None:
    parts = []
    for i, (uid, value) in enumerate(zip(uids, values), start=1):
        value = (value or "").strip()
        if value:
            parts.append(f"解法{i}（uid={uid}）：\n{value}")
    if not parts:
        return None
    return f"{heading}\n" + "\n\n".join(parts)


def build_data_block(decl: CheckDecl, problem: Problem, group: Solution | None, uid: str) -> str:
    """Assemble the single data block a quality prompt expects."""
    sections: list[str | None] = [
        _section(H_STATEMENT_ZH, problem.statement),
        _section(H_STATEMENT_EN, problem.statement_en or ""),
    ]

    if decl.include_short_answer:
        sections.append(_section(H_SHORT_ANSWER, problem.short_answer))
    if decl.include_subjects:
        sections.append(_section(H_SUBJECT, ", ".join(problem.subjects)))

    if decl.aggregate_groups:
        uids = problem.group_uids()
        sections.append(
            _aggregate(H_SOLUTIONS_AGG, uids, [s.text for s in problem.solutions])
        )
        sections.append(
            _aggregate(
                H_RUBRICS_AGG,
                uids,
                [serialize_rubric(s.rubric) if s.rubric else "" for s in problem.solutions],
            )
        )
    else:
        if decl.needs_solution and group is not None:
            sections.append(_section(H_SOLUTION, group.text))
        if decl.needs_rubric and group is not None and group.rubric:
            sections.append(_section(H_RUBRIC, serialize_rubric(group.rubric)))

    return "\n\n".join(s for s in sections if s)


def render_check(
    decl: CheckDecl,
    problem: Problem,
    group: Solution | None,
    uid: str,
    nonce: str,
    prompts_dir: Optional[Path] = None,
) -> str:
    template = load(decl.name, prompts_dir)
    data = wrap(nonce, build_data_block(decl, problem, group, uid))
    return (
        template.replace("{nonce_guard}", load("_nonce_guard_zh", prompts_dir))
        .replace("{json_out}", load("_json_out", prompts_dir))
        .replace("{data}", data)
    )


def render_solver(problem: Problem, nonce: str, prompts_dir: Optional[Path] = None) -> str:
    """Solver prompt.

    Deliberately statement-only: feeding it the reference solution, the short
    answer or the rubric would leak the answer and make the whole resistance
    signal meaningless.
    """
    return (
        load("solver", prompts_dir)
        .replace("{nonce_guard}", load("_nonce_guard_solver", prompts_dir))
        .replace("{statement}", wrap(nonce, problem.solver_statement()))
    )


def render_grader(
    problem: Problem,
    main: Solution,
    proposed: str,
    nonce: str,
    prompts_dir: Optional[Path] = None,
) -> str:
    """Grader prompt: all five inputs are wrapped individually.

    The proposed solution is wrapped too -- it is model output derived from a
    statement that may itself carry an injection attempt.
    """
    return (
        load("grader", prompts_dir)
        .replace("{nonce_guard}", load("_nonce_guard_grader", prompts_dir))
        .replace("{statement}", wrap(nonce, problem.statement_en or problem.statement))
        .replace("{ground_truth}", wrap(nonce, main.text))
        .replace("{short_answer}", wrap(nonce, problem.short_answer))
        .replace("{guidelines}", wrap(nonce, serialize_rubric(main.rubric)))
        .replace("{proposed}", wrap(nonce, proposed))
    )
