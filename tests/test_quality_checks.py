import pytest
from conftest import FakeClient, full_rubric, make_problem

from imo_qc.llm import SemanticError
from imo_qc.models import Solution, TokenUsage
from imo_qc.quality_checks import run_quality_checks
from imo_qc.registry import ALL_CHECKS, SEVEN_CHECKS, resolve_checks


async def run(problem, names, client):
    usage: dict[str, TokenUsage] = {}
    results = await run_quality_checks(problem, names, client, usage=usage)
    return results, usage


async def test_all_nine_dimensions_run_by_default():
    results, usage = await run(make_problem(), resolve_checks(None), FakeClient())
    assert list(results) == ALL_CHECKS
    assert len(results) == 9
    assert all(r.status == "ok" for r in results.values())
    assert all(u.calls >= 1 for u in usage.values())


async def test_subset_selection_and_order():
    results, _ = await run(make_problem(), ["consistency", "self_contained"], FakeClient())
    assert list(results) == ["consistency", "self_contained"]


def test_unknown_check_name_is_rejected():
    with pytest.raises(ValueError, match="unknown checks: nope"):
        resolve_checks(["nope"])


def test_seven_checks_excludes_the_subjective_pair():
    assert "elegance" not in SEVEN_CHECKS and "novelty" not in SEVEN_CHECKS
    assert len(SEVEN_CHECKS) == 7


# --------------------------------------------------------------- shape uniformity


async def test_per_group_dimensions_judge_every_group():
    problem = make_problem(
        solutions=[
            Solution(text="解法一", rubric=full_rubric()),
            Solution(text="解法二", rubric=full_rubric()),
        ]
    )
    client = FakeClient()
    results, _ = await run(problem, ["solvability", "self_contained"], client)

    solvability = results["solvability"]
    assert solvability.per_group is True
    assert [g.uid for g in solvability.groups] == ["main", "g2"]

    # Statement-level checks return one group so the shape never varies.
    self_contained = results["self_contained"]
    assert self_contained.per_group is False
    assert [g.uid for g in self_contained.groups] == ["main"]


async def test_verdict_lives_on_the_group():
    results, _ = await run(make_problem(), ["consistency"], FakeClient(verdict=("fail", "答案矛盾")))
    group = results["consistency"].groups[0]
    assert (group.verdict, group.reason) == ("fail", "答案矛盾")


# ------------------------------------------------------------------- skip paths


async def test_dimensions_needing_a_solution_are_skipped_without_one():
    problem = make_problem(solutions=[])
    results, _ = await run(problem, resolve_checks(None), FakeClient())

    for name in ("competition_scope", "solvability", "discrimination", "elegance"):
        assert results[name].status == "skipped", name
        assert results[name].skip_reason == "missing solution"

    # The statement-only dimensions still run, including consistency, which just
    # omits the sections it has no data for.
    for name in ("self_contained", "anti_trick", "expression", "novelty", "consistency"):
        assert results[name].status == "ok", name


async def test_discrimination_is_skipped_without_a_rubric():
    problem = make_problem(solutions=[Solution(text="证明")])
    results, _ = await run(problem, ["discrimination", "solvability"], FakeClient())
    assert results["discrimination"].skip_reason == "missing rubric"
    # Solvability does not read the rubric, so it still runs.
    assert results["solvability"].status == "ok"


async def test_skipped_dimensions_make_no_calls():
    client = FakeClient()
    await run(make_problem(solutions=[]), ["elegance"], client)
    assert client.verdict_prompts == []


# ------------------------------------------------------------------ failure paths


async def test_unusable_answer_is_an_error_not_a_fail():
    """A caller must be able to tell "the model judged fail" from "the call did
    not produce an answer"."""
    client = FakeClient(verdict=SemanticError("verdict must be pass or fail, got 'maybe'"))
    results, _ = await run(make_problem(), ["expression"], client)
    check = results["expression"]
    assert check.status == "error"
    assert check.groups[0].verdict is None
    assert check.groups[0].status == "error"


async def test_one_failing_group_does_not_sink_the_others():
    problem = make_problem(
        solutions=[
            Solution(text="解法一", rubric=full_rubric()),
            Solution(text="解法二", rubric=full_rubric()),
        ]
    )
    client = FakeClient(verdict=[("pass", "fine"), SemanticError("empty completion")])
    results, _ = await run(problem, ["solvability"], client)
    check = results["solvability"]
    assert check.status == "ok"
    assert sorted(g.status for g in check.groups) == ["error", "ok"]
