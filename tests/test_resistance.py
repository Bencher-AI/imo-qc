import asyncio

from conftest import FakeClient, full_rubric, make_problem

from imo_qc.config import ModelConfig, ResistanceConfig
from imo_qc.llm import SemanticError, TransportError
from imo_qc.models import Rubric, Solution, TokenUsage
from imo_qc.resistance import run_resistance


def cfg(**kw) -> ResistanceConfig:
    endpoint = ModelConfig(base_url="http://localhost/v1", model="m")
    return ResistanceConfig(solver=endpoint, grader=endpoint, **kw)


async def run(problem, solver, grader, **kw):
    usage_s, usage_g = TokenUsage(), TokenUsage()
    result = await run_resistance(
        problem, cfg(**kw), solver, grader, solver_usage=usage_s, grader_usage=usage_g
    )
    return result, usage_s, usage_g


# ------------------------------------------------------------------- skip paths


async def test_missing_solution_is_skipped():
    problem = make_problem(solutions=[])
    result, _, _ = await run(problem, FakeClient(), FakeClient())
    assert result.status == "skipped"
    assert result.skip_reason == "missing solution"


async def test_missing_rubric_is_skipped():
    problem = make_problem(solutions=[Solution(text="证明如下")])
    result, _, _ = await run(problem, FakeClient(), FakeClient())
    assert result.skip_reason == "missing rubric"


async def test_rubric_not_summing_to_ten_is_skipped():
    """The grader prompt and the parser are both fixed at 10 points, so any
    other total yields scores that cannot be compared with anything."""
    problem = make_problem(
        solutions=[Solution(text="证明", rubric=[Rubric(score=3, criterion="一半")])]
    )
    result, _, _ = await run(problem, FakeClient(), FakeClient())
    assert result.skip_reason == "rubric sum=3 != 10"


async def test_skipped_runs_make_no_calls():
    solver = FakeClient()
    await run(make_problem(solutions=[]), solver, FakeClient())
    assert solver.text_prompts == []


# ---------------------------------------------------------------------- happy path


async def test_attempt_count_is_configurable():
    solver, grader = FakeClient(), FakeClient(points=2)
    result, _, _ = await run(make_problem(), solver, grader, attempts=5)
    assert len(solver.text_prompts) == 5
    assert len(grader.points_prompts) == 5
    assert [a.attempt for a in result.attempts] == [1, 2, 3, 4, 5]
    assert all(a.points == 2 for a in result.attempts)


async def test_usage_is_tracked_per_role():
    _, usage_s, usage_g = await run(make_problem(), FakeClient(), FakeClient(), attempts=2)
    assert usage_s.calls == 2
    assert usage_g.calls == 2


async def test_attempts_run_concurrently():
    started = 0
    peak = 0

    async def slow_text(prompt):
        nonlocal started, peak
        started += 1
        peak = max(peak, started)
        await asyncio.sleep(0.01)
        started -= 1
        return "solution"

    solver = FakeClient()
    solver.complete_text = lambda p: _wrap(slow_text(p))  # type: ignore[assignment]
    result, _, _ = await run(make_problem(), solver, FakeClient(), attempts=3)
    assert peak > 1
    assert len(result.attempts) == 3


async def _wrap(coro):
    text = await coro
    return text, TokenUsage(calls=1)


# ------------------------------------------------------------------ failure paths


async def test_failed_attempts_are_kept_so_counts_stay_honest():
    solver = FakeClient(text=["ok", TransportError("http 500"), "ok"])
    result, _, _ = await run(make_problem(), solver, FakeClient(points=1), attempts=3)
    assert len(result.attempts) == 3
    assert result.status == "ok"
    assert sorted(a.status for a in result.attempts) == ["error", "ok", "ok"]


async def test_all_attempts_failing_is_an_error():
    solver = FakeClient(text=SemanticError("empty completion (after 6 attempts)"))
    result, _, _ = await run(make_problem(), solver, FakeClient(), attempts=2)
    assert result.status == "error"
    assert "empty completion" in result.error
    assert all(a.points is None for a in result.attempts)


# ---------------------------------------------------------------------- early stop


async def test_early_stop_cancels_remaining_attempts():
    calls = 0

    async def counted(prompt):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.02 * calls)  # stagger so the first finishes first
        return "solution", TokenUsage(calls=1)

    solver = FakeClient()
    solver.complete_text = counted  # type: ignore[assignment]
    result, _, _ = await run(
        make_problem(), solver, FakeClient(points=10), attempts=3, early_stop_at=10
    )
    assert len(result.attempts) == 3  # length is still the configured count
    assert any(a.status == "cancelled" for a in result.attempts)


async def test_without_early_stop_all_attempts_run_even_at_full_marks():
    solver, grader = FakeClient(), FakeClient(points=10)
    result, _, _ = await run(make_problem(), solver, grader, attempts=3)
    assert len(grader.points_prompts) == 3
    assert all(a.status == "ok" for a in result.attempts)


# -------------------------------------------------------------------- prompt shape


async def test_solver_prompt_never_leaks_the_answer():
    """If the reference solution, short answer or rubric reached the solver, the
    whole resistance signal would be meaningless."""
    problem = make_problem(
        statement="证明存在无穷多素数。",
        short_answer="ANSWER-SENTINEL",
        solutions=[Solution(text="欧几里得反证法：假设只有有限个……", rubric=full_rubric())],
    )
    solver = FakeClient()
    await run(problem, solver, FakeClient(), attempts=1)
    prompt = solver.text_prompts[0]
    assert "证明存在无穷多素数。" in prompt
    assert "欧几里得反证法" not in prompt
    assert "ANSWER-SENTINEL" not in prompt
    assert "步骤一" not in prompt


async def test_solver_prompt_prefers_english_statement():
    problem = make_problem(statement="中文题面", statement_en="Prove there are infinitely many primes.")
    solver = FakeClient()
    await run(problem, solver, FakeClient(), attempts=1)
    prompt = solver.text_prompts[0]
    assert "Prove there are infinitely many primes." in prompt
    assert "中文题面" not in prompt


async def test_grader_prompt_carries_all_five_inputs():
    problem = make_problem(
        short_answer="42",
        solutions=[Solution(text="参考解答", rubric=full_rubric())],
    )
    grader = FakeClient(points=6)
    await run(problem, FakeClient(text="学生解答"), grader, attempts=1)
    prompt = grader.points_prompts[0]
    # Statement, ground truth, short answer, guidelines, proposed solution.
    for value in (problem.statement, "参考解答", "42", "步骤一", "学生解答"):
        assert value in prompt
    for placeholder in ("{statement}", "{ground_truth}", "{short_answer}", "{guidelines}", "{proposed}"):
        assert placeholder not in prompt
