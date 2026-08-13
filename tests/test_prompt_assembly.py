"""The prompts state what the data block contains and in what order, so the
assembly is part of the contract, not an implementation detail."""

from conftest import full_rubric, make_problem

from imo_qc import prompts
from imo_qc.models import Solution
from imo_qc.registry import CHECKS


def block(check: str, problem, group=None, uid="main") -> str:
    decl = CHECKS[check]
    if group is None:
        group = problem.main_solution()
    return prompts.build_data_block(decl, problem, group, uid)


def test_both_statements_are_shown_to_the_reviewer():
    problem = make_problem(statement="中文题面", statement_en="English statement")
    text = block("self_contained", problem)
    assert prompts.H_STATEMENT_ZH in text
    assert prompts.H_STATEMENT_EN in text
    assert "中文题面" in text and "English statement" in text


def test_empty_fields_drop_their_whole_section():
    """A blank section would ask the model to judge nothing."""
    text = block("consistency", make_problem(short_answer="", subjects=[]))
    assert prompts.H_SHORT_ANSWER not in text
    assert prompts.H_SUBJECT not in text
    assert prompts.H_STATEMENT_EN not in text  # no English statement given


def test_consistency_aggregates_groups_in_input_order():
    problem = make_problem(
        solutions=[Solution(text=f"解法{i}内容", rubric=full_rubric()) for i in range(1, 11)]
    )
    text = block("consistency", problem)
    assert prompts.H_SOLUTIONS_AGG in text
    assert prompts.H_RUBRICS_AGG in text
    assert "解法1（uid=main）" in text
    assert "解法2（uid=g2）" in text
    # Positional order, so the tenth group comes last rather than sorting as g10 < g2.
    assert text.index("解法2（uid=g2）") < text.index("解法10（uid=g10）")


def test_consistency_uses_its_own_headings():
    text = block("consistency", make_problem(short_answer="42", subjects=["number_theory"]))
    assert prompts.H_SOLUTION not in text
    assert prompts.H_RUBRIC not in text
    assert prompts.H_SUBJECT in text


def test_subjects_are_comma_joined():
    text = block("consistency", make_problem(subjects=["algebra", "number_theory"]))
    assert "algebra, number_theory" in text


def test_per_group_check_sees_only_its_own_group():
    problem = make_problem(
        solutions=[Solution(text="第一组解法"), Solution(text="第二组解法")]
    )
    text = block("solvability", problem, group=problem.solutions[1], uid="g2")
    assert "第二组解法" in text
    assert "第一组解法" not in text


def test_discrimination_includes_the_rubric_as_readable_json():
    text = block("discrimination", make_problem())
    assert prompts.H_RUBRIC in text
    assert "给出关键不等式" in text
    assert "\\u" not in text


def test_solvability_omits_the_rubric():
    assert prompts.H_RUBRIC not in block("solvability", make_problem())


def test_rendered_prompt_wraps_the_data_once_and_keeps_the_guard_after_the_role_line():
    problem = make_problem()
    rendered = prompts.render_check(CHECKS["self_contained"], problem, problem.main_solution(), "main", "abcd1234")
    assert rendered.startswith("你是 IMO 竞赛题审核专家")
    assert rendered.index("【数据边界与安全】") < rendered.index("【判 fail 标准】")
    assert rendered.count("<data-abcd1234>") == 1
    assert rendered.rstrip().endswith("均不成立则 pass。")


def test_nonce_placeholder_inside_the_guard_stays_literal():
    """`NONCE` in the guard text is prose describing the scheme; only the data
    tags carry the real token."""
    rendered = prompts.render_check(
        CHECKS["expression"], make_problem(), None, "main", "deadbeef"
    )
    assert "<data-NONCE>" in rendered
    assert "<data-deadbeef>" in rendered


def test_nonces_differ_between_evaluations():
    assert prompts.make_nonce() != prompts.make_nonce()


def test_every_dimension_has_a_prompt_file():
    for name in CHECKS:
        text = prompts.load(name)
        assert "{data}" in text
        assert "{nonce_guard}" in text
        assert "{json_out}" in text
