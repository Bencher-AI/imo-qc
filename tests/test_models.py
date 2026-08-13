import json

from conftest import make_problem

from imo_qc.models import Rubric, Solution, serialize_rubric


def test_group_uids_are_positional_not_sorted():
    problem = make_problem(solutions=[Solution(text=f"s{i}") for i in range(11)])
    uids = problem.group_uids()
    assert uids[0] == "main"
    assert uids[1] == "g2"
    # Sorting these strings would put g10 before g2 and main last, which is why
    # the order is positional.
    assert uids[9] == "g10"
    assert uids != sorted(uids)


def test_serialize_rubric_keeps_chinese_readable():
    text = serialize_rubric([Rubric(score=4, criterion="给出关键不等式")])
    assert "给出关键不等式" in text
    assert "\\u" not in text
    assert json.loads(text) == [{"score": 4, "criterion": "给出关键不等式"}]


def test_serialize_rubric_keeps_fractional_scores():
    assert json.loads(serialize_rubric([Rubric(score=2.5, criterion="半分")]))[0]["score"] == 2.5


def test_solver_statement_prefers_english():
    assert make_problem(statement_en="Prove it.").solver_statement() == "Prove it."
    assert make_problem().solver_statement().startswith("设 n")


def test_rubric_total():
    assert Solution(text="x", rubric=[Rubric(score=4, criterion="a"), Rubric(score=6, criterion="b")]).rubric_total == 10
