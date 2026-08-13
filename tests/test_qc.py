import json

import pytest
from conftest import FakeClient, make_problem

from imo_qc import QC
from imo_qc.config import Config, QualityChecksConfig


def build(config, **clients) -> QC:
    return QC(config, **clients)


async def test_evaluate_runs_both_capabilities(config):
    qc = build(
        config,
        solver=FakeClient(),
        grader=FakeClient(points=8),
        reviewer=FakeClient(verdict=("fail", "记号写反")),
    )
    report = await qc.aevaluate(make_problem())

    assert report.problem_id == "p-001"
    assert report.resistance.status == "ok"
    assert [a.points for a in report.resistance.attempts] == [8, 8, 8]
    assert report.quality_checks["expression"].groups[0].verdict == "fail"
    assert len(report.quality_checks) == 9


async def test_report_is_json_serialisable(config):
    qc = build(config, solver=FakeClient(), grader=FakeClient(points=1), reviewer=FakeClient())
    report = await qc.aevaluate(make_problem())
    encoded = json.dumps(report.model_dump(), ensure_ascii=False)
    assert json.loads(encoded)["resistance"]["attempts"][0]["points"] == 1


async def test_usage_is_split_by_role(config):
    qc = build(config, solver=FakeClient(), grader=FakeClient(), reviewer=FakeClient())
    report = await qc.aevaluate(make_problem())
    usage = report.usage
    assert usage.solver.calls == 3
    assert usage.grader.calls == 3
    assert len(usage.checks) == 9
    assert usage.total_tokens == usage.solver.total_tokens + usage.grader.total_tokens + sum(
        u.total_tokens for u in usage.checks.values()
    )
    assert usage.latency_ms >= 0


async def test_checks_subset_reaches_quality_checks(config):
    qc = build(config, solver=FakeClient(), grader=FakeClient(), reviewer=FakeClient())
    report = await qc.aevaluate(make_problem(), checks=["novelty"])
    assert list(report.quality_checks) == ["novelty"]


async def test_quality_checks_only_config():
    config = Config(
        quality_checks=QualityChecksConfig(
            model={"base_url": "http://localhost/v1", "model": "m"}
        )
    )
    qc = build(config, reviewer=FakeClient())
    report = await qc.aevaluate(make_problem())
    assert report.resistance is None
    assert len(report.quality_checks) == 9


async def test_resistance_section_required_for_resistance_call():
    config = Config(
        quality_checks=QualityChecksConfig(
            model={"base_url": "http://localhost/v1", "model": "m"}
        )
    )
    with pytest.raises(ValueError, match="no `resistance` section"):
        await build(config, reviewer=FakeClient()).aresistance(make_problem())


async def test_empty_config_is_rejected():
    with pytest.raises(ValueError, match="neither"):
        await QC(Config()).aevaluate(make_problem())


async def test_sync_api_refuses_to_run_inside_a_loop(config):
    """Notebooks and web servers already have a loop; failing loudly beats
    RuntimeError from deep inside asyncio."""
    qc = build(config, solver=FakeClient(), grader=FakeClient(), reviewer=FakeClient())
    with pytest.raises(RuntimeError, match="await aevaluate"):
        qc.evaluate(make_problem())


def test_sync_api_works_outside_a_loop(config):
    qc = build(config, solver=FakeClient(), grader=FakeClient(), reviewer=FakeClient())
    report = qc.evaluate(make_problem())
    assert report.resistance.status == "ok"


def test_sync_quality_checks_only(config):
    qc = build(config, solver=FakeClient(), grader=FakeClient(), reviewer=FakeClient())
    results = qc.quality_checks(make_problem(), ["anti_trick"])
    assert list(results) == ["anti_trick"]
