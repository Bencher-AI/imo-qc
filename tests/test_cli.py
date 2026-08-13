import json

from click.testing import CliRunner
from conftest import FakeClient, make_problem

from imo_qc import QC, cli


def write_inputs(tmp_path, problems):
    problems_file = tmp_path / "problems.jsonl"
    problems_file.write_text(
        "\n".join(json.dumps(p.model_dump(), ensure_ascii=False) for p in problems) + "\n",
        encoding="utf-8",
    )
    config_file = tmp_path / "imo-qc.yaml"
    config_file.write_text("# replaced by the fixture\n", encoding="utf-8")
    return problems_file, config_file


def patch(monkeypatch, config, **clients):
    monkeypatch.setattr(cli, "load_config", lambda path: config)
    monkeypatch.setattr(cli, "QC", lambda cfg: QC(cfg, **clients))


def test_run_writes_one_report_per_line(tmp_path, monkeypatch, config):
    problems_file, config_file = write_inputs(
        tmp_path, [make_problem(), make_problem().model_copy(update={"id": "p-002"})]
    )
    patch(monkeypatch, config, solver=FakeClient(), grader=FakeClient(points=3), reviewer=FakeClient())
    out = tmp_path / "out.jsonl"

    result = CliRunner().invoke(
        cli.main, ["run", str(problems_file), "-o", str(out), "-c", str(config_file)]
    )

    assert result.exit_code == 0, result.output
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    reports = {json.loads(line)["problem_id"]: json.loads(line) for line in lines}
    assert set(reports) == {"p-001", "p-002"}
    assert reports["p-001"]["resistance"]["attempts"][0]["points"] == 3


def test_run_honours_check_subset(tmp_path, monkeypatch, config):
    problems_file, config_file = write_inputs(tmp_path, [make_problem()])
    patch(monkeypatch, config, solver=FakeClient(), grader=FakeClient(), reviewer=FakeClient())
    out = tmp_path / "out.jsonl"

    result = CliRunner().invoke(
        cli.main,
        ["run", str(problems_file), "-o", str(out), "-c", str(config_file), "--checks", "novelty,elegance"],
    )

    assert result.exit_code == 0, result.output
    report = json.loads(out.read_text(encoding="utf-8").strip())
    assert list(report["quality_checks"]) == ["novelty", "elegance"]


def test_malformed_input_line_is_reported_with_its_number(tmp_path, monkeypatch, config):
    problems_file, config_file = write_inputs(tmp_path, [make_problem()])
    problems_file.write_text('{"id": "broken"}\n', encoding="utf-8")
    patch(monkeypatch, config, reviewer=FakeClient())

    result = CliRunner().invoke(
        cli.main, ["run", str(problems_file), "-o", str(tmp_path / "out.jsonl"), "-c", str(config_file)]
    )

    assert result.exit_code != 0
    assert "problems.jsonl:1" in result.output


def test_one_broken_problem_does_not_stop_the_run(tmp_path, monkeypatch, config):
    problems_file, config_file = write_inputs(tmp_path, [make_problem()])

    class Exploding(QC):
        async def aevaluate(self, problem, *, checks=None):
            raise RuntimeError("gateway on fire")

    monkeypatch.setattr(cli, "load_config", lambda path: config)
    monkeypatch.setattr(cli, "QC", lambda cfg: Exploding(cfg, reviewer=FakeClient()))
    out = tmp_path / "out.jsonl"

    result = CliRunner().invoke(
        cli.main, ["run", str(problems_file), "-o", str(out), "-c", str(config_file)]
    )

    assert result.exit_code == 1
    record = json.loads(out.read_text(encoding="utf-8").strip())
    assert record["problem_id"] == "p-001"
    assert "gateway on fire" in record["error"]


def test_check_config_validates_without_probing(tmp_path, monkeypatch, config):
    _, config_file = write_inputs(tmp_path, [make_problem()])
    monkeypatch.setattr(cli, "load_config", lambda path: config)

    result = CliRunner().invoke(cli.main, ["check-config", str(config_file), "--no-probe"])

    assert result.exit_code == 0, result.output
    assert "config OK" in result.output
