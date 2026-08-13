"""End-to-end over real HTTP against a local stand-in endpoint.

Everything else stubs the client out, so this is the only place the actual SDK
call path is exercised: request shape, streaming, usage frames, parsing.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from conftest import make_problem

from imo_qc import QC
from imo_qc.config import Config, ModelConfig, QualityChecksConfig, ResistanceConfig

SOLVER_TEXT = "Assume the contrary and derive a contradiction. Hence the claim holds."
GRADER_TEXT = "The first rubric component is earned, the second is not.\n<points>4 out of 10</points>"
REVIEWER_TEXT = '{"verdict":"fail","reason":"记号 n 与 N 混用"}'

_USAGE = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}


def _content_for(model: str) -> str:
    if model == "solver-model":
        return SOLVER_TEXT
    if model == "grader-model":
        return GRADER_TEXT
    return REVIEWER_TEXT


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802 - required name
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        self.server.requests.append(body)  # type: ignore[attr-defined]
        content = _content_for(body.get("model", ""))
        if body.get("stream"):
            self._send_stream(body["model"], content)
        else:
            self._send_json(body["model"], content)

    def _send_json(self, model: str, content: str) -> None:
        payload = json.dumps(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": _USAGE,
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_stream(self, model: str, content: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def frame(obj: dict) -> None:
            data = f"data: {json.dumps(obj)}\n\n".encode()
            self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")

        base = {"id": "chatcmpl-test", "object": "chat.completion.chunk", "created": 0, "model": model}
        halfway = len(content) // 2
        for piece in (content[:halfway], content[halfway:]):
            frame({**base, "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]})
        # The final frame carries usage; its absence is what marks a truncated stream.
        frame({**base, "choices": [], "usage": _USAGE})
        done = b"data: [DONE]\n\n"
        self.wfile.write(f"{len(done):X}\r\n".encode() + done + b"\r\n")
        self.wfile.write(b"0\r\n\r\n")

    def log_message(self, *args) -> None:  # keep the test output clean
        pass


@pytest.fixture
def endpoint():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.requests = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()


def _config(base_url: str) -> Config:
    return Config(
        resistance=ResistanceConfig(
            attempts=2,
            solver=ModelConfig(
                base_url=base_url,
                api_key="k",
                model="solver-model",
                max_completion_tokens=256,
                timeout_sec=30,
                stream=True,  # the path used for long reasoning calls
                extra_body={"reasoning_effort": "xhigh"},
            ),
            grader=ModelConfig(
                base_url=base_url, api_key="k", model="grader-model", max_completion_tokens=256, timeout_sec=30
            ),
        ),
        quality_checks=QualityChecksConfig(
            model=ModelConfig(
                base_url=base_url, api_key="k", model="reviewer-model", max_tokens=512, timeout_sec=30
            )
        ),
        http_timeout_sec=60,
    )


async def test_full_evaluation_over_http(endpoint):
    server, base_url = endpoint
    qc = QC(_config(base_url))
    try:
        report = await qc.aevaluate(make_problem(), checks=["expression", "solvability"])
    finally:
        await qc.aclose()

    assert report.resistance.status == "ok"
    assert [a.points for a in report.resistance.attempts] == [4, 4]
    assert report.resistance.attempts[0].solution == SOLVER_TEXT  # reassembled from the stream
    assert "<points>4 out of 10</points>" in report.resistance.attempts[0].grader_raw

    expression = report.quality_checks["expression"]
    assert expression.groups[0].verdict == "fail"
    assert expression.groups[0].reason == "记号 n 与 N 混用"

    assert report.usage.solver.total_tokens == 60  # two attempts
    assert report.usage.grader.total_tokens == 60
    assert report.usage.checks["expression"].total_tokens == 30


async def test_request_bodies_carry_the_expected_fields(endpoint):
    server, base_url = endpoint
    qc = QC(_config(base_url))
    try:
        await qc.aresistance(make_problem())
    finally:
        await qc.aclose()

    solver_bodies = [b for b in server.requests if b["model"] == "solver-model"]
    grader_bodies = [b for b in server.requests if b["model"] == "grader-model"]
    assert len(solver_bodies) == 2 and len(grader_bodies) == 2

    solver = solver_bodies[0]
    assert solver["stream"] is True
    assert solver["stream_options"] == {"include_usage": True}
    assert solver["max_completion_tokens"] == 256
    assert "max_tokens" not in solver
    assert solver["reasoning_effort"] == "xhigh"  # extra_body is merged into the body
    assert "显然成立" not in solver["messages"][0]["content"]  # no ground truth to the solver

    assert grader_bodies[0].get("stream") in (None, False)
    assert "显然成立" in grader_bodies[0]["messages"][0]["content"]


def test_cli_runs_the_bundled_examples(endpoint, tmp_path):
    """Exercises the real config loader, the real CLI and the shipped examples."""
    from click.testing import CliRunner

    from imo_qc import cli

    server, base_url = endpoint
    config_file = tmp_path / "imo-qc.yaml"
    config_file.write_text(
        "resistance:\n"
        "  attempts: 1\n"
        "  solver:\n"
        f"    base_url: {base_url}\n"
        "    api_key: k\n"
        "    model: solver-model\n"
        "    max_completion_tokens: 256\n"
        "    timeout_sec: 30\n"
        "  grader:\n"
        "    model: grader-model\n"
        "quality_checks:\n"
        "  model:\n"
        f"    base_url: {base_url}\n"
        "    api_key: k\n"
        "    model: reviewer-model\n"
        "    max_tokens: 512\n"
        "    timeout_sec: 30\n"
        "http_timeout_sec: 60\n"
        "concurrency: 2\n",
        encoding="utf-8",
    )
    out = tmp_path / "results.jsonl"

    result = CliRunner().invoke(
        cli.main,
        [
            "run",
            "examples/problems.jsonl",
            "-o",
            str(out),
            "-c",
            str(config_file),
            "--checks",
            "self_contained,solvability",
        ],
    )

    assert result.exit_code == 0, result.output
    reports = {
        json.loads(line)["problem_id"]: json.loads(line)
        for line in out.read_text(encoding="utf-8").splitlines()
    }
    assert set(reports) == {"imo-1959-p1", "toy-parity", "toy-underspecified"}
    assert reports["imo-1959-p1"]["resistance"]["attempts"][0]["points"] == 4
    # The deliberately under-specified example has no solution at all.
    broken = reports["toy-underspecified"]
    assert broken["resistance"]["skip_reason"] == "missing solution"
    assert broken["quality_checks"]["solvability"]["status"] == "skipped"
    assert broken["quality_checks"]["self_contained"]["status"] == "ok"
