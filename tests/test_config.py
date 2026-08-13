import pytest
from pydantic import ValidationError

from imo_qc.config import ModelConfig, config_from_dict, load_config

BASE = {
    "resistance": {
        "attempts": 2,
        "solver": {
            "base_url": "http://localhost/v1",
            "api_key": "${TEST_KEY}",
            "model": "solver-model",
            "max_completion_tokens": 1024,
            "timeout_sec": 300,
            "extra_body": {"reasoning_effort": "xhigh"},
        },
        "grader": {"model": "grader-model"},
    },
    "quality_checks": {
        "model": {"base_url": "http://localhost/v1", "model": "reviewer", "timeout_sec": 120}
    },
    "http_timeout_sec": 3600,
}


def test_env_expansion(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "secret-value")
    config = config_from_dict(BASE)
    assert config.resistance.solver.api_key == "secret-value"


def test_missing_env_is_an_error(monkeypatch):
    monkeypatch.delenv("TEST_KEY", raising=False)
    with pytest.raises(ValueError, match="TEST_KEY"):
        config_from_dict(BASE)


def test_grader_inherits_solver_call_parameters(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "k")
    grader = config_from_dict(BASE).resistance.grader
    # Its own model, but the solver's heavy call parameters -- a grader with a
    # small token budget gets truncated and scores low.
    assert grader.model == "grader-model"
    assert grader.max_completion_tokens == 1024
    assert grader.timeout_sec == 300
    assert grader.extra_body == {"reasoning_effort": "xhigh"}
    assert grader.base_url == "http://localhost/v1"


def test_http_timeout_must_exceed_per_call_timeout(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "k")
    broken = {**BASE, "http_timeout_sec": 120}
    with pytest.raises(ValidationError, match="http_timeout_sec"):
        config_from_dict(broken)


def test_token_limits_are_mutually_exclusive():
    with pytest.raises(ValidationError, match="max_tokens or max_completion_tokens"):
        ModelConfig(base_url="u", model="m", max_tokens=10, max_completion_tokens=10)


def test_model_is_required():
    with pytest.raises(ValidationError):
        ModelConfig(base_url="u")


@pytest.mark.parametrize("value", [0, 11])
def test_early_stop_range(monkeypatch, value):
    monkeypatch.setenv("TEST_KEY", "k")
    raw = {**BASE}
    raw["resistance"] = {**BASE["resistance"], "early_stop_at": value}
    with pytest.raises(ValidationError, match="early_stop_at"):
        config_from_dict(raw)


def test_attempts_must_be_positive(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "k")
    raw = {**BASE}
    raw["resistance"] = {**BASE["resistance"], "attempts": 0}
    with pytest.raises(ValidationError, match="attempts"):
        config_from_dict(raw)


def test_example_config_is_valid(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("GRADER_API_KEY", "y")
    config = load_config("imo-qc.example.yaml")
    assert config.resistance is not None
    assert config.quality_checks is not None
