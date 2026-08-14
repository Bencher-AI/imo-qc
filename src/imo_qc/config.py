"""Configuration: any OpenAI-compatible endpoint, one section per role."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, model_validator

from .models import MAX_POINTS

_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ModelConfig(BaseModel):
    """One endpoint.

    ``model`` has no default on purpose: a wrong-but-plausible default would be
    charged to the user silently.
    """

    base_url: str
    api_key: str = ""
    model: str
    max_tokens: Optional[int] = None
    # o-series style endpoints reject max_tokens and want this instead; the two
    # are mutually exclusive.
    max_completion_tokens: Optional[int] = None
    #: Sent only when set, so an endpoint's own default applies otherwise. Worth
    #: setting for the solver: with greedy decoding, repeated attempts return
    #: near-identical answers and cost N times as much for one sample.
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    #: Where supported, fixes sampling for reproducible runs.
    seed: Optional[int] = None
    timeout_sec: float = 120.0
    stream: bool = False
    extra_body: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _one_token_limit(self) -> "ModelConfig":
        if self.max_tokens is not None and self.max_completion_tokens is not None:
            raise ValueError("set either max_tokens or max_completion_tokens, not both")
        return self


class RetryPolicy(BaseModel):
    """``max_attempts`` counts the first call, so 3 means at most 2 retries."""

    max_attempts: int = 3
    base_backoff_ms: int = 200


class RetryConfig(BaseModel):
    #: Transport failures: 429, 5xx, network errors, truncated streams.
    transport: RetryPolicy = RetryPolicy(max_attempts=3, base_backoff_ms=200)
    #: Answers that arrived but are unusable: empty content, no JSON object, no
    #: <points> block, out-of-range score. This is the layer that matters most in
    #: practice -- an empty completion returned with finish_reason=stop is the
    #: single most common judge failure.
    semantic: RetryPolicy = RetryPolicy(max_attempts=6, base_backoff_ms=2000)


class ResistanceConfig(BaseModel):
    attempts: int = 3
    #: Stop the remaining attempts once one scores at least this. Off by default
    #: (the report stays purely diagnostic). Note it only saves anything on
    #: problems the solver actually cracks.
    early_stop_at: Optional[int] = None
    solver: ModelConfig
    #: Fields left unset here are inherited from ``solver`` -- a grader with a
    #: small token limit gets truncated mid-verification and scores low, which
    #: reads as "the problem resisted the AI".
    grader: ModelConfig

    @model_validator(mode="after")
    def _check(self) -> "ResistanceConfig":
        if self.attempts < 1:
            raise ValueError("resistance.attempts must be >= 1")
        if self.early_stop_at is not None and not (1 <= self.early_stop_at <= MAX_POINTS):
            raise ValueError(f"resistance.early_stop_at must be within 1..{MAX_POINTS}")
        return self


class QualityChecksConfig(BaseModel):
    model: ModelConfig


class Config(BaseModel):
    resistance: Optional[ResistanceConfig] = None
    quality_checks: Optional[QualityChecksConfig] = None
    #: Directory of replacement prompt files. Any name found here wins over the
    #: bundled one, so a translated set can live with your experiment instead of
    #: inside site-packages.
    prompts_dir: Optional[Path] = None
    #: HTTP-level ceiling. Must exceed every per-call timeout, otherwise a call
    #: can be cut off by the transport before its own deadline is reached.
    http_timeout_sec: float = 3600.0
    #: Problems in flight (used by the CLI).
    concurrency: int = 8
    #: Hard ceiling on concurrent LLM calls. Separate from ``concurrency``
    #: because one problem can issue many calls at once.
    max_inflight_calls: int = 10
    retry: RetryConfig = Field(default_factory=RetryConfig)

    @model_validator(mode="after")
    def _timeout_layers(self) -> "Config":
        per_call = []
        if self.resistance:
            per_call += [self.resistance.solver.timeout_sec, self.resistance.grader.timeout_sec]
        if self.quality_checks:
            per_call.append(self.quality_checks.model.timeout_sec)
        if per_call and self.http_timeout_sec <= max(per_call):
            raise ValueError(
                f"http_timeout_sec ({self.http_timeout_sec}) must be greater than the "
                f"largest per-call timeout_sec ({max(per_call)})"
            )
        if self.max_inflight_calls < 1 or self.concurrency < 1:
            raise ValueError("concurrency and max_inflight_calls must be >= 1")
        return self


def expand_env(value: Any) -> Any:
    """Replace ``${VAR}`` with the environment value, erroring when unset."""
    if isinstance(value, str):

        def sub(m: re.Match[str]) -> str:
            name = m.group(1)
            if name not in os.environ:
                raise ValueError(f"environment variable {name} is referenced but not set")
            return os.environ[name]

        return _ENV_REF.sub(sub, value)
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    return value


def _inherit(child: dict[str, Any], parent: dict[str, Any]) -> dict[str, Any]:
    merged = dict(parent)
    merged.update({k: v for k, v in child.items() if v is not None})
    return merged


def config_from_dict(raw: dict[str, Any]) -> Config:
    raw = expand_env(raw)
    resistance = raw.get("resistance")
    if isinstance(resistance, dict):
        solver = resistance.get("solver") or {}
        grader = resistance.get("grader") or {}
        resistance = dict(resistance)
        resistance["grader"] = _inherit(grader, solver)
    if resistance is not None:
        raw = {**raw, "resistance": resistance}
    return Config.model_validate(raw)


def load_config(path: str | Path) -> Config:
    with open(path, encoding="utf-8") as f:
        return config_from_dict(yaml.safe_load(f) or {})
