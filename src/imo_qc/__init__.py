"""AI-resistance and quality checks for olympiad math problems."""

from .config import (
    Config,
    ModelConfig,
    QualityChecksConfig,
    ResistanceConfig,
    RetryConfig,
    RetryPolicy,
    config_from_dict,
    load_config,
)
from .llm import LLMClient, SemanticError, TransportError
from .models import (
    MAX_POINTS,
    Attempt,
    CheckResult,
    GroupResult,
    Problem,
    Report,
    ResistanceResult,
    Rubric,
    Solution,
    TokenUsage,
    Usage,
)
from .qc import QC
from .registry import ALL_CHECKS, CHECKS, SEVEN_CHECKS

__version__ = "0.1.0"

__all__ = [
    "ALL_CHECKS",
    "CHECKS",
    "MAX_POINTS",
    "SEVEN_CHECKS",
    "Attempt",
    "CheckResult",
    "Config",
    "GroupResult",
    "LLMClient",
    "ModelConfig",
    "Problem",
    "QC",
    "QualityChecksConfig",
    "Report",
    "ResistanceConfig",
    "ResistanceResult",
    "RetryConfig",
    "RetryPolicy",
    "Rubric",
    "SemanticError",
    "Solution",
    "TokenUsage",
    "TransportError",
    "Usage",
    "config_from_dict",
    "load_config",
]
