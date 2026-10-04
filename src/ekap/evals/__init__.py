"""Evaluation harness and golden set."""

from ekap.evals.golden import CORPUS, GOLDEN_QUESTIONS, GoldenQuestion
from ekap.evals.runner import (
    CaseResult,
    EvalReport,
    EvalRunner,
    build_retriever,
    main,
    run_eval,
)

__all__ = [
    "CORPUS",
    "GOLDEN_QUESTIONS",
    "CaseResult",
    "EvalReport",
    "EvalRunner",
    "GoldenQuestion",
    "build_retriever",
    "main",
    "run_eval",
]
