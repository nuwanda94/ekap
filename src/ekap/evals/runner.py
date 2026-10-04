"""Score golden questions against the in-memory retriever.

The runner is pure Python: it builds the bundled corpus, retrieves the top
chunk for each question, and fails closed when the source or required phrase
is wrong. ``main`` returns a process exit code so CI can invoke it later.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass

from ekap.evals.golden import CORPUS, GOLDEN_QUESTIONS, GoldenQuestion
from ekap.rag import Ingester, Retriever

_DEFAULT_MIN_PASS_RATE = 1.0


@dataclass(frozen=True, slots=True)
class CaseResult:
    """Outcome of scoring one golden question."""

    question_id: str
    passed: bool
    expected_source_id: str
    actual_source_id: str | None
    detail: str


@dataclass(frozen=True, slots=True)
class EvalReport:
    """Aggregate score for a golden-set run."""

    total: int
    passed: int
    failed: int
    min_pass_rate: float
    results: tuple[CaseResult, ...]

    @property
    def pass_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.passed / self.total

    @property
    def ok(self) -> bool:
        return self.total > 0 and self.failed == 0 and self.pass_rate >= self.min_pass_rate


def build_retriever() -> Retriever:
    """Ingest the bundled corpus. No network calls."""
    ingester = Ingester()
    for source_id, text in CORPUS:
        ingester.add(source_id, text)
    return Retriever(ingester)


class EvalRunner:
    """Score golden questions by top-1 source id and a required phrase."""

    def __init__(
        self,
        retriever: Retriever | None = None,
        *,
        min_pass_rate: float = _DEFAULT_MIN_PASS_RATE,
    ) -> None:
        if not 0.0 <= min_pass_rate <= 1.0:
            raise ValueError("min_pass_rate must be between 0 and 1")
        self._retriever = retriever if retriever is not None else build_retriever()
        self.min_pass_rate = min_pass_rate

    def score(self, question: GoldenQuestion) -> CaseResult:
        hits = self._retriever.query(question.question, top_k=1)
        if not hits:
            return CaseResult(
                question_id=question.question_id,
                passed=False,
                expected_source_id=question.expected_source_id,
                actual_source_id=None,
                detail="no hit",
            )
        hit = hits[0]
        actual = hit.chunk.source_id
        phrase = question.must_contain.lower()
        contains = phrase in hit.chunk.text.lower()
        source_ok = actual == question.expected_source_id
        passed = source_ok and contains
        if passed:
            detail = "ok"
        elif not source_ok:
            detail = f"source {actual} != {question.expected_source_id}"
        else:
            detail = f"missing phrase {question.must_contain!r}"
        return CaseResult(
            question_id=question.question_id,
            passed=passed,
            expected_source_id=question.expected_source_id,
            actual_source_id=actual,
            detail=detail,
        )

    def run(self, questions: Sequence[GoldenQuestion] | None = None) -> EvalReport:
        selected = tuple(questions if questions is not None else GOLDEN_QUESTIONS)
        results = tuple(self.score(question) for question in selected)
        passed = sum(1 for result in results if result.passed)
        return EvalReport(
            total=len(results),
            passed=passed,
            failed=len(results) - passed,
            min_pass_rate=self.min_pass_rate,
            results=results,
        )


def format_report(report: EvalReport) -> str:
    lines = [
        (
            f"golden {report.passed}/{report.total} passed "
            f"(rate={report.pass_rate:.0%}, min={report.min_pass_rate:.0%})"
        )
    ]
    for result in report.results:
        if not result.passed:
            lines.append(f"FAIL {result.question_id}: {result.detail}")
    lines.append("ok" if report.ok else "regression")
    return "\n".join(lines)


def run_eval(
    questions: Sequence[GoldenQuestion] | None = None,
    retriever: Retriever | None = None,
    *,
    min_pass_rate: float = _DEFAULT_MIN_PASS_RATE,
) -> int:
    """Score questions and return 0 on pass, 1 on regression."""
    report = EvalRunner(retriever, min_pass_rate=min_pass_rate).run(questions)
    print(format_report(report))
    return 0 if report.ok else 1


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry. Exits non-zero when the golden set regresses."""
    parser = argparse.ArgumentParser(description="Score the EKAP golden retrieval set")
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        default=_DEFAULT_MIN_PASS_RATE,
        help="Minimum pass rate required to exit 0 (default: 1.0)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    return run_eval(min_pass_rate=args.min_pass_rate)


if __name__ == "__main__":
    sys.exit(main())
