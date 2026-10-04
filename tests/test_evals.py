"""Unit tests for the golden-set runner."""

from ekap.evals import (
    GOLDEN_QUESTIONS,
    EvalRunner,
    build_retriever,
    run_eval,
)
from ekap.evals.golden import CORPUS
from ekap.rag import Ingester, Retriever


def test_golden_set_has_at_least_twenty_questions() -> None:
    assert len(GOLDEN_QUESTIONS) >= 20
    assert len({question.question_id for question in GOLDEN_QUESTIONS}) == len(GOLDEN_QUESTIONS)
    assert {question.expected_source_id for question in GOLDEN_QUESTIONS} <= {
        source_id for source_id, _text in CORPUS
    }


def test_subset_of_golden_set_passes() -> None:
    subset = GOLDEN_QUESTIONS[:5]
    report = EvalRunner(build_retriever()).run(subset)
    assert report.total == 5
    assert report.passed == 5
    assert report.ok
    assert all(result.actual_source_id == result.expected_source_id for result in report.results)


def test_full_golden_set_passes() -> None:
    report = EvalRunner().run()
    assert report.total >= 20
    assert report.failed == 0
    assert report.ok


def test_regression_returns_nonzero_exit() -> None:
    ingester = Ingester()
    ingester.add("unrelated", "Cafeteria menu lists soup on Tuesdays.")
    code = run_eval(GOLDEN_QUESTIONS[:3], Retriever(ingester))
    assert code == 1


def test_passing_run_returns_zero_exit() -> None:
    assert run_eval(GOLDEN_QUESTIONS[:2], build_retriever()) == 0
