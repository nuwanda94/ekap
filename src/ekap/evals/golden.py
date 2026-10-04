"""Golden questions and the corpus they are scored against.

Questions use distinctive tokens so the in-memory retriever can be graded
without a model or network call. Each question expects one source id and a
phrase that must appear in the top chunk.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GoldenQuestion:
    """One retrieval case in the golden set."""

    question_id: str
    question: str
    expected_source_id: str
    must_contain: str


CORPUS: tuple[tuple[str, str], ...] = (
    (
        "expense-policy",
        (
            "Expense reports over 500 dollars require manager approval before reimbursement. "
            "Itemized receipts are mandatory for every expense claim."
        ),
    ),
    (
        "travel-policy",
        (
            "Business travel must be booked through the corporate portal. "
            "Airfare is limited to economy class unless a director grants an exception."
        ),
    ),
    (
        "vacation-handbook",
        (
            "Vacation accrues at 1.5 days per month. "
            "Unused vacation may roll over up to 10 days into the next calendar year."
        ),
    ),
    (
        "security-policy",
        (
            "Multi-factor authentication is required for VPN access. "
            "Passwords must be rotated every 90 days and must not be shared."
        ),
    ),
    (
        "retention-policy",
        (
            "Customer records are retained for 7 years. "
            "Deletion requests must be completed within 30 days of verification."
        ),
    ),
    (
        "incident-policy",
        (
            "Priority one incidents page the on-call engineer within 15 minutes. "
            "A written postmortem is due within 48 hours."
        ),
    ),
    (
        "remote-policy",
        (
            "Remote work requires a written agreement. "
            "Core collaboration hours are 10 to 16 in the local office timezone."
        ),
    ),
    (
        "vendor-policy",
        (
            "Vendors must complete a security questionnaire before any system access is granted. "
            "Contracts renew annually."
        ),
    ),
    (
        "review-policy",
        (
            "Production changes require two reviewers. "
            "Emergency hotfixes still need a follow-up review within one business day."
        ),
    ),
    (
        "onboarding-policy",
        (
            "New hires complete security training within 5 days of their start date. "
            "Laptops are issued only after training is recorded."
        ),
    ),
    (
        "benefits-policy",
        (
            "Medical enrollment must be submitted within 30 days of hire. "
            "The employer matches retirement contributions up to 4 percent."
        ),
    ),
)

GOLDEN_QUESTIONS: tuple[GoldenQuestion, ...] = (
    GoldenQuestion(
        "q01",
        "Who must approve expense reports over 500 dollars?",
        "expense-policy",
        "manager approval",
    ),
    GoldenQuestion(
        "q02",
        "Are itemized receipts mandatory for an expense claim?",
        "expense-policy",
        "itemized receipts",
    ),
    GoldenQuestion(
        "q03",
        "How is business travel booked in the corporate portal?",
        "travel-policy",
        "corporate portal",
    ),
    GoldenQuestion(
        "q04",
        "When is airfare limited to economy class?",
        "travel-policy",
        "economy class",
    ),
    GoldenQuestion(
        "q05",
        "How fast does vacation accrue each month?",
        "vacation-handbook",
        "1.5 days",
    ),
    GoldenQuestion(
        "q06",
        "How many unused vacation days may roll over?",
        "vacation-handbook",
        "10 days",
    ),
    GoldenQuestion(
        "q07",
        "Is multi-factor authentication required for VPN access?",
        "security-policy",
        "VPN access",
    ),
    GoldenQuestion(
        "q08",
        "How often must passwords be rotated?",
        "security-policy",
        "90 days",
    ),
    GoldenQuestion(
        "q09",
        "How long are customer records retained?",
        "retention-policy",
        "7 years",
    ),
    GoldenQuestion(
        "q10",
        "When must deletion requests be completed?",
        "retention-policy",
        "deletion requests",
    ),
    GoldenQuestion(
        "q11",
        "How quickly do priority one incidents page on-call?",
        "incident-policy",
        "15 minutes",
    ),
    GoldenQuestion(
        "q12",
        "When is a written postmortem due?",
        "incident-policy",
        "postmortem",
    ),
    GoldenQuestion(
        "q13",
        "Does remote work require a written agreement?",
        "remote-policy",
        "written agreement",
    ),
    GoldenQuestion(
        "q14",
        "What are the core collaboration hours?",
        "remote-policy",
        "collaboration hours",
    ),
    GoldenQuestion(
        "q15",
        "What must vendors complete before system access?",
        "vendor-policy",
        "security questionnaire",
    ),
    GoldenQuestion(
        "q16",
        "How often do vendor contracts renew?",
        "vendor-policy",
        "renew annually",
    ),
    GoldenQuestion(
        "q17",
        "How many reviewers do production changes require?",
        "review-policy",
        "two reviewers",
    ),
    GoldenQuestion(
        "q18",
        "Do emergency hotfixes need a follow-up review?",
        "review-policy",
        "hotfixes",
    ),
    GoldenQuestion(
        "q19",
        "When must new hires complete security training?",
        "onboarding-policy",
        "security training",
    ),
    GoldenQuestion(
        "q20",
        "When are laptops issued to new hires?",
        "onboarding-policy",
        "laptops",
    ),
    GoldenQuestion(
        "q21",
        "When must medical enrollment be submitted after hire?",
        "benefits-policy",
        "medical enrollment",
    ),
    GoldenQuestion(
        "q22",
        "How much does the employer match retirement contributions?",
        "benefits-policy",
        "4 percent",
    ),
)
