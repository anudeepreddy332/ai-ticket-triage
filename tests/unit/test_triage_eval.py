from app.core.models import (
    RoutingDestination,
    Severity,
    SupportTicket,
    TicketCategory,
    TriageResult,
)
from app.ports.errors import ClassifierUnavailableError
from evals.triage_eval import EvalCase, run_eval


class FakeEvalClassifier:
    def __init__(
        self,
        results: dict[str, TriageResult],
        failing_ids: set[str] | None = None,
    ) -> None:
        self._results = results
        self._failing_ids = failing_ids or set()

    def classify(
        self,
        ticket: SupportTicket,
    ) -> TriageResult:
        if ticket.ticket_id in self._failing_ids:
            raise ClassifierUnavailableError(
                "classifier unavailable"
            )

        return self._results[ticket.ticket_id]


def make_case(
    ticket_id: str,
    severity: Severity,
    category: TicketCategory,
    route: RoutingDestination,
) -> EvalCase:
    return EvalCase(
        ticket=SupportTicket(
            ticket_id=ticket_id,
            subject="Test ticket",
            body="Test ticket body",
        ),
        expected_severity=severity,
        expected_category=category,
        expected_route=route,
    )


def test_eval_calculates_quality_metrics() -> None:
    cases = [
        make_case(
            "ticket-1",
            Severity.P1,
            TicketCategory.PAYMENTS,
            RoutingDestination.PAYMENTS_ONCALL,
        ),
        make_case(
            "ticket-2",
            Severity.P3,
            TicketCategory.BILLING,
            RoutingDestination.BILLING_SUPPORT,
        ),
    ]

    classifier = FakeEvalClassifier(
        results={
            "ticket-1": TriageResult(
                ticket_id="ticket-1",
                severity=Severity.P1,
                category=TicketCategory.PAYMENTS,
                summary="Payment outage.",
                route_to=RoutingDestination.PAYMENTS_ONCALL,
                confidence=0.95,
            ),
            "ticket-2": TriageResult(
                ticket_id="ticket-2",
                severity=Severity.P2,
                category=TicketCategory.BILLING,
                summary="Billing issue.",
                route_to=RoutingDestination.BILLING_SUPPORT,
                confidence=0.8,
            ),
        }
    )

    report = run_eval(classifier, cases)

    assert report.total == 2
    assert report.completed == 2
    assert report.failures == 0

    assert report.severity_accuracy == 0.5
    assert report.category_accuracy == 1.0
    assert report.route_accuracy == 1.0
    assert report.p1_recall == 1.0


def test_eval_counts_classifier_failure_and_continues() -> None:
    cases = [
        make_case(
            "ticket-1",
            Severity.P1,
            TicketCategory.PAYMENTS,
            RoutingDestination.PAYMENTS_ONCALL,
        ),
        make_case(
            "ticket-2",
            Severity.P3,
            TicketCategory.BILLING,
            RoutingDestination.BILLING_SUPPORT,
        ),
    ]

    classifier = FakeEvalClassifier(
        results={
            "ticket-1": TriageResult(
                ticket_id="ticket-1",
                severity=Severity.P1,
                category=TicketCategory.PAYMENTS,
                summary="Payment outage.",
                route_to=RoutingDestination.PAYMENTS_ONCALL,
                confidence=0.95,
            ),
        },
        failing_ids={"ticket-2"},
    )

    report = run_eval(classifier, cases)

    assert report.total == 2
    assert report.completed == 1
    assert report.failures == 1
    assert report.failure_rate == 0.5