from app.core.models import (
    RoutingDestination,
    Severity,
    SupportTicket,
    TicketCategory,
    TriageResult,
)
from app.core.triage import TriageService


class FakeClassifier:
    def __init__(self, result: TriageResult) -> None:
        self.result = result
        self.received_ticket: SupportTicket | None = None

    def classify(self, ticket: SupportTicket) -> TriageResult:
        self.received_ticket = ticket
        return self.result


class FakeRepository:
    def __init__(self) -> None:
        self.saved_results: list[TriageResult] = []

    def save(self, result: TriageResult) -> None:
        self.saved_results.append(result)


def test_triage_classifies_and_persists_ticket() -> None:
    ticket = SupportTicket(
        ticket_id="ticket-123",
        customer_id="customer-456",
        subject="Payments failing",
        body="Our checkout has returned 500 errors for fifteen minutes.",
    )

    expected_result = TriageResult(
        ticket_id="ticket-123",
        severity=Severity.P1,
        category=TicketCategory.PAYMENTS,
        summary="Payment processing is failing.",
        route_to=RoutingDestination.PAYMENTS_ONCALL,
        confidence=0.97,
    )

    classifier = FakeClassifier(expected_result)
    repository = FakeRepository()

    service = TriageService(
        classifier=classifier,
        repository=repository,
    )

    result = service.triage(ticket)

    assert classifier.received_ticket == ticket
    assert result == expected_result
    assert repository.saved_results == [expected_result]


def test_triage_rejects_result_for_wrong_ticket() -> None:
    ticket = SupportTicket(
        ticket_id="ticket-123",
        subject="Billing issue",
        body="I was charged twice.",
    )

    wrong_result = TriageResult(
        ticket_id="ticket-999",
        severity=Severity.P2,
        category=TicketCategory.BILLING,
        summary="Duplicate billing charge.",
        route_to=RoutingDestination.BILLING_SUPPORT,
        confidence=0.92,
    )

    classifier = FakeClassifier(wrong_result)
    repository = FakeRepository()

    service = TriageService(
        classifier=classifier,
        repository=repository,
    )

    try:
        service.triage(ticket)
    except ValueError as exc:
        assert str(exc) == "classifier returned a result for a different ticket"
    else:
        raise AssertionError("Expected ValueError")

    assert repository.saved_results == []