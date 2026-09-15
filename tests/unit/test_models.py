import pytest

from app.core.models import (
    RoutingDestination,
    Severity,
    SupportTicket,
    TicketCategory,
    TriageResult,
)


def test_support_ticket_accepts_valid_data() -> None:
    ticket = SupportTicket(
        ticket_id="ticket-123",
        customer_id="customer-456",
        subject="payment failure",
        body="My payment has failed three times.",
    )
    assert ticket.ticket_id == "ticket-123"
    assert ticket.customer_id == "customer-456"

@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ticket_id", ""),
        ("subject", ""),
        ("body", ""),
    ],
)

def test_support_ticket_rejects_required_empty_fields(
    field: str,
    value: str,
) -> None:
    data = {
        "ticket_id": "ticket-123",
        "subject": "Payment failure",
        "body": "Payment failed.",
    }

    data[field] = value

    with pytest.raises(ValueError):
        SupportTicket(**data)

def test_triage_result_accepts_valid_data() -> None:
    result = TriageResult(
        ticket_id="ticket-123",
        severity=Severity.P1,
        category=TicketCategory.PAYMENTS,
        summary="Payments are failing.",
        route_to=RoutingDestination.PAYMENTS_ONCALL,
        confidence=0.96,
    )

    assert result.severity == Severity.P1
    assert result.confidence == 0.96


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_triage_result_rejects_invalid_confidence(
    confidence: float,
) -> None:
    with pytest.raises(ValueError):
        TriageResult(
            ticket_id="ticket-123",
            severity=Severity.P2,
            category=TicketCategory.BILLING,
            summary="Billing issue.",
            route_to=RoutingDestination.BILLING_SUPPORT,
            confidence=confidence,
        )
