from dataclasses import dataclass
from enum import StrEnum


class Severity(StrEnum):
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"

class TicketCategory(StrEnum):
    PAYMENTS = "payments"
    BILLING = "billing"
    ACCOUNT = "account"
    TECHNICAL = "technical"
    OTHER = "other"


class RoutingDestination(StrEnum):
    PAYMENTS_ONCALL = "payments-oncall"
    BILLING_SUPPORT = "billing-support"
    ACCOUNT_SUPPORT = "account-support"
    TECH_SUPPORT = "technical-support"
    GENERAL_SUPPORT = "general-support"

@dataclass(frozen=True, slots=True)
class SupportTicket:
    ticket_id: str
    subject: str
    body: str
    customer_id: str | None = None

    def __post_init__(self) -> None:
        if not self.ticket_id.strip():
            raise ValueError("ticket_id must not be empty")

        if not self.subject.strip():
            raise ValueError("subject must not be empty")

        if not self.body.strip():
            raise ValueError("body must not be empty")

@dataclass(frozen=True, slots=True)
class TriageResult:
    ticket_id: str
    severity: Severity
    category: TicketCategory
    summary: str
    route_to: RoutingDestination
    confidence: float

    def __post_init__(self) -> None:
        if not self.ticket_id.strip():
            raise ValueError("ticket_id must not be empty")

        if not self.summary.strip():
            raise ValueError("summary must not be empty")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")