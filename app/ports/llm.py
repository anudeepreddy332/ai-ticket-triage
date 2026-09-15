from typing import Protocol

from app.core.models import SupportTicket, TriageResult


class TicketClassifier(Protocol):
    def classify(self, ticket: SupportTicket) -> TriageResult:
        """Classify a support ticket into a structured triage result."""
        ...
