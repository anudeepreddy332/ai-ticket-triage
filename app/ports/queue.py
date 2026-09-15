from typing import Protocol

from app.core.models import SupportTicket


class TicketQueue(Protocol):
    def publish(self, ticket: SupportTicket) -> None:
        """Publish a ticket for asynchronous processing."""
        ...
