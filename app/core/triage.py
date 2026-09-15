from app.core.models import SupportTicket, TriageResult
from app.ports.llm import TicketClassifier
from app.ports.repository import TriageResultRepository


class TriageService:
    def __init__(
            self,
            classifier: TicketClassifier,
            repository: TriageResultRepository,
    ) -> None:
        self._classifier = classifier
        self._repository = repository

    def triage(self, ticket: SupportTicket) -> TriageResult:
        result = self._classifier.classify(ticket)

        if result.ticket_id != ticket.ticket_id:
            raise ValueError(
                "classifier returned a result for a different ticket"
            )
        self._repository.save(result)

        return result
