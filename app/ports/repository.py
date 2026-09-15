from typing import Protocol

from app.core.models import TriageResult


class TriageResultRepository(Protocol):
    def save(self, result: TriageResult) -> None:
        """Persist a completed triage result."""
        ...

