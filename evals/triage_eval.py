import json
import time
from dataclasses import dataclass
from pathlib import Path

from app.core.models import (
    RoutingDestination,
    Severity,
    SupportTicket,
    TicketCategory,
)
from app.ports.errors import ClassificationError
from app.ports.llm import TicketClassifier


@dataclass(frozen=True, slots=True)
class EvalCase:
    ticket: SupportTicket
    expected_severity: Severity
    expected_category: TicketCategory
    expected_route: RoutingDestination


@dataclass(frozen=True, slots=True)
class EvalReport:
    total: int
    completed: int
    failures: int
    severity_correct: int
    category_correct: int
    route_correct: int
    p1_total: int
    p1_correct: int
    latencies_ms: tuple[int, ...]

    @property
    def severity_accuracy(self) -> float:
        return self._ratio(self.severity_correct, self.total)

    @property
    def category_accuracy(self) -> float:
        return self._ratio(self.category_correct, self.total)

    @property
    def route_accuracy(self) -> float:
        return self._ratio(self.route_correct, self.total)

    @property
    def failure_rate(self) -> float:
        return self._ratio(self.failures, self.total)

    @property
    def p1_recall(self) -> float:
        return self._ratio(self.p1_correct, self.p1_total)

    @property
    def average_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0

        return sum(self.latencies_ms) / len(self.latencies_ms)

    @staticmethod
    def _ratio(numerator: int, denominator: int) -> float:
        if denominator == 0:
            return 0.0

        return numerator / denominator


def load_eval_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                payload = json.loads(line)

                cases.append(
                    EvalCase(
                        ticket=SupportTicket(
                            ticket_id=payload["ticket_id"],
                            subject=payload["subject"],
                            body=payload["body"],
                        ),
                        expected_severity=Severity(
                            payload["expected_severity"]
                        ),
                        expected_category=TicketCategory(
                            payload["expected_category"]
                        ),
                        expected_route=RoutingDestination(
                            payload["expected_route"]
                        ),
                    )
                )

            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid eval case on line {line_number}"
                ) from exc

    if not cases:
        raise ValueError("Eval dataset contains no cases")

    return cases


def run_eval(
    classifier: TicketClassifier,
    cases: list[EvalCase],
) -> EvalReport:
    severity_correct = 0
    category_correct = 0
    route_correct = 0
    failures = 0
    completed = 0

    p1_total = sum(
        case.expected_severity == Severity.P1
        for case in cases
    )
    p1_correct = 0

    latencies: list[int] = []

    for case in cases:
        started_at = time.perf_counter()

        try:
            result = classifier.classify(case.ticket)

        except ClassificationError:
            failures += 1
            continue

        elapsed_ms = round(
            (time.perf_counter() - started_at) * 1000
        )

        latencies.append(elapsed_ms)
        completed += 1

        if result.severity == case.expected_severity:
            severity_correct += 1

            if case.expected_severity == Severity.P1:
                p1_correct += 1

        if result.category == case.expected_category:
            category_correct += 1

        if result.route_to == case.expected_route:
            route_correct += 1

    return EvalReport(
        total=len(cases),
        completed=completed,
        failures=failures,
        severity_correct=severity_correct,
        category_correct=category_correct,
        route_correct=route_correct,
        p1_total=p1_total,
        p1_correct=p1_correct,
        latencies_ms=tuple(latencies),
    )