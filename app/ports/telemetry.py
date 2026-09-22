from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ClassificationTelemetry:
    outcome: str
    stop_reason: str | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    model_latency_ms: int | None
    adapter_latency_ms: int


class ClassificationTelemetrySink(Protocol):
    def record(
        self,
        telemetry: ClassificationTelemetry,
    ) -> None:
        """Record one classifier invocation's operational metadata."""
        ...
