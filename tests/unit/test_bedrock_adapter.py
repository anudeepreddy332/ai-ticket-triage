import json
import logging
from typing import Any

import pytest
from botocore.exceptions import ClientError

from app.adapters.aws.bedrock import BedrockTicketClassifier
from app.core.models import (
    RoutingDestination,
    Severity,
    SupportTicket,
    TicketCategory,
)
from app.ports.errors import (
    ClassificationBlockedError,
    ClassifierResponseError,
    ClassifierUnavailableError,
)


class FakeBedrockClient:
    def __init__(
        self,
        *,
        response: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)

        if self.error is not None:
            raise self.error

        if self.response is None:
            raise AssertionError("Fake Bedrock response was not configured")

        return self.response


def make_ticket() -> SupportTicket:
    return SupportTicket(
        ticket_id="ticket-123",
        customer_id="customer-secret-456",
        subject="Payment outage",
        body="Checkout has returned 500 errors for fifteen minutes.",
    )


def test_bedrock_classifier_returns_structured_result() -> None:
    client = FakeBedrockClient(
        response={
            "stopReason": "end_turn",
            "output": {
                "message": {
                    "content": [
                        {
                            "reasoningContent": {
                                "reasoningText": {
                                    "text": "internal reasoning"
                                }
                            }
                        },
                        {
                            "text": json.dumps(
                                {
                                    "severity": "P1",
                                    "category": "payments",
                                    "summary": "Checkout payments are failing.",
                                    "confidence": 0.97,
                                }
                            )
                        },
                    ]
                }
            }
        }
    )

    classifier = BedrockTicketClassifier(
        client=client,
        model_id="openai.gpt-oss-120b-1:0",
    )

    result = classifier.classify(make_ticket())

    assert result.ticket_id == "ticket-123"
    assert result.severity == Severity.P1
    assert result.category == TicketCategory.PAYMENTS
    assert result.route_to == RoutingDestination.PAYMENTS_ONCALL
    assert result.confidence == 0.97


def test_bedrock_classifier_does_not_send_customer_id() -> None:
    client = FakeBedrockClient(
        response={
            "stopReason": "end_turn",
            "output": {
                "message": {
                    "content": [
                        {
                            "text": json.dumps(
                                {
                                    "severity": "P3",
                                    "category": "technical",
                                    "summary": "Technical issue.",
                                    "confidence": 0.8,
                                }
                            )
                        }
                    ]
                }
            }
        }
    )

    classifier = BedrockTicketClassifier(
        client=client,
        model_id="openai.gpt-oss-120b-1:0",
    )

    classifier.classify(make_ticket())

    serialized_request = json.dumps(client.calls[0])

    assert "customer-secret-456" not in serialized_request
    assert "outputConfig" in client.calls[0]


def test_bedrock_classifier_translates_aws_failure() -> None:
    aws_error = ClientError(
        {
            "Error": {
                "Code": "ThrottlingException",
                "Message": "Rate exceeded",
            }
        },
        "Converse",
    )

    classifier = BedrockTicketClassifier(
        client=FakeBedrockClient(error=aws_error),
        model_id="openai.gpt-oss-120b-1:0",
    )

    with pytest.raises(ClassifierUnavailableError):
        classifier.classify(make_ticket())


def test_bedrock_classifier_rejects_malformed_output() -> None:
    classifier = BedrockTicketClassifier(
        client=FakeBedrockClient(
            response={
                "output": {
                    "message": {
                        "content": [
                            {
                                "text": "not valid JSON",
                            }
                        ]
                    }
                }
            }
        ),
        model_id="openai.gpt-oss-120b-1:0",
    )

    with pytest.raises(ClassifierResponseError):
        classifier.classify(make_ticket())


def test_bedrock_classifier_rejects_invalid_domain_values() -> None:
    classifier = BedrockTicketClassifier(
        client=FakeBedrockClient(
            response={
                "output": {
                    "message": {
                        "content": [
                            {
                                "text": json.dumps(
                                    {
                                        "severity": "P0",
                                        "category": "payments",
                                        "summary": "Invalid severity.",
                                        "confidence": 1.4,
                                    }
                                )
                            }
                        ]
                    }
                }
            }
        ),
        model_id="openai.gpt-oss-120b-1:0",
    )

    with pytest.raises(ClassifierResponseError):
        classifier.classify(make_ticket())

def test_bedrock_classifier_adds_guardrail_configuration() -> None:
    client = FakeBedrockClient(
        response={
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": 20,
                "outputTokens": 10,
                "totalTokens": 30,
            },
            "metrics": {
                "latencyMs": 100,
            },
            "output": {
                "message": {
                    "content": [
                        {
                            "text": json.dumps(
                                {
                                    "severity": "P3",
                                    "category": "technical",
                                    "summary": "Technical issue.",
                                    "confidence": 0.8,
                                }
                            )
                        }
                    ]
                }
            },
        }
    )

    classifier = BedrockTicketClassifier(
        client=client,
        model_id="openai.gpt-oss-120b-1:0",
        guardrail_id="guardrail-123",
        guardrail_version="1",
    )

    classifier.classify(make_ticket())

    assert client.calls[0]["guardrailConfig"] == {
        "guardrailIdentifier": "guardrail-123",
        "guardrailVersion": "1",
        "trace": "enabled",
    }


@pytest.mark.parametrize(
    ("guardrail_id", "guardrail_version"),
    [
        ("guardrail-123", None),
        (None, "1"),
    ],
)
def test_bedrock_classifier_rejects_incomplete_guardrail_configuration(
    guardrail_id: str | None,
    guardrail_version: str | None,
) -> None:
    with pytest.raises(ValueError):
        BedrockTicketClassifier(
            client=FakeBedrockClient(),
            model_id="openai.gpt-oss-120b-1:0",
            guardrail_id=guardrail_id,
            guardrail_version=guardrail_version,
        )


def test_bedrock_classifier_raises_when_guardrail_intervenes() -> None:
    classifier = BedrockTicketClassifier(
        client=FakeBedrockClient(
            response={
                "stopReason": "guardrail_intervened",
                "usage": {
                    "inputTokens": 0,
                    "outputTokens": 0,
                    "totalTokens": 0,
                },
                "metrics": {
                    "latencyMs": 50,
                },
                "output": {
                    "message": {
                        "content": [
                            {
                                "text": "Request blocked.",
                            }
                        ]
                    }
                },
            }
        ),
        model_id="openai.gpt-oss-120b-1:0",
    )

    with pytest.raises(ClassificationBlockedError):
        classifier.classify(make_ticket())


def test_bedrock_classifier_logs_safe_operational_metadata(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = FakeBedrockClient(
        response={
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": 20,
                "outputTokens": 10,
                "totalTokens": 30,
            },
            "metrics": {
                "latencyMs": 100,
            },
            "output": {
                "message": {
                    "content": [
                        {
                            "text": json.dumps(
                                {
                                    "severity": "P3",
                                    "category": "technical",
                                    "summary": "Technical issue.",
                                    "confidence": 0.8,
                                }
                            )
                        }
                    ]
                }
            },
        }
    )

    classifier = BedrockTicketClassifier(
        client=client,
        model_id="openai.gpt-oss-120b-1:0",
    )

    with caplog.at_level(
        logging.INFO,
        logger="app.adapters.aws.bedrock",
    ):
        classifier.classify(make_ticket())

    logs = caplog.text

    assert "bedrock_classification_succeeded" in logs
    assert '"total_tokens": 30' in logs
    assert '"bedrock_latency_ms": 100' in logs

    assert "customer-secret-456" not in logs
    assert "Payment outage" not in logs
    assert "Checkout has returned" not in logs

def test_bedrock_classifier_rejects_missing_stop_reason() -> None:
    classifier = BedrockTicketClassifier(
        client=FakeBedrockClient(
            response={
                "output": {
                    "message": {
                        "content": [
                            {
                                "text": json.dumps(
                                    {
                                        "severity": "P3",
                                        "category": "technical",
                                        "summary": "Technical issue.",
                                        "confidence": 0.8,
                                    }
                                )
                            }
                        ]
                    }
                }
            }
        ),
        model_id="openai.gpt-oss-120b-1:0",
    )

    with pytest.raises(ClassifierResponseError):
        classifier.classify(make_ticket())