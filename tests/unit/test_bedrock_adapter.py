import json
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