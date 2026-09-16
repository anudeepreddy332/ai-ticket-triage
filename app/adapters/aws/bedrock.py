import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from app.core.models import (
    RoutingDestination,
    Severity,
    SupportTicket,
    TicketCategory,
    TriageResult,
)
from app.ports.errors import (
    ClassifierResponseError,
    ClassifierUnavailableError,
)

if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime.client import BedrockRuntimeClient
else:
    BedrockRuntimeClient = Any


SYSTEM_PROMPT = """
You are a support-ticket triage classifier.

Classify each ticket using these severity rules:

P1:
Active critical outage, security incident, safety issue, or widespread
business-critical failure requiring immediate response.

P2:
Major degradation or urgent customer-impacting issue that is not a full
critical outage.

P3:
Standard support issue requiring normal investigation.

P4:
Low-priority question, informational request, or minor issue.

Available categories:
payments, billing, account, technical, other.

Produce a concise factual summary.

Confidence must be a number from 0.0 to 1.0.

Do not invent facts that are not present in the ticket.
""".strip()


TRIAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "severity": {
            "type": "string",
            "enum": [severity.value for severity in Severity],
        },
        "category": {
            "type": "string",
            "enum": [category.value for category in TicketCategory],
        },
        "summary": {
            "type": "string",
        },
        "confidence": {
            "type": "number",
        },
    },
    "required": [
        "severity",
        "category",
        "summary",
        "confidence",
    ],
    "additionalProperties": False,
}


ROUTE_BY_CATEGORY = {
    TicketCategory.PAYMENTS: RoutingDestination.PAYMENTS_ONCALL,
    TicketCategory.BILLING: RoutingDestination.BILLING_SUPPORT,
    TicketCategory.ACCOUNT: RoutingDestination.ACCOUNT_SUPPORT,
    TicketCategory.TECHNICAL: RoutingDestination.TECH_SUPPORT,
    TicketCategory.OTHER: RoutingDestination.GENERAL_SUPPORT,
}


class BedrockTicketClassifier:
    def __init__(
        self,
        client: BedrockRuntimeClient,
        model_id: str,
    ) -> None:
        self._client = client
        self._model_id = model_id

    def classify(self, ticket: SupportTicket) -> TriageResult:
        try:
            response = self._client.converse(
                modelId=self._model_id,
                system=[
                    {
                        "text": SYSTEM_PROMPT,
                    }
                ],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "text": self._build_ticket_prompt(ticket),
                            }
                        ],
                    }
                ],
                inferenceConfig={
                    "maxTokens": 512,
                    "temperature": 0.0,
                },
                outputConfig={
                    "textFormat": {
                        "type": "json_schema",
                        "structure": {
                            "jsonSchema": {
                                "name": "ticket_triage",
                                "description": (
                                    "Structured support-ticket triage result"
                                ),
                                "schema": json.dumps(TRIAGE_SCHEMA),
                            }
                        },
                    }
                },
            )

        except (ClientError, BotoCoreError) as exc:
            raise ClassifierUnavailableError(
                "Bedrock classification request failed"
            ) from exc

        try:
            payload = json.loads(self._extract_text(response))

            category = TicketCategory(payload["category"])

            return TriageResult(
                ticket_id=ticket.ticket_id,
                severity=Severity(payload["severity"]),
                category=category,
                summary=str(payload["summary"]).strip(),
                route_to=ROUTE_BY_CATEGORY[category],
                confidence=float(payload["confidence"]),
            )

        except (KeyError, TypeError, ValueError) as exc:
            raise ClassifierResponseError(
                "Bedrock returned an invalid classification response"
            ) from exc

    @staticmethod
    def _build_ticket_prompt(ticket: SupportTicket) -> str:
        return (
            f"Subject:\n{ticket.subject}\n\n"
            f"Ticket body:\n{ticket.body}"
        )

    @staticmethod
    def _extract_text(response: Mapping[str, Any]) -> str:
        try:
            content = response["output"]["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise ClassifierResponseError(
                "Bedrock response did not contain message content"
            ) from exc

        text_parts = [
            block["text"]
            for block in content
            if isinstance(block, dict)
            and isinstance(block.get("text"), str)
        ]

        if not text_parts:
            raise ClassifierResponseError(
                "Bedrock response contained no text output"
            )

        return "".join(text_parts)


def create_bedrock_classifier(
    *,
    model_id: str,
    region: str,
) -> BedrockTicketClassifier:
    config = Config(
        connect_timeout=5,
        read_timeout=180,
        retries={
            "mode": "standard",
            "total_max_attempts": 3,
        },
    )

    client: BedrockRuntimeClient = boto3.client(
        "bedrock-runtime",
        region_name=region,
        config=config,
    )

    return BedrockTicketClassifier(
        client=client,
        model_id=model_id,
    )