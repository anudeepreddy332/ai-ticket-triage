# AI Ticket Triage

Production-style asynchronous AI support-ticket triage pipeline built to learn and demonstrate practical cloud-native AI architecture.

## Goal

Accept support tickets asynchronously and classify them using an LLM into structured operational information such as:

- severity
- category
- summary
- routing destination

## Target architecture

Client
→ API Gateway
→ Ingress Lambda
→ SQS
→ Worker Lambda
→ Amazon Bedrock
→ DynamoDB

Repeated processing failures
→ Dead-Letter Queue

Observability
→ CloudWatch

## Architecture principles

- Core business logic remains independent of AWS.
- Cloud services are accessed through adapters.
- AWS infrastructure is defined using AWS SAM.
- Infrastructure should be deployable and removable without deleting application code.
- Unit tests should run without AWS credentials.
- External side effects should remain outside the core domain layer.

## Development

Python: 3.14

Create/sync environment:

```bash
uv sync