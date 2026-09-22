import json
import os
from pathlib import Path

from app.adapters.aws.bedrock import create_bedrock_classifier
from evals.triage_eval import load_eval_cases, run_eval


DATASET = Path(
    "evals/datasets/ticket_triage_v1.jsonl"
)


def main() -> None:
    model_id = os.environ["BEDROCK_MODEL_ID"]

    region = os.environ.get(
        "AWS_REGION",
        "us-east-1",
    )

    classifier = create_bedrock_classifier(
        model_id=model_id,
        region=region,
    )

    cases = load_eval_cases(DATASET)

    report = run_eval(
        classifier,
        cases,
    )

    output = {
        "total": report.total,
        "completed": report.completed,
        "failures": report.failures,
        "failure_rate": round(
            report.failure_rate,
            4,
        ),
        "severity_accuracy": round(
            report.severity_accuracy,
            4,
        ),
        "category_accuracy": round(
            report.category_accuracy,
            4,
        ),
        "route_accuracy": round(
            report.route_accuracy,
            4,
        ),
        "p1_recall": round(
            report.p1_recall,
            4,
        ),
        "average_latency_ms": round(
            report.average_latency_ms,
            2,
        ),
    }

    print(
        json.dumps(
            output,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()