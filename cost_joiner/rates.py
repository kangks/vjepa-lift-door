"""List rates for every priced line, with the source of each number.

Two kinds of rate live here:

1. Bedrock tokens: the APP's frozen constants, copied verbatim from
   workshop/lift-door/agentic/lda_core/reasoner.py (PRICE_IN 4.4e-6, PRICE_OUT
   22.0e-6, model anthropic.claude-opus-5-5 via bedrock-mantle). Queried
   2026-11-09: the AWS Pricing API has NO Opus 5.x SKU in AmazonBedrock or
   AmazonBedrockService (Anthropic models bill as the Marketplace product
   "Claude Opus 5.5 (Amazon Bedrock Edition)"), so this rate is an app constant,
   not catalogue-verified. Reconciliation against a bill is the only check.
   Same exact-key discipline as attributes/app/cost_joiner/rates.py: an unknown
   model id RAISES rather than falling back to another model's price.

2. AWS list rates: fetched read-only from the AWS Pricing API (us-east-1) by
   collect.py into out/raw/pricing.json, keyed by the usage types below. The
   report refuses to run if a key is missing from the snapshot. No silent
   default constants: a wrong-but-plausible rate is the failure this guards.

Free tiers are NOT applied. These are marginal list rates. Cost Explorer shows
Lambda, SQS, Step Functions and Scheduler billing $0.00 on 7-8 Nov because
of free tiers; the report says so next to those columns.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- Bedrock (app constants) ---------------------------------------------------------------
BEDROCK_MODEL_ID = "anthropic.claude-opus-5-5"
_BEDROCK_RATES_PER_TOKEN = {
    # model id: (input $/token, output $/token)
    BEDROCK_MODEL_ID: (4.4e-6, 22.0e-6),
}
BEDROCK_RATE_SOURCE = (
    "app constant lda_core/reasoner.py PRICE_IN=4.4e-6 PRICE_OUT=22.0e-6 "
    "(no Opus 5.x SKU in AWS Pricing API on 2026-11-09; Marketplace-billed)"
)


class UnknownModelRateError(ValueError):
    pass


def bedrock_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """input*rate_in + output*rate_out. output_tokens already includes thinking tokens
    (Anthropic usage semantics; output_tokens_details.thinking_tokens is a subset)."""
    if model_id not in _BEDROCK_RATES_PER_TOKEN:
        raise UnknownModelRateError(f"no rate for model_id={model_id!r}; known={sorted(_BEDROCK_RATES_PER_TOKEN)}")
    rin, rout = _BEDROCK_RATES_PER_TOKEN[model_id]
    return input_tokens * rin + output_tokens * rout


# --- AWS Pricing API queries -----------------------------------------------------------------
# key: (ServiceCode, usagetype, beginRange of the tier to use)
PRICE_QUERIES: dict[str, tuple[str, str, str]] = {
    "fargate_arm_vcpu_hour": ("AmazonECS", "USE1-Fargate-ARM-vCPU-Hours:perCPU", "0"),
    "fargate_arm_gb_hour": ("AmazonECS", "USE1-Fargate-ARM-GB-Hours", "0"),
    "fargate_ephemeral_gb_hour": ("AmazonECS", "USE1-Fargate-EphemeralStorage-GB-Hours", "0"),
    "public_ipv4_hour": ("AmazonVPC", "USE1-PublicIPv4:InUseAddress", "0"),
    "s3_tier1_request": ("AmazonS3", "Requests-Tier1", "0"),  # PUT, COPY, POST, LIST
    "s3_tier2_request": ("AmazonS3", "Requests-Tier2", "0"),  # GET, HEAD and all other
    "s3_standard_gb_month": ("AmazonS3", "TimedStorage-ByteHrs", "0"),  # first 50 TB tier
    "sqs_standard_request": ("AWSQueueService", "Requests-RBP", "0"),  # Tier1
    "lambda_arm_gb_second": ("AWSLambda", "Lambda-GB-Second-ARM", "0"),  # Tier-1
    "lambda_x86_gb_second": ("AWSLambda", "Lambda-GB-Second", "0"),  # Tier-1
    "lambda_arm_request": ("AWSLambda", "Request-ARM", "0"),
    "lambda_x86_request": ("AWSLambda", "Request", "0"),
    "lambda_arm_storage_gb_second": ("AWSLambda", "Lambda-Storage-GB-Second-ARM", "0"),
    "sfn_state_transition": ("AmazonStates", "USE1-StateTransition", "0"),
    "scheduler_invocation": ("AWSEvents", "USE1-ScheduledInvocation", "14000000"),  # first 14M are $0
    "logs_ingest_gb": ("AmazonCloudWatch", "USE1-DataProcessing-Bytes", "0"),
    "logs_otel_ingest_gb": ("AmazonCloudWatch", "USE1-DataProcessingOtel-Bytes", "0"),
    "logs_storage_gb_month": ("AmazonCloudWatch", "USE1-TimedStorage-ByteHrs", "0"),
    "xray_spans_indexed": ("AmazonCloudWatch", "USE1-XRay-Spans-Indexed", "0"),
    "ddb_write_request_unit": ("AmazonDynamoDB", "WriteRequestUnits", "0"),
    "ddb_read_request_unit": ("AmazonDynamoDB", "ReadRequestUnits", "0"),
}

# AWS service events (S3 "Object Created" to the default bus) and same-account rule
# delivery are not charged by EventBridge. Documented AWS pricing, not a Pricing API SKU.
EVENTBRIDGE_AWS_SERVICE_EVENT_USD = 0.0
EVENTBRIDGE_RATE_SOURCE = "AWS EventBridge pricing: AWS service (state change) events are free"


@dataclass(frozen=True)
class Rate:
    key: str
    usd: float
    unit: str
    usagetype: str
    service: str
    description: str
    publication_date: str


class RateCard:
    """Read-only view over out/raw/pricing.json."""

    def __init__(self, snapshot: dict):
        self._rates: dict[str, Rate] = {}
        for key, row in snapshot.get("rates", {}).items():
            self._rates[key] = Rate(key=key, usd=float(row["usd"]), unit=row["unit"], usagetype=row["usagetype"],
                                    service=row["service"], description=row["description"],
                                    publication_date=row.get("publication_date", ""))
        missing = sorted(set(PRICE_QUERIES) - set(self._rates))
        if missing:
            raise KeyError(f"pricing snapshot is missing rates {missing}; re-run collect")

    def __getitem__(self, key: str) -> float:
        return self._rates[key].usd

    def rate(self, key: str) -> Rate:
        return self._rates[key]

    def all(self) -> list[Rate]:
        return [self._rates[k] for k in sorted(self._rates)]


# --- pure rate math (unit tested) -------------------------------------------------------------
GIB = 1024 ** 3
HOURS_PER_MONTH_NOV = 30 * 24  # S3/CloudWatch storage ByteHrs are pro-rated over the month's hours
FARGATE_FREE_EPHEMERAL_GIB = 20
LAMBDA_FREE_EPHEMERAL_MB = 512


def fargate_task_rate_per_second(vcpu: float, memory_gib: float, ephemeral_gib: float, rates: RateCard) -> float:
    """Fargate Linux/ARM: per-second billing of vCPU + memory + ephemeral storage above 20 GiB."""
    extra_eph = max(0.0, ephemeral_gib - FARGATE_FREE_EPHEMERAL_GIB)
    per_hour = (vcpu * rates["fargate_arm_vcpu_hour"] + memory_gib * rates["fargate_arm_gb_hour"]
                + extra_eph * rates["fargate_ephemeral_gb_hour"])
    return per_hour / 3600.0


def lambda_invocation_cost(duration_ms: float, memory_mb: int, arch: str, rates: RateCard,
                           ephemeral_mb: int = LAMBDA_FREE_EPHEMERAL_MB, requests: int = 1) -> float:
    """GB-seconds x duration rate + requests x request rate + ephemeral storage above 512 MB.
    duration_ms is rounded up to the next 1 ms (Lambda billing granularity)."""
    import math

    arm = arch == "arm64"
    billed_s = math.ceil(duration_ms) / 1000.0
    gb = memory_mb / 1024.0
    compute = gb * billed_s * rates["lambda_arm_gb_second" if arm else "lambda_x86_gb_second"]
    req = requests * rates["lambda_arm_request" if arm else "lambda_x86_request"]
    eph_gb = max(0, ephemeral_mb - LAMBDA_FREE_EPHEMERAL_MB) / 1024.0
    storage = eph_gb * billed_s * rates["lambda_arm_storage_gb_second"] if arm else 0.0
    return compute + req + storage


def storage_cost_per_day(bytes_stored: int, gb_month_rate: float) -> float:
    """One day of S3/CloudWatch storage at a GB-month rate (GB = 2^30 bytes, 30-day month)."""
    return bytes_stored / GIB * gb_month_rate * 24 / HOURS_PER_MONTH_NOV


def ingest_cost(bytes_ingested: float, gb_rate: float) -> float:
    return bytes_ingested / GIB * gb_rate
