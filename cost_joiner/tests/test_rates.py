import pytest

from lda_cost_joiner.rates import (PRICE_QUERIES, RateCard, UnknownModelRateError, bedrock_cost,
                                   fargate_task_rate_per_second, ingest_cost, lambda_invocation_cost,
                                   storage_cost_per_day)

# Values as returned by the AWS Pricing API for us-east-1 on 2026-11-09.
LIST = {
    "fargate_arm_vcpu_hour": 0.03238, "fargate_arm_gb_hour": 0.00356, "fargate_ephemeral_gb_hour": 0.000111,
    "public_ipv4_hour": 0.005, "s3_tier1_request": 5e-06, "s3_tier2_request": 4e-07, "s3_standard_gb_month": 0.023,
    "sqs_standard_request": 4e-07, "lambda_arm_gb_second": 1.33334e-05, "lambda_x86_gb_second": 1.66667e-05,
    "lambda_arm_request": 2e-07, "lambda_x86_request": 2e-07, "lambda_arm_storage_gb_second": 3.09e-08,
    "sfn_state_transition": 2.5e-05, "scheduler_invocation": 1e-06, "logs_ingest_gb": 0.5,
    "logs_otel_ingest_gb": 0.5, "logs_storage_gb_month": 0.03, "xray_spans_indexed": 7.5e-07,
    "ddb_write_request_unit": 6.25e-07, "ddb_read_request_unit": 1.25e-07,
}


@pytest.fixture
def card():
    return RateCard({"rates": {k: {"usd": v, "unit": "u", "usagetype": PRICE_QUERIES[k][1],
                                   "service": PRICE_QUERIES[k][0], "description": ""} for k, v in LIST.items()}})


def test_bedrock_cost_matches_app_constants():
    # run-20261108T143425-anomaly_door_desync-2d3f: 7318 in / 1420 out; app rounded to 0.0634
    assert bedrock_cost("anthropic.claude-opus-5-5", 7318, 1420) == pytest.approx(7318 * 4.4e-6 + 1420 * 22e-6)
    assert round(bedrock_cost("anthropic.claude-opus-5-5", 7318, 1420), 4) == 0.0634


def test_unknown_model_raises_not_defaults():
    with pytest.raises(UnknownModelRateError):
        bedrock_cost("us.anthropic.claude-opus-5", 1, 1)


def test_rate_card_refuses_missing_keys():
    with pytest.raises(KeyError):
        RateCard({"rates": {}})


def test_fargate_embed_task_rate(card):
    # 16 vCPU, 32 GiB, 21 GiB ephemeral (1 GiB above the free 20)
    per_hour = 16 * 0.03238 + 32 * 0.00356 + 1 * 0.000111
    assert fargate_task_rate_per_second(16, 32, 21, card) * 3600 == pytest.approx(per_hour)
    assert per_hour == pytest.approx(0.632111, abs=1e-6)


def test_fargate_coordinator_task_rate_has_no_ephemeral_charge(card):
    assert fargate_task_rate_per_second(1, 2, 20, card) * 3600 == pytest.approx(0.03238 + 2 * 0.00356)


def test_lambda_arm_with_ephemeral_storage(card):
    # lda-knn: 4096 MB ARM, 2048 MB ephemeral; 1850 ms billed
    c = lambda_invocation_cost(1849.57, 4096, "arm64", card, ephemeral_mb=2048)
    expected = 4 * 1.850 * 1.33334e-05 + 2e-07 + 1.5 * 1.850 * 3.09e-08
    assert c == pytest.approx(expected)


def test_lambda_x86_rounds_up_to_ms_and_ignores_storage(card):
    c = lambda_invocation_cost(1000.2, 256, "x86_64", card, ephemeral_mb=512)
    assert c == pytest.approx(0.25 * 1.001 * 1.66667e-05 + 2e-07)


def test_storage_per_day_and_ingest(card):
    gib = 1024 ** 3
    assert storage_cost_per_day(gib, 0.023) == pytest.approx(0.023 / 30)
    assert ingest_cost(gib, 0.5) == pytest.approx(0.5)
