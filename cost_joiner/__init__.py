"""Per-invocation, fully loaded cost joiner for the lift-door agentic pipeline (lda).

Method (from attributes/design/cost-per-successful-agentic-outcome-methodology-and-findings.md
sections 1.2, 4.1, 4.2, 4.8): Attribute's daily workload cost, apportioned by lda's own
per-run telemetry (busy-seconds), plus app-metered token cost per call, plus per-run
request-priced services at list rates. Never "Attribute attributed $X to this run".
"""

SOURCE_ATTRIBUTE_APPORTIONED = "ATTRIBUTE_APPORTIONED"
SOURCE_APP_METERED = "APP_METERED"
SOURCE_LIST_RATE_ESTIMATE = "LIST_RATE_ESTIMATE"
SOURCE_PENDING_ATTRIBUTE = "PENDING_ATTRIBUTE"
SOURCE_TAGS = (
    SOURCE_ATTRIBUTE_APPORTIONED, SOURCE_APP_METERED, SOURCE_LIST_RATE_ESTIMATE, SOURCE_PENDING_ATTRIBUTE,
)
