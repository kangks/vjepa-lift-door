# Rates used

AWS list rates: AWS Pricing API pricing:GetProducts us-east-1 (read-only), fetched 2026-11-09T07:06:30Z. Free tiers are not applied (Cost Explorer shows Lambda, SQS, Step Functions and Scheduler billing $0.00 on 7-8 Nov because of them).

| key | USD | unit | usage type | publication date | description |
|---|---|---|---|---|---|
| ddb_read_request_unit | 1.25e-07 | ReadRequestUnits | AmazonDynamoDB ReadRequestUnits | 2026-11-05 | $0.125 per million read request units (N. Virginia) |
| ddb_write_request_unit | 6.25e-07 | WriteRequestUnits | AmazonDynamoDB WriteRequestUnits | 2026-11-05 | $0.625 per million write request units (N. Virginia) |
| fargate_arm_gb_hour | 0.00356 | hours | AmazonECS USE1-Fargate-ARM-GB-Hours | 2026-10-19 | AWS Fargate - ARM - Memory - US East (N. Virginia) |
| fargate_arm_vcpu_hour | 0.03238 | hours | AmazonECS USE1-Fargate-ARM-vCPU-Hours:perCPU | 2026-10-19 | AWS Fargate - ARM - vCPU - US East (N. Virginia) |
| fargate_ephemeral_gb_hour | 0.000111 | GB-Hours | AmazonECS USE1-Fargate-EphemeralStorage-GB-Hours | 2026-10-19 | AWS Fargate - Ephemeral Storage - US East (N. Virginia) |
| lambda_arm_gb_second | 1.33334e-05 | Lambda-GB-Second | AWSLambda Lambda-GB-Second-ARM | 2026-10-30 | AWS Lambda - Total Compute for ARM - US East (Northern Virginia)-Tier-1 |
| lambda_arm_request | 2e-07 | Requests | AWSLambda Request-ARM | 2026-10-30 | AWS Lambda - Total Requests for ARM - US East (Northern Virginia) |
| lambda_arm_storage_gb_second | 3.09e-08 | GB-Seconds | AWSLambda Lambda-Storage-GB-Second-ARM | 2026-10-30 | AWS Lambda - Ephemeral Storage for ARM - US East (N. Virginia) |
| lambda_x86_gb_second | 1.66667e-05 | Lambda-GB-Second | AWSLambda Lambda-GB-Second | 2026-10-30 | AWS Lambda - Total Compute - US East (Northern Virginia)-Tier-1 |
| lambda_x86_request | 2e-07 | Requests | AWSLambda Request | 2026-10-30 | AWS Lambda - Total Requests - US East (Northern Virginia) |
| logs_ingest_gb | 0.5 | GB | AmazonCloudWatch USE1-DataProcessing-Bytes | 2026-11-04 | $0.50 per GB custom log data ingested in Standard log class - US East (Northern Virginia) |
| logs_otel_ingest_gb | 0.5 | GB | AmazonCloudWatch USE1-DataProcessingOtel-Bytes | 2026-11-04 | $0.50 per GB of USE1-DataProcessingOtel-Bytes in US East (N. Virginia) |
| logs_storage_gb_month | 0.03 | GB-Mo | AmazonCloudWatch USE1-TimedStorage-ByteHrs | 2026-11-04 | $0.03 per GB-mo of log storage - US East (Northern Virginia) |
| public_ipv4_hour | 0.005 | Hrs | AmazonVPC USE1-PublicIPv4:InUseAddress | 2026-11-05 | $0.005 per In-use public IPv4 address per hour |
| s3_standard_gb_month | 0.023 | GB-Mo | AmazonS3 TimedStorage-ByteHrs | 2026-11-04 | $0.023 per GB - first 50 TB / month of storage used |
| s3_tier1_request | 5e-06 | Requests | AmazonS3 Requests-Tier1 | 2026-11-04 | $0.005 per 1,000 PUT, COPY, POST, or LIST requests |
| s3_tier2_request | 4e-07 | Requests | AmazonS3 Requests-Tier2 | 2026-11-04 | $0.004 per 10,000 GET and all other requests |
| scheduler_invocation | 1e-06 | Invocations | AWSEvents USE1-ScheduledInvocation | 2026-11-08 | $1.00 per million scheduled invocations in US East (N. Virginia) |
| sfn_state_transition | 2.5e-05 | StateTransitions | AmazonStates USE1-StateTransition | 2026-11-04 | $0.000025 per state transition |
| sqs_standard_request | 4e-07 | Requests | AWSQueueService Requests-RBP | 2026-09-11 | $0.40 per million Amazon SQS standard requests in Tier1 in US East (N Virginia) |
| xray_spans_indexed | 7.5e-07 | spans indexed | AmazonCloudWatch USE1-XRay-Spans-Indexed | 2026-11-04 | $0.00000075 per spans indexed for USE1-XRay-Spans-Indexed in US East (N. Virginia) |

Bedrock: app constant lda_core/reasoner.py PRICE_IN=4.4e-6 PRICE_OUT=22.0e-6 (no Opus 5.x SKU in AWS Pricing API on 2026-11-09; Marketplace-billed).
EventBridge: AWS EventBridge pricing: AWS service (state change) events are free.
X-Ray: spans go to the `aws/spans` log group (`GetTraceSegmentDestination` = CloudWatchLogs, Transaction Search). Priced at the CloudWatch OTel ingest rate on the trace's segment-document bytes (a proxy for ingested span bytes). Span indexing is assumed to be the free default 1%.

Derived Fargate per-second rates (ARM64, Linux):
- lda-embed {'vcpu': 16.0, 'memory_gib': 32.0, 'ephemeral_gib': 21.0}: US$0.00017559/s = US$0.63211/task-hour
- lda-coordinator {'vcpu': 1.0, 'memory_gib': 2.0, 'ephemeral_gib': 20.0}: US$0.00001097/s = US$0.03950/task-hour
