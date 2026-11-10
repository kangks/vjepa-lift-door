# Reconciliation

Attribute REST API status at collection: **UNAUTHORIZED: REST API JWT expired at 2026-11-06T05:58:10+00:00**. The token in SSM `/luminara/attribute/api_token` (issued 2026-08-08, 90-day expiry) returns 401 on `/api/v1/health`, so no Attribute figure is available for any day. Every Attribute comparison below is therefore **UNAVAILABLE**, not zero. A new REST token (dashboard Setup > API Tokens) written to that SSM parameter, then `cli.py all --cost-explorer`, fills these in.

Cost Explorer (`ce:GetCostAndUsage`, tag `Project=lift-door-agentic` plus the untagged Marketplace Bedrock line) is shown as a second, AWS-native billing cross-check. It is NOT one of the four cell sources. All Cost Explorer days below are flagged `Estimated` by AWS and still filling in.

## 1. Bedrock tokens: Σ app-metered vs billing

| UTC day | runs calling Opus | input tok | output tok | Σ app-metered USD | Attribute aicost Bedrock | delta vs Attribute | Cost Explorer 'Claude Opus 5.5 (Amazon Bedrock Edition)' (account-wide) | delta vs CE |
|---|---|---|---|---|---|---|---|---|
| 2026-11-07 | 3 | 21954 | 4028 | 0.18521 | UNAVAILABLE | n/a | 0.18536 | -0.00015 (-0.1%) |
| 2026-11-08 | 30 | 219540 | 41410 | 1.87700 | UNAVAILABLE | n/a | 1.44502 | +0.43197 (+29.9%) |
| 2026-11-09 | 0 | 0 | 0 | 0.00000 | UNAVAILABLE | n/a | no line | n/a |

Cumulative Σ app-metered Opus on 2026-11-08 by pass, against CE 1.44502 (hypothesis check: where does the partial CE day stop?):

| through | Σ app-metered | minus CE |
|---|---|---|
| (not batch-driven) | 0.06027 | -1.38475 |
| lda-batch-20261108-e2etest/t-e2e | 0.12056 | -1.32446 |
| lda-batch-20261108/pass1 | 0.56631 | -0.87871 |
| lda-batch-20261108/pass2 | 1.00643 | -0.43860 |
| lda-batch-20261108/pass3 | 1.44502 | +0.00000 |
| lda-batch-20261108/pass4 | 1.87700 | +0.43197 |

Notes: the Cost Explorer Bedrock line is account-wide and untagged (Marketplace product), so any other Opus 5.5 caller in <ACCOUNT_ID> on that day lands in it too, and any lda Opus call made outside these 55 result.json records (for example `scripts/bedrock_preflight.py`) is in CE but not in Σ app-metered. The 8 Nov CE figure is partial. The app rate is a frozen constant (see rates_used.md); a persistent same-sign delta on a settled day would mean the constant is wrong.

## 2. Fargate: Σ per-run shares vs workload day cost

| UTC day | workload | billed task time | method | list-rate day cost | Σ per-run fully-absorbed | Σ per-run marginal | idle | Attribute EC2-ECS day cost | delta (list vs Attribute) |
|---|---|---|---|---|---|---|---|---|---|
| 2026-11-07 | lda-embed | 1005 s | estimate | 0.17648 | 0.17648 | 0.02392 | 0.15256 | UNAVAILABLE | n/a |
| 2026-11-07 | lda-coordinator | 1159 s | estimate | 0.01272 | 0.01272 | 0.00244 | 0.01028 | UNAVAILABLE | n/a |
| 2026-11-08 | lda-embed | 4395 s | estimate/exact | 0.77174 | 0.77174 | 0.34469 | 0.42705 | UNAVAILABLE | n/a |
| 2026-11-08 | lda-coordinator | 4438 s | estimate/exact | 0.04870 | 0.04870 | 0.03115 | 0.01754 | UNAVAILABLE | n/a |

Σ fully-absorbed equals the day cost by construction (runs carry the idle). Task-time windows:

- lda-coordinator 2026-11-07T01:01Z..01:11Z: 670 s billed, LiveTaskCount minutes + calibrated offset 70s (estimate)
- lda-coordinator 2026-11-07T01:16Z..01:23Z: 490 s billed, LiveTaskCount minutes + calibrated offset 70s (estimate)
- lda-coordinator 2026-11-08T11:35Z..11:39Z: 310 s billed, LiveTaskCount minutes + calibrated offset 70s (estimate)
- lda-coordinator 2026-11-08T12:25Z..12:31Z: 410 s billed, exact ECS task record (t-e2e)
- lda-coordinator 2026-11-08T14:32Z..14:46Z: 927 s billed, exact ECS task record (pass1)
- lda-coordinator 2026-11-08T17:01Z..17:15Z: 895 s billed, exact ECS task record (pass2)
- lda-coordinator 2026-11-08T19:32Z..19:46Z: 911 s billed, exact ECS task record (pass3)
- lda-coordinator 2026-11-08T22:02Z..22:17Z: 986 s billed, exact ECS task record (pass4)
- lda-embed 2026-11-07T01:02Z..01:11Z: 593 s billed, LiveTaskCount minutes + calibrated offset 53s (estimate)
- lda-embed 2026-11-07T01:16Z..01:22Z: 413 s billed, LiveTaskCount minutes + calibrated offset 53s (estimate)
- lda-embed 2026-11-08T11:36Z..11:39Z: 233 s billed, LiveTaskCount minutes + calibrated offset 53s (estimate)
- lda-embed 2026-11-08T12:25Z..12:31Z: 415 s billed, exact ECS task record (t-e2e)
- lda-embed 2026-11-08T14:32Z..14:46Z: 940 s billed, exact ECS task record (pass1)
- lda-embed 2026-11-08T17:01Z..17:16Z: 909 s billed, exact ECS task record (pass2)
- lda-embed 2026-11-08T19:32Z..19:46Z: 916 s billed, exact ECS task record (pass3)
- lda-embed 2026-11-08T22:02Z..22:18Z: 982 s billed, exact ECS task record (pass4)

- calibration lda-coordinator: LiveTaskCount misses on average 70 s per window (image pull + stopping) across 5 windows with exact records (50, 87, 55, 71, 86 s)
- calibration lda-embed: LiveTaskCount misses on average 53 s per window (image pull + stopping) across 5 windows with exact records (55, 100, 9, 76, 22 s)

## 3. Cost Explorer quantities on the lda tag vs this report's own counts

| UTC day | usage type | CE quantity | CE USD (after free tier) | this report |
|---|---|---|---|---|
| 2026-11-07 | DataTransfer-In-Bytes | 1.1475 | 0.000000 |  |
| 2026-11-07 | DataTransfer-Out-Bytes | 0.0036 | 0.000000 |  |
| 2026-11-07 | Global-Bucket-Hrs-FreeTier | 0.0014 | 0.000000 |  |
| 2026-11-07 | Lambda-GB-Second-ARM | 213.3446 | 0.000000 |  |
| 2026-11-07 | Lambda-Storage-GB-Second-ARM | 48.4665 | 0.000001 |  |
| 2026-11-07 | ReadRequestUnits | 7.0000 | 0.000001 |  |
| 2026-11-07 | Request-ARM | 99.0000 | 0.000000 |  |
| 2026-11-07 | Requests-RBP | 79.0000 | 0.000000 |  |
| 2026-11-07 | Requests-Tier1 | 384.0000 | 0.001920 | 46 pipeline PUT/COPY (runs/+incoming/ listing; excludes batch/ driver writes and bank/model uploads) |
| 2026-11-07 | Requests-Tier2 | 741.0000 | 0.000296 | 35 pipeline GET (code path; excludes driver HEAD polling, checkpoint download, operator reads) |
| 2026-11-07 | TimedStorage-ByteHrs | 0.0304 | 0.003041 |  |
| 2026-11-07 | USE1-Build-Min:ARM:g1.large | 32.0000 | 0.480000 |  |
| 2026-11-07 | USE1-DataProcessing-Bytes | 0.0005 | 0.000000 |  |
| 2026-11-07 | USE1-DataProcessingOtel-Bytes | 0.0000 | 0.000008 |  |
| 2026-11-07 | USE1-VendedLog-Bytes | 0.0000 | 0.000000 |  |
| 2026-11-07 | WriteRequestUnits | 4.0000 | 0.000003 | 4 (2 per ticket: table + GSI) |
| 2026-11-08 | DataTransfer-In-Bytes | 0.0120 | 0.000000 |  |
| 2026-11-08 | DataTransfer-Out-Bytes | 0.0005 | 0.000000 |  |
| 2026-11-08 | DataTransfer-Regional-Bytes | 0.0795 | 0.000795 |  |
| 2026-11-08 | Lambda-GB-Second | 74.2507 | 0.000000 |  |
| 2026-11-08 | Lambda-GB-Second-ARM | 416.0969 | 0.000000 |  |
| 2026-11-08 | Lambda-Storage-GB-Second-ARM | 124.6890 | 0.000004 |  |
| 2026-11-08 | ReadRequestUnits | 2.0000 | 0.000000 |  |
| 2026-11-08 | Request | 154.0000 | 0.000000 |  |
| 2026-11-08 | Request-ARM | 146.0000 | 0.000000 |  |
| 2026-11-08 | Requests-RBP | 174.0000 | 0.000000 |  |
| 2026-11-08 | Requests-Tier1 | 562.0000 | 0.002810 | 537 pipeline PUT/COPY (runs/+incoming/ listing; excludes batch/ driver writes and bank/model uploads) |
| 2026-11-08 | Requests-Tier2 | 699.0000 | 0.000280 | 366 pipeline GET (code path; excludes driver HEAD polling, checkpoint download, operator reads) |
| 2026-11-08 | TimedStorage-ByteHrs | 0.0932 | 0.004422 |  |
| 2026-11-08 | USE1-DataProcessing-Bytes | 0.0001 | 0.000000 |  |
| 2026-11-08 | USE1-DataProcessingOtel-Bytes | 0.0000 | 0.000022 |  |
| 2026-11-08 | USE1-Fargate-ARM-GB-Hours | 33.1661 | 0.118071 | 41.5341 GB-h |
| 2026-11-08 | USE1-Fargate-ARM-vCPU-Hours:perCPU | 16.5831 | 0.536959 | 20.7670 vCPU-h (task records) |
| 2026-11-08 | USE1-Fargate-EphemeralStorage-GB-Hours | 0.9736 | 0.000108 |  |
| 2026-11-08 | USE1-PublicIPv4:InUseAddress | 2.1364 | 0.010682 | 2.4537 h |
| 2026-11-08 | USE1-ScheduledInvocation | 56.0000 | 0.000000 |  |
| 2026-11-08 | USE1-StateTransition | 294.0000 | 0.000000 | 357 (GetExecutionHistory *StateEntered) |
| 2026-11-08 | USE1-TimedStorage-ByteHrs | 0.0000 | 0.000000 |  |
| 2026-11-08 | USE1-VendedLog-Bytes | 0.0001 | 0.000000 |  |
| 2026-11-08 | WriteRequestUnits | 38.0000 | 0.000024 | 50 (2 per ticket: table + GSI) |

Cost Explorer tag-filtered ECS/Fargate cost per day:
- 2026-11-07 AWS Lambda: US$0.00000 (Estimated=True)
- 2026-11-07 Amazon DynamoDB: US$0.00000 (Estimated=True)
- 2026-11-07 Amazon EC2 Container Registry (ECR): US$0.00304 (Estimated=True)
- 2026-11-07 Amazon Simple Storage Service: US$0.00222 (Estimated=True)
- 2026-11-07 AmazonCloudWatch: US$0.00001 (Estimated=True)
- 2026-11-07 CodeBuild: US$0.48000 (Estimated=True)
- 2026-11-08 AWS Lambda: US$0.00000 (Estimated=True)
- 2026-11-08 Amazon DynamoDB: US$0.00002 (Estimated=True)
- 2026-11-08 Amazon EC2 Container Registry (ECR): US$0.00296 (Estimated=True)
- 2026-11-08 Amazon Elastic Container Service: US$0.65593 (Estimated=True)
- 2026-11-08 Amazon Simple Storage Service: US$0.00455 (Estimated=True)
- 2026-11-08 Amazon Virtual Private Cloud: US$0.01068 (Estimated=True)
- 2026-11-08 AmazonCloudWatch: US$0.00002 (Estimated=True)

## 4. Telemetry self-checks (do the counted quantities agree with each other?)

- S3 PUT: listed objects under `runs/<id>/` + `incoming/<id>/` equal the code-path PUT prediction (coordinator claim/result[/reasoner meta], embed frames/tokens, knn anomaly map/detector[/map+diffs], incoming clip) for **55/55** runs.
- S3 GET: X-Ray `GetObject` spans from coordinator+embed equal the code path (1 clip GET, +3 on alert) for **55/55** traced runs. lda-knn's S3 calls are not instrumented in X-Ray, so its 3-4 GETs per run (+7 bank GETs on a cold start, 7 cold starts among run invocations) remain code-path estimates. The coordinator's `claim.json` PUT happens before its root span opens, which is why X-Ray shows one PUT fewer than the listing.
- Lambda: lda-knn invocations on run traces matched to a REPORT line by request id: **55/55**. Billed ms includes cold-start init (image functions bill init).
- X-Ray: traces retrieved for 55/55 runs.

## 5. Lambda billed time vs CloudWatch

| UTC day | function | REPORT lines | Σ billed ms (incl. init) | CloudWatch Invocations | CloudWatch Duration ms (excl. init) | Σ billed ms on run traces |
|---|---|---|---|---|---|---|
| 2026-11-07 | lda-knn | 5 | 32311 | 5 | 32307 | 11757 |
| 2026-11-07 | lda-ticket | 2 | 7005 | 2 | 7004 | 7005 |
| 2026-11-08 | lda-knn | 51 | 104715 | 51 | 104691 | 104715 |
| 2026-11-08 | lda-ticket | 25 | 25546 | 25 | 25533 | 25546 |
