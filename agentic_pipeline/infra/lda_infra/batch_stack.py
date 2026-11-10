"""LiftDoorAgenticBatch: unattended 12-clip batch for 8 Nov 2026 (attrb_sandbox only).

Separate stack. It references the accepted LiftDoorAgentic / LiftDoorAgenticBuild resources by literal
name/ARN only (no cross-stack exports), so neither accepted stack's template changes.
Lambda code lives in workshop/lift-door/agentic_batch_8nov/lambda.
"""
from __future__ import annotations

import json
from pathlib import Path

from aws_cdk import (
    CfnOutput, Duration, RemovalPolicy, Stack, Tags,
    aws_iam as iam, aws_lambda as lambda_, aws_logs as logs, aws_scheduler as scheduler,
    aws_ssm as ssm, aws_stepfunctions as sfn, aws_stepfunctions_tasks as tasks,
)
from cdk_nag import NagPackSuppression, NagSuppressions
from constructs import Construct

BATCH_ID = "lda-batch-20261108"
PASS_TIMES = [  # UTC, see agentic_batch_8nov/PLAN.md
    ("pass1", "2026-11-08T14:30:00"),
    ("pass2", "2026-11-08T17:00:00"),
    ("pass3", "2026-11-08T19:30:00"),
    ("pass4", "2026-11-08T22:00:00"),
]
WATCHDOG_WINDOW = ("2026-11-08T14:00:00.000Z", "2026-11-09T00:40:00.000Z")
DEADMAN_TIMES = [("deadman1", "2026-11-09T00:30:00"), ("deadman2", "2026-11-09T00:50:00")]
GROUP, DEADMAN_GROUP, TEST_GROUP = "lda-batch", "lda-batch-deadman", "lda-batch-test"


class BatchStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, *, profile: str, account: str, region: str, **kwargs):
        super().__init__(scope, construct_id, **kwargs)
        if profile != "attrb_sandbox":
            raise ValueError("LiftDoorAgenticBatch is for attrb_sandbox only")
        Tags.of(self).add("Component", "batch")
        outputs_path = Path(__file__).resolve().parents[2] / "outputs" / f"{profile}.json"
        main = json.loads(outputs_path.read_text())["LiftDoorAgentic"]
        bucket = main["BucketName"]
        cluster = main["ClusterName"]
        embed, coord = main["EmbedServiceName"], main["CoordinatorServiceName"]
        queue_url = main["QueueUrl"]
        queue_arn = f"arn:aws:sqs:{region}:{account}:{queue_url.rsplit('/', 1)[1]}"
        bucket_arn = f"arn:aws:s3:::{bucket}"
        service_arns = [f"arn:aws:ecs:{region}:{account}:service/{cluster}/{name}" for name in (embed, coord)]
        task_arn = f"arn:aws:ecs:{region}:{account}:task/{cluster}/*"
        cluster_arn = f"arn:aws:ecs:{region}:{account}:cluster/{cluster}"
        param_arn = lambda name: f"arn:aws:ssm:{region}:{account}:parameter{name}"  # noqa: E731
        batch_params = [param_arn("/lda/batch/*")]
        lease_param = param_arn("/lda/lease_until")
        sm_name = "lda-batch-pass"
        sm_arn = f"arn:aws:states:{region}:{account}:stateMachine:{sm_name}"
        code_dir = Path(__file__).resolve().parents[3] / "agentic_batch_8nov" / "lambda"

        # SSM parameters. The kill switch starts DISABLED; it is set to true by hand after the tests.
        for cid, name, value, desc in (
            ("EnabledParam", "/lda/batch/enabled", "false", "lda batch kill switch: true = passes may run"),
            ("LockParam", "/lda/batch/lock", "free", "lda batch pass lock (JSON owner+expires or free)"),
            ("SpendParam", "/lda/batch/spend_usd", "0", "lda batch cost ledger estimate (USD)"),
            ("LeaseWrittenParam", "/lda/batch/lease_written", "none", "last /lda/lease_until value written by the batch"),
        ):
            ssm.StringParameter(self, cid, parameter_name=name, string_value=value, description=desc)

        env = {
            "LDA_BUCKET": bucket, "LDA_CLUSTER": cluster, "LDA_EMBED_SERVICE": embed, "LDA_COORD_SERVICE": coord,
            "LDA_QUEUE_URL": queue_url, "LDA_WATCHDOG_FUNCTION": "lda-batch-watchdog",
            "LDA_STOP_AT_USD": "8.0", "LDA_CAP_USD": "10.0", "LDA_PASS_WORST_USD": "1.40",
            "LDA_SCHEDULE_GROUP": GROUP, "LDA_STATE_MACHINE_ARN": sm_arn, "LDA_BATCH_ID": BATCH_ID,
        }

        def role(cid, name):
            r = iam.Role(self, cid, role_name=name, assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"))
            return r

        def fn(cid, name, handler, r, timeout):
            # Log groups were created (1 month retention, retained) by the first, rolled-back deploy at
            # 12:08Z; they are referenced by name so the retained groups are reused.
            group = logs.LogGroup.from_log_group_name(self, f"{cid}Logs", f"/aws/lambda/{name}")
            r.add_to_policy(iam.PolicyStatement(actions=["logs:CreateLogStream", "logs:PutLogEvents"],
                                                resources=[group.log_group_arn + ":*"]))
            return lambda_.Function(
                self, cid, function_name=name, runtime=lambda_.Runtime.PYTHON_3_12,
                handler=f"batch_lambda.{handler}", role=r, timeout=Duration.seconds(timeout),
                memory_size=256, environment=env, log_group=group,
                code=lambda_.Code.from_asset(str(code_dir), exclude=["__pycache__", "*.pyc"]),
            )

        ssm_rw = iam.PolicyStatement(actions=["ssm:GetParameter", "ssm:PutParameter"], resources=batch_params + [lease_param])
        ecs_svc = iam.PolicyStatement(actions=["ecs:UpdateService", "ecs:DescribeServices"], resources=service_arns)
        s3_put_batch = iam.PolicyStatement(actions=["s3:PutObject"], resources=[f"{bucket_arn}/batch/*"])

        # Watchdog: stops only.
        watchdog_role = role("WatchdogRole", "lda-batch-watchdog-role")
        watchdog_role.add_to_policy(ssm_rw)
        watchdog_role.add_to_policy(ecs_svc)
        watchdog_role.add_to_policy(s3_put_batch)
        watchdog_role.add_to_policy(iam.PolicyStatement(actions=["sqs:GetQueueAttributes"], resources=[queue_arn]))
        watchdog = fn("WatchdogFunction", "lda-batch-watchdog", "watchdog_handler", watchdog_role, 60)

        # Step: drives one pass.
        step_role = role("StepRole", "lda-batch-step-role")
        step_role.add_to_policy(ssm_rw)
        step_role.add_to_policy(ecs_svc)
        step_role.add_to_policy(iam.PolicyStatement(actions=["ecs:ListTasks"], resources=["*"],
                                                    conditions={"ArnEquals": {"ecs:cluster": cluster_arn}}))
        step_role.add_to_policy(iam.PolicyStatement(actions=["ecs:DescribeTasks"], resources=[task_arn]))
        step_role.add_to_policy(iam.PolicyStatement(
            actions=["s3:GetObject"],
            resources=[f"{bucket_arn}/batch/*", f"{bucket_arn}/runs/*"]))
        step_role.add_to_policy(iam.PolicyStatement(
            actions=["s3:PutObject"], resources=[f"{bucket_arn}/batch/*", f"{bucket_arn}/incoming/*"]))
        step_role.add_to_policy(iam.PolicyStatement(
            actions=["s3:ListBucket"], resources=[bucket_arn]))  # HeadObject on a missing key -> 404, not 403
        step_role.add_to_policy(iam.PolicyStatement(actions=["sqs:GetQueueAttributes"], resources=[queue_arn]))
        step_role.add_to_policy(iam.PolicyStatement(actions=["cloudtrail:LookupEvents", "xray:BatchGetTraces"], resources=["*"]))
        step_role.add_to_policy(iam.PolicyStatement(actions=["lambda:InvokeFunction"], resources=[watchdog.function_arn]))
        step = fn("StepFunction", "lda-batch-step", "step_handler", step_role, 120)

        # State machine: Step -> Choice(done) -> Wait($.wait) -> Step. Errors -> EmergencyStop -> Fail.
        emergency = tasks.LambdaInvoke(
            self, "EmergencyStop", lambda_function=watchdog, payload_response_only=True,
            payload=sfn.TaskInput.from_object({"mode": "emergency", "force": True, "batch_id": BATCH_ID}),
            result_path="$.emergency",
        ).next(sfn.Fail(self, "PassFailed", cause="unhandled error in lda-batch-step; services forced to 0"))
        step_task = tasks.LambdaInvoke(
            self, "Step", lambda_function=step, payload_response_only=True,  # whole state is the payload
            retry_on_service_exceptions=False,
        )
        step_task.add_retry(errors=["States.ALL"], max_attempts=3, interval=Duration.seconds(20), backoff_rate=2)
        step_task.add_catch(emergency, errors=["States.ALL"], result_path="$.error")
        wait = sfn.Wait(self, "Wait", time=sfn.WaitTime.seconds_path("$.wait"))
        done = sfn.Choice(self, "Done?").when(sfn.Condition.boolean_equals("$.done", True), sfn.Succeed(self, "PassDone"))
        done.otherwise(wait.next(step_task))
        sm_logs = logs.LogGroup.from_log_group_name(self, "PassLogs", "/aws/vendedlogs/states/lda-batch-pass")
        machine = sfn.StateMachine(
            self, "PassMachine", state_machine_name=sm_name,
            definition_body=sfn.DefinitionBody.from_chainable(step_task.next(done)),
            timeout=Duration.hours(3),
            logs=sfn.LogOptions(destination=sm_logs, level=sfn.LogLevel.ERROR),
        )

        # Dead-man: own role, depends on nothing else in this stack at run time.
        deadman_role = role("DeadmanRole", "lda-batch-deadman-role")
        deadman_role.add_to_policy(ssm_rw)
        deadman_role.add_to_policy(ecs_svc)
        deadman_role.add_to_policy(s3_put_batch)
        deadman_role.add_to_policy(iam.PolicyStatement(actions=["scheduler:ListSchedules"], resources=["*"]))
        deadman_role.add_to_policy(iam.PolicyStatement(
            actions=["scheduler:GetSchedule", "scheduler:UpdateSchedule"],
            resources=[f"arn:aws:scheduler:{region}:{account}:schedule/{GROUP}/*"]))
        deadman_role.add_to_policy(iam.PolicyStatement(actions=["states:ListExecutions"], resources=[sm_arn]))
        deadman_role.add_to_policy(iam.PolicyStatement(
            actions=["states:StopExecution"], resources=[f"arn:aws:states:{region}:{account}:execution:{sm_name}:*"]))
        deadman = fn("DeadmanFunction", "lda-batch-deadman", "deadman_handler", deadman_role, 120)

        # EventBridge Scheduler.
        sched_role = iam.Role(self, "SchedulerRole", role_name="lda-batch-scheduler-role",
                              assumed_by=iam.ServicePrincipal("scheduler.amazonaws.com",
                                                              conditions={"StringEquals": {"aws:SourceAccount": account}}))
        sched_role.add_to_policy(iam.PolicyStatement(actions=["states:StartExecution"], resources=[sm_arn]))
        sched_role.add_to_policy(iam.PolicyStatement(actions=["lambda:InvokeFunction"],
                                                     resources=[watchdog.function_arn, deadman.function_arn]))
        deadman_role.add_to_policy(iam.PolicyStatement(actions=["iam:PassRole"], resources=[sched_role.role_arn],
                                                       conditions={"StringEquals": {"iam:PassedToService": "scheduler.amazonaws.com"}}))
        groups = {}
        for cid, name in (("Group", GROUP), ("DeadmanGroup", DEADMAN_GROUP), ("TestGroup", TEST_GROUP)):
            groups[name] = scheduler.CfnScheduleGroup(self, cid, name=name)
        off = scheduler.CfnSchedule.FlexibleTimeWindowProperty(mode="OFF")
        retry = scheduler.CfnSchedule.RetryPolicyProperty(maximum_event_age_in_seconds=3600, maximum_retry_attempts=3)
        for index, (pass_id, at) in enumerate(PASS_TIMES):
            s = scheduler.CfnSchedule(
                self, f"Schedule{pass_id.title()}", name=f"lda-batch-{pass_id}", group_name=GROUP,
                schedule_expression=f"at({at})", schedule_expression_timezone="UTC", flexible_time_window=off,
                state="ENABLED", description=f"lda batch {BATCH_ID} {pass_id}: all 12 clips",
                target=scheduler.CfnSchedule.TargetProperty(
                    arn=machine.state_machine_arn, role_arn=sched_role.role_arn, retry_policy=retry,
                    input=json.dumps({"batch_id": BATCH_ID, "pass_id": pass_id, "clips": "all",
                                      "final_pass": index == len(PASS_TIMES) - 1})),
            )
            s.add_dependency(groups[GROUP])
        s = scheduler.CfnSchedule(
            self, "ScheduleWatchdog", name="lda-batch-watchdog", group_name=GROUP,
            schedule_expression="rate(10 minutes)", flexible_time_window=off, state="ENABLED",
            start_date=WATCHDOG_WINDOW[0], end_date=WATCHDOG_WINDOW[1],
            description="lda batch watchdog tick (stops idle services held by the batch lease)",
            target=scheduler.CfnSchedule.TargetProperty(
                arn=watchdog.function_arn, role_arn=sched_role.role_arn, retry_policy=retry,
                input=json.dumps({"mode": "tick", "batch_id": BATCH_ID})),
        )
        s.add_dependency(groups[GROUP])
        for name, at in DEADMAN_TIMES:
            s = scheduler.CfnSchedule(
                self, f"Schedule{name.title()}", name=f"lda-batch-{name}", group_name=DEADMAN_GROUP,
                schedule_expression=f"at({at})", schedule_expression_timezone="UTC", flexible_time_window=off,
                state="ENABLED", description="lda batch dead-man: kill switch off, schedules off, services 0",
                target=scheduler.CfnSchedule.TargetProperty(
                    arn=deadman.function_arn, role_arn=sched_role.role_arn, retry_policy=retry,
                    input=json.dumps({"dry_run": False, "batch_id": BATCH_ID, "label": name})),
            )
            s.add_dependency(groups[DEADMAN_GROUP])

        for key, value in (("StateMachineArn", machine.state_machine_arn), ("StepFunctionName", step.function_name),
                           ("WatchdogFunctionName", watchdog.function_name), ("DeadmanFunctionName", deadman.function_name),
                           ("SchedulerRoleArn", sched_role.role_arn), ("BatchId", BATCH_ID), ("BucketName", bucket),
                           ("ScheduleGroup", GROUP), ("DeadmanScheduleGroup", DEADMAN_GROUP), ("TestScheduleGroup", TEST_GROUP)):
            CfnOutput(self, key, value=value)

        NagSuppressions.add_stack_suppressions(self, [
            NagPackSuppression(id="AwsSolutions-IAM5", reason="Prefix-scoped S3/SSM/ECS-task ARNs; CloudTrail LookupEvents, X-Ray BatchGetTraces, ecs:ListTasks (cluster-conditioned) and scheduler:ListSchedules have no resource-level scoping."),
            NagPackSuppression(id="AwsSolutions-L1", reason="Python 3.12 matches the existing lda-reaper runtime; one-day batch."),
            NagPackSuppression(id="AwsSolutions-SF1", reason="ERROR-level logging is enough for a one-day batch; every pass writes its own manifest to S3."),
            NagPackSuppression(id="AwsSolutions-SF2", reason="X-Ray tracing of the batch driver is not needed; the pipeline itself is traced."),
            NagPackSuppression(id="AwsSolutions-IAM4", reason="No managed policies are attached; suppression is defensive."),
        ])
