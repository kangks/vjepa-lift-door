from __future__ import annotations

import json
import re
from pathlib import Path

from aws_cdk import (
    CfnOutput, Duration, Fn, RemovalPolicy, Size, Stack, Tags,
    aws_servicediscovery as cloudmap, aws_dynamodb as dynamodb, aws_ec2 as ec2,
    aws_ecr as ecr, aws_ecs as ecs, aws_events as events,
    aws_events_targets as event_targets, aws_iam as iam, aws_lambda as lambda_,
    aws_logs as logs, aws_s3 as s3, aws_sns as sns, aws_sqs as sqs,
    aws_ssm as ssm,
)
from cdk_nag import NagPackSuppression, NagSuppressions
from constructs import Construct
from .pins import CKPT_SHA256

# DoiT Attribute (attrb) ECS Fargate sensor. Enabled only for the profiles listed
# here (the bearer-token SSM parameter exists only in attrb_sandbox), so the
# ml-sandbox template is unchanged. Pattern copied from the working
# luminara-v2-coordinator-fargate task definition in attrb_sandbox.
ATTRB_SENSOR_PROFILES = frozenset({"attrb_sandbox"})
ATTRB_SENSOR_IMAGE = "quay.io/attribute/sensor:0.0.286"  # OCI index includes linux/arm64
ATTRB_TOKEN_PARAMETER = "/lift-door-agentic/attribute/bearer_token"
ATTRB_OTEL_ADDR = "otel-endpoint.app.attrb.io:443"
ATTRB_VOLUME = "attribute-shared"
ATTRB_MOUNT = "/opt/attrb"
ATTRB_LAUNCHER = f"{ATTRB_MOUNT}/launch/ecs-fargate"
ATTRB_LOADER_NAME = "attribute-sensor-loader"


class MainStack(Stack):
    def __init__(
        self, scope: Construct, construct_id: str, *, profile: str, account: str, region: str,
        azs: list[str], build_stack, image_tag: str, image_digests_path: str | None = None, **kwargs,
    ):
        super().__init__(scope, construct_id, **kwargs)
        self.profile = profile
        self.attrb_sensor = profile in ATTRB_SENSOR_PROFILES
        self.build_stack = build_stack
        self.data_bucket = build_stack.data_bucket
        self.repositories = dict(build_stack.repositories)
        for name in ("knn", "ticket"):
            repo_name = f"lda-{name}"
            self.repositories[name] = ecr.Repository.from_repository_attributes(
                self, f"{name.title()}LambdaRepository",
                repository_arn=build_stack.repositories[name].repository_arn,
                repository_name=repo_name,
            )
        self.image_tag = image_tag
        self.image_digests = self._load_image_digests(image_digests_path)

        # Network: public subnets supply outbound IPv4 without a NAT gateway.
        self.vpc = ec2.Vpc(
            self, "Vpc", vpc_name="lda-vpc", availability_zones=azs,
            nat_gateways=0, restrict_default_security_group=False,
            subnet_configuration=[
                ec2.SubnetConfiguration(name="public", subnet_type=ec2.SubnetType.PUBLIC),
            ],
        )
        self.namespace = cloudmap.PrivateDnsNamespace(
            self, "PrivateNamespace", name="lda.local", vpc=self.vpc,
        )
        self.cluster = ecs.Cluster(
            self, "Cluster", cluster_name="lda",
            vpc=self.vpc, container_insights=False,
        )

        self.log_groups = {
            "coordinator": self._log_group("CoordinatorLogGroup", "/lda/ecs/coordinator"),
            "embed": self._log_group("EmbedLogGroup", "/lda/ecs/embed"),
            "knn": self._log_group("KnnLogGroup", "/lda/lambda/knn"),
            "ticket": self._log_group("TicketLogGroup", "/lda/lambda/ticket"),
            "reaper": self._log_group("ReaperLogGroup", "/lda/lambda/reaper"),
            "agent": self._log_group("AgentLogGroup", "/lda/agent/coordinator"),
        }
        self.agent_log_stream = logs.CfnLogStream(
            self, "AgentRuntimeLogStream", log_group_name=self.log_groups["agent"].log_group_name,
            log_stream_name="runtime-logs",
        )

        self.queue = sqs.Queue(
            self, "DoorEventsQueue", queue_name="lda-door-events",
            encryption=sqs.QueueEncryption.SQS_MANAGED,
            enforce_ssl=True,
            visibility_timeout=Duration.seconds(1800),
            retention_period=Duration.days(4),
            dead_letter_queue=sqs.DeadLetterQueue(
                max_receive_count=2,
                queue=sqs.Queue(
                    self, "DoorEventsDlq", queue_name="lda-door-events-dlq",
                    encryption=sqs.QueueEncryption.SQS_MANAGED,
                    enforce_ssl=True, retention_period=Duration.days(4),
                ),
            ),
        )
        events.Rule(
            self, "IncomingObjectCreatedRule",
            rule_name="lda-incoming-object-created",
            event_bus=events.EventBus.from_event_bus_name(self, "DefaultEventBus", "default"),
            event_pattern=events.EventPattern(
                source=["aws.s3"],
                detail_type=["Object Created"],
                detail={
                    "bucket": {"name": [self.data_bucket.bucket_name]},
                    "object": {"key": [{"prefix": "incoming/"}]},
                },
            ),
            targets=[event_targets.SqsQueue(self.queue)],
        )

        self.coordinator_security_group = ec2.SecurityGroup(
            self, "CoordinatorSecurityGroup", vpc=self.vpc,
            security_group_name="lda-coordinator", description="Coordinator task outbound access",
            allow_all_outbound=True,
        )
        self.embed_security_group = ec2.SecurityGroup(
            self, "EmbedSecurityGroup", vpc=self.vpc,
            security_group_name="lda-embed", description="Private embed API ingress from coordinator",
            allow_all_outbound=True,
        )
        self.embed_security_group.add_ingress_rule(
            self.coordinator_security_group, ec2.Port.tcp(8080), "Coordinator calls embed API",
        )

        self.ticket_table = dynamodb.Table(
            self, "MockTicketsTable",
            table_name="lda-mock-tickets",
            partition_key=dynamodb.Attribute(name="ticket_id", type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            encryption=dynamodb.TableEncryption.DEFAULT,
            point_in_time_recovery=False,
            removal_policy=RemovalPolicy.DESTROY,
        )
        self.ticket_table.add_global_secondary_index(
            index_name="by_run_id",
            partition_key=dynamodb.Attribute(name="run_id", type=dynamodb.AttributeType.STRING),
            projection_type=dynamodb.ProjectionType.ALL,
        )
        NagSuppressions.add_resource_suppressions(
            self.ticket_table,
            [NagPackSuppression(
                id="AwsSolutions-DDB3",
                reason="PITR is disabled for the tiny mock ticket table to avoid unnecessary demo cost; tickets are demonstration records and the plan freezes PITR off.",
            )],
        )

        self.parameter = ssm.StringParameter(
            self, "LeaseParameter",
            parameter_name="/lda/lease_until",
            string_value="1970-01-01T00:00:00Z",
            tier=ssm.ParameterTier.STANDARD,
        )

        self.embed_role = iam.Role(
            self, "EmbedTaskRole", role_name="lda-embed-task",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
        )
        self.coordinator_role = iam.Role(
            self, "CoordinatorTaskRole", role_name="lda-coordinator-task",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
        )
        self._add_s3_policy(
            self.embed_role, ["s3:GetObject"],
            ["models/*", "incoming/*", "dataset/*"],
        )
        self._add_s3_policy(self.embed_role, ["s3:PutObject"], ["runs/*", "bank/*"])
        self._add_s3_policy(self.coordinator_role, ["s3:GetObject"], ["runs/*", "bank/*", "incoming/*"])
        self._add_s3_policy(self.coordinator_role, ["s3:PutObject"], ["runs/*"])
        self.coordinator_role.add_to_policy(iam.PolicyStatement(
            actions=[
                "sqs:ReceiveMessage", "sqs:DeleteMessage", "sqs:ChangeMessageVisibility",
                "sqs:GetQueueAttributes",
            ],
            resources=[self.queue.queue_arn],
        ))
        self._add_telemetry_policy(self.embed_role, [self.log_groups["embed"]])
        self._add_telemetry_policy(self.coordinator_role, [self.log_groups["agent"]])

        # attrb sensor: env + secret for the traced app containers. The embed task is
        # already at the Fargate 16 vCPU maximum with 32 GiB, which leaves ample room
        # for the sensor (~0.1-0.2 vCPU, 0.3-1 GiB); the coordinator tasks get
        # 0.5 -> 1 vCPU and 1 -> 2 GiB (a valid Fargate combination).
        sensor_env = {}
        sensor_secrets = {}
        coordinator_cpu, coordinator_memory = 512, 1024
        if self.attrb_sensor:
            token_parameter = ssm.StringParameter.from_secure_string_parameter_attributes(
                self, "AttrbSensorTokenParameter", parameter_name=ATTRB_TOKEN_PARAMETER,
            )
            sensor_env = {"CONFIG_ZPROBE_OTEL_ADDR": ATTRB_OTEL_ADDR}
            sensor_secrets = {
                "CONFIG_ZPROBE_BEARER_TOKEN": ecs.Secret.from_ssm_parameter(token_parameter),
            }
            coordinator_cpu, coordinator_memory = 1024, 2048

        self.embed_task = ecs.FargateTaskDefinition(
            self, "EmbedTaskDefinition", family="lda-embed",
            cpu=16384, memory_limit_mib=32768, ephemeral_storage_gib=21,
            runtime_platform=ecs.RuntimePlatform(
                cpu_architecture=ecs.CpuArchitecture.ARM64,
                operating_system_family=ecs.OperatingSystemFamily.LINUX,
            ),
            task_role=self.embed_role,
        )
        self.coordinator_task = ecs.FargateTaskDefinition(
            self, "CoordinatorTaskDefinition", family="lda-coordinator",
            cpu=coordinator_cpu, memory_limit_mib=coordinator_memory,
            runtime_platform=ecs.RuntimePlatform(
                cpu_architecture=ecs.CpuArchitecture.ARM64,
                operating_system_family=ecs.OperatingSystemFamily.LINUX,
            ),
            task_role=self.coordinator_role,
        )
        for task in (self.embed_task, self.coordinator_task):
            NagSuppressions.add_resource_suppressions(
                task,
                [NagPackSuppression(
                    id="AwsSolutions-ECS2",
                    reason="Task environment values are non-secret service configuration; all AWS credentials come from the task role.",
                )],
            )
        self.embed_container = self.embed_task.add_container(
            "EmbedContainer",
            image=ecs.ContainerImage.from_ecr_repository(self.repositories["embed"], image_tag),
            environment={
                "LDA_BUCKET": self.data_bucket.bucket_name,
                "LDA_CKPT_PATH": "/tmp/model/vitl_ema_encoder_only.pt",
                "LDA_CKPT_SHA256": CKPT_SHA256,
                "LDA_CKPT_S3_KEY": "models/vjepa21/vitl_ema_encoder_only.pt",
                "LDA_OTEL_MODE": "aws",
                "AWS_REGION": region,
                "OTEL_PYTHON_DISTRO": "aws_distro",
                "OTEL_PYTHON_CONFIGURATOR": "aws_configurator",
                "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
                "OTEL_TRACES_EXPORTER": "otlp",
                "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": (
                    f"https://xray.{region}.amazonaws.com/v1/traces"
                ),
                "OTEL_LOGS_EXPORTER": "none",
                "OTEL_METRICS_EXPORTER": "none",
                "OTEL_RESOURCE_ATTRIBUTES": "service.name=lda-embed",
                **({"OTEL_SERVICE_NAME": "lda-embed", **sensor_env} if self.attrb_sensor else {}),
            },
            **self._sensor_wrap(["/app/services/embed/entrypoint.sh"], sensor_secrets),
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix="embed", log_group=self.log_groups["embed"],
            ),
            health_check=ecs.HealthCheck(
                command=[
                    "CMD-SHELL",
                    "python -c \"import urllib.request,sys; "
                    "sys.exit(0 if urllib.request.urlopen('http://localhost:8080/health', timeout=5).status==200 else 1)\"",
                ],
                interval=Duration.seconds(30),
                timeout=Duration.seconds(10),
                retries=3,
                start_period=Duration.seconds(300),
            ),
        )
        self.embed_container.add_port_mappings(ecs.PortMapping(container_port=8080))
        self._add_sensor_loader(
            self.embed_task, self.embed_container,
            ecs.LogDrivers.aws_logs(stream_prefix="attrb-loader", log_group=self.log_groups["embed"]),
        )

        coordinator_environment = {
            "LDA_BUCKET": self.data_bucket.bucket_name,
            "LDA_QUEUE_URL": self.queue.queue_url,
            "LDA_EMBED_URL": "http://embed.lda.local:8080",
            "LDA_KNN_FUNCTION": "lda-knn",
            "LDA_TICKET_FUNCTION": "lda-ticket",
            "LDA_TICKET_TABLE": self.ticket_table.table_name,
            "LDA_REASONER_MODEL": "anthropic.claude-opus-5-5",
            "LDA_REASONER_MODE": "bedrock",
            "LDA_CKPT_SHA256": CKPT_SHA256,
            "LDA_OTEL_MODE": "aws",
            "AWS_REGION": region,
            "AGENT_OBSERVABILITY_ENABLED": "true",
            "OTEL_PYTHON_DISTRO": "aws_distro",
            "OTEL_PYTHON_CONFIGURATOR": "aws_configurator",
            "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
            "OTEL_TRACES_EXPORTER": "otlp",
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": (
                f"https://xray.{region}.amazonaws.com/v1/traces"
            ),
            "OTEL_LOGS_EXPORTER": "otlp",
            "OTEL_RESOURCE_ATTRIBUTES": (
                "service.name=lda-coordinator,aws.log.group.names=/lda/agent/coordinator"
            ),
            "OTEL_EXPORTER_OTLP_LOGS_HEADERS": (
                "x-aws-log-group=/lda/agent/coordinator,"
                "x-aws-log-stream=runtime-logs,x-aws-metric-namespace=bedrock-agentcore"
            ),
        }
        if self.attrb_sensor:
            # OTEL_SERVICE_NAME equals the ECS service name and the existing
            # service.name resource attribute (the one-off shares this env).
            coordinator_environment.update({"OTEL_SERVICE_NAME": "lda-coordinator", **sensor_env})
        self.coordinator_container = self.coordinator_task.add_container(
            "CoordinatorContainer",
            image=ecs.ContainerImage.from_ecr_repository(self.repositories["coordinator"], image_tag),
            environment=coordinator_environment,
            **self._sensor_wrap(["/app/services/coordinator/entrypoint.sh"], sensor_secrets),
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix="coordinator", log_group=self.log_groups["coordinator"],
            ),
        )
        self.coordinator_container.add_environment("LDA_KNN_FUNCTION", "lda-knn")
        self.coordinator_container.add_environment("LDA_TICKET_FUNCTION", "lda-ticket")
        self._add_sensor_loader(
            self.coordinator_task, self.coordinator_container,
            ecs.LogDrivers.aws_logs(stream_prefix="attrb-loader", log_group=self.log_groups["coordinator"]),
        )
        self.oneoff_task = ecs.FargateTaskDefinition(
            self, "OneoffTaskDefinition", family="lda-coordinator-oneoff",
            cpu=coordinator_cpu, memory_limit_mib=coordinator_memory,
            execution_role=self.coordinator_task.execution_role,
            task_role=self.coordinator_role,
            runtime_platform=ecs.RuntimePlatform(
                cpu_architecture=ecs.CpuArchitecture.ARM64,
                operating_system_family=ecs.OperatingSystemFamily.LINUX,
            ),
        )
        NagSuppressions.add_resource_suppressions(
            self.oneoff_task,
            [NagPackSuppression(
                id="AwsSolutions-ECS2",
                reason="Task environment values are non-secret service configuration; all AWS credentials come from the task role.",
            )],
        )
        self.oneoff_container = self.oneoff_task.add_container(
            "OneoffContainer",
            image=ecs.ContainerImage.from_registry(
                self.repositories["coordinator"].repository_uri + f":{image_tag}"
            ),
            **(
                self._sensor_wrap(["opentelemetry-instrument", "python", "-u"], sensor_secrets)
                or {"entry_point": ["opentelemetry-instrument", "python", "-u"]}
            ),
            command=["-c", "print('lda oneoff: pass a command override')"],
            environment=coordinator_environment,
            logging=ecs.GenericLogDriver(
                log_driver="awslogs",
                options={
                    "awslogs-group": self.log_groups["coordinator"].log_group_name,
                    "awslogs-region": region,
                    "awslogs-stream-prefix": "oneoff",
                },
            ),
        )
        self._add_sensor_loader(
            self.oneoff_task, self.oneoff_container,
            ecs.GenericLogDriver(
                log_driver="awslogs",
                options={
                    "awslogs-group": self.log_groups["coordinator"].log_group_name,
                    "awslogs-region": region,
                    "awslogs-stream-prefix": "oneoff-attrb-loader",
                },
            ),
        )
        for task in (self.embed_task, self.coordinator_task):
            NagSuppressions.add_resource_suppressions(
                task.execution_role,
                [NagPackSuppression(
                    id="AwsSolutions-IAM5",
                    reason="The CDK-generated task execution role uses the approved AWS-managed execution policy for ECR pulls and awslogs.",
                    applies_to=["Resource::*"],
                )],
                apply_to_children=True,
            )

        self.embed_service = ecs.FargateService(
            self, "EmbedService", service_name="lda-embed",
            cluster=self.cluster, task_definition=self.embed_task, desired_count=0,
            assign_public_ip=True, security_groups=[self.embed_security_group],
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
            min_healthy_percent=0, max_healthy_percent=100,
            health_check_grace_period=Duration.seconds(300),
            circuit_breaker=ecs.DeploymentCircuitBreaker(enable=True, rollback=True),
            **self._service_tag_options(),
            cloud_map_options=ecs.CloudMapOptions(
                name="embed", cloud_map_namespace=self.namespace,
                dns_record_type=cloudmap.DnsRecordType.A,
                container=self.embed_container, container_port=8080,
            ),
        )
        self.coordinator_service = ecs.FargateService(
            self, "CoordinatorService", service_name="lda-coordinator",
            cluster=self.cluster, task_definition=self.coordinator_task, desired_count=0,
            assign_public_ip=True, security_groups=[self.coordinator_security_group],
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
            min_healthy_percent=0, max_healthy_percent=100,
            circuit_breaker=ecs.DeploymentCircuitBreaker(enable=True, rollback=True),
            **self._service_tag_options(),
        )

        self.knn_role = iam.Role(
            self, "KnnFunctionRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
        )
        self.ticket_role = iam.Role(
            self, "TicketFunctionRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
        )
        self.reaper_role = iam.Role(
            self, "ReaperFunctionRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
        )
        self._add_s3_policy(self.knn_role, ["s3:GetObject"], ["bank/*", "runs/*"])
        self._add_s3_policy(self.knn_role, ["s3:PutObject"], ["runs/*", "bank/*"])
        self._add_telemetry_policy(self.knn_role, [self.log_groups["knn"]])
        self._add_telemetry_policy(self.ticket_role, [self.log_groups["ticket"]])
        self.ticket_role.add_to_policy(iam.PolicyStatement(
            actions=["dynamodb:PutItem", "dynamodb:GetItem"],
            resources=[self.ticket_table.table_arn],
        ))
        self.reaper_role.add_to_policy(iam.PolicyStatement(
            actions=["ssm:GetParameter"], resources=[self.parameter.parameter_arn],
        ))
        self.reaper_role.add_to_policy(iam.PolicyStatement(
            actions=["ecs:UpdateService", "ecs:DescribeServices"],
            resources=[self.embed_service.service_arn, self.coordinator_service.service_arn],
        ))

        self.knn_function = lambda_.DockerImageFunction(
            self, "KnnFunction", function_name="lda-knn",
            code=self._docker_code("knn"),
            architecture=lambda_.Architecture.ARM_64,
            memory_size=4096, timeout=Duration.seconds(120),
            ephemeral_storage_size=Size.mebibytes(2048),
            tracing=lambda_.Tracing.PASS_THROUGH,
            role=self.knn_role,
            log_group=self.log_groups["knn"],
            environment={
                "LDA_BUCKET": self.data_bucket.bucket_name,
                "LDA_OTEL_MODE": "aws",
                "OTEL_SERVICE_NAME": "lda-knn",
            },
        )
        self.ticket_function = lambda_.DockerImageFunction(
            self, "TicketFunction", function_name="lda-ticket",
            code=self._docker_code("ticket"),
            architecture=lambda_.Architecture.ARM_64,
            memory_size=256, timeout=Duration.seconds(30),
            tracing=lambda_.Tracing.PASS_THROUGH,
            role=self.ticket_role,
            log_group=self.log_groups["ticket"],
            environment={
                "LDA_BUCKET": self.data_bucket.bucket_name,
                "LDA_TICKET_TABLE": self.ticket_table.table_name,
                "LDA_OTEL_MODE": "aws",
                "OTEL_SERVICE_NAME": "lda-ticket",
            },
        )
        self.reaper_function = lambda_.Function(
            self, "ReaperFunction", function_name="lda-reaper",
            runtime=lambda_.Runtime.PYTHON_3_12,
            architecture=lambda_.Architecture.ARM_64,
            handler="handler.handler",
            code=lambda_.Code.from_asset(str(Path(__file__).resolve().parents[2] / "services/reaper")),
            memory_size=128, timeout=Duration.seconds(30),
            tracing=lambda_.Tracing.PASS_THROUGH,
            role=self.reaper_role,
            log_group=self.log_groups["reaper"],
            environment={
                "LDA_LEASE_PARAMETER": self.parameter.parameter_name,
                "LDA_SERVICES": json.dumps([
                    {"cluster": self.cluster.cluster_name, "service": self.embed_service.service_name},
                    {"cluster": self.cluster.cluster_name, "service": self.coordinator_service.service_name},
                ]),
            },
        )
        NagSuppressions.add_resource_suppressions(
            self.reaper_function,
            [NagPackSuppression(
                id="AwsSolutions-L1",
                reason="Python 3.12 is the runtime explicitly frozen for the reaper in STEP_04.",
            )],
        )
        events.Rule(
            self, "IdleReaperSchedule",
            rule_name="lda-idle-reaper",
            schedule=events.Schedule.rate(Duration.minutes(15)),
            targets=[event_targets.LambdaFunction(self.reaper_function)],
        )

        self.coordinator_role.add_to_policy(iam.PolicyStatement(
            actions=["lambda:InvokeFunction"],
            resources=[self.knn_function.function_arn, self.ticket_function.function_arn],
        ))
        self.coordinator_role.add_to_policy(iam.PolicyStatement(
            actions=["bedrock-mantle:CreateInference"],
            resources=[f"arn:aws:bedrock-mantle:{region}:{account}:project/*"],
            conditions={"StringEquals": {"bedrock-mantle:Model": "anthropic.claude-opus-5-5"}},
        ))

        CfnOutput(self, "BucketName", value=self.data_bucket.bucket_name)
        CfnOutput(self, "QueueUrl", value=self.queue.queue_url)
        CfnOutput(self, "ClusterName", value=self.cluster.cluster_name)
        CfnOutput(self, "EmbedServiceName", value=self.embed_service.service_name)
        CfnOutput(self, "CoordinatorServiceName", value=self.coordinator_service.service_name)
        CfnOutput(self, "CoordinatorTaskDefinitionArn", value=self.coordinator_task.task_definition_arn)
        CfnOutput(self, "OneoffTaskDefinitionArn", value=self.oneoff_task.task_definition_arn)
        CfnOutput(
            self, "PublicSubnetIds",
            value=Fn.join(",", [subnet.subnet_id for subnet in self.vpc.public_subnets]),
        )
        CfnOutput(self, "CoordinatorSecurityGroupId", value=self.coordinator_security_group.security_group_id)
        CfnOutput(self, "KnnFunctionName", value=self.knn_function.function_name)
        CfnOutput(self, "TicketFunctionName", value=self.ticket_function.function_name)
        CfnOutput(self, "TicketTableName", value=self.ticket_table.table_name)

        self._suppress_findings()

    @staticmethod
    def _load_image_digests(path):
        if not path:
            return {}
        source = Path(path)
        if not source.is_absolute():
            source = Path.cwd() / source
        values = json.loads(source.read_text())
        images = values.get("images", values)
        parsed = {}
        for name in ("knn", "ticket"):
            item = images.get(name, {})
            digest = item.get("digest")
            if digest:
                parsed[name] = digest.rsplit("@", 1)[-1]
                if not re.fullmatch(r"sha256:[0-9a-f]{64}", parsed[name]):
                    raise ValueError(f"invalid image digest for {name}: {digest}")
        return parsed

    def _sensor_wrap(self, original_entrypoint, secrets):
        """Container kwargs that run the original entrypoint under the attrb launcher.

        The launcher is prefixed to the ENTRYPOINT (rather than moving the original
        entrypoint into COMMAND) so RunTask command overrides keep working: the
        process argv is launcher + original entrypoint + command either way.
        """
        if not self.attrb_sensor:
            return {}
        return {"entry_point": [ATTRB_LAUNCHER, *original_entrypoint], "secrets": secrets}

    def _add_sensor_loader(self, task, app_container, logging):
        if not self.attrb_sensor:
            return None
        task.add_volume(name=ATTRB_VOLUME, host=ecs.Host())
        loader = task.add_container(
            "AttrbSensorLoader", container_name=ATTRB_LOADER_NAME,
            image=ecs.ContainerImage.from_registry(ATTRB_SENSOR_IMAGE),
            essential=False, cpu=0,
            entry_point=["load_attribute"], command=[ATTRB_MOUNT],
            logging=logging,
        )
        loader.add_mount_points(ecs.MountPoint(
            container_path=ATTRB_MOUNT, source_volume=ATTRB_VOLUME, read_only=False,
        ))
        app_container.add_mount_points(ecs.MountPoint(
            container_path=ATTRB_MOUNT, source_volume=ATTRB_VOLUME, read_only=True,
        ))
        app_container.add_container_dependencies(ecs.ContainerDependency(
            container=loader, condition=ecs.ContainerDependencyCondition.COMPLETE,
        ))
        return loader

    def _service_tag_options(self):
        if not self.attrb_sensor:
            return {}
        return {"propagate_tags": ecs.PropagatedTagSource.SERVICE, "enable_ecs_managed_tags": True}

    def _docker_code(self, name):
        digest = self.image_digests.get(name)
        return lambda_.DockerImageCode.from_ecr(
            self.repositories[name],
            tag_or_digest=digest or self.image_tag,
        )

    def _log_group(self, construct_id, group_name):
        return logs.LogGroup(
            self, construct_id, log_group_name=group_name,
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.RETAIN,
        )

    def _add_s3_policy(self, role, actions, prefixes):
        role.add_to_policy(iam.PolicyStatement(
            actions=actions,
            resources=[f"{self.data_bucket.bucket_arn}/{prefix}" for prefix in prefixes],
        ))

    def _add_telemetry_policy(self, role, log_groups):
        role.add_to_policy(iam.PolicyStatement(
            actions=[
                "xray:PutTraceSegments", "xray:PutTelemetryRecords",
                "xray:GetSamplingRules", "xray:GetSamplingTargets",
            ],
            resources=["*"],
        ))
        group_arns = [group.log_group_arn + ":*" for group in log_groups]
        role.add_to_policy(iam.PolicyStatement(
            actions=["logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams"],
            resources=group_arns,
        ))

    def _suppress_findings(self):
        # The application uses explicit names, short-lived demo services, and
        # public-subnet egress without a NAT; these are all documented choices.
        NagSuppressions.add_resource_suppressions(
            self.vpc,
            [
                NagPackSuppression(
                    id="AwsSolutions-VPC7",
                    reason="VPC flow logs add cost; the workshop acceptance traces and application logs provide the required demo evidence.",
                ),
            ],
            apply_to_children=True,
        )
        NagSuppressions.add_resource_suppressions(
            self.cluster,
            [
                NagPackSuppression(
                    id="AwsSolutions-ECS4",
                    reason="Container Insights is disabled to avoid idle CloudWatch cost; detailed spans and service logs are retained.",
                ),
            ],
            apply_to_children=True,
        )
        for role in (self.embed_role, self.coordinator_role, self.knn_role, self.ticket_role, self.reaper_role):
            NagSuppressions.add_resource_suppressions(
                role,
                [
                    NagPackSuppression(
                        id="AwsSolutions-IAM5",
                        reason="Prefix-scoped S3 and CloudWatch resources are least-privilege; X-Ray ingestion and sampling actions require Resource '*'.",
                    ),
                ],
                apply_to_children=True,
            )
        for service in (self.embed_service, self.coordinator_service):
            NagSuppressions.add_resource_suppressions(
                service,
                [
                    NagPackSuppression(
                        id="AwsSolutions-ECS2",
                        reason="The task environment contains only non-secret configuration; AWS credentials come from the task role.",
                    ),
                    NagPackSuppression(
                        id="AwsSolutions-EC23",
                        reason="Task egress uses public IPv4 without NAT for cost; inbound is restricted to coordinator-to-embed TCP 8080.",
                    ),
                ],
                apply_to_children=True,
            )
        for function in (self.knn_function, self.ticket_function, self.reaper_function):
            NagSuppressions.add_resource_suppressions(
                function,
                [
                    NagPackSuppression(
                        id="AwsSolutions-IAM4",
                        reason="Lambda execution roles use only the AWSLambdaBasicExecutionRole logging policy; service permissions are explicit.",
                    ),
                    NagPackSuppression(
                        id="AwsSolutions-IAM5",
                        reason="X-Ray ingestion/sampling and scoped log-stream resources require the documented telemetry actions.",
                    ),
                ],
                apply_to_children=True,
            )
