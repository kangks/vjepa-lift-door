from __future__ import annotations

from aws_cdk import (
    ArnFormat, CfnOutput, Duration, RemovalPolicy, Stack, Tags, aws_codebuild as codebuild,
    aws_ecr as ecr, aws_iam as iam, aws_logs as logs, aws_s3 as s3,
)
from cdk_nag import NagPackSuppression, NagSuppressions
from constructs import Construct


class BuildStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, *, profile: str, account: str, region: str, **kwargs):
        super().__init__(scope, construct_id, **kwargs)
        self.profile = profile

        self.data_bucket = s3.Bucket(
            self, "DataBucket",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            versioned=False,
            removal_policy=RemovalPolicy.RETAIN,
            auto_delete_objects=False,
        )
        cfn_bucket = self.data_bucket.node.default_child
        cfn_bucket.add_property_override(
            "NotificationConfiguration.EventBridgeConfiguration.EventBridgeEnabled", True,
        )

        self.repositories = {
            name: ecr.Repository(
                self, f"{name.title()}Repository",
                repository_name=f"lda-{name}",
                image_scan_on_push=True,
                image_tag_mutability=ecr.TagMutability.IMMUTABLE,
                encryption=ecr.RepositoryEncryption.AES_256,
                lifecycle_rules=[ecr.LifecycleRule(max_image_count=5)],
                empty_on_delete=True,
                removal_policy=RemovalPolicy.DESTROY,
            )
            for name in ("embed", "knn", "ticket", "coordinator")
        }
        for name in ("knn", "ticket"):
            function_name = f"lda-{name}"
            self.repositories[name].add_to_resource_policy(iam.PolicyStatement(
                sid="LambdaImagePullScoped",
                principals=[iam.ServicePrincipal("lambda.amazonaws.com")],
                actions=["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"],
                conditions={
                    "StringEquals": {"aws:SourceAccount": Stack.of(self).account},
                    "ArnLike": {
                        "aws:SourceArn": [
                            Stack.of(self).format_arn(
                                service="lambda", resource="function",
                                resource_name=function_name,
                                arn_format=ArnFormat.COLON_RESOURCE_NAME,
                            ),
                            Stack.of(self).format_arn(
                                service="lambda", resource="function",
                                resource_name=f"{function_name}:*",
                                arn_format=ArnFormat.COLON_RESOURCE_NAME,
                            ),
                        ],
                    },
                },
            ))
        # CloudFormation currently omits the default AES256 value from the
        # synthesized resource. Keep the requested encryption mode explicit.
        for repository in self.repositories.values():
            repository.node.default_child.add_property_override(
                "EncryptionConfiguration.EncryptionType", "AES256",
            )

        self.image_log_group = logs.LogGroup(
            self, "ImageBuildLogGroup", log_group_name="/lda/codebuild/images",
            retention=logs.RetentionDays.ONE_MONTH, removal_policy=RemovalPolicy.RETAIN,
        )
        self.e2e_log_group = logs.LogGroup(
            self, "E2eBuildLogGroup", log_group_name="/lda/codebuild/e2e",
            retention=logs.RetentionDays.ONE_MONTH, removal_policy=RemovalPolicy.RETAIN,
        )

        image_project = self._project(
            "ImageBuildCodeBuildProject", "lda-image-build", "buildspec-images.yml", 60,
            self.image_log_group, account, region,
        )
        e2e_project = self._project(
            "E2eCodeBuildProject", "lda-e2e-test", "buildspec-e2e.yml", 120,
            self.e2e_log_group, account, region,
        )
        self.image_project = image_project
        self.e2e_project = e2e_project

        for project, mode in ((image_project, "images"), (e2e_project, "e2e")):
            self._grant_codebuild(project, mode)

        NagSuppressions.add_resource_suppressions(
            self.data_bucket,
            [NagPackSuppression(
                id="AwsSolutions-S1",
                reason="The private project bucket is SSL-only and retained; server access logging would add a second bucket and fixed log-storage costs beyond the frozen acceptance.",
            )],
        )

        CfnOutput(self, "BucketName", value=self.data_bucket.bucket_name)
        CfnOutput(self, "RepoEmbedUri", value=self.repositories["embed"].repository_uri)
        CfnOutput(self, "RepoKnnUri", value=self.repositories["knn"].repository_uri)
        CfnOutput(self, "RepoTicketUri", value=self.repositories["ticket"].repository_uri)
        CfnOutput(self, "RepoCoordinatorUri", value=self.repositories["coordinator"].repository_uri)
        CfnOutput(self, "ImageBuildProject", value=image_project.project_name)
        CfnOutput(self, "E2eTestProject", value=e2e_project.project_name)

        NagSuppressions.add_stack_suppressions(
            self,
            [
                NagPackSuppression(
                    id="AwsSolutions-CB4",
                    reason="CodeBuild artifact and source storage use the project S3 bucket with SSE-S3; a customer KMS key adds fixed cost.",
                ),
                NagPackSuppression(
                    id="AwsSolutions-IAM5",
                    reason="The image role needs wildcard resource scope only for ecr:GetAuthorizationToken; all other actions are repo, bucket-prefix, or service-log scoped.",
                ),
                NagPackSuppression(
                    id="AwsSolutions-IAM4",
                    reason="CodeBuild service roles keep only the CDK-generated scoped logging/report permissions; project permissions are explicit and least-privilege.",
                ),
            ],
            apply_to_nested_stacks=True,
        )

    def _project(self, construct_id, project_name, buildspec_name, timeout, log_group, account, region):
        variables = {
            "ACCOUNT": codebuild.BuildEnvironmentVariable(value=account),
            "REGION": codebuild.BuildEnvironmentVariable(value=region),
            "BUCKET": codebuild.BuildEnvironmentVariable(value=self.data_bucket.bucket_name),
            "REPO_EMBED": codebuild.BuildEnvironmentVariable(value=self.repositories["embed"].repository_uri),
            "REPO_KNN": codebuild.BuildEnvironmentVariable(value=self.repositories["knn"].repository_uri),
            "REPO_TICKET": codebuild.BuildEnvironmentVariable(value=self.repositories["ticket"].repository_uri),
            "REPO_COORDINATOR": codebuild.BuildEnvironmentVariable(value=self.repositories["coordinator"].repository_uri),
        }
        project = codebuild.Project(
            self, construct_id,
            project_name=project_name,
            source=codebuild.Source.s3(bucket=self.data_bucket, path="build/src/placeholder.zip"),
            build_spec=codebuild.BuildSpec.from_source_filename(buildspec_name),
            environment=codebuild.BuildEnvironment(
                build_image=codebuild.LinuxArmBuildImage.AMAZON_LINUX_2_STANDARD_3_0,
                compute_type=codebuild.ComputeType.LARGE,
                # Reason: docker build requires the Docker daemon; PLAN §12.1.
                privileged=True,
            ),
            timeout=Duration.minutes(timeout),
            environment_variables=variables,
            logging=codebuild.LoggingOptions(
                cloud_watch=codebuild.CloudWatchLoggingOptions(log_group=log_group),
            ),
        )
        Tags.of(project).add("Project", "lift-door-agentic")
        return project

    def _grant_codebuild(self, project, mode):
        repo_arns = [repo.repository_arn for repo in self.repositories.values()]
        ecr_pull = [
            "ecr:BatchCheckLayerAvailability",
            "ecr:BatchGetImage",
            "ecr:GetDownloadUrlForLayer",
        ]
        ecr_push = ecr_pull + [
            "ecr:CompleteLayerUpload",
            "ecr:InitiateLayerUpload",
            "ecr:PutImage",
            "ecr:UploadLayerPart",
        ]
        project.add_to_role_policy(
            iam.PolicyStatement(actions=["ecr:GetAuthorizationToken"], resources=["*"])
        )
        project.add_to_role_policy(
            iam.PolicyStatement(actions=ecr_push if mode == "images" else ecr_pull, resources=repo_arns)
        )
        bucket_arn = self.data_bucket.bucket_arn
        source_resources = [
            f"{bucket_arn}/build/src/*",
            f"{bucket_arn}/models/*",
        ]
        if mode == "e2e":
            source_resources.append(f"{bucket_arn}/dataset/*")
        project.add_to_role_policy(
            iam.PolicyStatement(actions=["s3:GetObject"], resources=source_resources)
        )
        project.add_to_role_policy(
            iam.PolicyStatement(
                actions=["s3:GetObjectVersion"], resources=[f"{bucket_arn}/build/src/*"],
            )
        )
        project.add_to_role_policy(
            iam.PolicyStatement(
                actions=["s3:GetBucketLocation", "s3:GetBucketAcl"], resources=[bucket_arn],
            )
        )
        project.add_to_role_policy(
            iam.PolicyStatement(actions=["s3:PutObject"], resources=[f"{bucket_arn}/build/out/*"])
        )
