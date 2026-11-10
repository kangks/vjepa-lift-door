from __future__ import annotations

import json
import re
from pathlib import Path

from aws_cdk.assertions import Template

from app import create_app
from lda_infra import pins

ROOT = Path(__file__).resolve().parents[2]
INFRA = Path(__file__).resolve().parents[1]


def _app_templates(profile: str):
    app = create_app({
        "profile": profile,
        "region": "us-east-1",
        "azs": "us-east-1a,us-east-1b",
        "imageTag": "src-000000000000",
        "imageDigests": str(INFRA / "tests/fixtures/images.json"),
    })
    stacks = {
        name: app.node.find_child(name)
        for name in ("LiftDoorAgenticBudget", "LiftDoorAgenticBuild", "LiftDoorAgentic")
    }
    templates = {name: Template.from_stack(stack).to_json() for name, stack in stacks.items()}
    return templates


def _resources(template, resource_type):
    return {
        logical_id: resource
        for logical_id, resource in template["Resources"].items()
        if resource["Type"] == resource_type
    }


def _as_list(value):
    return value if isinstance(value, list) else [value]


def _inline_statements(template):
    role_names = {
        logical_id: resource.get("Properties", {}).get("RoleName", logical_id)
        for logical_id, resource in _resources(template, "AWS::IAM::Role").items()
    }
    statements = []
    for logical_id, resource in _resources(template, "AWS::IAM::Role").items():
        for policy in resource.get("Properties", {}).get("Policies", []):
            for statement in _as_list(policy.get("PolicyDocument", {}).get("Statement", [])):
                statements.append((role_names[logical_id], statement))
    for policy in _resources(template, "AWS::IAM::Policy").values():
        refs = []
        for role in _as_list(policy.get("Properties", {}).get("Roles", [])):
            if isinstance(role, dict) and "Ref" in role:
                refs.append(role["Ref"])
        names = [role_names.get(ref, ref) for ref in refs]
        role_name = ",".join(names) if names else "attached-policy"
        for statement in _as_list(policy["Properties"]["PolicyDocument"].get("Statement", [])):
            statements.append((role_name, statement))
    return statements


def _actions(statement):
    return [str(action) for action in _as_list(statement.get("Action", []))]


def _render(value):
    return json.dumps(value, sort_keys=True)


def test_01_all_account_templates_are_synthesizable_and_tagged():
    for profile in ("ml-sandbox", "attrb_sandbox"):
        templates = _app_templates(profile)
        for template in templates.values():
            for resource in template["Resources"].values():
                tags = resource.get("Properties", {}).get("Tags")
                if tags is not None:
                    if isinstance(tags, dict):
                        values = tags
                    else:
                        values = {tag["Key"]: tag["Value"] for tag in tags}
                    assert values["Project"] == "lift-door-agentic"
            types = {resource["Type"] for resource in template["Resources"].values()}
            assert "AWS::Logs::ResourcePolicy" not in types
            assert not any(resource_type.startswith("AWS::XRay::") for resource_type in types)

        budgets = _resources(templates["LiftDoorAgenticBudget"], "AWS::Budgets::Budget")
        assert len(budgets) == 1
        budget_props = next(iter(budgets.values()))["Properties"]
        assert budget_props["ResourceTags"] == [{"Key": "Project", "Value": "lift-door-agentic"}]


def test_02_bucket_security_and_no_public_ingress():
    templates = _app_templates("ml-sandbox")
    build = templates["LiftDoorAgenticBuild"]
    main = templates["LiftDoorAgentic"]
    buckets = _resources(build, "AWS::S3::Bucket")
    assert len(buckets) == 1
    bucket = next(iter(buckets.values()))
    props = bucket["Properties"]
    assert props["PublicAccessBlockConfiguration"] == {
        "BlockPublicAcls": True, "BlockPublicPolicy": True,
        "IgnorePublicAcls": True, "RestrictPublicBuckets": True,
    }
    assert props["BucketEncryption"]["ServerSideEncryptionConfiguration"][0][
        "ServerSideEncryptionByDefault"
    ]["SSEAlgorithm"] == "AES256"
    assert bucket["DeletionPolicy"] == "Retain"
    assert props["NotificationConfiguration"]["EventBridgeConfiguration"]["EventBridgeEnabled"] is True
    bucket_policies = _resources(build, "AWS::S3::BucketPolicy")
    assert len(bucket_policies) == 1
    statements = next(iter(bucket_policies.values()))["Properties"]["PolicyDocument"]["Statement"]
    assert any(
        statement.get("Condition", {}).get("Bool", {}).get("aws:SecureTransport") == "false"
        for statement in _as_list(statements)
    )

    for resource in main["Resources"].values():
        props = resource.get("Properties", {})
        if resource["Type"] == "AWS::EC2::SecurityGroup":
            for rule in props.get("SecurityGroupIngress", []):
                assert rule.get("CidrIp") != "0.0.0.0/0"
                assert rule.get("CidrIpv6") != "::/0"
        elif resource["Type"] == "AWS::EC2::SecurityGroupIngress":
            assert props.get("CidrIp") != "0.0.0.0/0"
            assert props.get("CidrIpv6") != "::/0"


def test_03_services_are_idle_arm64_and_lambdas_use_pass_through():
    main = _app_templates("ml-sandbox")["LiftDoorAgentic"]
    services = _resources(main, "AWS::ECS::Service")
    assert len(services) == 2
    assert all(service["Properties"]["DesiredCount"] == 0 for service in services.values())
    taskdefs = _resources(main, "AWS::ECS::TaskDefinition")
    assert len(taskdefs) == 3
    assert all(
        task["Properties"]["RuntimePlatform"]["CpuArchitecture"] == "ARM64"
        for task in taskdefs.values()
    )
    embed_task = next(task for task in taskdefs.values() if task["Properties"]["Cpu"] == "16384")
    assert embed_task["Properties"]["Memory"] == "32768"
    assert embed_task["Properties"]["EphemeralStorage"]["SizeInGiB"] == 21
    functions = _resources(main, "AWS::Lambda::Function")
    assert len(functions) == 3
    log_group_ids = {
        logical_id for logical_id, resource in _resources(main, "AWS::Logs::LogGroup").items()
        if resource["Properties"].get("LogGroupName", "").startswith("/lda/lambda/")
    }
    for function in functions.values():
        assert function["Properties"]["TracingConfig"]["Mode"] == "PassThrough"
        assert function["Properties"]["LoggingConfig"]["LogGroup"]["Ref"] in log_group_ids


def test_04_all_eight_log_groups_are_retained_and_used():
    templates = _app_templates("ml-sandbox")
    all_log_groups = {}
    for template in templates.values():
        all_log_groups.update(_resources(template, "AWS::Logs::LogGroup"))
    assert len(all_log_groups) == 8
    log_group_ids = {
        logical_id
        for template in templates.values()
        for logical_id, resource in _resources(template, "AWS::Logs::LogGroup").items()
    }
    for group in all_log_groups.values():
        assert group["Properties"]["RetentionInDays"] >= 30
        assert group["DeletionPolicy"] == "Retain"
    main = templates["LiftDoorAgentic"]
    for task in _resources(main, "AWS::ECS::TaskDefinition").values():
        for container in task["Properties"]["ContainerDefinitions"]:
            group = container["LogConfiguration"]["Options"]["awslogs-group"]
            assert isinstance(group, dict) and group.get("Ref") in log_group_ids


def test_05_iam_env_and_resource_scoping():
    templates = _app_templates("ml-sandbox")
    main = templates["LiftDoorAgentic"]
    functions = _resources(main, "AWS::Lambda::Function")
    lambda_roles = {
        resource["Properties"]["FunctionName"]: resource["Properties"]["Role"]["Fn::GetAtt"][0]
        for resource in functions.values()
    }
    bedrock_roles = []
    put_roles = []
    all_statements = []
    for template in (main,):
        for role_name, statement in _inline_statements(template):
            actions = _actions(statement)
            all_statements.append((role_name, statement))
            assert "*" not in actions
            assert not any(action.endswith(":*") for action in actions)
            if "bedrock-mantle:CreateInference" in actions:
                bedrock_roles.append((role_name, statement))
                assert statement["Condition"]["StringEquals"]["bedrock-mantle:Model"] == "anthropic.claude-opus-5-5"
                assert all(
                    ":project/" in str(resource)
                    for resource in _as_list(statement.get("Resource"))
                )
            if "dynamodb:PutItem" in actions:
                put_roles.append(role_name)
            if "*" in _as_list(statement.get("Resource")):
                allowed = {
                    "xray:PutTraceSegments", "xray:PutTelemetryRecords",
                    "xray:GetSamplingRules", "xray:GetSamplingTargets",
                    "ecr:GetAuthorizationToken",
                }
                assert set(actions) <= allowed
            assert "logs:CreateLogGroup" not in actions
    assert len(bedrock_roles) == 1 and "lda-coordinator-task" in bedrock_roles[0][0]
    assert len(put_roles) == 1 and lambda_roles["lda-ticket"] in put_roles[0]
    for function in functions.values():
        env = function["Properties"].get("Environment", {}).get("Variables", {})
        assert not {"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"} & set(env)
    for task in _resources(main, "AWS::ECS::TaskDefinition").values():
        for container in task["Properties"]["ContainerDefinitions"]:
            names = {entry["Name"] for entry in container.get("Environment", [])}
            assert not {"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"} & names
            assert "OTEL_EXPORTER_OTLP_TRACES_HEADERS" not in names


def test_06_no_collisions_hard_coded_ids_or_unapproved_network_resources():
    templates = _app_templates("ml-sandbox")
    rendered = json.dumps(templates).lower()
    for forbidden in (
        "round10", "round-10", "round12", "round-12", "lift-lobby",
        "liftlobby", "cosmos", "nano",
    ):
        assert forbidden not in rendered
    code = "\n".join(
        path.read_text()
        for path in list((INFRA / "lda_infra").glob("*.py")) + list((ROOT / "services").rglob("*.py"))
    )
    account_ids = json.loads((INFRA / "accounts.json").read_text()).values()
    assert all(account_id not in code for account_id in account_ids)
    main = templates["LiftDoorAgentic"]
    azs = {
        resource["Properties"].get("AvailabilityZone")
        for resource in _resources(main, "AWS::EC2::Subnet").values()
    }
    assert azs == {"us-east-1a", "us-east-1b"}
    prohibited = {
        "AWS::EC2::NatGateway", "AWS::EC2::VPCEndpoint",
        "AWS::ElasticLoadBalancingV2::LoadBalancer",
    }
    assert not prohibited & {resource["Type"] for resource in main["Resources"].values()}


def test_07_budget_and_retain_delete_summary():
    templates = _app_templates("ml-sandbox")
    budget = next(iter(_resources(templates["LiftDoorAgenticBudget"], "AWS::Budgets::Budget").values()))
    data = budget["Properties"]["Budget"]
    assert data["BudgetName"] == "lift-door-agentic-ml-sandbox"
    assert data["BudgetType"] == "COST" and data["TimeUnit"] == "MONTHLY"
    assert data["BudgetLimit"] == {"Amount": 20.0, "Unit": "USD"}
    assert data["CostFilters"]["TagKeyValue"] == ["user:Project$lift-door-agentic"]
    notifications = budget["Properties"]["NotificationsWithSubscribers"]
    assert len(notifications) == 4
    assert any(
        row["Notification"]["NotificationType"] == "FORECASTED"
        and row["Notification"]["Threshold"] == 100
        for row in notifications
    )
    assert all(row["Subscribers"][0]["Address"] == "<EMAIL>m" for row in notifications)

    retained = []
    for stack_name, template in templates.items():
        for logical_id, resource in template["Resources"].items():
            if resource.get("DeletionPolicy") == "Retain":
                retained.append((stack_name, logical_id, resource["Type"]))
    retain_types = [row[2] for row in retained]
    assert retain_types.count("AWS::S3::Bucket") == 1
    assert retain_types.count("AWS::Logs::LogGroup") == 8
    print("DeletionPolicy Retain resources:", retained)


def test_08_eventbridge_s3_rule_targets_queue_with_scoped_policy():
    main = _app_templates("ml-sandbox")["LiftDoorAgentic"]
    rules = _resources(main, "AWS::Events::Rule")
    incoming = [
        rule for rule in rules.values()
        if rule["Properties"].get("EventPattern", {}).get("detail-type") == ["Object Created"]
    ]
    assert len(incoming) == 1
    target_arn = incoming[0]["Properties"]["Targets"][0]["Arn"]
    assert isinstance(target_arn, dict) and "Fn::GetAtt" in target_arn
    statements = [
        statement
        for policy in _resources(main, "AWS::SQS::QueuePolicy").values()
        for statement in _as_list(policy["Properties"]["PolicyDocument"]["Statement"])
    ]
    assert any(
        statement.get("Principal", {}).get("Service") == "events.amazonaws.com"
        and "aws:SourceArn" in statement.get("Condition", {}).get("ArnEquals", {})
        for statement in statements
    )


def test_17_build_stack_repositories_are_immutable_and_auto_delete_native():
    build = _app_templates("ml-sandbox")["LiftDoorAgenticBuild"]
    repositories = _resources(build, "AWS::ECR::Repository")
    assert len(repositories) == 4
    for repo in repositories.values():
        props = repo["Properties"]
        assert props["ImageScanningConfiguration"]["ScanOnPush"] is True
        assert props["ImageTagMutability"] == "IMMUTABLE"
        assert props["EncryptionConfiguration"]["EncryptionType"] == "AES256"
        assert props["EmptyOnDelete"] is True
        assert repo["DeletionPolicy"] == "Delete"
        lifecycle = json.loads(props["LifecyclePolicy"]["LifecyclePolicyText"])
        assert lifecycle["rules"][0]["selection"]["countNumber"] == 5
    assert not any("Custom::ECRAutoDeleteImages" in row["Type"] for row in build["Resources"].values())


def test_18_build_projects_are_arm_large_privileged_and_logged():
    build = _app_templates("ml-sandbox")["LiftDoorAgenticBuild"]
    projects = _resources(build, "AWS::CodeBuild::Project")
    assert len(projects) == 2
    by_name = {row["Properties"]["Name"]: row["Properties"] for row in projects.values()}
    assert set(by_name) == {"lda-image-build", "lda-e2e-test"}
    for name, props in by_name.items():
        assert props["Environment"]["Type"] == "ARM_CONTAINER"
        assert props["Environment"]["ComputeType"] == "BUILD_GENERAL1_LARGE"
        assert props["Environment"]["PrivilegedMode"] is True
        assert "VpcConfig" not in props
        assert props["TimeoutInMinutes"] == (60 if name == "lda-image-build" else 120)
        assert props["LogsConfig"]["CloudWatchLogs"]["Status"] == "ENABLED"
        group_ref = props["LogsConfig"]["CloudWatchLogs"]["GroupName"].get("Ref")
        groups = _resources(build, "AWS::Logs::LogGroup")
        assert group_ref in groups
        assert groups[group_ref]["Properties"]["LogGroupName"].startswith("/lda/codebuild/")


def test_19_codebuild_role_permissions_are_scoped_and_print_extras():
    build = _app_templates("ml-sandbox")["LiftDoorAgenticBuild"]
    projects = {p["Properties"]["Name"]: p["Properties"] for p in _resources(build, "AWS::CodeBuild::Project").values()}
    policies = _resources(build, "AWS::IAM::Policy")
    role_names = {
        logical_id: resource.get("Properties", {}).get("RoleName", logical_id)
        for logical_id, resource in _resources(build, "AWS::IAM::Role").items()
    }
    for name, project in projects.items():
        role_ref = project["ServiceRole"]["Fn::GetAtt"][0]
        statements = []
        for policy in policies.values():
            refs = [item.get("Ref") for item in _as_list(policy["Properties"].get("Roles", [])) if isinstance(item, dict)]
            if role_ref in refs:
                statements.extend(_as_list(policy["Properties"]["PolicyDocument"].get("Statement", [])))
        for role_id, role in _resources(build, "AWS::IAM::Role").items():
            if role_id == role_ref:
                for policy in role.get("Properties", {}).get("Policies", []):
                    statements.extend(_as_list(policy["PolicyDocument"].get("Statement", [])))
        assert statements, f"no policies associated with {name}"
        required_actions = {
            "ecr:GetAuthorizationToken",
            "ecr:BatchCheckLayerAvailability", "ecr:BatchGetImage",
            "ecr:GetDownloadUrlForLayer", "s3:GetObject", "s3:GetObjectVersion",
            "s3:GetBucketLocation", "s3:GetBucketAcl", "s3:PutObject",
            "logs:CreateLogStream", "logs:PutLogEvents",
        }
        if name == "lda-image-build":
            required_actions |= {
                "ecr:CompleteLayerUpload", "ecr:InitiateLayerUpload",
                "ecr:PutImage", "ecr:UploadLayerPart",
            }
        else:
            assert not {
                "ecr:CompleteLayerUpload", "ecr:InitiateLayerUpload",
                "ecr:PutImage", "ecr:UploadLayerPart",
            } & {action for statement in statements for action in _actions(statement)}
        assert required_actions <= {action for statement in statements for action in _actions(statement)}
        bucket_ids = {
            logical_id for logical_id, resource in _resources(build, "AWS::S3::Bucket").items()
        }
        repo_ids = {
            logical_id for logical_id, resource in _resources(build, "AWS::ECR::Repository").items()
        }
        own_log_group_ids = {
            logical_id
            for logical_id, resource in _resources(build, "AWS::Logs::LogGroup").items()
            if resource["Properties"].get("LogGroupName") == (
                "/lda/codebuild/images" if name == "lda-image-build" else "/lda/codebuild/e2e"
            )
        }
        project_id = next(
            logical_id for logical_id, resource in _resources(build, "AWS::CodeBuild::Project").items()
            if resource["Properties"]["Name"] == name
        )
        extras = []
        required_present = set()
        permitted_actions = {
            "ecr:GetAuthorizationToken", "ecr:BatchCheckLayerAvailability", "ecr:BatchGetImage",
            "ecr:GetDownloadUrlForLayer", "ecr:CompleteLayerUpload", "ecr:InitiateLayerUpload",
            "ecr:PutImage", "ecr:UploadLayerPart", "s3:GetObject", "s3:GetObjectVersion",
            "s3:GetBucketLocation", "s3:GetBucketAcl", "s3:PutObject",
            "s3:GetBucket*", "s3:GetObject*", "s3:List*",
            "logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents",
            "codebuild:CreateReportGroup", "codebuild:CreateReport", "codebuild:UpdateReport",
            "codebuild:BatchPutTestCases", "codebuild:BatchPutCodeCoverages",
        }
        for statement in statements:
            actions = set(_actions(statement))
            resources = _as_list(statement.get("Resource", []))
            assert actions.isdisjoint({"s3:*", "ecr:*", "logs:*"})
            if "*" in resources:
                assert actions == {"ecr:GetAuthorizationToken"}
            assert not any(action.startswith(("bedrock:", "dynamodb:", "lambda:")) for action in actions)
            assert actions <= permitted_actions
            rendered_resources = _render(resources)
            if "ecr:GetAuthorizationToken" not in actions and any(
                action.startswith("ecr:") for action in actions
            ):
                assert all(f'"{repo_id}"' in rendered_resources for repo_id in repo_ids)
            if any(action.startswith("s3:") for action in actions):
                # CDK's S3 source integration grants bucket-level source access to
                # the exact project bucket; explicit data permissions stay prefix scoped.
                assert all(
                    any(f'"{bucket_id}"' in _render(resource) for bucket_id in bucket_ids)
                    for resource in resources
                )
                if actions & {"s3:GetBucket*", "s3:GetObject*", "s3:List*"}:
                    assert len(resources) == 2
                    assert "build/src/placeholder.zip" in rendered_resources
                else:
                    assert all(
                        any(prefix in _render(resource) for prefix in (
                            "/build/src/*", "/models/*", "/dataset/*", "/build/out/*",
                        ))
                        or actions <= {"s3:GetBucketLocation", "s3:GetBucketAcl"}
                        for resource in resources
                    )
            if any(action.startswith("logs:") for action in actions):
                for resource in resources:
                    rendered = _render(resource)
                    assert (
                        any(f'"{group_id}"' in rendered for group_id in own_log_group_ids)
                        or f"/aws/codebuild/" in rendered and f'"{project_id}"' in rendered
                    )
            if any(action.startswith("codebuild:") for action in actions):
                rendered = _render(resources)
                assert "report-group/" in rendered and f'"{project_id}"' in rendered
            # Capture the statements not explicitly authored by this stack. They
            # are CDK's source, log, and report grants, each checked above.
            if not actions <= required_actions:
                extras.append({"actions": sorted(actions), "resources": resources})
            for action in actions:
                if action.startswith("ecr:") and action != "ecr:GetAuthorizationToken":
                    required_present.add(action)
        print(f"{name} CDK default/extra statements:", extras)
        assert not any("ecr:DescribeImages" in _actions(statement) for statement in statements)


def test_20_main_stack_uses_ecr_images_and_lambda_digests():
    templates = _app_templates("ml-sandbox")
    main = templates["LiftDoorAgentic"]
    assert not _resources(main, "AWS::ECR::Repository")
    rendered = json.dumps(main)
    assert "src-000000000000" in rendered
    repository_ids = {
        "embed": next(
            logical_id for logical_id, resource in _resources(templates["LiftDoorAgenticBuild"], "AWS::ECR::Repository").items()
            if resource["Properties"]["RepositoryName"] == "lda-embed"
        ),
        "coordinator": next(
            logical_id for logical_id, resource in _resources(templates["LiftDoorAgenticBuild"], "AWS::ECR::Repository").items()
            if resource["Properties"]["RepositoryName"] == "lda-coordinator"
        ),
    }
    task_images = {
        container["Name"]: container["Image"]
        for task in _resources(main, "AWS::ECS::TaskDefinition").values()
        for container in task["Properties"]["ContainerDefinitions"]
    }
    assert len(task_images) == 3
    for container_name, repository_name in (
        ("EmbedContainer", "embed"), ("CoordinatorContainer", "coordinator"),
        ("OneoffContainer", "coordinator"),
    ):
        image = _render(task_images[container_name])
        assert repository_ids[repository_name] in image
        assert ":src-000000000000" in image
    functions = _resources(main, "AWS::Lambda::Function")
    docker_functions = [f for f in functions.values() if "ImageUri" in f["Properties"].get("Code", {})]
    assert len(docker_functions) == 2
    fixture = json.loads((INFRA / "tests/fixtures/images.json").read_text())
    for function, component in zip(docker_functions, ("knn", "ticket")):
        image_uri = function["Properties"]["Code"]["ImageUri"]
        digest = fixture["images"][component]["digest"]
        assert f"@{digest}" in _render(image_uri)
    assert not re.search(r"docker-image:[0-9a-f]{64}", rendered)


def test_21_exact_lambda_and_custom_resource_counts():
    templates = _app_templates("ml-sandbox")
    all_resources = [
        resource
        for template in templates.values()
        for resource in template["Resources"].values()
    ]
    assert sum(resource["Type"] == "AWS::Lambda::Function" for resource in all_resources) == 3
    custom = [resource["Type"] for resource in all_resources if resource["Type"].startswith("Custom::")]
    assert custom == []
    assert not any(resource["Type"] == "Custom::S3BucketNotifications" for resource in all_resources)


def test_22_no_auto_codebuild_log_groups_are_synthesized():
    templates = _app_templates("ml-sandbox")
    groups = [
        resource["Properties"]["LogGroupName"]
        for template in templates.values()
        for resource in _resources(template, "AWS::Logs::LogGroup").values()
    ]
    assert len(groups) == 8
    assert not any(name.startswith("/aws/codebuild/lda") for name in groups)


def test_23_lambda_image_pull_repository_policies_are_conditioned():
    templates = _app_templates("ml-sandbox")
    build = templates["LiftDoorAgenticBuild"]
    main = templates["LiftDoorAgentic"]
    accounts = json.loads((INFRA / "accounts.json").read_text())
    expected_account = accounts["ml-sandbox"]
    expected_actions = {
        "ecr:BatchGetImage",
        "ecr:GetDownloadUrlForLayer",
    }
    repositories = {
        resource["Properties"]["RepositoryName"]: resource
        for resource in _resources(build, "AWS::ECR::Repository").values()
    }
    for repo_name in ("lda-knn", "lda-ticket"):
        policy = repositories[repo_name]["Properties"]["RepositoryPolicyText"]
        statements = _as_list(policy["Statement"])
        assert len(statements) == 1
        statement = statements[0]
        assert statement["Sid"] == "LambdaImagePullScoped"
        assert statement["Principal"]["Service"] == "lambda.amazonaws.com"
        assert set(_as_list(statement["Action"])) == expected_actions
        condition = statement["Condition"]
        assert condition["StringEquals"]["aws:SourceAccount"] == expected_account
        expected_function = repo_name
        expected_source_arns = [
            {
                "Fn::Join": [
                    "",
                    [
                        "arn:", {"Ref": "AWS::Partition"},
                        f":lambda:us-east-1:{expected_account}:function:{expected_function}",
                    ],
                ],
            },
            {
                "Fn::Join": [
                    "",
                    [
                        "arn:", {"Ref": "AWS::Partition"},
                        f":lambda:us-east-1:{expected_account}:function:{expected_function}:*",
                    ],
                ],
            },
        ]
        assert condition["ArnLike"]["aws:SourceArn"] == expected_source_arns

    for repository in repositories.values():
        for statement in _as_list(
            repository["Properties"].get("RepositoryPolicyText", {}).get("Statement", [])
        ):
            assert "Condition" in statement
            assert "aws:SourceAccount" in statement["Condition"].get("StringEquals", {})
            assert "aws:SourceArn" in statement["Condition"].get("ArnLike", {})

    assert not _resources(main, "AWS::ECR::Repository")
    assert not _resources(main, "AWS::ECR::RepositoryPolicy")
    functions = _resources(main, "AWS::Lambda::Function").values()
    function_names = {function["Properties"]["FunctionName"] for function in functions}
    assert {"lda-knn", "lda-ticket"} <= function_names
    assert all(re.fullmatch(r"lda-[a-z0-9-]+", name) for name in function_names)


def test_24_telemetry_exporters_and_pinned_checkpoint_sha():
    main = _app_templates("ml-sandbox")["LiftDoorAgentic"]
    taskdefs = _resources(main, "AWS::ECS::TaskDefinition").values()
    envs = {
        container["Name"]: {
            row["Name"]: row["Value"]
            for row in container.get("Environment", [])
        }
        for task in taskdefs
        for container in task["Properties"]["ContainerDefinitions"]
    }
    embed = envs["EmbedContainer"]
    coordinator = envs["CoordinatorContainer"]

    acceptance_path = (
        ROOT.parents[2]
        / ".claude/worktrees/vjepa-talk-draft/agentic_plan/ACCEPTANCE.md"
    )
    acceptance_section0 = acceptance_path.read_text().split("## 1.", 1)[0]
    checkpoint_rows = [
        line for line in acceptance_section0.splitlines()
        if "vitl_ema_encoder_only.pt" in line
    ]
    assert len(checkpoint_rows) == 1
    acceptance_sha = re.search(r"\b[0-9a-f]{64}\b", checkpoint_rows[0])
    assert acceptance_sha is not None
    assert pins.CKPT_SHA256 == acceptance_sha.group(0)

    assert embed["LDA_CKPT_SHA256"] == pins.CKPT_SHA256
    assert embed["OTEL_LOGS_EXPORTER"] == "none"
    assert embed["OTEL_METRICS_EXPORTER"] == "none"
    assert "AGENT_OBSERVABILITY_ENABLED" not in embed
    assert "OTEL_EXPORTER_OTLP_LOGS_HEADERS" not in embed
    assert embed["OTEL_TRACES_EXPORTER"] == "otlp"

    assert coordinator["LDA_CKPT_SHA256"] == pins.CKPT_SHA256
    assert coordinator["AGENT_OBSERVABILITY_ENABLED"] == "true"
    assert coordinator["OTEL_LOGS_EXPORTER"] == "otlp"
    assert "x-aws-log-group=/lda/agent/coordinator" in (
        coordinator["OTEL_EXPORTER_OTLP_LOGS_HEADERS"]
    )


def test_25_oneoff_task_uses_coordinator_roles_env_and_image():
    main = _app_templates("ml-sandbox")["LiftDoorAgentic"]
    taskdefs = _resources(main, "AWS::ECS::TaskDefinition")
    coordinator = next(
        row for row in taskdefs.values()
        if row["Properties"]["Family"] == "lda-coordinator"
    )
    oneoff = next(
        row for row in taskdefs.values()
        if row["Properties"]["Family"] == "lda-coordinator-oneoff"
    )
    coordinator_container = coordinator["Properties"]["ContainerDefinitions"][0]
    oneoff_container = oneoff["Properties"]["ContainerDefinitions"][0]

    assert oneoff_container["Name"] == "OneoffContainer"
    assert oneoff_container["EntryPoint"] == ["opentelemetry-instrument", "python", "-u"]
    assert oneoff_container["Command"] == [
        "-c", "print('lda oneoff: pass a command override')"
    ]
    assert oneoff_container["Image"] == coordinator_container["Image"]
    assert oneoff["Properties"]["TaskRoleArn"] == coordinator["Properties"]["TaskRoleArn"]
    assert oneoff["Properties"]["ExecutionRoleArn"] == coordinator["Properties"]["ExecutionRoleArn"]
    assert oneoff_container["Environment"] == coordinator_container["Environment"]
    assert oneoff_container["LogConfiguration"]["Options"]["awslogs-group"] == (
        coordinator_container["LogConfiguration"]["Options"]["awslogs-group"]
    )
    assert oneoff_container["LogConfiguration"]["Options"]["awslogs-stream-prefix"] == "oneoff"

    iam_resources = [
        resource for resource in main["Resources"].values()
        if resource["Type"].startswith("AWS::IAM::")
    ]
    assert len(iam_resources) == 14
    # 29 is the pre-OneoffTaskDefinition synthesized baseline; the deployed
    # processed template normalizes some adjacent statements into fewer rows.
    assert len(_inline_statements(main)) == 29


def test_26_xray_trace_exporter_endpoint_uses_stack_region():
    main = _app_templates("ml-sandbox")["LiftDoorAgentic"]
    taskdefs = _resources(main, "AWS::ECS::TaskDefinition").values()
    by_family = {
        task["Properties"]["Family"]: task
        for task in taskdefs
    }
    for family in ("lda-embed", "lda-coordinator", "lda-coordinator-oneoff"):
        container = by_family[family]["Properties"]["ContainerDefinitions"][0]
        env = {item["Name"]: item["Value"] for item in container["Environment"]}
        region = env["AWS_REGION"]
        assert env["OTEL_TRACES_EXPORTER"] == "otlp"
        assert env["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] == (
            f"https://xray.{region}.amazonaws.com/v1/traces"
        )
