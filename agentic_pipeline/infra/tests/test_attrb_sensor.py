"""attrb (DoiT Attribute) ECS Fargate sensor: present in attrb_sandbox only."""
from __future__ import annotations

from aws_cdk.assertions import Template

from app import create_app
from lda_infra.main_stack import (
    ATTRB_LAUNCHER, ATTRB_LOADER_NAME, ATTRB_MOUNT, ATTRB_OTEL_ADDR, ATTRB_SENSOR_IMAGE,
    ATTRB_TOKEN_PARAMETER, ATTRB_VOLUME,
)
from tests.test_template import INFRA, _inline_statements, _resources

ORIGINAL_ENTRYPOINTS = {
    "lda-embed": ["/app/services/embed/entrypoint.sh"],
    "lda-coordinator": ["/app/services/coordinator/entrypoint.sh"],
    "lda-coordinator-oneoff": ["opentelemetry-instrument", "python", "-u"],
}
APP_CONTAINERS = {
    "lda-embed": "EmbedContainer",
    "lda-coordinator": "CoordinatorContainer",
    "lda-coordinator-oneoff": "OneoffContainer",
}
SERVICE_NAMES = {
    "lda-embed": "lda-embed",
    "lda-coordinator": "lda-coordinator",
    "lda-coordinator-oneoff": "lda-coordinator",
}


def _main(profile):
    app = create_app({
        "profile": profile,
        "region": "us-east-1",
        "azs": "us-east-1a,us-east-1b",
        "imageTag": "src-000000000000",
        "imageDigests": str(INFRA / "tests/fixtures/images.json"),
    })
    return Template.from_stack(app.node.find_child("LiftDoorAgentic")).to_json()


def test_attrb_sidecar_wraps_all_three_task_definitions():
    main = _main("attrb_sandbox")
    taskdefs = {
        task["Properties"]["Family"]: task["Properties"]
        for task in _resources(main, "AWS::ECS::TaskDefinition").values()
    }
    assert set(taskdefs) == set(APP_CONTAINERS)
    for family, props in taskdefs.items():
        assert props["RuntimePlatform"]["CpuArchitecture"] == "ARM64"
        assert {"Name": ATTRB_VOLUME, "Host": {}} in props["Volumes"]
        containers = {c["Name"]: c for c in props["ContainerDefinitions"]}
        assert props["ContainerDefinitions"][0]["Name"] == APP_CONTAINERS[family]
        loader = containers[ATTRB_LOADER_NAME]
        assert loader["Image"] == ATTRB_SENSOR_IMAGE
        assert loader["Essential"] is False
        assert loader["EntryPoint"] == ["load_attribute"]
        assert loader["Command"] == [ATTRB_MOUNT]
        assert loader["MountPoints"] == [
            {"ContainerPath": ATTRB_MOUNT, "ReadOnly": False, "SourceVolume": ATTRB_VOLUME},
        ]
        app = containers[APP_CONTAINERS[family]]
        assert app["EntryPoint"] == [ATTRB_LAUNCHER, *ORIGINAL_ENTRYPOINTS[family]]
        assert {"ContainerPath": ATTRB_MOUNT, "ReadOnly": True, "SourceVolume": ATTRB_VOLUME} in app["MountPoints"]
        assert app["DependsOn"] == [{"Condition": "COMPLETE", "ContainerName": ATTRB_LOADER_NAME}]
        env = {row["Name"]: row["Value"] for row in app["Environment"]}
        assert env["CONFIG_ZPROBE_OTEL_ADDR"] == ATTRB_OTEL_ADDR
        assert env["OTEL_SERVICE_NAME"] == SERVICE_NAMES[family]
        assert f"service.name={SERVICE_NAMES[family]}" in env["OTEL_RESOURCE_ATTRIBUTES"]
        assert "CONFIG_ZPROBE_BEARER_TOKEN" not in env
        secrets = {row["Name"]: row["ValueFrom"] for row in app["Secrets"]}
        assert set(secrets) == {"CONFIG_ZPROBE_BEARER_TOKEN"}
        assert ATTRB_TOKEN_PARAMETER in str(secrets["CONFIG_ZPROBE_BEARER_TOKEN"])
    assert (taskdefs["lda-embed"]["Cpu"], taskdefs["lda-embed"]["Memory"]) == ("16384", "32768")
    for family in ("lda-coordinator", "lda-coordinator-oneoff"):
        assert (taskdefs[family]["Cpu"], taskdefs[family]["Memory"]) == ("1024", "2048")


def test_attrb_services_propagate_tags_and_token_grant_is_scoped():
    main = _main("attrb_sandbox")
    for service in _resources(main, "AWS::ECS::Service").values():
        assert service["Properties"]["PropagateTags"] == "SERVICE"
        assert service["Properties"]["EnableECSManagedTags"] is True
    token_statements = [
        statement for _, statement in _inline_statements(main)
        if ATTRB_TOKEN_PARAMETER in str(statement.get("Resource"))
    ]
    assert token_statements
    for statement in token_statements:
        assert all(str(action).startswith("ssm:") for action in statement["Action"])
        assert "*" not in str(statement["Resource"]).replace("arn:", "")


def test_ml_sandbox_has_no_attrb_sensor():
    rendered = str(_main("ml-sandbox"))
    for needle in (ATTRB_LOADER_NAME, ATTRB_SENSOR_IMAGE, ATTRB_TOKEN_PARAMETER, "CONFIG_ZPROBE", "PropagateTags"):
        assert needle not in rendered
