#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


def _list(value):
    return value if isinstance(value, list) else [value]


def summarize(path: Path) -> str:
    template = json.loads(path.read_text())
    resources = template.get("Resources", {})
    role_names = {
        logical_id: props.get("Properties", {}).get("RoleName", logical_id)
        for logical_id, props in resources.items()
        if props.get("Type") == "AWS::IAM::Role"
    }
    lines = [f"Template: {path}", "IAM statements:"]

    for logical_id, resource in resources.items():
        props = resource.get("Properties", {})
        if resource.get("Type") == "AWS::IAM::Role":
            for policy in props.get("Policies", []):
                for statement in _list(policy.get("PolicyDocument", {}).get("Statement", [])):
                    lines.append(json.dumps({
                        "role": role_names[logical_id],
                        "policy": policy.get("PolicyName"),
                        "actions": statement.get("Action"),
                        "resources": statement.get("Resource"),
                        "conditions": statement.get("Condition"),
                    }, sort_keys=True))
        if resource.get("Type") == "AWS::IAM::Policy":
            role_refs = [
                role.get("Ref", json.dumps(role, sort_keys=True))
                for role in _list(props.get("Roles", []))
            ]
            for statement in _list(props.get("PolicyDocument", {}).get("Statement", [])):
                lines.append(json.dumps({
                    "role": [role_names.get(ref, ref) for ref in role_refs],
                    "policy": props.get("PolicyName"),
                    "actions": statement.get("Action"),
                    "resources": statement.get("Resource"),
                    "conditions": statement.get("Condition"),
                }, sort_keys=True))

    lines.append("Security group rules:")
    for logical_id, resource in resources.items():
        resource_type = resource.get("Type")
        props = resource.get("Properties", {})
        if resource_type == "AWS::EC2::SecurityGroup":
            for direction in ("SecurityGroupIngress", "SecurityGroupEgress"):
                for rule in props.get(direction, []):
                    lines.append(json.dumps({
                        "group": props.get("GroupName", logical_id),
                        "direction": direction,
                        **rule,
                    }, sort_keys=True))
        elif resource_type in ("AWS::EC2::SecurityGroupIngress", "AWS::EC2::SecurityGroupEgress"):
            lines.append(json.dumps({
                "group": props.get("GroupId"),
                "direction": resource_type.rsplit("::", 1)[-1],
                **props,
            }, sort_keys=True))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} <cloudformation-template.json>")
    print(summarize(Path(sys.argv[1])), end="")
