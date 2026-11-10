from __future__ import annotations

import json
from pathlib import Path

from aws_cdk import App, Aspects, Environment, Tags
from cdk_nag import AwsSolutionsChecks

from lda_infra.budget_stack import BudgetStack
from lda_infra.build_stack import BuildStack
from lda_infra.main_stack import MainStack
from lda_infra.batch_stack import BatchStack


def create_app(context: dict | None = None):
    app = App(context=context)
    infra_dir = Path(__file__).resolve().parent
    accounts = json.loads((infra_dir / "accounts.json").read_text())
    profile = app.node.try_get_context("profile")
    if profile not in accounts:
        raise ValueError(f"-c profile must be one of: {', '.join(sorted(accounts))}")

    account = accounts[profile]
    region = app.node.try_get_context("region") or "us-east-1"
    azs = [item.strip() for item in (app.node.try_get_context("azs") or "us-east-1a,us-east-1b").split(",")]
    if "us-east-1e" in azs:
        raise ValueError("us-east-1e maps to use1-az3; choose us-east-1a and us-east-1b")
    app.node.set_context(f"availability-zones:account={account}:region={region}", azs)
    budget_usd = float(app.node.try_get_context("budgetUsd") or 20)
    alert_email = app.node.try_get_context("alertEmail") or "<EMAIL>m"
    selected = app.node.try_get_context("stacks") or "all"
    if selected not in {"all", "build", "main", "batch"}:
        raise ValueError("-c stacks must be all, build, main, or batch")

    env = Environment(account=account, region=region)
    Tags.of(app).add("Project", "lift-door-agentic")
    Tags.of(app).add("Owner", "agentic-executor")
    Tags.of(app).add("Profile", profile)
    Aspects.of(app).add(AwsSolutionsChecks(verbose=True))

    BudgetStack(
        app, "LiftDoorAgenticBudget", profile=profile, budget_usd=budget_usd,
        alert_email=alert_email, env=env,
    )
    build = BuildStack(
        app, "LiftDoorAgenticBuild", profile=profile, account=account,
        region=region, env=env,
    )
    if selected == "batch":
        BatchStack(app, "LiftDoorAgenticBatch", profile=profile, account=account, region=region, env=env)
        return app
    if selected != "build":
        image_tag = app.node.try_get_context("imageTag")
        if not image_tag:
            raise ValueError("main stack synth requires -c imageTag=<tag>")
        digest_path = app.node.try_get_context("imageDigests")
        MainStack(
            app, "LiftDoorAgentic", profile=profile, account=account,
            region=region, azs=azs, build_stack=build, image_tag=image_tag,
            image_digests_path=digest_path, env=env,
        )
    return app


if __name__ == "__main__":
    create_app().synth()
