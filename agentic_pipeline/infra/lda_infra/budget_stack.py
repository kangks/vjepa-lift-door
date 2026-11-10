from aws_cdk import Stack
from aws_cdk import aws_budgets as budgets
from constructs import Construct


class BudgetStack(Stack):
    def __init__(
        self, scope: Construct, construct_id: str, *, profile: str,
        budget_usd: float, alert_email: str, **kwargs,
    ):
        super().__init__(scope, construct_id, **kwargs)
        name = f"lift-door-agentic-{profile}"
        notifications = []
        for threshold, kind in ((50, "ACTUAL"), (80, "ACTUAL"), (100, "ACTUAL"), (100, "FORECASTED")):
            notifications.append(
                budgets.CfnBudget.NotificationWithSubscribersProperty(
                    notification=budgets.CfnBudget.NotificationProperty(
                        notification_type=kind,
                        comparison_operator="GREATER_THAN",
                        threshold=threshold,
                        threshold_type="PERCENTAGE",
                    ),
                    subscribers=[
                        budgets.CfnBudget.SubscriberProperty(
                            subscription_type="EMAIL", address=alert_email,
                        )
                    ],
                )
            )
        budget = budgets.CfnBudget(
            self, "MonthlyBudget",
            budget=budgets.CfnBudget.BudgetDataProperty(
                budget_name=name,
                budget_type="COST",
                time_unit="MONTHLY",
                budget_limit=budgets.CfnBudget.SpendProperty(amount=budget_usd, unit="USD"),
                cost_filters={"TagKeyValue": ["user:Project$lift-door-agentic"]},
            ),
            notifications_with_subscribers=notifications,
            resource_tags=[
                budgets.CfnBudget.ResourceTagProperty(
                    key="Project", value="lift-door-agentic",
                )
            ],
        )
        budget.override_logical_id("LiftDoorAgenticMonthlyBudget")
