#!/usr/bin/env bash
set -euo pipefail
: "${EMAIL:?set EMAIL for budget alerts}"
REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-west-2}}"
BUDGET_NAME="${BUDGET_NAME:-tiny-leela-first-cloud-guardrail}"
LIMIT_USD="${LIMIT_USD:-100}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --region "$REGION")
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
cat > "$TMP/budget.json" <<JSON
{
  "BudgetName": "$BUDGET_NAME",
  "BudgetLimit": {"Amount": "$LIMIT_USD", "Unit": "USD"},
  "TimeUnit": "MONTHLY",
  "BudgetType": "COST"
}
JSON
cat > "$TMP/notifications.json" <<JSON
[
  {
    "Notification": {"NotificationType": "ACTUAL", "ComparisonOperator": "GREATER_THAN", "Threshold": 50, "ThresholdType": "PERCENTAGE"},
    "Subscribers": [{"SubscriptionType": "EMAIL", "Address": "$EMAIL"}]
  },
  {
    "Notification": {"NotificationType": "FORECASTED", "ComparisonOperator": "GREATER_THAN", "Threshold": 80, "ThresholdType": "PERCENTAGE"},
    "Subscribers": [{"SubscriptionType": "EMAIL", "Address": "$EMAIL"}]
  }
]
JSON
aws budgets create-budget --account-id "$ACCOUNT_ID" --budget file://"$TMP/budget.json" --notifications-with-subscribers file://"$TMP/notifications.json"
echo "Created budget $BUDGET_NAME limit=$LIMIT_USD; confirm subscription email from AWS Budgets."
