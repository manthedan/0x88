#!/usr/bin/env bash
set -euo pipefail

USER_NAME="${USER_NAME:-tiny-leela-operator}"
PROFILE="${PROFILE:-tiny-leela}"
POLICY_NAME="${POLICY_NAME:-TinyLeelaBootstrapPolicy}"
REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-west-2}}"
POLICY_DOC="${POLICY_DOC:-cloud/aws/tiny_leela_bootstrap_policy.json}"

command -v aws >/dev/null || { echo "aws CLI not found" >&2; exit 2; }
[[ -s "$POLICY_DOC" ]] || { echo "missing policy doc: $POLICY_DOC" >&2; exit 2; }

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/${POLICY_NAME}"

if aws iam get-user --user-name "$USER_NAME" >/dev/null 2>&1; then
  echo "IAM user exists: $USER_NAME"
else
  echo "Creating IAM user: $USER_NAME"
  aws iam create-user --user-name "$USER_NAME" >/dev/null
fi

if aws iam get-policy --policy-arn "$POLICY_ARN" >/dev/null 2>&1; then
  echo "Updating managed policy: $POLICY_ARN"
  versions=$(aws iam list-policy-versions --policy-arn "$POLICY_ARN" --query 'Versions[?IsDefaultVersion==`false`]|sort_by(@,&CreateDate)[].VersionId' --output text)
  count=$(aws iam list-policy-versions --policy-arn "$POLICY_ARN" --query 'length(Versions)' --output text)
  if [[ "$count" -ge 5 ]]; then
    oldest=$(echo "$versions" | awk '{print $1}')
    [[ -n "$oldest" ]] && aws iam delete-policy-version --policy-arn "$POLICY_ARN" --version-id "$oldest"
  fi
  aws iam create-policy-version --policy-arn "$POLICY_ARN" --policy-document "file://$POLICY_DOC" --set-as-default >/dev/null
else
  echo "Creating managed policy: $POLICY_ARN"
  aws iam create-policy --policy-name "$POLICY_NAME" --policy-document "file://$POLICY_DOC" >/dev/null
fi

aws iam attach-user-policy --user-name "$USER_NAME" --policy-arn "$POLICY_ARN"

KEY_COUNT=$(aws iam list-access-keys --user-name "$USER_NAME" --query 'length(AccessKeyMetadata)' --output text)
if [[ "$KEY_COUNT" -eq 0 ]]; then
  echo "Creating one access key and storing it in local AWS profile: $PROFILE"
  tmp=$(mktemp)
  chmod 600 "$tmp"
  aws iam create-access-key --user-name "$USER_NAME" > "$tmp"
  AKID=$(python3 - <<'PY' "$tmp"
import json,sys
print(json.load(open(sys.argv[1]))['AccessKey']['AccessKeyId'])
PY
)
  SAK=$(python3 - <<'PY' "$tmp"
import json,sys
print(json.load(open(sys.argv[1]))['AccessKey']['SecretAccessKey'])
PY
)
  aws configure set aws_access_key_id "$AKID" --profile "$PROFILE"
  aws configure set aws_secret_access_key "$SAK" --profile "$PROFILE"
  rm -f "$tmp"
else
  echo "Access key already exists for $USER_NAME; not creating another."
fi
aws configure set region "$REGION" --profile "$PROFILE"

echo "Profile identity:"
AWS_PROFILE="$PROFILE" aws sts get-caller-identity --output json
