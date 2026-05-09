#!/usr/bin/env bash
set -euo pipefail
REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-west-2}}"
ACCOUNT_ID="${ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"
REPO="${REPO:-tiny-leela-cache-worker}"
TAG="${TAG:-dataset-latest}"
IMAGE_URI="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO:$TAG"

echo "LOGIN ECR $REGION account=$ACCOUNT_ID"
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com" >/dev/null

echo "BUILD $IMAGE_URI"
docker build -f cloud/aws/Dockerfile.dataset-worker -t "$IMAGE_URI" .

echo "PUSH $IMAGE_URI"
docker push "$IMAGE_URI"

echo "$IMAGE_URI"
