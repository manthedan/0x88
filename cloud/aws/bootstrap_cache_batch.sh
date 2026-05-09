#!/usr/bin/env bash
set -euo pipefail

: "${BUCKET:?set BUCKET=tiny-leela-distributed-...}"
REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-west-2}}"
PROJECT="${PROJECT:-tiny-leela-cache}"
MAX_VCPUS="${MAX_VCPUS:-32}"
BID_PERCENTAGE="${BID_PERCENTAGE:-50}"

command -v aws >/dev/null || { echo "aws CLI not found" >&2; exit 2; }
command -v docker >/dev/null || { echo "docker not found" >&2; exit 2; }
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --region "$REGION")
ECR_REPO="$PROJECT-worker"
IMAGE_URI="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPO:latest"

echo "CREATE/VERIFY bucket s3://$BUCKET region=$REGION"
if ! aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  aws s3api create-bucket --bucket "$BUCKET" --region "$REGION" --create-bucket-configuration LocationConstraint="$REGION"
fi
aws s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true

echo "CREATE/VERIFY ECR repo $ECR_REPO"
aws ecr describe-repositories --repository-names "$ECR_REPO" --region "$REGION" >/dev/null 2>&1 || aws ecr create-repository --repository-name "$ECR_REPO" --region "$REGION" >/dev/null
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"
docker build -f cloud/aws/Dockerfile.cache-worker -t "$IMAGE_URI" .
docker push "$IMAGE_URI"

VPC_ID=$(aws ec2 describe-vpcs --region "$REGION" --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
if [[ -z "$VPC_ID" || "$VPC_ID" == "None" ]]; then echo "No default VPC found; pass parameters manually to CloudFormation." >&2; exit 3; fi
SUBNET_IDS=$(aws ec2 describe-subnets --region "$REGION" --filters Name=vpc-id,Values="$VPC_ID" --query 'Subnets[].SubnetId' --output text | tr '\t' ',')
SG_ID=$(aws ec2 describe-security-groups --region "$REGION" --filters Name=vpc-id,Values="$VPC_ID" Name=group-name,Values=default --query 'SecurityGroups[0].GroupId' --output text)

echo "DEPLOY CloudFormation stack $PROJECT"
aws cloudformation deploy \
  --region "$REGION" \
  --stack-name "$PROJECT" \
  --template-file cloud/aws/batch_cpu_spot_cloudformation.yaml \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
    ProjectName="$PROJECT" \
    BucketName="$BUCKET" \
    ContainerImage="$IMAGE_URI" \
    VpcId="$VPC_ID" \
    SubnetIds="$SUBNET_IDS" \
    SecurityGroupIds="$SG_ID" \
    MaxVCpus="$MAX_VCPUS" \
    BidPercentage="$BID_PERCENTAGE"

echo "DONE"
aws cloudformation describe-stacks --region "$REGION" --stack-name "$PROJECT" --query 'Stacks[0].Outputs' --output table
