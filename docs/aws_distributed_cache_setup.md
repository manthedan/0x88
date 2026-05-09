# AWS distributed cache worker setup

Status: first cloud scaffold for CPU-only shard jobs in `us-west-2` with a $100 guardrail target.  The first intended workload is SquareFormer h7/h8 token-cache generation, not full self-play.

Current deployed setup:

```text
AWS profile: tiny-leela
IAM user: tiny-leela-operator
bucket: s3://tiny-leela-distributed-ddbb
region: us-west-2
Batch compute env: tiny-leela-cache-cpu-spot
Batch job queue: tiny-leela-cache-queue
Batch job definition: tiny-leela-cache-squareformer-cache
ECR repo: tiny-leela-cache-worker
max vCPUs: 32
worker shape: 2 vCPU / 4096 MiB
budget alert: tiny-leela-first-cloud-guardrail, $100, d.dubats@gmail.com
```

Confirm the AWS Budgets subscription email before relying on alerts.

## Goal

Use AWS for boring distributed CPU work:

```text
S3                 dataset shards, job manifests, cache archives
ECR                cache-worker container image
AWS Batch CPU Spot shard workers
CloudWatch Logs    worker logs
AWS Budgets        $100 guardrail alert
```

This is deliberately smaller than the future distributed self-play system.  It implements the safe part first: shard-parallel cache/reanalysis workers with manifest/checksum outputs.

## What you set locally

Do not paste secrets into chat.  Configure the AWS CLI locally.  The safer local profile is:

```bash
export AWS_PROFILE=tiny-leela
export AWS_DEFAULT_REGION=us-west-2
aws sts get-caller-identity
```

The root credentials were only used to create the restricted `tiny-leela-operator` IAM user and attach `cloud/aws/tiny_leela_bootstrap_policy.json`.  The reproducible helper is:

```bash
cloud/aws/create_operator_user.sh
```

Optional budget guardrail:

```bash
EMAIL=you@example.com LIMIT_USD=100 \
  cloud/aws/create_budget_guardrail.sh
```

AWS will send an email confirmation for budget alerts.

## Bootstrap Batch + ECR + S3

Pick a globally unique bucket name:

```bash
export AWS_PROFILE=tiny-leela
export AWS_DEFAULT_REGION=us-west-2
export BUCKET=tiny-leela-distributed-ddbb
export PROJECT=tiny-leela-cache
export MAX_VCPUS=32
export BID_PERCENTAGE=50

cloud/aws/bootstrap_cache_batch.sh
```

The bootstrap script:

```text
creates/verifies S3 bucket
blocks public bucket access
creates/verifies ECR repo
builds and pushes cloud/aws/Dockerfile.cache-worker
discovers the default VPC/subnets/security group
deploys cloud/aws/batch_cpu_spot_cloudformation.yaml
prints JobQueueName and JobDefinitionName
```

If your account has no default VPC, create one or deploy the CloudFormation template manually with chosen subnet/security-group parameters.

## Submit a SquareFormer cache job

Example for a true h8 dataset, deriving an h7 cache:

```bash
cloud/aws/submit_squareformer_cache_jobs.sh \
  --dataset-dir data/datasets/supervised_10m_elite_tcec_h8_v1 \
  --bucket-uri s3://$BUCKET/h7h8_10m \
  --history 7 \
  --job-queue tiny-leela-cache-queue \
  --job-definition tiny-leela-cache-squareformer \
  --upload-dataset
```

Then submit h8 similarly:

```bash
cloud/aws/submit_squareformer_cache_jobs.sh \
  --dataset-dir data/datasets/supervised_10m_elite_tcec_h8_v1 \
  --bucket-uri s3://$BUCKET/h7h8_10m \
  --history 8 \
  --job-queue tiny-leela-cache-queue \
  --job-definition tiny-leela-cache-squareformer
```

`--upload-dataset` is only needed once if the dataset already exists in S3.

Outputs land under:

```text
s3://$BUCKET/h7h8_10m/caches/DATASET/cache_squareformer_h7/
s3://$BUCKET/h7h8_10m/caches/DATASET/cache_squareformer_h8/
```

Each shard writes:

```text
meta.json
cache.tar.zst
cache.tar.zst.sha256
worker_manifest.json
```

## Download/extract cache locally

After jobs complete:

```bash
cloud/aws/download_squareformer_cache.sh \
  --cache-prefix s3://$BUCKET/h7h8_10m/caches/supervised_10m_elite_tcec_h8_v1/cache_squareformer_h7 \
  --out-dir data/datasets/supervised_10m_elite_tcec_h8_v1/cache_squareformer_h7 \
  --train-shards 40 \
  --history 7 \
  --dataset-manifest data/datasets/supervised_10m_elite_tcec_h8_v1/manifest.json
```

The downloader extracts each archive into the local training layout and writes:

```text
cache_manifest.json
```

## Cost notes

For SquareFormer compact-token caches, h7/h8 are modestly larger than h2:

```text
h2 token_features = 11
h7 token_features = 16  (~1.44x token bytes)
h8 token_features = 17  (~1.53x token bytes)
```

The expensive part is usually dataset upload/download and redundant IO, not CPU.  CPU Spot workers should be far below the $100 guardrail for a 10M cache if the job is kept to one pass and artifacts are cleaned up.

## Stop / cleanup

Cancel jobs:

```bash
aws batch list-jobs --job-queue tiny-leela-cache-queue --job-status RUNNING
aws batch cancel-job --job-id JOB_ID --reason "manual stop"
```

Scale down happens automatically because Batch `MinvCpus=0` and `DesiredvCpus=0`.

Delete stack when done:

```bash
aws cloudformation delete-stack --stack-name tiny-leela-cache --region us-west-2
```

Delete S3 data only when you are sure:

```bash
aws s3 rm s3://$BUCKET/h7h8_10m --recursive
```

## Current limitations

- This scaffold is CPU-only.
- It assumes AWS Batch EC2 Spot, not Fargate, because cache archives can be large.
- It uses JSON/S3 text shard lists for simplicity.
- Worker outputs are S3 archives; download/extract before local PyTorch training.
- Full distributed self-play is intentionally out of scope for this first cloud milestone.
