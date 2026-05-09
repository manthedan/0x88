# Tiny Leela cloud job guardrails

This note captures the operational guardrails added after the 10M successes and 100M row-shortfall failures.

## Failure modes to prevent

- Late row shortfall, e.g. a 100M h8 job reaching only ~83.8M train rows.
- Accidental uncompressed 100M raw input uploads.
- Cache fanout from a missing or non-h8/non-100M dataset manifest.
- Ambiguous state spread across AWS Batch, CloudWatch, S3, local logs, and chat.
- Repair submissions that add inputs or loosen filters without a written contract.

## Required pre-submit contract

Before a 100M h8 dataset submit, run:

```bash
./scripts/tlops cloud preflight-h8-dataset \
  --base-dataset-dir data/datasets/supervised_100m_elite_tcec_v1 \
  --max-rows 100000000 \
  --dev-rows 1000000 \
  --compress-inputs \
  --estimate-capacity
```

This checks:

- local raw inputs exist;
- 100M submissions declare compressed-input handling;
- target train/dev/history contract;
- selection-rule capacity using the same core row-selection logic as the AWS worker;
- a default 1.15x safety margin.

For quick, non-submit inspection only, use `--allow-without-capacity-estimate`.

## Manifest gate before cache fanout

Before h7/h8 SquareFormer cache jobs:

```bash
./scripts/tlops cloud validate-h8-manifest \
  --s3-prefix s3://tiny-leela-distributed-ddbb/h8_dataset_100m/datasets/supervised_100m_elite_tcec_h8_v1 \
  --expect-history 8 \
  --expect-train 100000000 \
  --expect-dev 1000000 \
  --expect-train-shards 100
```

Only submit cache fanout if this passes.

## DAG status dashboard

Use:

```bash
./scripts/tlops cloud status-h8-100m \
  --write-md artifacts/cloud_h8_dataset_100m/status.md
```

The command summarizes active AWS jobs, S3 manifest validity, and an optional local log tail.

## Row-shortfall repair workflow

If a dataset job fails from row shortfall:

```bash
./scripts/tlops cloud repair-plan-h8-dataset --actual-train ACTUAL_ROWS
```

Preferred order:

1. Add more high-quality elite/TCEC raw months.
2. Regenerate/extend the input manifest.
3. Rerun `preflight-h8-dataset --estimate-capacity`.
4. Resubmit with compressed inputs and a clear output prefix/overwrite decision.
5. Validate the final S3 manifest before cache fanout.

Fallbacks requiring explicit approval:

- increase `--max-rows-per-game`;
- lower `--skip-plies`;
- loosen opening/source/dedupe rules;
- accept a sub-100M dataset.
