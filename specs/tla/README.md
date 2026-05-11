# Tiny Leela TLA+ specs

Small formal models for the orchestration invariants that are easier to get wrong than they look.

Current specs:

- `ShardLifecycle.tla` — AWS Batch/S3 shard lifecycle; a run cannot be committed until every shard validates.
- `ModelPromotionLanes.tla` — accepted/candidate/rejected model discipline and Zero vs SUP-SP replay separation.

These specs intentionally model the control plane, not chess rules or neural search quality.

Expected future check command, once TLC is installed:

```bash
java -jar tla2tools.jar -config specs/tla/ShardLifecycle.cfg specs/tla/ShardLifecycle.tla
java -jar tla2tools.jar -config specs/tla/ModelPromotionLanes.cfg specs/tla/ModelPromotionLanes.tla
```

At the moment this repo/harness does not have TLC installed, so the `.tla` files are scaffolds until a TLA+ toolchain is added.
