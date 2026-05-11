# Mac mini CPU offload plan

Status: Current  
Scope: bounded CPU evaluation, arena, and search-calibration jobs for Tiny Leela.

## Executive rule

Use a three-tier compute split:

- **Local workstation:** GPU training, model export, parity/correctness gates, tiny smoke runs, and anything that must touch active training state.
- **Mac mini:** bounded CPU post-training evaluation: ONNX PUCT audits, visit curves, aux-PUCT tuning, anchor arenas, latency/search benchmarks, and other jobs that are expensive enough to interrupt local work but small enough to finish on one CPU box.
- **Cloud:** massive data/caches/reanalysis/self-play: dataset construction, SquareFormer cache fanout, large teacher jobs, large self-play generation, and any arena/self-play target needing many workers.

This keeps the local GPU saturated on training while the Mac mini turns completed checkpoints into eval evidence.

## Why Mac mini is now the default CPU eval box

Measured on CNN96x8 100M e08 ONNX PUCT batch benchmark:

```text
artifact: artifacts/remote_offload/mac_mini_cnn96_puct_visit_curve_20260509T191129Z/puct_batch_benchmark.log
local:    artifacts/local_cpu_compare/cnn96_puct_batch_20260509T200043Z/puct_batch_benchmark.log
```

The Mac mini M4 is about **3.0x faster** than the local Ryzen 3900XT for this ONNX/PUCT CPU workload.

Representative median times:

```text
visits  batch  mac_ms  local_ms  mac_speedup
32      32     9       29        3.2x
128     32     100     308       3.1x
256     32     304     924       3.0x
512     32     735     2279      3.1x
1024    32     1512    4459      3.0x
```

Operational interpretation:

- Mac mini should receive CPU eval jobs by default.
- Local CPU should only run eval jobs when Mac is unavailable, when the job is a tiny smoke, or when the result is needed before remote sync overhead is worth it.
- Keep Mac eval jobs bounded and artifact-producing; do not turn it into an unmanaged always-on self-play farm.

## Jobs to offload to Mac mini

### 1. Post-training release/eval packet

After a new candidate checkpoint is exported to ONNX and passes local correctness gates, run on the Mac mini:

- PUCT core tests and root-prior parity.
- PUCT consistency over a fixed opening/position set.
- PUCT batch benchmark over visits/batch sizes.
- Visit curve versus a nearby Tiny Leela baseline.
- Optional quick Stockfish-js anchor check.

Current helper:

```bash
MODEL=path/to/model.onnx \
META=path/to/model.meta.json \
ANCHOR_MODEL=path/to/baseline.onnx \
ANCHOR_META=path/to/baseline.meta.json \
VISIT_STEPS="32 64 128 256 512 1024" \
GAMES_PER_PAIR=4 \
./scripts/remote_cpu_offload_puct_visit_curve.sh
```

Pull partial/final results:

```bash
ACTION=pull \
RDIR='/Users/minime/<remote-run-dir>' \
LOCAL_OUT='artifacts/remote_offload/<job-name>' \
./scripts/remote_cpu_offload_puct_visit_curve.sh
```

### 2. Search-parameter tuning

Use Mac mini for bounded tuning sweeps that would otherwise steal local CPU and interfere with training:

- Bayesian aux-PUCT-by-visit tuning.
- Classic PUCT cpuct/visit sweeps.
- Batch-size and ORT-thread sensitivity checks.
- Shallow Gumbel/root probes if they are CPU-only and bounded.

Current helper:

```bash
MODEL=path/to/model.onnx \
META=path/to/model.meta.json \
VISIT_STEPS="32 64 128 256 512 1024" \
ITERATIONS=8 \
GAMES_PER_CANDIDATE=4 \
./scripts/remote_cpu_offload_puct_bayes_by_visit.sh
```

### 3. Anchor arenas

Use Mac mini for CPU UCI anchor sweeps when the required engines are available there:

- Stockfish-js Elo-limited anchors.
- Stockfish-js full-single low-node anchors.
- Maia/lc0 anchors only after remote wrappers and weights are installed/synced.

Current helper after a Bayesian tuning run:

```bash
BAYES_LOCAL='artifacts/remote_offload/<bayes-job>' \
PAIRS=4 \
STOCKFISH_ELO_LEVELS=1320,1600,1800 \
STOCKFISH_ELO_NODES=64 \
FULL_STOCKFISH_NODES='8 16 32' \
./scripts/remote_cpu_offload_tuned_puct_anchor_sweep.sh
```

If Maia is not installed on the Mac, run Maia legs locally with `.local_engines/maia/*.sh`, or install/sync remote lc0/Maia wrappers before enabling `INCLUDE_MAIA=1`.

### 4. CPU-only diagnostics

Good Mac candidates:

- evaluator byte-vs-path parity over many positions,
- mirrored-FEN consistency sweeps,
- move encoding/policy-map stress checks,
- ONNX bucket eval JSONL runs,
- opening-suite sensitivity checks,
- protocol-card generation and summarization.

Keep local versions for quick smoke only.

## Jobs that should stay local

Keep these on the local workstation:

- GPU training and resume logic.
- Export immediately after training, especially when it depends on local checkpoint paths.
- Correctness gates before remote sync:
  ```bash
  npm run typecheck
  node --experimental-strip-types eval/puct_core_tests.mjs
  node --experimental-strip-types --test tests/encoding_parity.test.mjs tests/policy_map.test.mjs
  ```
- Tiny smoke runs needed to catch broken exports before copying ONNX files to the Mac.
- Any job that needs local-only Maia/lc0 wrappers until the Mac is provisioned.

Do not resume Tactical MoveFormer GPU work without explicit approval, even if Mac CPU is idle.

## Jobs that should go to cloud

Cloud is the default for work that is throughput-oriented, S3-native, or embarrassingly parallel:

- h8 dataset builds and repair jobs,
- h7/h8 SquareFormer cache fanout,
- large teacher overlay generation,
- large Stockfish reanalysis,
- massive self-play generation,
- large arena matrices with thousands to millions of games,
- large distributed evaluation intended to approximate OpenBench-style evidence.

Mac mini should not be used as a substitute for distributed cloud when the goal is massive volume.

## Self-play between local and Mac mini

Two-machine self-play is useful for **smoke and calibration**, not for the super-massive target.

From current Mac measurements on CNN96 PUCT visit curves:

```text
32 visits:   4 games / ~5s   => ~2,900 games/hour on Mac for short bounded games
512 visits:  4 games / ~90s  => ~160 games/hour on Mac
1024 visits: 4 games / ~196s => ~73 games/hour on Mac
```

Using the measured ~3x Mac-vs-local speed ratio, local CPU would add roughly one-third of the Mac throughput if it were idle. Combined rough envelope:

```text
low visits:  a few thousand games/hour
512 visits:  ~200 games/hour
1024 visits: ~100 games/hour
```

That is enough for:

- self-play smoke tests,
- resign/adjudication calibration,
- policy-temperature experiments,
- small Gumbel/root-selection tests,
- debugging chunk schemas and validation.

It is not enough for the desired large-scale self-play program. For serious self-play, keep the accepted-model loop, chunk schema, validation, and replay ingestion design, but target cloud workers.

Recommended policy:

- **Local + Mac self-play:** only for infrastructure smoke and small calibration batches.
- **Cloud self-play:** required for large-scale training data generation and serious rating evidence.

## Default post-training process

1. Train/export locally.
2. Run local correctness/parity smoke.
3. Launch Mac mini visit curve and benchmark packet.
4. Pull Mac results into `artifacts/remote_offload/<job>/`.
5. If promising, launch Mac Bayesian/tuned anchor packet.
6. If still promising and the question needs volume, escalate to cloud distributed arena/self-play.
7. Record paths and outcome in the relevant status file, model manifest, or release-gate summary.

## Operational conventions

- Use `REMOTE=mac-mini` unless intentionally targeting another host.
- Keep `ORT_THREADS=6` by default on the Mac mini; increase only after a specific benchmark shows improvement.
- Prefer one heavy Mac offload job at a time. Multiple small pulls/status checks are fine.
- Always write results under `artifacts/remote_offload/<job-name>/`.
- For detached jobs, preserve `remote_info.env` and use `ACTION=pull` rather than ad-hoc rsync.
- Keep remote workdirs until results are pulled and summarized; clean with `ACTION=clean` only after confirming local artifacts.
- Do not commit generated `artifacts/remote_offload/*` outputs.

## Promotion implications

Mac mini results are valid as bounded CPU evidence when protocol files are present, but they are not automatically promotion decisions. A promotion candidate should still satisfy the normal funnel:

1. supervised metrics,
2. local export/parity gates,
3. Mac CPU smoke/visit curve,
4. Mac anchor sweep or local Maia sweep,
5. larger cloud/distributed gate if the model is close to promotion.
