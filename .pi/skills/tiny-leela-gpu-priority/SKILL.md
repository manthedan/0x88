---
name: tiny-leela-gpu-priority
description: GPU scheduling and safety rules for Tiny Leela training/eval processes. Use before launching, resuming, pausing, or killing GPU jobs, especially BT4/SquareFormer, Tactical MoveFormer, and local CUDA training.
---

# Tiny Leela GPU Priority

Use this skill whenever touching local GPU processes.

## Priority order

Current priority is:

```text
1. BT4 / SquareFormer h7+h8 training
2. Required evaluation/promote gates for BT4 outputs
3. Tactical MoveFormer GPU work
4. Opportunistic experiments
```

Do not resume Tactical MoveFormer while BT4 is actively using the RTX 3090 unless the user explicitly changes priority.

## Important interpretation

Low VRAM use does **not** mean the GPU is idle. Tiny BT4 can saturate SM/compute with low memory use.

Before launching another CUDA job, check:

```bash
nvidia-smi
```

If available, prefer detailed utilization too:

```bash
nvidia-smi dmon -s pucmt -c 5
```

Signs BT4 owns the GPU:

```text
GPU util near 90-100%
SM util high
power high, e.g. ~350-420W on RTX 3090
python training PID from BT4 log/status
```

## Required launch hygiene

Any new long-running GPU job must have:

```text
output directory
run log
pid file
status file or registry entry
clear terminal marker: done/failed/cancelled
```

Prefer registry integration:

```bash
./scripts/tlops run list --active
```

## Tactical MoveFormer pause state

Known BT4-priority pause markers may exist here:

```text
artifacts/head_ablation_1m/tactical_moveformer64_1m/paused_for_bt4
artifacts/head_ablation_1m/tactical_moveformer64_1m/bt4_priority_stop
```

If these markers exist, do not remove them or SIGCONT related processes unless:

1. BT4 has finished or been cancelled,
2. GPU is available,
3. user explicitly agrees to resume Tactical MoveFormer.

## Safe workflow before action

1. Inspect active registry/processes:
   ```bash
   ./scripts/tlops run list --active
   nvidia-smi
   ```
2. Inspect relevant pid/log/status files.
3. If pausing, prefer graceful script-defined controls; otherwise record why SIGSTOP/SIGTERM was used.
4. If killing/cancelling, write/append a durable failed/cancelled marker.
5. Summarize exactly which PIDs/jobs changed.

## Never do this

- Do not start a second heavy training job just because VRAM is free.
- Do not silently resume paused jobs.
- Do not delete pid/log/status files to make a run look inactive.
- Do not mix h2/h8 cache assumptions when launching BT4 training.
