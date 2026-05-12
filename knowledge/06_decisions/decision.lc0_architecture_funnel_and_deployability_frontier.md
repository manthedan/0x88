---
id: decision.lc0_architecture_funnel_and_deployability_frontier
type: decision
title: Decision - LC0 architecture funnel and deployability frontier
status: active
created: 2026-05-12
updated: 2026-05-12
priority: high
depends_on:
  - "[[Design - LC0 search-distillation pipeline]]"
  - "[[Design - Candidate frontier cards]]"
  - "[[Experiment - BT4 architecture roadmap next ablations]]"
supports:
  - "[[Roadmap - Current Tiny Leela portfolio]]"
related:
  - "[[Design - Inference optimization]]"
  - "[[Design - Runtime target matrix and workflow delegation]]"
agent_summary: >
  Freeze current 10M architecture ablation churn and promote one BT4/SquareFormer
  winner plus one MF80 winner into LC0 smoke, 10M LC0 pilot, and 100M+ LC0
  scaling if sanity gates pass. Promotion is based on a multi-axis strength,
  latency, FLOPs, params, and deployment frontier; quantization is deployment
  polish, not an architecture-selection gate.
---

# Decision - LC0 architecture funnel and deployability frontier

## Decision

Stop broad 10M architecture ablation churn for now. Promote exactly one clear winner per active architecture lane into the LC0-distillation path, pending necessary correctness/eval sanity:

```text
BT4/SquareFormer lane:
  provisional winner = bt4_h2_flip_av_relbank_d256_l8

MF80 lane:
  provisional winner = mf80_av_top48_10m_flipped_moverel_gate
```

Do not start new 10M architecture variants unless there is a specific reopen trigger. The near-term goal is to validate and scale these candidates, not to keep expanding the architecture matrix.

## Reopen criteria

Further 10M ablations are allowed later only if one of these happens:

- the promoted candidate has a correctness/export/runtime blocker,
- eval exposes a catastrophic weakness not shared by the lane,
- LC0 distillation changes the training regime enough to invalidate the 10M supervised choice,
- a frontier-card metric reveals a poor strength/runtime/complexity tradeoff,
- a specific new idea has a written kill criterion and planned anchor protocol.

## LC0 scaling funnel

Use a narrow LC0 funnel:

1. **Adapter/data proof** on a tiny/fast model or stable path. This proves chunk parsing, policy mapping, WDL/Q perspective, normalization, manifests, and tiny overfit. It is not architecture evidence.
2. **10k-100k LC0 sanity** on the promoted BT4/SquareFormer and MF80 winners.
3. **10M LC0 pilot** on both promoted winners if sanity gates pass.
4. **100M+ LC0 scale** for both promoted winners if no correctness/eval blocker appears.
5. **500M/1B scale** for both only if both remain frontier-relevant; otherwise scale the clear winner and keep the other as a supported baseline/original-project lane.

This allows two serious lanes while preventing matrix explosion.

## Deployability frontier

Do not use a hard MB cap as the architecture-selection rule. "Tiny" is multi-axis and hardware/runtime dependent. Candidate cards should separate:

```text
architecture-intrinsic:
  params
  estimated FLOPs/MACs
  activation/memory shape
  legal-action/top-k cost

deployment-realized:
  raw ONNX bytes
  quantized bytes when available
  browser/native latency
  load/warmup behavior
  memory pressure
  fixed-time and fixed-visit strength
```

Promotion is based on the strength/runtime/size/complexity frontier, not dev loss alone. A model can be larger or slower than another Tiny Leela candidate only if it buys clear chess value at the relevant wall-clock budget.

The primary product frontier is chess quality per real wall-clock move in browser/native search. Current 128-visit models feel instant on good hardware; future LC0-distilled policies may make 128 visits much stronger, and 512 visits may still be acceptable on many devices. Serious deployable candidates should therefore report both fixed-visit and fixed-time behavior.

## Quantization policy

Quantization is separate from architecture selection for now:

- PTQ/QAT is deployment polish and a possible bonus, not a 10M/100M architecture gate.
- Quantized models should replace FP32 only with effectively zero quality loss.
- QAT may move earlier only if evidence shows it is necessary to preserve quality or unlock a target runtime.
- LC0-style QAT gains are worth investigating later, but Tiny Leela should first find strong FP32/BF16 candidates under a loose tiny heuristic.

## Browser adaptive deployment

Adaptive browser deployment is a final-deploy requirement, not routine ablation paperwork. Final deployable candidates should support:

- capability detection: WebGPU, ORT WebGPU graph support, WASM SIMD/threads, and rough device class,
- lazy loading of the best compatible export variant,
- a small warmup/benchmark to estimate eval speed,
- automatic visit selection for a target latency range,
- manual user override and recorded chosen visits in analysis metadata.

This belongs in the optional deployment section of [[Design - Candidate frontier cards]]. Training candidates only need the lightweight core card.
