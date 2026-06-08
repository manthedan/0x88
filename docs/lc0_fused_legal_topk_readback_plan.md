# LC0 fused legal-mask/top-k/readback-shrink plan

Status: scoped only; not implemented. Stable defaults remain unchanged.

## Goal

Replace the standalone GPU-legal-prior experiment with a single opt-in WGSL-head postprocess lane that fuses enough work to justify the extra GPU dispatches:

1. apply legal-move mask to the mapped policy logits,
2. compute legal softmax/top-k for search priors,
3. read back only compact legal candidates needed by PUCT, plus WDL,
4. preserve current JS-legal semantics and drift gates.

The prior standalone `legalPriorsBackend=gpu` toggle reduced bytes but did not improve fixed-suite throughput. A future attempt must therefore shrink both bytes and CPU/queue work as one fused path, not just move legal-prior prep to GPU.

## Non-goals

- Do not change stable ORT or hybrid defaults.
- Do not use `batchPipelineDepth>1` as promotion evidence.
- Do not accept speed without best-move and policy/value drift gates.
- Do not repeat standalone GPU legal priors unless a new fused implementation shape depends on it as an internal substage.

## Proposed opt-in shape

Add a new explicit legal-prior backend such as `legalPriorsBackend=gpu-topk` or a more specific `policyReadbackBackend=legal-topk` for `hybrid-wgsl-heads` only.

Per evaluated position, the fused route should output:

- WDL `[3]`, unchanged semantics;
- compact legal candidates sorted by prior, with UCI/policy index/prior/logit;
- enough metadata to reconstruct `Lc0Evaluation.priors` exactly for search, or a documented top-K mode that is never used for promotion until search parity passes.

Implementation sketch:

1. Keep WGSL mapped-policy logits on GPU after the current heads dispatch.
2. Upload or reuse a compact legal-move descriptor buffer produced by JS/WASM movegen. Do not attempt GPU movegen first.
3. Run a legal-mask + max-reduction pass over legal descriptors.
4. Run exp/sum normalization over legal descriptors.
5. Run a bounded top-K selection or partial sort over legal descriptors.
6. Copy one compact candidate buffer plus WDL to the readback buffer.

## Gates

Minimum acceptance gates for any implementation PR:

1. Isolated parity: compare GPU compact candidates to JS `legalPolicyPriors` on all existing FEN/history legal-prior fixtures, including castling, en passant, checks, and promotions.
2. Hybrid drift: `npm run lc0:browser-hybrid-drift -- --head-backend wgsl --input-backend wasm --encoder-kernel mixed-tvm-ffn --legal-priors-backend <new-mode> --limit 9` must match f32/native best moves and stay within WDL/top-prior drift thresholds.
3. Fixture search parity: depth1-only `lc0:browser-hybrid-search-fixture-parity` with WASM input, WGSL heads, b4, `batchPipelineDepth=1`, and progress timeout must pass native/depth-baseline checks.
4. Fixed-suite speed: cleaned b4/depth1 fixed suite must beat a same-session JS-legal control by a meaningful margin; byte reduction alone is insufficient.
5. Diagnostics: artifact must report legal descriptor count, top-K count, readback bytes, map count, dispatch count, legal-prior GPU time if measurable, and JS/WASM prep time for descriptor creation.

## Main risks

- Extra dispatches and queue synchronization can erase byte savings.
- Descriptor upload/prep can replace, not remove, JS legal-prior overhead.
- Top-K truncation can perturb search if the tail priors matter; full legal candidate readback may be required for parity.
- GPU sort/reduction complexity may exceed the small CPU legal-prior cost in b4/depth1.

## First safe milestone

Before any fixed-suite benchmark, implement only an isolated browser route that consumes mapped-policy logits and a legal descriptor list for fixtures, then reports max prior/logit drift and compact readback bytes. Promote to search wiring only after that route is parity-clean.
