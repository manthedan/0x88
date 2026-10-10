# Centipawn: encode real rows, copy padding

## Change and scope

`SquareformerTvmjsWebgpuEvaluator.evaluatePhysicalBatch` previously repeated the
last prepared row to physical batch 16, then re-encoded its tokens and attack
summary for every padding slot. It now encodes each logical row once, applies its
channel mask once, and copies the last real row's typed-array contents into the
remaining slots. Normalization, history, int32 tokens, float32 attack summaries,
physical batch 16, model precision, search and GPU execution are unchanged.

The other Squareformer paths were inspected:

- `src/nn/squareformerEvaluator.ts` (ORT fallback) allocates for logical rows only;
  static models use separate batch-1 evaluations. No repeated padding encoding.
- `src/nn/squareformerTvmHybridEvaluator.ts` (custom-WebGPU hybrid) evaluates one
  position at a time, without physical padding. No matching change needed.

## Byte parity

`tests/centipawn_tvmjs_evaluator.test.mjs` captures the actual token and attack
bytes passed to the evaluator's fake TVM runtime. An independent original-loop
reference in `tests/fixtures/centipawn_encoding.mjs` normalizes real rows, pads
those rows, then runs the unchanged pure TypeScript encoders for every slot.

There are 128 comparisons: b1/b4/b8/b16 × normalization on/off × masks on/off ×
eight position rotations. Coverage includes White/Black to move, full seven-ply,
partial and empty history, repetition tokens, en passant, castling, promotions,
halfmove limits, fractional/negative/negative-zero masks and distinct last rows.
A 17-position test checks chunk boundaries. A mask-read-count test proves that
padding does not reapply the last real mask; invalid-mask and empty-input behavior
are also checked.

## Artifacts and environment

Model: **bt4-soap-rem-c19000-final**, production **v2-shape-k16**, f32,
physical batch **16**. SHA-256 values verified against the local files:

| Artifact | SHA-256 |
| --- | --- |
| ONNX model | `de2dd68518f6e2ed162da5ee27c67a5a4795567d6e39972e3705824572b7660e` |
| Metadata | `5b78093585ea34d3bcda599ed62d2c01fa4c1839114d9e81ee5b1f6dac4b1b22` |
| TVMJS batch-16 model WASM | `fd3d8635c8368338c3a0d1b57a725a70b4f8a74f8d130ee831848c641cba75b5` |
| TVMJS runtime JS | `e5225b46a98cc6c5ab133233a67f830e856ac0a23b01022768e7f94ed58a8111` |
| TVMJS runtime WASM | `f5d0b00f63ee878a2144f576ab41825a7c3d13a79621782dcc98b14f4d1c58e5` |
| v2 manifest | `3b097749e62cdaf88165d01d187b56fafc4a497b0564272be67d95202696dfe9` |
| Before evaluator source | `1fa7705eb24219eb73097792f395f6cfa9480ce20e0fdea5d4d443f625ab5170` |
| After evaluator source | `841d6992e2960f343e3b41a29d885772435a32ab19c7dc57d388362816b0b338` |

Baseline commit: `5761490ef5533d82870bbaf0b0d3529a3298ddd3` (main at branch start).
Node **v24.16.0**, Apple **M4**, macOS/Darwin **25.4.0 arm64**.
Browser: **HeadlessChrome/149.0.0.0**, WebGPU adapter **apple / metal-3**,
cross-origin isolated. The ONNX model identifies the weights; the browser search
actually uses the TVMJS WASM above, with no ORT fallback (strict variant).
Raw samples and the browser measurement expression are in
`docs/centipawn_padded_encoding.evidence.json`.

## Preprocessing measurement

`scripts/bench_centipawn_encoding.mjs` imports the actual before evaluator from
`git show <baseline-ref>:src/nn/squareformerTvmjsWebgpuEvaluator.ts` into a temporary
file in **this** worktree and removes it in `finally`. The after evaluator is
imported directly. Both use the current, unchanged encoder dependencies.

Timing begins at the exclusive callback entry and ends at the first VM function
lookup, immediately after preprocessing. It includes normalization, history
parsing, allocation, encoding, masking and padding; it excludes GPU work,
postprocessing, and serialization/promise queue latency. A stub aborts execution
at that boundary, so this is CPU preprocessing per physical call, not inference.

Each configuration uses 500 warmup calls per variant, followed by 12 rounds of
200 calls per variant. A/B and B/A order alternate. Reported medians and min–max
spread are over the 12 round means (not individual-call percentiles). All timing
runs use the shared machine-wide benchlock; no build or test was run by this
agent concurrently with the final preprocessing run.

Two workloads use mixed distinct real rows from the parity fixture (repeated
only when more than eight real rows are needed):

- **No history**: b1 is the White starting position; larger batches mix sides and
  positions, with empty histories and no masks.
- **Seven history + mask**: b1 is Black to move after e4; larger batches mix sides
  and positions. Every real row has seven history FENs and a fractional mask.

| Workload | Logical batch | Before ms (min–max) | After ms (min–max) | Speedup |
| --- | ---: | ---: | ---: | ---: |
| No history | 1 | 0.3203 (0.3090–0.3331) | 0.0286 (0.0251–0.0331) | 11.19× |
| No history | 4 | 0.3249 (0.3149–0.3482) | 0.0933 (0.0863–0.0963) | 3.48× |
| No history | 8 | 0.1765 (0.1687–0.1822) | 0.1241 (0.1177–0.1333) | 1.42× |
| No history | 16 | 0.2426 (0.2289–0.2576) | 0.2403 (0.2256–0.2522) | 1.01× |
| Seven history + mask | 1 | 0.8082 (0.7905–0.8286) | 0.0807 (0.0763–0.0864) | 10.01× |
| Seven history + mask | 4 | 0.6763 (0.6553–0.7186) | 0.2390 (0.2281–0.2656) | 2.83× |
| Seven history + mask | 8 | 0.8212 (0.7985–0.8601) | 0.4408 (0.4256–0.4777) | 1.86× |
| Seven history + mask | 16 | 0.8813 (0.8526–0.9094) | 0.8757 (0.8517–0.9124) | 1.01× |

The short logical batches improve consistently in the isolated preprocessing
measurement; b16 is effectively unchanged. Absolute times across logical sizes
are not monotonic because different final rows supply padding (and attack-summary
cost depends on the board). Compare before/after **within** each row.

Reproduce from the repository root (set `BENCHLOCK` to the assigned shared mutex):

```sh
"$BENCHLOCK" node --experimental-strip-types scripts/bench_centipawn_encoding.mjs \
  --baseline-ref 5761490 --rounds 12 --iterations 200 --warmup 500 \
  --out .local-dev-artifacts/centipawn-encoding.json
```

## Production-browser Analysis A/B

Both versions were built with `BUILD_SCOPE=product npm run build:client`.
The before build used the exact baseline evaluator blob; all other source was
unchanged. Built directories were retained in this worktree and selected through
`dist-client` for `node scripts/serve_isolated_static.mjs dist-client 5191`.
The server and all native `agent_browser` operations ran inside the lifetime of
a benchlock-wrapped holder command. Unique browser session:
`centipawn-encode-sol-5191`. The browser and server were closed afterward.

Configuration: Centipawn **TVMJS strict**, **256 visits**, logical search batch
**4**, physical batch **16**, **MultiPV 1**, White starting position. For every
sample a full navigation created a fresh evaluator, first warmed on the Black
position after e4. The MultiPV change's scheduled warmup was allowed to finish
before loading the timed position, with auto-analysis off. This avoids measuring
initial compilation or reusing an evaluator cache for the same starting-position
search. The timer surrounds the real Analyze click until
`__analysisSchedulingState().analyzing` becomes false, polled every 5 ms.

Order: **before, after, after, before, before, after, after, before**.

| Variant | Median wall ms | Min–max ms | Samples ms |
| --- | ---: | ---: | --- |
| Before | 916.29 | 815.24–1123.39 | 920.54, 912.03, 815.24, 1123.39 |
| After | 864.33 | 779.89–1082.32 | 1082.32, 890.77, 837.90, 779.89 |

All eight produced **+0.71**, best move **e4**, and the same PV:
`e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Nxe4 d4 b5`. No page errors were reported.
The median is 5.7% lower, but spreads overlap substantially with only four
samples per variant: **no conclusive end-to-end speedup claim**. The isolated
CPU preprocessing improvement is the reason to retain this change.

## Gates and remaining questions

- `npm run typecheck`: zero errors or warnings.
- Focused evaluator/input/release/kernel-asset tests: 16 passed.
- `npm test`: 774 passed, 3 skipped, zero failures (777 total).
- `git diff --check`: clean.
- Both production builds succeeded.

Remaining question: how much of the preprocessing saving survives GPU-dominated
Play/Analysis workloads across devices and positions? The browser A/B here is a
small wall-time smoke, not a broad NPS or search-strength result.
