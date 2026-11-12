# Maia3 ONNX graph rewrite for ORT WebGPU (November 2026)

The shipped Maia3 model (`maia3_simplified.qdq8.onnx`) has two graph-level
costs on ORT Web 1.27 WebGPU, both found in the kernel roofline
([`kernel_hillclimb_roofline_2026-11.md`](kernel_hillclimb_roofline_2026-11.md)).
First, eight `Einsum bhi,oi->bho` attention-bias projections run as a generic
Einsum kernel. Second, there are hundreds of tiny launches.
`scripts/maia3_graph_rewrite.py` rewrites the existing ONNX and keeps the
int8 QDQ weights, the model I/O and the dequant fold intact.

**Result:** `einsum,promo` is the candidate worth releasing. On the M4 it cuts
b1 move-eval wall time by about 24% and b11 grid wall time by about 32%
(paired). It halves dispatches (817 to 424). Its outputs are bit-identical on
ORT WASM and match the shipped model on the WebGPU gate. It is **not shipped**:
the shipped model, manifest and defaults are unchanged. The release steps are
in [§6](#6-release-steps).

All numbers: Apple M4, headless Chrome, production `dist-client` with the lab
overlay, every timing under `notes/kernel-hillclimb/benchlock.sh`. The machine
is shared (load average 5-11 during the runs), so timings are paired A/B
ratios, not absolute times.

## 1. What the graph actually spends launches on

The brief guessed that dynamic-batch shape arithmetic was the launch-bound
part. It is not. ORT runs `Shape`/`Gather`/`Concat` on shape tensors on the
CPU, so they cost no GPU dispatches. Most of the small launches come from the
**promotion head**:

- 256 promotion logits, each built by a scalar chain:
  `Gather(axis 1)` → `Gather` → `Add` → `Unsqueeze`.
- The chains index the square-pair logits (`/model/Div_2_output_0`, `[B,64,64]`)
  and a per-piece bias (`/model/Mul_6_output_0`, `[B,8,4]`).
- They feed one 257-input `Concat`, which ORT WebGPU splits into 28-29 Concat
  dispatches.
- Together that is about 400 of the 817 dispatches per b1 invoke.

## 2. Passes (`scripts/maia3_graph_rewrite.py`)

Passes always run in the order `einsum, promo, splitk, static`. The output
passes `onnx.checker.check_model(full_check=True)`, and a JSON report records
bytes, sha256 and op counts before and after. The rewrite is deterministic:
re-running it gives the same sha256 (onnx 1.21.0, numpy 2.4.6).

| pass | what it does | op delta (vs shipped) |
|---|---|---|
| `einsum` | `Einsum bhi,oi->bho` with a constant 2-D weight becomes `MatMul(X, Wᵀ)`. The shared `gab_shared_weight` [4096,128] f16 is transposed once into a new initializer. The remaining `bid,bjd->bij` Einsum is left as is. | Einsum 9→1, MatMul 60→68 |
| `promo` | Traces the 256 scalar chains back to their two bases and computes flat indices with `ravel_multi_index`. Replaces them with: `Reshape(P,[0,-1])`, two vector `Gather`s with constant index tensors, one `Add`, and `Concat(head, logits)` under the original output name. | Add 347→92, Unsqueeze 287→31, Gather 158→48; nodes 1419→799 |
| `splitk` (experimental) | `Gemm(transB=1)` with a constant weight, K ≥ 1024 and N ≤ 256 becomes a split-K `MatMul` over 16 chunks plus `ReduceSum`. Targets the smolgen compress Gemm (1×2048→128, 4 workgroups, about 246 µs ×8 at b1). `--splitk-f32` accumulates the partials in f32. | Gemm 18→10, ReduceSum +8 |
| `static --batch N` | Pins input/output dim 0 to N, then repeats shape inference and constant folding (Shape/Size of static shapes, all-constant nodes ≤ 64k elements, never Q/DQ) until nothing changes. | Shape 27→0, Concat 28→3, nodes 799→678 |

The 43 int8 `DequantizeLinear` nodes are untouched in every variant, so the
browser dequant fold (`src/nn/onnxDequantFold.ts`) still folds all 43.
`scripts/maia3_graph_parity.mjs` checks the fold count.

## 3. Gates

### In-page kernel-bench gate (ORT WebGPU candidate vs ORT WASM shipped model)

`--gate` on the `ort-maia3-graph*` candidates (`scripts/kernel_hillclimb/candidates.mjs`).
The reference is the shipped model on ORT WASM. The 10-position fixture corpus
runs under the 4 Elo conditions (40 rows).

- b1 ceilings: move 1.3, value 0.15.
- grid ceilings: move 0.9, value 0.2.
- Legal-probability ceiling: 0.05. Tie epsilon: 0.02.
- Raw logit max-abs; value logits are in model order L, D, W.

| model | b1 move / value / legal prob | b1 best | grid move / value / legal prob | grid best | gate |
|---|---|---|---|---|---|
| shipped | 0.492 / 0.078 / 0.0263 | 38/40 (2 tolerated ties) | 0.500 / 0.110 / 0.0236 | 38/40 | pass |
| einsum | 0.492 / 0.078 / 0.0263 | 38/40 | 0.500 / 0.110 / 0.0236 | 38/40 | pass |
| **einsum,promo** | **0.492 / 0.078 / 0.0263** | **38/40** | **0.500 / 0.110 / 0.0236** | **38/40** | **pass** |
| einsum,promo,static (b1 / b11) | 0.492 / 0.078 / 0.0263 | 38/40 | 0.500 / 0.110 / 0.0236 | 38/40 | pass |
| einsum,promo,splitk | 0.578 / 0.096 / 0.0313 | 38/40 | 0.656 / 0.146 / 0.0266 | 39/40 | **b1 FAIL** (row 28 top-3 order reversed) |
| einsum,promo,splitk-f32 | 0.563 / **0.162** / 0.0254 | 38/40 | 0.594 / 0.107 / 0.0280 | 38/40 | **b1 FAIL** (value > 0.15) |

The two tolerated ties are `h2h1q` vs `a2a1q`, which the shipped model also
"misses": reference gaps 0 and 0.0134. The einsum and promo rewrites leave the
WebGPU outputs identical to the shipped model's.

### Node parity (ORT WASM candidate vs ORT WASM shipped model)

`scripts/maia3_graph_parity.mjs` runs both models through the product load path
(dequant fold, then ORT `graphOptimizationLevel: 'all'`).

- Corpus: 57 positions (10 fixtures, 7 promotion-heavy FENs, 40 seeded playout
  positions).
  - "conditions": 228 rows, the 4 Elo conditions.
  - "grid" (b11): 627 rows, eloSelf 600..2600 vs 1500.
- Limits:
  - legal probability ≤ 0.01;
  - value probability ≤ 0.01 (softmax over L, D, W);
  - tie epsilon 0.02;
  - fold count equal.

| model | batch / set | logits_move (legal) | logits_value | legal prob | value prob | E[score] | best | top-3 |
|---|---|---|---|---|---|---|---|---|
| einsum | b1, b11 conditions, b11 grid | 0 (0) | 0 | 0 | 0 | 0 | all | all |
| **einsum,promo** | b1, b11 conditions, b11 grid | **0 (0)** | **0** | **0** | **0** | **0** | **228/228, 627/627** | all |
| static b1 | b1 conditions | 0.125 (0.039) | 0.039 | 0.0048 | 0.0038 | 0.0019 | 228/228 | 228/228 |
| static b11 | b11 grid | 0.125 (0.055) | 0.021 | 0.0063 | 0.0043 | 0.0021 | 624/627 | 619/627 |
| splitk | b11 grid | 0.133 (0.047) | 0.022 | 0.0050 | 0.0039 | 0.0019 | 625/627 | 620/627 |

All rows pass. For einsum and einsum,promo the outputs are bit-identical on
WASM. The static and splitk variants drift a little, because ORT's WASM fusions
change once shapes are static or the Gemm becomes MatMul + ReduceSum. All 3
static b11 best-move flips are within the tie epsilon.

Sanity check that the gate can fail: passing the upstream fp16 model as the
"candidate" fails it. Legal prob is 0.0172, value prob 0.0167, and the fold
count is 0 vs 43.

## 4. Paired timing (M4)

Paired ABAB in one page session: A = candidate, B = shipped model, 10 iters,
3 page loads. The table gives the median `wallRatioAoverB` with p10-p90 in
brackets, and the GPU-timestamp busy ratio. Below 1 means the candidate is
faster.

| model | b1 wall A/B (3 loads) | b1 GPU busy | b11 grid wall A/B (3 loads) | grid GPU busy |
|---|---|---|---|---|
| einsum | 0.983 [0.91-1.12], 0.948 [0.85-1.03], 0.956 [0.83-1.04] | 0.87-0.89 | 0.726 [0.69-0.79], 0.710 [0.60-0.82], 0.717 [0.69-0.77] | 0.68 |
| **einsum,promo** | **0.759 [0.71-0.81], 0.783 [0.61-0.86], 0.762 [0.65-0.83]** | **0.80-0.81** | **0.673 [0.60-0.80], 0.680 [0.59-0.75], 0.689 [0.57-0.80]** | **0.65-0.67** |
| einsum,promo,static (b1 / b11) | 0.748 [0.67-0.80], 0.758 [0.62-0.81], 0.752 [0.61-0.83] | 0.78-0.79 | 0.672 [0.60-0.73], 0.654 [0.57-0.74], 0.670 [0.61-0.79] | 0.64-0.66 |
| splitk (gated out) | 0.651, 0.696, 0.678 | 0.62-0.64 | 0.633, 0.643, 0.642 | 0.63-0.64 |
| splitk-f32 (gated out) | 0.675, 0.659, 0.654 | 0.63-0.65 | 0.677, 0.668, 0.685 | 0.65-0.68 |

Per position, einsum,promo runs b1 at about 8.6-8.9 ms vs 11.5-12.4 ms for the
shipped model. The b11 grid runs at about 4.2-4.5 ms vs 6.2-6.8 ms.

**Dispatches and submits per invoke** (b1 profile; ORT splits the same way at b11):

| model | dispatches | submits | b1 kernel ms (timestamp sum) |
|---|---|---|---|
| shipped | 817 | 59 | 8.39 |
| einsum | 817 | 59 | 7.13 |
| **einsum,promo** | **424** | **35** | **7.92** |
| einsum,promo,static | 395 | 29 | 9.36 |
| einsum,promo,splitk | 448 | 36 | 6.76 |
| einsum,promo,splitk-f32 | 464 | 37 | 5.97 |

The kernel-ms column comes from single gate runs on a loaded machine. Use the
paired ratios above for speed and this table for dispatch counts.

The Einsum fix mostly matters at batch: in the grid profile the attention-bias
Einsums take 26.0 ms of 78.7 ms in the shipped model, and 1.0 ms after the
rewrite (that 1.0 ms is the remaining `bid,bjd->bij`). At b1 the gain is small
because the kernel is launch-bound. The promo fix is what moves b1.

Yukon (RTX 3090, Vulkan) was not measured. Both submissions were refused
because another 0x88 job was already waiting in the shared queue, and the
retry window ran out before this record was written.

## 5. Verdicts

- **einsum: applied and kept.** Bit-identical, and the grid is about 28% faster.
- **promo: applied and kept** (not in the brief; it is the real launch-bound
  part). Bit-identical. Together with einsum it gives b1 about 0.76 and the
  grid about 0.68.
- **static shapes: applied, not recommended.** It folds the shape arithmetic
  (Shape 27→0, 29 fewer dispatches) and passes both gates. But the paired
  timing is within noise of the dynamic einsum,promo model (b1 0.75 vs 0.76,
  grid 0.67 vs 0.68). It would also need one model per batch size, and the
  product runs Maia3 at b1 (move, critical moments), b6 (Analysis panel
  ratings) and b11 (bench grid). Static b1 also lets ORT fuse the smolgen
  Gemm into a slower kernel (4.94 ms).
- **splitk: gated out.** It is the fastest variant (b1 about 0.67, grid about
  0.64), but it fails the b1 WebGPU gate: an f16 top-3 reversal, and with f32
  partials the value logits go over 0.15. Worth revisiting only with a
  reduction whose rounding matches the shipped Gemm, or with a gate decision
  to loosen the b1 value ceiling.

## 6. Release steps

1. Derive: `python -I scripts/maia3_graph_rewrite.py public/models/maia3/maia3_simplified.qdq8.onnx public/models/maia3/maia3_simplified.qdq8.graph.onnx --passes einsum,promo --report <report.json>`.
   The expected sha256 is `3be705330f38fb0309c82e0ea2f95734ee6bd14fd72eabbad694356bba443246`
   (27,971,924 bytes) with onnx 1.21.0.
2. Gate:
   - `node --experimental-strip-types scripts/maia3_graph_parity.mjs --candidate <new file> --batch 1 --batch 11`.
   - Then `--gate` on `ort-maia3-graph` and `ort-maia3-grid-graph` from a built preview.
   - Then the paired `ort-maia3-ab-graph` / `ort-maia3-grid-ab-graph` on at least one non-Apple GPU (Yukon).
3. Add a `public/models/maia3/manifest.json` entry (file, url, bytes, sha256,
   `derivedFrom: "maia3_simplified.qdq8.onnx"`, source naming the script).
   Extend `scripts/stage_maia3_assets.mjs` to derive it from the qdq8 file and
   check the sha256.
4. Publish the new file to R2 under the new name: add it to
   `scripts/r2_brotli_publish_assets.mjs` with immutable caching. Keep the old
   file live for cached clients.
5. Point `MAIA3_DEFAULT_MODEL_URL` (`src/lc0/maia3.ts`) at the new file.
   Update the `maia3` / `maia3-grid` presets in `lab/kernel-hillclimb/ort-profile.mjs`,
   `docs/model_provenance/maia3.md`, `public/models/maia3/README.md`,
   `docs/hosted_artifacts.md`, and the artifact index (`scripts/write_artifact_index.mjs`).
6. Run the Maia3 browser smoke and the Play/Analysis Maia paths on a **built**
   preview (dev masks worker bundling problems). Keep the model dynamic-batch:
   the product needs b1, b6 and b11 from one file.

## 7. Reproduce

```sh
# candidates (gitignored, <checkout>/artifacts/maia3-graph/)
.venv-onnx/bin/python -I scripts/maia3_graph_rewrite.py public/models/maia3/maia3_simplified.qdq8.onnx \
  artifacts/maia3-graph/maia3.einsum-promo.onnx --passes einsum,promo \
  --report artifacts/maia3-graph/maia3.einsum-promo.report.json
# add ,static --batch 1|11, or ,splitk [--splitk-f32] for the other variants

# node parity
node --experimental-strip-types scripts/maia3_graph_parity.mjs \
  --candidate artifacts/maia3-graph/maia3.einsum-promo.onnx --batch 1 --batch 11

# browser gates and paired timing (built dist-client + lab overlay)
mkdir -p dist-client/models/maia3/candidates && cp artifacts/maia3-graph/maia3.einsum-promo.onnx dist-client/models/maia3/candidates/
node scripts/serve_isolated_static.mjs dist-client 5277 &
B='--base-url=http://127.0.0.1:5277 --browser-args=--enable-unsafe-webgpu,--enable-webgpu-developer-features'
notes/kernel-hillclimb/benchlock.sh maia3-graph -- node scripts/kernel_hillclimb/bench.mjs --candidate ort-maia3-graph --gate $B
notes/kernel-hillclimb/benchlock.sh maia3-graph -- node scripts/kernel_hillclimb/bench.mjs --candidate ort-maia3-ab-graph --repeat 3 $B
# other rewrites: --query model=/models/maia3/candidates/<file>
```

The `benchlock.sh` path is relative to `<workspace>`, and the other commands
run from `<checkout>`. Results are written to
`artifacts/kernel-hillclimb/results/ort-maia3*-graph*.json`.
