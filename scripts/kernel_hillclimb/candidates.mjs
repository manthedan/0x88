// Named bench candidates for the kernel hill-climb. A candidate is a runtime /
// config / WGSL variant plus the correctness gate it must pass before its
// timing counts. Baselines are the shipped paths (docs/kernel_hillclimb_roofline_2026-11.md).
//
// Gate kinds:
//   in-page   the bench page itself compares outputs (query param parity=...);
//             the result's top-level `ok` is false when the gate fails. Works on
//             remote devices (static dist + COOP/COEP only).
//   command   an existing repo smoke run locally from the checkout (needs vite
//             dev / node_modules); recorded under result.gates by bench.mjs.
//   none      measurement page, nothing to gate (peak microbenchmarks).

/** ORT WebGPU paths: raw outputs vs ORT WASM on identical inputs (same model bytes). */
const ORT_WASM_PARITY = {
  kind: 'in-page',
  query: 'parity=wasm',
  description:
    'Raw output max-abs diff, ORT WebGPU vs ORT WASM over the whole 10-position fixture corpus (physical-batch chunks, separate from the timing feeds). npm test separately proves ORT WASM vs native lc0 fixtures, so this chains the shipped GPU path to the native reference.',
  // Per-output max-abs ceilings live with the presets (lab/kernel-hillclimb/ort-profile.mjs
  // parityMaxAbs): ~2x the shipped baseline on the M4 so remote devices get the same gate.
  thresholds: 'ort-profile.mjs PRESETS[*].parityMaxAbs',
};

/** Maia3 QDQ8 export before the einsum,promo graph rewrite (the shipped default until the rewrite). */
const MAIA3_PRE_GRAPH_MODEL = '/models/maia3/maia3_simplified.qdq8.onnx';

const SEARCH_TIE_GATE =
  'Fixed-visit search parity, A vs the shipped model B on ORT WebGPU (lab/kernel-hillclimb/search-ab.mjs): same best move, or a tie under the f16 protocol (equal root visits for both moves in either tree, or reference prior gap <= tieEpsilon).';

function bigNetSplitKCandidates(files) {
  const out = {};
  for (const [preset, file] of Object.entries(files)) {
    const model = `/models/lc0-splitk/${file}`;
    out[`ort-${preset}-splitk`] = { page: 'ort-profile', query: `preset=${preset}&model=${model}`, path: 'ort-webgpu', gate: ORT_WASM_PARITY };
    out[`ort-${preset}-splitk-ab`] = {
      page: 'ort-profile',
      query: `preset=${preset}&model=${model}&ab=${preset}&abRounds=20&profileInvokes=0&iters=10`,
      path: 'ort-webgpu',
      gate: ORT_WASM_PARITY,
    };
    out[`search-${preset}-splitk-ab`] = {
      page: 'search-ab',
      query: `preset=${preset}&model=${model}`,
      path: 'ort-webgpu',
      gate: { kind: 'in-page', query: 'parity=search&tieEpsilon=0.01', description: SEARCH_TIE_GATE },
    };
  }
  return out;
}

export const CANDIDATES = {
  peak: { page: 'peak', query: '', path: 'device', gate: { kind: 'none' } },
  'peak-gemm': { page: 'peak', query: 'tests=gemm,sgmatrix', path: 'device', gate: { kind: 'none' } },

  // Shipped ORT WebGPU baselines.
  'ort-lc0-small': { page: 'ort-profile', query: 'preset=lc0-small', path: 'ort-webgpu', gate: ORT_WASM_PARITY },
  'ort-lc0-small-b8': { page: 'ort-profile', query: 'preset=lc0-small-b8', path: 'ort-webgpu', gate: ORT_WASM_PARITY },
  'ort-t3': { page: 'ort-profile', query: 'preset=t3', path: 'ort-webgpu', gate: ORT_WASM_PARITY },
  'ort-bt4': { page: 'ort-profile', query: 'preset=bt4&iters=15', path: 'ort-webgpu', gate: ORT_WASM_PARITY },
  'ort-lqo': { page: 'ort-profile', query: 'preset=lqo', path: 'ort-webgpu', gate: ORT_WASM_PARITY },
  'ort-maia3': { page: 'ort-profile', query: 'preset=maia3', path: 'ort-webgpu', gate: ORT_WASM_PARITY },
  'ort-maia3-grid': { page: 'ort-profile', query: 'preset=maia3-grid', path: 'ort-webgpu', gate: ORT_WASM_PARITY },

  // Maia3 graph rewrite (scripts/maia3_graph_rewrite.py, docs/maia3_graph_rewrite_2026-11.md).
  // The presets now ship the rewritten model; these candidates keep judging it
  // against the pre-rewrite QDQ8 model on ORT WASM. `--query model=...` swaps
  // in another rewrite (copy it under <dist>/models/maia3/candidates/ first).
  'ort-maia3-graph': { page: 'ort-profile', query: `preset=maia3&referenceModel=${MAIA3_PRE_GRAPH_MODEL}`, path: 'ort-webgpu', gate: ORT_WASM_PARITY },
  'ort-maia3-grid-graph': {
    page: 'ort-profile',
    query: `preset=maia3-grid&referenceModel=${MAIA3_PRE_GRAPH_MODEL}`,
    path: 'ort-webgpu',
    gate: ORT_WASM_PARITY,
  },
  // Paired rewrite-vs-pre-rewrite (A = shipped rewrite, B = pre-rewrite QDQ8), ABAB in one page session.
  'ort-maia3-ab-graph': {
    page: 'ort-profile',
    query: `preset=maia3&referenceModel=${MAIA3_PRE_GRAPH_MODEL}&ab=maia3&abModel=${MAIA3_PRE_GRAPH_MODEL}&parity=wasm&profileInvokes=0&iters=10`,
    path: 'ort-webgpu',
    gate: ORT_WASM_PARITY,
  },
  'ort-maia3-grid-ab-graph': {
    page: 'ort-profile',
    query: `preset=maia3-grid&referenceModel=${MAIA3_PRE_GRAPH_MODEL}&ab=maia3-grid&abModel=${MAIA3_PRE_GRAPH_MODEL}&parity=wasm&profileInvokes=0&iters=10`,
    path: 'ort-webgpu',
    gate: ORT_WASM_PARITY,
  },

  // Shipped TVMJS baseline (Centipawn) and the LC0 TVMJS research lanes.
  // Paired b1-vs-b8 Lc0 Small comparison (ABAB interleaved in one page session).
  'ort-lc0-small-ab-b8': {
    page: 'ort-profile',
    query: 'preset=lc0-small&ab=lc0-small-b8&profileInvokes=0&iters=10',
    path: 'ort-webgpu',
    gate: ORT_WASM_PARITY,
  },
  // Split-K graph rewrite of Lc0 Small b1 (scripts/kernel_hillclimb/splitk_rewrite.py, default
  // options; docs/lc0_small_splitk_2026-11.md). The model is a local, gitignored candidate that
  // build_lab.mjs stages under /models/lc0-splitk/; parity runs the shipped model on ORT WASM.
  'ort-lc0-small-splitk': {
    page: 'ort-profile',
    query: 'preset=lc0-small&model=/models/lc0-splitk/t1.b1.qdq8.splitk-all.onnx',
    path: 'ort-webgpu',
    gate: ORT_WASM_PARITY,
  },
  // Paired split-K (A) vs shipped (B), ABAB in one page session.
  'ort-lc0-small-splitk-ab': {
    page: 'ort-profile',
    query: 'preset=lc0-small&model=/models/lc0-splitk/t1.b1.qdq8.splitk-all.onnx&ab=lc0-small&abRounds=30&profileInvokes=0&iters=10',
    path: 'ort-webgpu',
    gate: ORT_WASM_PARITY,
  },
  // Smolgen-compress split-K of the big nets (splitk_rewrite.py --include 'smolgen/dense1/w$'
  // --assume-m 8; docs/bignet_smolgen_splitk_2026-11.md). Gated against each preset's shipped model
  // on ORT WASM; the -ab candidates pair split-K (A) with shipped (B).
  ...bigNetSplitKCandidates({
    t3: 't3-512x15x16h-distill-swa-2767500.batch8.f16.splitk.onnx',
    bt4: 'BT4-1024x15x32h-swa-6147500-policytune-332.batch8.f16.qdq8.splitk.onnx',
    lqo: 'lqo_v2.f16.qdq8.splitk.onnx',
  }),
  // MultiHeadAttention graph rewrite of the big nets (scripts/kernel_hillclimb/mha_rewrite.py,
  // docs/bignet_fused_attention_2026-11.md): ORT's WebGPU FlashAttention replaces the per-layer
  // Transpose/MatMul/Mul/Add/Softmax chain. Local gitignored models staged by build_lab.mjs under
  // /models/lc0-mha/; parity runs the shipped model on ORT WASM. The -ab variants pair the rewrite
  // (A) with the shipped model (B), ABAB in one page session.
  'ort-bt4-mha': { page: 'ort-profile', query: 'preset=bt4&iters=15&model=/models/lc0-mha/bt4.b8.qdq8.mha.onnx', path: 'ort-webgpu', gate: ORT_WASM_PARITY },
  'ort-bt4-mha-ab': {
    page: 'ort-profile',
    query: 'preset=bt4&model=/models/lc0-mha/bt4.b8.qdq8.mha.onnx&ab=bt4&abRounds=12&abBlock=3&profileInvokes=3&iters=10',
    path: 'ort-webgpu',
    gate: ORT_WASM_PARITY,
  },
  'ort-t3-mha': { page: 'ort-profile', query: 'preset=t3&model=/models/lc0-mha/t3.b8.f16.mha.onnx', path: 'ort-webgpu', gate: ORT_WASM_PARITY },
  'ort-t3-mha-ab': {
    page: 'ort-profile',
    query: 'preset=t3&model=/models/lc0-mha/t3.b8.f16.mha.onnx&ab=t3&abRounds=20&abBlock=3&profileInvokes=3&iters=10',
    path: 'ort-webgpu',
    gate: ORT_WASM_PARITY,
  },
  // --compute f32 rewrites (Cast to f32 around each MHA, so FlashAttention accumulates in f32).
  'ort-bt4-mha32-ab': {
    page: 'ort-profile',
    query: 'preset=bt4&model=/models/lc0-mha/bt4.b8.qdq8.mha32.onnx&ab=bt4&abRounds=12&abBlock=3&profileInvokes=3&iters=10',
    path: 'ort-webgpu',
    gate: ORT_WASM_PARITY,
  },
  'ort-t3-mha32-ab': {
    page: 'ort-profile',
    query: 'preset=t3&model=/models/lc0-mha/t3.b8.f16.mha32.onnx&ab=t3&abRounds=20&abBlock=3&profileInvokes=3&iters=10',
    path: 'ort-webgpu',
    gate: ORT_WASM_PARITY,
  },
  'ort-lqo-mha-ab': {
    page: 'ort-profile',
    query: 'preset=lqo&model=/models/lc0-mha/lqo.qdq8.mha.onnx&ab=lqo&abRounds=16&abBlock=3&profileInvokes=3&iters=10',
    path: 'ort-webgpu',
    gate: ORT_WASM_PARITY,
  },
  'ort-lc0-small-mha-ab': {
    page: 'ort-profile',
    query: 'preset=lc0-small&model=/models/lc0-mha/t1.b1.qdq8.mha.onnx&ab=lc0-small&abRounds=30&profileInvokes=5&iters=10',
    path: 'ort-webgpu',
    gate: ORT_WASM_PARITY,
  },
  'tvmjs-centipawn': {
    page: 'tvmjs-profile',
    query: 'target=centipawn',
    path: 'tvmjs-webgpu',
    gate: {
      kind: 'in-page',
      query: 'parity=ort',
      description:
        'Legal-policy and WDL max-abs <= 0.001 vs the ORT Centipawn evaluator over the 10-position fixture corpus (the docs/centipawn_tvmjs_subgroups_2026-10.md whole-output gate, evaluator form). Promotion additionally needs the search-parity runs in that doc.',
    },
  },
  'tvmjs-centipawn-v3-subgroups': {
    page: 'tvmjs-profile',
    query: 'target=centipawn&manifest=/runtimes/centipawn-tvmjs-webgpu/bt4-soap-rem-c19000-final/f32/v3-subgroups/manifest.json',
    path: 'tvmjs-webgpu',
    gate: { kind: 'in-page', query: 'parity=ort', description: 'as tvmjs-centipawn' },
  },
  'tvmjs-lc0-t1-b8': {
    page: 'tvmjs-profile',
    query: 'target=lc0-t1&batch=8',
    path: 'tvmjs-webgpu',
    gate: {
      kind: 'in-page',
      query: 'parity=ort&tieEpsilon=0.01',
      description:
        'vs ORT f16 b1 over the 10-position fixture corpus: same legal-move set, finite priors/WDL, max |prior diff| <= 0.03 and max |WDL diff| <= 0.01 (lab/kernel-hillclimb/gates.mjs compareLc0), same best move unless the reference prior gap is <= tie-epsilon 0.01.',
      command: [
        'npm',
        'run',
        'lc0:tvmjs-webgpu-smoke',
        '--',
        '--batch',
        '8',
        '--ort-compare',
        'f16',
        '--ort-ep',
        'webgpu',
        '--search-visits',
        '16',
        '--search-fixtures',
        '4',
        '--tie-epsilon',
        '0.01',
        '--port',
        '5242',
      ],
    },
  },
  'tvmjs-lc0-t1-b16': {
    page: 'tvmjs-profile',
    query: 'target=lc0-t1&batch=16',
    path: 'tvmjs-webgpu',
    gate: { kind: 'in-page', query: 'parity=ort&tieEpsilon=0.01', description: 'as tvmjs-lc0-t1-b8' },
  },
  'tvmjs-lc0-bt4-b8': {
    page: 'tvmjs-profile',
    query: 'target=lc0-bt4&batch=8&iters=15',
    path: 'tvmjs-webgpu',
    gate: {
      kind: 'in-page',
      query: 'parity=ort&tieEpsilon=0.01',
      description:
        'As t1 with max |prior diff| <= 0.05; BT4 strict best-move parity fails on f16 near-ties by design (docs/lc0_tvmjs_bt4it332_onboarding_2026-06-09.md), so flips are tolerated only where the reference prior gap is <= 0.01.',
    },
  },
};

/** Custom WGSL heads (hybrid-wgsl-heads runtime, not in the v0 deploy profile). */
export const WGSL_HEADS_GATE = {
  kind: 'command',
  command: ['npm', 'run', 'lc0:browser-wgsl-smokes'],
  description: 'Existing custom-WGSL head/encoder smokes vs ORT (scripts/lc0_browser_wgsl_smokes.mjs). Applies only to the hybrid WGSL runtime.',
};
