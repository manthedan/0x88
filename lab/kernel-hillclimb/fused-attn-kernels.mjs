// Fused 64-token attention for the Lc0 big nets (docs/bignet_fused_attention_2026-11.md).
//
// One dispatch per encoder replaces ORT's Q/K/V Transposes, QK MatMul, scale Mul,
// smolgen-bias Add, Softmax, P.V MatMul and output Transpose:
//
//   O[b*64+i, h*d + c] = sum_j softmax_j(scale * Q_i.K_j + bias[b,h,i,j]) * V[b*64+j, h*d + c]
//
// Q, K, V and O stay in the Gemm layout [B*64, H*d] (no transposes); bias is the
// smolgen output [B, H, 64, 64]. All storage is f16 (vec4). One workgroup per
// (batch, head); K and V for that head are staged in workgroup memory (2 x 4 KB
// at d = 32).
//
// Layouts ('rows' parameter):
//   rows=1  64 threads, one query row per thread (ORT's FlashAttention uses the same
//           mapping with 128 threads on Apple, half of them idle at 64 tokens).
//   rows=2  128 threads, two threads per query row, each owning 32 of the 64 keys;
//           the pair merges (max, sum, o) through workgroup memory.
//
// acc='f32': scores, softmax and P.V accumulate in f32 (scale applied after the dot,
// as the ONNX graph does). acc='f16': dots and P.V accumulate in f16 with Q
// pre-scaled (ORT FlashAttention's order); exp is f32 in both.
export function fusedAttentionKernel({ heads, depth = 32, tokens = 64, acc = 'f32', rows = 1 }) {
  if (depth % 4) throw new Error('depth must be a multiple of 4');
  if (tokens !== 64) throw new Error('kernel is specialised for 64 tokens');
  if (rows !== 1 && rows !== 2) throw new Error('rows must be 1 or 2');
  const A = acc === 'f16' ? 'f16' : 'f32';
  const dv4 = depth / 4;
  const d4 = (heads * depth) / 4;
  const keysPerThread = tokens / rows;
  const wg = tokens * rows;
  const scaleQ = acc === 'f16';
  return /* wgsl */ `
enable f16;
const H : u32 = ${heads}u;
const T : u32 = ${tokens}u;
const DV4 : u32 = ${dv4}u;      // vec4s per head row
const D4 : u32 = ${d4}u;        // vec4s per token row (all heads)
const KPT : u32 = ${keysPerThread}u;
const WG : u32 = ${wg}u;
struct Params { scale : f32, pad0 : f32, pad1 : f32, pad2 : f32 };
@group(0) @binding(0) var<storage, read> q : array<vec4<f16>>;
@group(0) @binding(1) var<storage, read> k : array<vec4<f16>>;
@group(0) @binding(2) var<storage, read> v : array<vec4<f16>>;
@group(0) @binding(3) var<storage, read> bias : array<vec4<f16>>;
@group(0) @binding(4) var<storage, read_write> o : array<vec4<f16>>;
@group(0) @binding(5) var<uniform> params : Params;
var<workgroup> ks : array<vec4<f16>, ${tokens * dv4}>;
var<workgroup> vs : array<vec4<f16>, ${tokens * dv4}>;
${
  rows === 2
    ? `var<workgroup> red_m : array<f32, ${wg}>;
var<workgroup> red_s : array<f32, ${wg}>;
var<workgroup> red_o : array<vec4<${A}>, ${wg * dv4}>;`
    : ''
}

@compute @workgroup_size(${wg})
fn main(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_index) lid : u32) {
  let bh = wid.x;
  let b = bh / H;
  let h = bh % H;
  // Stage this head's K and V rows (T x DV4 vec4 each).
  for (var e = lid; e < T * DV4; e += WG) {
    let g = (b * T + (e / DV4)) * D4 + h * DV4 + (e % DV4);
    ks[e] = k[g];
    vs[e] = v[g];
  }
  let row = lid % T;          // query token
  let part = lid / T;         // which key half (rows=2), else 0
  let k0 = part * KPT;
  var qr : array<vec4<${A}>, ${dv4}>;
  let qbase = (b * T + row) * D4 + h * DV4;
  ${scaleQ ? `let qs = f16(params.scale);` : ''}
  for (var c = 0u; c < DV4; c++) {
    qr[c] = vec4<${A}>(q[qbase + c])${scaleQ ? ' * qs' : ''};
  }
  workgroupBarrier();

  var s : array<f32, ${keysPerThread}>;
  var mx = -3.0e38;
  let bbase = ((b * H + h) * T + row) * (T / 4u) + k0 / 4u;
  for (var j4 = 0u; j4 < KPT / 4u; j4++) {
    let bv = vec4<f32>(bias[bbase + j4]);
    for (var jj = 0u; jj < 4u; jj++) {
      let j = j4 * 4u + jj;
      var dacc = ${A}(0);
      for (var c = 0u; c < DV4; c++) {
        dacc += dot(qr[c], vec4<${A}>(ks[(k0 + j) * DV4 + c]));
      }
      let sc = ${scaleQ ? 'f32(dacc)' : 'dacc * params.scale'} + bv[jj];
      s[j] = sc;
      mx = max(mx, sc);
    }
  }
${
  rows === 2
    ? `  red_m[lid] = mx;
  workgroupBarrier();
  mx = max(red_m[row], red_m[row + T]);`
    : ''
}
  var sum = 0.0;
  var acc_o : array<vec4<${A}>, ${dv4}>;
  for (var j = 0u; j < KPT; j++) {
    let p = exp(s[j] - mx);
    sum += p;
    let pa = ${A}(p);
    for (var c = 0u; c < DV4; c++) {
      acc_o[c] += vec4<${A}>(vs[(k0 + j) * DV4 + c]) * pa;
    }
  }
${
  rows === 2
    ? `  red_s[lid] = sum;
  for (var c = 0u; c < DV4; c++) { red_o[lid * DV4 + c] = acc_o[c]; }
  workgroupBarrier();
  if (part != 0u) { return; }
  sum = red_s[row] + red_s[row + T];
  for (var c = 0u; c < DV4; c++) { acc_o[c] = red_o[row * DV4 + c] + red_o[(row + T) * DV4 + c]; }`
    : ''
}
  let inv = 1.0 / sum;
  for (var c = 0u; c < DV4; c++) {
    o[qbase + c] = vec4<f16>(vec4<f32>(acc_o[c]) * inv);
  }
}
`;
}

/** Bytes a fused dispatch must move (Q, K, V, bias read once; O written once). */
export function fusedAttentionBytes({ batch, heads, depth = 32, tokens = 64 }) {
  const qkvo = 4 * batch * tokens * heads * depth * 2;
  const bias = batch * heads * tokens * tokens * 2;
  return qkvo + bias;
}

/** FLOPs of QK^T and P.V (2 per MAC). */
export function fusedAttentionFlops({ batch, heads, depth = 32, tokens = 64 }) {
  return 2 * 2 * batch * heads * tokens * tokens * depth;
}

/**
 * Parity gate for the f32-accumulate variants against the ORT WASM run of the
 * exported subgraph. Both sides store f16, so the unit is the f16 ULP of the
 * reference value: the largest measured f32 differences (BT4/t3 L0/L7/L14, M4 and
 * RTX 3090) are 1 ULP (2.4e-4 @ |0.26|, 9.8e-4 @ |1.34|, 1.95e-3 @ |2.45|).
 * Per element: |cand - wasm| <= ATOL + RTOL * |wasm| + |wasm - f64|.
 *   - RTOL = 2^-9 is >= 2 f16 ULPs at any magnitude (ULP <= 2^-10 * |x|).
 *   - ATOL = 2^-12 (1 ULP at |0.25..0.5|) covers near-zero outputs.
 *   - |wasm - f64| is the reference's own measured error against the float64 numpy
 *     result on the same f16 inputs. WASM's f16 path is not exact either: at BT4
 *     L14 one output is 5.26e-4 where f64 and the kernel both give 9.37e-4
 *     (cancellation), which a WASM-only tolerance would blame on the kernel.
 * Measured on the M4 (BT4 and t3, L0/L7/L14): every element sits at <= 0.33 of its
 * allowance (3x headroom) except that BT4 L14 element, at 0.63 (the kernel is 7e-7
 * from f64 there). A zeroed output fails on every element with |ref| > ~2.5e-4, a 1%
 * rescale on every element with |ref| > ~0.03.
 * f16-accumulate variants exceed this by design (up to 2.9 at BT4 L14); they are
 * diagnostic only and never gate.
 */
export const FUSED_ATTN_PARITY_TOL = Object.freeze({ atol: 2 ** -12, rtol: 2 ** -9 });

/**
 * max |a-b|, mean |a-b|, the worst element's reference value, non-finite count, and
 * the tolerance check: violations (elements over the allowance, non-finite counted
 * too) and worstTolRatio (max |a-b| / allowance, < 1 passes). The allowance is
 * atol + rtol*|ref|, plus |ref - truth| per element when a float64 `truth` is given.
 */
export function fusedAttnDiffStats(cand, ref, truth = null, tol = FUSED_ATTN_PARITY_TOL) {
  if (cand.length !== ref.length) throw new Error(`length mismatch ${cand.length} vs ${ref.length}`);
  if (truth && truth.length !== ref.length) throw new Error(`truth length mismatch ${truth.length} vs ${ref.length}`);
  let max = 0;
  let sum = 0;
  let at = 0;
  let nonFinite = 0;
  let violations = 0;
  let worstTolRatio = 0;
  let worst = 0;
  for (let i = 0; i < ref.length; i++) {
    if (!Number.isFinite(cand[i])) {
      nonFinite++;
      continue;
    }
    const d = Math.abs(cand[i] - ref[i]);
    sum += d;
    if (d > max) {
      max = d;
      at = i;
    }
    const allow = tol.atol + tol.rtol * Math.abs(ref[i]) + (truth ? Math.abs(ref[i] - truth[i]) : 0);
    const ratio = d / allow;
    if (ratio > 1) violations++;
    if (ratio > worstTolRatio) {
      worstTolRatio = ratio;
      worst = i;
    }
  }
  return {
    maxAbs: max,
    meanAbs: sum / ref.length,
    refAtMax: ref[at],
    nonFinite,
    violations: violations + nonFinite,
    worstTolRatio,
    worstIndex: worst,
    worstRef: ref[worst],
    worstCand: cand[worst],
  };
}

/** Gate a kernel variant's per-layer parity rows ({ layer, vsWasm }); f16-accumulate variants never gate. */
export function fusedAttnGate(name, acc, parity, tol = FUSED_ATTN_PARITY_TOL) {
  if (acc !== 'f32') return { gated: false, pass: null, problems: [] };
  const problems = [];
  if (!parity.length) problems.push(`${name}: no parity rows`);
  for (const p of parity) {
    const s = p.vsWasm;
    if (s.nonFinite) problems.push(`${name} L${p.layer}: ${s.nonFinite} non-finite outputs`);
    if (s.violations > s.nonFinite)
      problems.push(
        `${name} L${p.layer}: ${s.violations - s.nonFinite} elements over ${tol.atol} + ${tol.rtol}*|wasm| + |wasm - f64| (worst ratio ${s.worstTolRatio.toFixed(2)}: ${s.worstCand} vs ${s.worstRef}; max-abs ${s.maxAbs} @ ${s.refAtMax})`,
      );
  }
  return { gated: true, pass: problems.length === 0, problems };
}
