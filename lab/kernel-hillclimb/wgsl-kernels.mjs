// WGSL generators for the peak/ceiling microbenchmarks. These are measurement
// kernels, not product kernels: they establish what a straightforward
// hand-written shader reaches on this device at our real shapes.

/** Independent-accumulator FMA chain; 8 vec4 accumulators per thread. */
export function fmaKernel(type, iters) {
  const t = type === 'f16' ? 'f16' : 'f32';
  const enable = t === 'f16' ? 'enable f16;\n' : '';
  const acc = Array.from({ length: 8 }, (_, i) => i);
  return `${enable}
@group(0) @binding(0) var<storage, read_write> outBuf: array<vec4<f32>>;
@group(0) @binding(1) var<uniform> P: vec4<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let seed = ${t}(f32(gid.x & 255u) * 0.001);
  let b = vec4<${t}>(${t}(P.x));
  let c = vec4<${t}>(${t}(P.y));
${acc.map((i) => `  var a${i} = vec4<${t}>(seed + ${t}(${i}.0 * 0.01));`).join('\n')}
  for (var i = 0u; i < ${iters}u; i = i + 1u) {
${acc.map((i) => `    a${i} = fma(a${i}, b, c);`).join('\n')}
${acc.map((i) => `    a${i} = fma(a${i}, b, c);`).join('\n')}
  }
  let s = ${acc.map((i) => `a${i}`).join(' + ')};
  if (P.z > 1.0e30) { outBuf[gid.x] = vec4<f32>(s); }
  if (gid.x == 0u) { outBuf[0] = vec4<f32>(s); }
}`;
}
/** FLOPs per thread of fmaKernel: iters * 2 (unroll) * 8 accumulators * 4 lanes * 2 flops. */
export const fmaFlopsPerThread = (iters) => iters * 2 * 8 * 4 * 2;

/** Grid-stride read: every thread sums vec4<f32>; tiny write. */
export function readKernel() {
  return `
@group(0) @binding(0) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> P: vec4<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let stride = nwg.x * 256u;
  var acc = vec4<f32>(0.0);
  for (var i = gid.x; i < P.x; i = i + stride) { acc = acc + src[i]; }
  if (acc.x == 1234.5678) { dst[gid.x] = acc; }
}`;
}

export function copyKernel() {
  return `
@group(0) @binding(0) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> P: vec4<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let stride = nwg.x * 256u;
  for (var i = gid.x; i < P.x; i = i + stride) { dst[i] = src[i]; }
}`;
}

export function tinyKernel() {
  return `
@group(0) @binding(0) var<storage, read_write> dst: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) li: u32) { if (li == 0u) { dst[0] = dst[0] + 1u; } }`;
}

/**
 * Register-blocked shared-memory GEMM, C[M,N] = A[M,K] * B[K,N], all
 * row-major. 16x16 threads; each thread owns (4*tm4) x (4*tn4) outputs, so
 * BM = 64*tm4, BN = 64*tn4, BK = 16. Requires M%BM == N%BN == K%16 == 0.
 * storage: 'f32' | 'f16'; accum: 'f32' | 'f16'.
 */
export function gemmKernel({ M, N, K, storage, accum, tm4, tn4 }) {
  const T = storage;
  const AT = accum;
  const BM = 64 * tm4;
  const BN = 64 * tn4;
  const BK = 16;
  if (M % BM || N % BN || K % BK) throw new Error(`gemm shape ${M}x${N}x${K} not divisible by ${BM}x${BN}x${BK}`);
  const enable = T === 'f16' || AT === 'f16' ? 'enable f16;\n' : '';
  const accs = [];
  for (let i = 0; i < tm4; i++) for (let r = 0; r < 4; r++) for (let j = 0; j < tn4; j++) accs.push({ i, r, j, name: `c${i}_${r}_${j}` });
  const lines = [];
  lines.push(`${enable}
@group(0) @binding(0) var<storage, read> A: array<vec4<${T}>>;
@group(0) @binding(1) var<storage, read> B: array<vec4<${T}>>;
@group(0) @binding(2) var<storage, read_write> C: array<vec4<${T}>>;
const M = ${M}u; const N = ${N}u; const K = ${K}u;
const BM4 = ${BM / 4}u; const BN4 = ${BN / 4}u;
var<workgroup> As: array<vec4<${AT}>, ${(BK * BM) / 4}>;
var<workgroup> Bs: array<vec4<${AT}>, ${(BK * BN) / 4}>;
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let row0 = wg.y * ${BM}u;
  let col0 = wg.x * ${BN}u;
  let tx = lid.x; let ty = lid.y;`);
  for (const a of accs) lines.push(`  var ${a.name} = vec4<${AT}>();`);
  lines.push(`  for (var k0 = 0u; k0 < K; k0 = k0 + ${BK}u) {`);
  for (let l = 0; l < tm4; l++) {
    lines.push(`    {
      let idx = li + ${l * 256}u; let m = idx / 4u; let kq = idx % 4u;
      let a = vec4<${AT}>(A[((row0 + m) * K + k0) / 4u + kq]);
      As[(kq * 4u + 0u) * BM4 + m / 4u][m % 4u] = a.x;
      As[(kq * 4u + 1u) * BM4 + m / 4u][m % 4u] = a.y;
      As[(kq * 4u + 2u) * BM4 + m / 4u][m % 4u] = a.z;
      As[(kq * 4u + 3u) * BM4 + m / 4u][m % 4u] = a.w;
    }`);
  }
  for (let l = 0; l < tn4; l++) {
    lines.push(`    {
      let idx = li + ${l * 256}u; let k = idx / BN4; let nq = idx % BN4;
      Bs[k * BN4 + nq] = vec4<${AT}>(B[((k0 + k) * N + col0) / 4u + nq]);
    }`);
  }
  lines.push(`    workgroupBarrier();
    for (var k = 0u; k < ${BK}u; k = k + 1u) {`);
  for (let i = 0; i < tm4; i++) lines.push(`      let a${i} = As[k * BM4 + ${i * 16}u + ty];`);
  for (let j = 0; j < tn4; j++) lines.push(`      let b${j} = Bs[k * BN4 + ${j * 16}u + tx];`);
  const comp = ['x', 'y', 'z', 'w'];
  for (const a of accs) lines.push(`      ${a.name} = fma(vec4<${AT}>(a${a.i}.${comp[a.r]}), b${a.j}, ${a.name});`);
  lines.push(`    }
    workgroupBarrier();
  }`);
  for (const a of accs) {
    lines.push(`  C[((row0 + ${a.i * 64}u + ty * 4u + ${a.r}u) * N + col0) / 4u + ${a.j * 16}u + tx] = vec4<${T}>(${a.name});`);
  }
  lines.push('}');
  return { code: lines.join('\n'), dispatch: [N / BN, M / BM, 1], bm: BM, bn: BN };
}

/**
 * Subgroup-matrix GEMM (chromium-experimental-subgroup-matrix). One subgroup
 * per workgroup (matrix load/store offsets must be uniform), owning a
 * (8*rt) x (8*ct) tile of C and streaming A/B straight from storage.
 * Requires a fixed subgroup size sg.
 */
export function subgroupMatrixGemmKernel({ M, N, K, type, resultType, mm, nn, kk, sg, rt = 2, ct = 2 }) {
  const tileM = mm * rt;
  const tileN = nn * ct;
  if (M % tileM || N % tileN || K % kk) throw new Error(`sgmatrix shape ${M}x${N}x${K} not divisible by ${tileM}x${tileN}x${kk}`);
  const enable = ['enable chromium_experimental_subgroup_matrix;', type === 'f16' || resultType === 'f16' ? 'enable f16;' : '', 'enable subgroups;']
    .filter(Boolean)
    .join('\n');
  const L = `subgroup_matrix_left<${type}, ${kk}, ${mm}>`;
  const R = `subgroup_matrix_right<${type}, ${nn}, ${kk}>`;
  const S = `subgroup_matrix_result<${resultType}, ${nn}, ${mm}>`;
  const lines = [];
  lines.push(`${enable}
@group(0) @binding(0) var<storage, read> A: array<${type}>;
@group(0) @binding(1) var<storage, read> B: array<${type}>;
@group(0) @binding(2) var<storage, read_write> C: array<${resultType}>;
const M = ${M}u; const N = ${N}u; const K = ${K}u;
@compute @workgroup_size(${sg})
fn main(@builtin(workgroup_id) wg: vec3<u32>) {
  let r0 = wg.y * ${tileM}u;
  let c0 = wg.x * ${tileN}u;`);
  for (let i = 0; i < rt; i++) for (let j = 0; j < ct; j++) lines.push(`  var acc${i}_${j} = ${S}();`);
  lines.push(`  for (var k = 0u; k < K; k = k + ${kk}u) {`);
  for (let i = 0; i < rt; i++) lines.push(`    let l${i} = subgroupMatrixLoad<${L}>(&A, (r0 + ${i * mm}u) * K + k, false, K);`);
  for (let j = 0; j < ct; j++) lines.push(`    let q${j} = subgroupMatrixLoad<${R}>(&B, k * N + c0 + ${j * nn}u, false, N);`);
  for (let i = 0; i < rt; i++) for (let j = 0; j < ct; j++) lines.push(`    acc${i}_${j} = subgroupMatrixMultiplyAccumulate(l${i}, q${j}, acc${i}_${j});`);
  lines.push('  }');
  for (let i = 0; i < rt; i++)
    for (let j = 0; j < ct; j++) lines.push(`  subgroupMatrixStore(&C, (r0 + ${i * mm}u) * N + c0 + ${j * nn}u, acc${i}_${j}, false, N);`);
  lines.push('}');
  return { code: lines.join('\n'), dispatch: [N / tileN, M / tileM, 1] };
}
