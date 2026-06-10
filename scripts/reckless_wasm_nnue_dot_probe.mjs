#!/usr/bin/env node
// Standalone probe for the propagate_l1 sparse dot-product shape: per nonzero
// group, splat 4 packed u8 activations and dpbusd them against 64 i8 weights
// into 4 i32x4 accumulators (L2_SIZE = 16). Compares the previous lane-extract
// dpbusd emulation, the shuffle-based emulation, and the relaxed integer dot
// for exact parity and throughput. Activations stay in [0, 127] to match the
// proven activate_ft output range (255 * 255 >> 9 = 127), which is what makes
// the relaxed i7x16 dot exact rather than implementation-defined.
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { performance } from 'node:perf_hooks';
import { execFileSync } from 'node:child_process';

const rustSource = String.raw`#![allow(static_mut_refs, unused_imports, dead_code)]
use std::arch::wasm32::*;

#[unsafe(no_mangle)]
pub static mut SCRATCH: [u8; 65536] = [0; 65536];

#[unsafe(no_mangle)]
pub extern "C" fn scratch_ptr() -> *mut u8 {
    unsafe { SCRATCH.as_mut_ptr() }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn dot_scalar(acts: *const u8, weights: *const i8, out: *mut i32, groups: usize) {
    for j in 0..16 {
        *out.add(j) = 0;
    }
    for g in 0..groups {
        for j in 0..16usize {
            let mut sum = 0i32;
            for k in 0..4usize {
                sum += (*acts.add(g * 4 + k)) as i32 * (*weights.add(g * 64 + j * 4 + k)) as i32;
            }
            *out.add(j) += sum;
        }
    }
}

#[cfg(target_feature = "simd128")]
mod simd_kernels {
    use std::arch::wasm32::*;

    unsafe fn dpbusd_extract(i32s: v128, u8s: v128, i8s: v128) -> v128 {
        let prod_lo = i16x8_mul(u16x8_extend_low_u8x16(u8s), i16x8_extend_low_i8x16(i8s));
        let prod_hi = i16x8_mul(u16x8_extend_high_u8x16(u8s), i16x8_extend_high_i8x16(i8s));
        let pair_lo = i32x4_extadd_pairwise_i16x8(prod_lo);
        let pair_hi = i32x4_extadd_pairwise_i16x8(prod_hi);
        let sums = i32x4(
            i32x4_extract_lane::<0>(pair_lo) + i32x4_extract_lane::<1>(pair_lo),
            i32x4_extract_lane::<2>(pair_lo) + i32x4_extract_lane::<3>(pair_lo),
            i32x4_extract_lane::<0>(pair_hi) + i32x4_extract_lane::<1>(pair_hi),
            i32x4_extract_lane::<2>(pair_hi) + i32x4_extract_lane::<3>(pair_hi),
        );
        i32x4_add(i32s, sums)
    }

    unsafe fn dpbusd_shuffle(i32s: v128, u8s: v128, i8s: v128) -> v128 {
        let prod_lo = i16x8_mul(u16x8_extend_low_u8x16(u8s), i16x8_extend_low_i8x16(i8s));
        let prod_hi = i16x8_mul(u16x8_extend_high_u8x16(u8s), i16x8_extend_high_i8x16(i8s));
        let pair_lo = i32x4_extadd_pairwise_i16x8(prod_lo);
        let pair_hi = i32x4_extadd_pairwise_i16x8(prod_hi);
        let sums = i32x4_add(
            i32x4_shuffle::<0, 2, 4, 6>(pair_lo, pair_hi),
            i32x4_shuffle::<1, 3, 5, 7>(pair_lo, pair_hi),
        );
        i32x4_add(i32s, sums)
    }

    #[cfg(target_feature = "relaxed-simd")]
    unsafe fn dpbusd_relaxed(i32s: v128, u8s: v128, i8s: v128) -> v128 {
        i32x4_relaxed_dot_i8x16_i7x16_add(i8s, u8s, i32s)
    }

    #[cfg(not(target_feature = "relaxed-simd"))]
    unsafe fn dpbusd_relaxed(i32s: v128, u8s: v128, i8s: v128) -> v128 {
        dpbusd_shuffle(i32s, u8s, i8s)
    }

    macro_rules! dot_kernel {
        ($name:ident, $dpbusd:ident) => {
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn $name(acts: *const u8, weights: *const i8, out: *mut i32, groups: usize) {
                let mut acc = [i32x4_splat(0); 4];
                for g in 0..groups {
                    let input = i32x4_splat((acts.add(g * 4) as *const i32).read_unaligned());
                    for v in 0..4 {
                        let w = v128_load(weights.add(g * 64 + v * 16) as *const v128);
                        acc[v] = $dpbusd(acc[v], input, w);
                    }
                }
                for v in 0..4 {
                    v128_store(out.add(v * 4) as *mut v128, acc[v]);
                }
            }
        };
    }

    dot_kernel!(dot_extract, dpbusd_extract);
    dot_kernel!(dot_shuffle, dpbusd_shuffle);
    dot_kernel!(dot_relaxed, dpbusd_relaxed);
}

#[cfg(not(target_feature = "simd128"))]
mod simd_kernels {
    macro_rules! dot_fallback {
        ($name:ident) => {
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn $name(acts: *const u8, weights: *const i8, out: *mut i32, groups: usize) {
                super::dot_scalar(acts, weights, out, groups);
            }
        };
    }

    dot_fallback!(dot_extract);
    dot_fallback!(dot_shuffle);
    dot_fallback!(dot_relaxed);
}
`;

const groups = Number(process.env.RECKLESS_NNUE_DOT_GROUPS ?? 96);
const iterations = Number(process.env.RECKLESS_NNUE_DOT_ITERS ?? 200_000);
if (!Number.isInteger(groups) || groups <= 0 || groups > 192) throw new Error('RECKLESS_NNUE_DOT_GROUPS must be 1..192');
if (!Number.isInteger(iterations) || iterations <= 0) throw new Error('RECKLESS_NNUE_DOT_ITERS must be a positive integer');

const ACT_BYTES = groups * 4;
const WEIGHT_BYTES = groups * 64;
const OUT_OFFSET_BYTES = 49152;

function compileRust(source, out, features) {
  const args = ['--target', 'wasm32-wasip1', '--crate-type', 'cdylib', '-O'];
  if (features) args.push('-C', `target-feature=${features}`);
  args.push(source, '-o', out);
  execFileSync('rustc', args, { stdio: 'inherit' });
}

async function instantiate(file) {
  const { instance } = await WebAssembly.instantiate(readFileSync(file), {});
  for (const name of ['memory', 'scratch_ptr', 'dot_scalar', 'dot_extract', 'dot_shuffle', 'dot_relaxed']) {
    if (!(name in instance.exports)) throw new Error(`${file} did not export ${name}`);
  }
  return instance.exports;
}

function fillInputs(exports, seed) {
  const base = exports.scratch_ptr();
  const bytes = new Uint8Array(exports.memory.buffer);
  let x = seed >>> 0;
  const next = () => {
    x = (Math.imul(x, 1664525) + 1013904223) >>> 0;
    return x >>> 24;
  };
  // Activations mimic ft_out: u8 in [0, 127] with some zero groups sprinkled in.
  for (let i = 0; i < ACT_BYTES; i += 1) bytes[base + i] = next() & 0x7f;
  for (let g = 0; g < groups; g += 7) bytes.fill(0, base + g * 4, base + g * 4 + 4);
  // Weights are full-range signed i8.
  for (let i = 0; i < WEIGHT_BYTES; i += 1) bytes[base + 16384 + i] = next();
  return {
    acts: bytes.slice(base, base + ACT_BYTES),
    weights: bytes.slice(base + 16384, base + 16384 + WEIGHT_BYTES),
  };
}

function installInputs(exports, vectors) {
  const base = exports.scratch_ptr();
  const bytes = new Uint8Array(exports.memory.buffer);
  bytes.set(vectors.acts, base);
  bytes.set(vectors.weights, base + 16384);
  return {
    actsPtr: base,
    weightsPtr: base + 16384,
    outPtr: base + OUT_OFFSET_BYTES,
    view: new Int32Array(exports.memory.buffer),
  };
}

function runKernel(exports, exportName, layout) {
  exports[exportName](layout.actsPtr, layout.weightsPtr, layout.outPtr, groups);
  return Array.from(layout.view.slice(layout.outPtr >> 2, (layout.outPtr >> 2) + 16));
}

function benchmark(exports, exportName, layout) {
  const fn = exports[exportName];
  const start = performance.now();
  for (let i = 0; i < iterations; i += 1) fn(layout.actsPtr, layout.weightsPtr, layout.outPtr, groups);
  const elapsedMs = performance.now() - start;
  return {
    elapsedMs: Number(elapsedMs.toFixed(3)),
    macsPerSecond: Math.round((groups * 64 * iterations) / Math.max(1e-9, elapsedMs / 1000)),
  };
}

const workdir = mkdtempSync(join(tmpdir(), 'reckless-nnue-dot-probe-'));
try {
  const source = join(workdir, 'dot_probe.rs');
  const scalarWasm = join(workdir, 'dot_scalar.wasm');
  const simdWasm = join(workdir, 'dot_simd.wasm');
  const relaxedWasm = join(workdir, 'dot_relaxed.wasm');
  writeFileSync(source, rustSource);
  compileRust(source, scalarWasm, null);
  compileRust(source, simdWasm, '+simd128');
  compileRust(source, relaxedWasm, '+simd128,+relaxed-simd');

  const scalar = await instantiate(scalarWasm);
  const simd = await instantiate(simdWasm);
  let relaxed = null;
  let relaxedUnsupportedReason = null;
  try {
    relaxed = await instantiate(relaxedWasm);
  } catch (error) {
    relaxedUnsupportedReason = String(error?.message ?? error);
  }

  const vectors = fillInputs(scalar, 0xdecafbad);
  const reference = runKernel(scalar, 'dot_scalar', installInputs(scalar, vectors));

  const runs = [
    { name: 'scalar', exports: scalar, exportName: 'dot_scalar' },
    { name: 'extract', exports: simd, exportName: 'dot_extract' },
    { name: 'shuffle', exports: simd, exportName: 'dot_shuffle' },
    ...(relaxed ? [{ name: 'relaxed', exports: relaxed, exportName: 'dot_relaxed' }] : []),
  ];

  const report = {
    target: 'wasm32-wasip1',
    kernel: 'propagate_l1 sparse u8xi8 group dot (4x i32x4 accumulators)',
    groups,
    iterations,
    reference,
    ...(relaxedUnsupportedReason ? { relaxedUnsupportedReason } : {}),
    kernels: {},
  };

  let mismatchedKernels = 0;
  for (const run of runs) {
    const layout = installInputs(run.exports, vectors);
    const output = runKernel(run.exports, run.exportName, layout);
    const mismatches = output.reduce((sum, value, i) => sum + (value !== reference[i] ? 1 : 0), 0);
    if (mismatches !== 0) mismatchedKernels += 1;
    const bench = benchmark(run.exports, run.exportName, layout);
    report.kernels[run.name] = { mismatches, ...bench };
  }
  for (const [name, data] of Object.entries(report.kernels)) {
    if (name === 'scalar') continue;
    data.speedupVsScalar = Number((report.kernels.scalar.elapsedMs / Math.max(1e-9, data.elapsedMs)).toFixed(3));
    data.speedupVsExtract = Number((report.kernels.extract.elapsedMs / Math.max(1e-9, data.elapsedMs)).toFixed(3));
  }

  console.log(JSON.stringify(report, null, 2));
  if (mismatchedKernels !== 0) process.exitCode = 1;
} finally {
  rmSync(workdir, { recursive: true, force: true });
}
