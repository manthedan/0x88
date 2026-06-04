#!/usr/bin/env node
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { performance } from 'node:perf_hooks';
import { execFileSync } from 'node:child_process';

const rustSource = String.raw`#![allow(static_mut_refs, unused_imports)]
use std::arch::wasm32::*;

const FT_QUANT: i16 = 255;
const FT_SHIFT: u32 = 9;

#[unsafe(no_mangle)]
pub static mut SCRATCH_I16: [i16; 4096] = [0; 4096];
#[unsafe(no_mangle)]
pub static mut SCRATCH_U8: [u8; 2048] = [0; 2048];

#[unsafe(no_mangle)]
pub extern "C" fn scratch_i16_ptr() -> *mut i16 { unsafe { SCRATCH_I16.as_mut_ptr() } }
#[unsafe(no_mangle)]
pub extern "C" fn scratch_u8_ptr() -> *mut u8 { unsafe { SCRATCH_U8.as_mut_ptr() } }

#[unsafe(no_mangle)]
pub unsafe extern "C" fn activate_ft_scalar(pst_left: *const i16, threat_left: *const i16, pst_right: *const i16, threat_right: *const i16, out: *mut u8, len: usize) {
    for i in 0..len {
        let left = (*pst_left.add(i) + *threat_left.add(i)).clamp(0, FT_QUANT);
        let right = (*pst_right.add(i) + *threat_right.add(i)).clamp(0, FT_QUANT);
        *out.add(i) = (((left as i32 * right as i32) >> FT_SHIFT) & 0xff) as u8;
    }
}

#[cfg(target_feature = "simd128")]
#[target_feature(enable = "simd128")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn activate_ft_simd(pst_left: *const i16, threat_left: *const i16, pst_right: *const i16, threat_right: *const i16, out: *mut u8, len: usize) {
    let zero = i16x8_splat(0);
    let max = i16x8_splat(FT_QUANT);
    let chunks = len / 16;
    for chunk in 0..chunks {
        let offset = chunk * 16;
        let left0 = clamp_i16x8(i16x8_add(v128_load(pst_left.add(offset) as *const v128), v128_load(threat_left.add(offset) as *const v128)), zero, max);
        let right0 = clamp_i16x8(i16x8_add(v128_load(pst_right.add(offset) as *const v128), v128_load(threat_right.add(offset) as *const v128)), zero, max);
        let left1 = clamp_i16x8(i16x8_add(v128_load(pst_left.add(offset + 8) as *const v128), v128_load(threat_left.add(offset + 8) as *const v128)), zero, max);
        let right1 = clamp_i16x8(i16x8_add(v128_load(pst_right.add(offset + 8) as *const v128), v128_load(threat_right.add(offset + 8) as *const v128)), zero, max);

        let prod0_lo = u32x4_shr(u32x4_extmul_low_u16x8(left0, right0), FT_SHIFT);
        let prod0_hi = u32x4_shr(u32x4_extmul_high_u16x8(left0, right0), FT_SHIFT);
        let prod1_lo = u32x4_shr(u32x4_extmul_low_u16x8(left1, right1), FT_SHIFT);
        let prod1_hi = u32x4_shr(u32x4_extmul_high_u16x8(left1, right1), FT_SHIFT);
        let packed0 = u16x8_narrow_i32x4(prod0_lo, prod0_hi);
        let packed1 = u16x8_narrow_i32x4(prod1_lo, prod1_hi);
        let packed = u8x16_narrow_i16x8(packed0, packed1);
        v128_store(out.add(offset) as *mut v128, packed);
    }
    for i in chunks * 16..len {
        let left = (*pst_left.add(i) + *threat_left.add(i)).clamp(0, FT_QUANT);
        let right = (*pst_right.add(i) + *threat_right.add(i)).clamp(0, FT_QUANT);
        *out.add(i) = (((left as i32 * right as i32) >> FT_SHIFT) & 0xff) as u8;
    }
}

#[cfg(target_feature = "simd128")]
#[target_feature(enable = "simd128")]
unsafe fn clamp_i16x8(v: v128, zero: v128, max: v128) -> v128 {
    i16x8_min(i16x8_max(v, zero), max)
}

#[cfg(not(target_feature = "simd128"))]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn activate_ft_simd(pst_left: *const i16, threat_left: *const i16, pst_right: *const i16, threat_right: *const i16, out: *mut u8, len: usize) {
    activate_ft_scalar(pst_left, threat_left, pst_right, threat_right, out, len);
}
`;

const length = Number(process.env.RECKLESS_NNUE_ACTIVATE_PROBE_LEN ?? 384);
const iterations = Number(process.env.RECKLESS_NNUE_ACTIVATE_PROBE_ITERS ?? 250_000);
if (!Number.isInteger(length) || length <= 0 || length > 1024) throw new Error('RECKLESS_NNUE_ACTIVATE_PROBE_LEN must be 1..1024');
if (!Number.isInteger(iterations) || iterations <= 0) throw new Error('RECKLESS_NNUE_ACTIVATE_PROBE_ITERS must be a positive integer');

function compileRust(source, out, simd) {
  const args = ['--target', 'wasm32-wasip1', '--crate-type', 'cdylib', '-O'];
  if (simd) args.push('-C', 'target-feature=+simd128');
  args.push(source, '-o', out);
  execFileSync('rustc', args, { stdio: 'inherit' });
}

async function instantiate(file) {
  const { instance } = await WebAssembly.instantiate(readFileSync(file), {});
  for (const name of ['memory', 'scratch_i16_ptr', 'scratch_u8_ptr', 'activate_ft_scalar', 'activate_ft_simd']) {
    if (!(name in instance.exports)) throw new Error(`${file} did not export ${name}`);
  }
  return instance.exports;
}

function fillVectors(exports, seed) {
  const base = exports.scratch_i16_ptr() >> 1;
  const pstLeftOffset = base;
  const threatLeftOffset = base + length;
  const pstRightOffset = base + length * 2;
  const threatRightOffset = base + length * 3;
  const outOffset = exports.scratch_u8_ptr();
  const i16 = new Int16Array(exports.memory.buffer);
  const u8 = new Uint8Array(exports.memory.buffer);
  let x = seed >>> 0;
  for (let lane = 0; lane < length * 4; lane += 1) {
    x = (Math.imul(x, 1664525) + 1013904223) >>> 0;
    // Values straddle the 0..255 clamp range like real accumulated feature sums.
    i16[base + lane] = ((x >>> 16) % 384) - 64;
  }
  u8.fill(0, outOffset, outOffset + length);
  return {
    pstLeft: i16.slice(pstLeftOffset, pstLeftOffset + length),
    threatLeft: i16.slice(threatLeftOffset, threatLeftOffset + length),
    pstRight: i16.slice(pstRightOffset, pstRightOffset + length),
    threatRight: i16.slice(threatRightOffset, threatRightOffset + length),
  };
}

function installVectors(exports, vectors) {
  const base = exports.scratch_i16_ptr() >> 1;
  const pstLeftOffset = base;
  const threatLeftOffset = base + length;
  const pstRightOffset = base + length * 2;
  const threatRightOffset = base + length * 3;
  const outOffset = exports.scratch_u8_ptr();
  const i16 = new Int16Array(exports.memory.buffer);
  const u8 = new Uint8Array(exports.memory.buffer);
  i16.set(vectors.pstLeft, pstLeftOffset);
  i16.set(vectors.threatLeft, threatLeftOffset);
  i16.set(vectors.pstRight, pstRightOffset);
  i16.set(vectors.threatRight, threatRightOffset);
  u8.fill(0, outOffset, outOffset + length);
  return { u8, pstLeftOffset, threatLeftOffset, pstRightOffset, threatRightOffset, outOffset };
}

function callKernel(fn, offsets) {
  fn(offsets.pstLeftOffset << 1, offsets.threatLeftOffset << 1, offsets.pstRightOffset << 1, offsets.threatRightOffset << 1, offsets.outOffset, length);
}

function resultVector(offsets) {
  return offsets.u8.slice(offsets.outOffset, offsets.outOffset + length);
}

function checksum(values) {
  let hash = 0x811c9dc5;
  for (const value of values) {
    hash ^= value;
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash >>> 0;
}

function benchmark(exports, exportName, vectors) {
  const offsets = installVectors(exports, vectors);
  const fn = exports[exportName];
  const start = performance.now();
  for (let i = 0; i < iterations; i += 1) callKernel(fn, offsets);
  const elapsedMs = performance.now() - start;
  const values = resultVector(offsets);
  return {
    elapsedMs,
    lanesPerSecond: (length * iterations) / Math.max(1e-9, elapsedMs / 1000),
    checksum: checksum(values),
  };
}

function inspectSimdOpcodeCounts(files) {
  const inspector = new URL('./inspect_wasm_simd.mjs', import.meta.url);
  const output = execFileSync(process.execPath, [inspector.pathname, ...files], { encoding: 'utf8' });
  const counts = {};
  let current = null;
  for (const line of output.split('\n')) {
    const file = files.find((candidate) => line.trim() === candidate);
    if (file) { current = file; continue; }
    const match = line.match(/simdOpcodeCount=(\d+)/);
    if (current && match) counts[current] = Number(match[1]);
  }
  return counts;
}

const workdir = mkdtempSync(join(tmpdir(), 'reckless-nnue-activate-wasm-probe-'));
try {
  const source = join(workdir, 'activate_probe.rs');
  const scalarWasm = join(workdir, 'activate_scalar.wasm');
  const simdWasm = join(workdir, 'activate_simd.wasm');
  writeFileSync(source, rustSource);
  compileRust(source, scalarWasm, false);
  compileRust(source, simdWasm, true);

  const scalar = await instantiate(scalarWasm);
  const simd = await instantiate(simdWasm);
  const vectors = fillVectors(scalar, 0xfeed5eed);

  const scalarOnce = installVectors(scalar, vectors);
  const simdOnce = installVectors(simd, vectors);
  callKernel(scalar.activate_ft_scalar, scalarOnce);
  callKernel(simd.activate_ft_simd, simdOnce);
  const scalarResult = resultVector(scalarOnce);
  const simdResult = resultVector(simdOnce);
  let mismatches = 0;
  for (let i = 0; i < length; i += 1) if (scalarResult[i] !== simdResult[i]) mismatches += 1;

  const scalarBench = benchmark(scalar, 'activate_ft_scalar', vectors);
  const simdBench = benchmark(simd, 'activate_ft_simd', vectors);
  const speedup = scalarBench.elapsedMs / Math.max(1e-9, simdBench.elapsedMs);
  const simdOpcodeCounts = inspectSimdOpcodeCounts([scalarWasm, simdWasm]);
  const report = {
    target: 'wasm32-wasip1',
    kernel: 'NNUE activate_ft clipped pair-product u8 feature output',
    length,
    iterations,
    parity: { mismatches, scalarChecksum: checksum(scalarResult), simdChecksum: checksum(simdResult) },
    simdOpcodeCounts: { scalar: simdOpcodeCounts[scalarWasm], simd: simdOpcodeCounts[simdWasm] },
    scalar: { elapsedMs: Number(scalarBench.elapsedMs.toFixed(3)), lanesPerSecond: Math.round(scalarBench.lanesPerSecond), checksum: scalarBench.checksum },
    simd: { elapsedMs: Number(simdBench.elapsedMs.toFixed(3)), lanesPerSecond: Math.round(simdBench.lanesPerSecond), checksum: simdBench.checksum },
    speedup: Number(speedup.toFixed(3)),
  };
  console.log(JSON.stringify(report, null, 2));
  if (mismatches !== 0 || scalarBench.checksum !== simdBench.checksum) process.exitCode = 1;
} finally {
  rmSync(workdir, { recursive: true, force: true });
}
