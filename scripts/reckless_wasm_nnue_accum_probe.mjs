#!/usr/bin/env node
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { performance } from 'node:perf_hooks';
import { execFileSync } from 'node:child_process';

const rustSource = String.raw`#![allow(static_mut_refs, unused_imports)]
use std::arch::wasm32::*;

#[unsafe(no_mangle)]
pub static mut SCRATCH: [i16; 4096] = [0; 4096];

#[unsafe(no_mangle)]
pub extern "C" fn scratch_ptr() -> *mut i16 {
    unsafe { SCRATCH.as_mut_ptr() }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn accum_add_sub_scalar(acc: *mut i16, add: *const i16, sub: *const i16, len: usize) {
    for i in 0..len {
        *acc.add(i) = (*acc.add(i)).wrapping_add((*add.add(i)).wrapping_sub(*sub.add(i)));
    }
}

#[cfg(target_feature = "simd128")]
#[target_feature(enable = "simd128")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn accum_add_sub_simd(acc: *mut i16, add: *const i16, sub: *const i16, len: usize) {
    let chunks = len / 8;
    for chunk in 0..chunks {
        let offset = chunk * 8;
        let current = v128_load(acc.add(offset) as *const v128);
        let add_weights = v128_load(add.add(offset) as *const v128);
        let sub_weights = v128_load(sub.add(offset) as *const v128);
        let updated = i16x8_add(current, i16x8_sub(add_weights, sub_weights));
        v128_store(acc.add(offset) as *mut v128, updated);
    }
    for i in chunks * 8..len {
        *acc.add(i) = (*acc.add(i)).wrapping_add((*add.add(i)).wrapping_sub(*sub.add(i)));
    }
}

#[cfg(not(target_feature = "simd128"))]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn accum_add_sub_simd(acc: *mut i16, add: *const i16, sub: *const i16, len: usize) {
    accum_add_sub_scalar(acc, add, sub, len);
}
`;

const length = Number(process.env.RECKLESS_NNUE_PROBE_LEN ?? 1024);
const iterations = Number(process.env.RECKLESS_NNUE_PROBE_ITERS ?? 200_000);
if (!Number.isInteger(length) || length <= 0 || length > 1365) throw new Error('RECKLESS_NNUE_PROBE_LEN must be 1..1365');
if (!Number.isInteger(iterations) || iterations <= 0) throw new Error('RECKLESS_NNUE_PROBE_ITERS must be a positive integer');

function compileRust(source, out, simd) {
  const args = ['--target', 'wasm32-wasip1', '--crate-type', 'cdylib', '-O'];
  if (simd) args.push('-C', 'target-feature=+simd128');
  args.push(source, '-o', out);
  execFileSync('rustc', args, { stdio: 'inherit' });
}

async function instantiate(file) {
  const { instance } = await WebAssembly.instantiate(readFileSync(file), {});
  for (const name of ['memory', 'scratch_ptr', 'accum_add_sub_scalar', 'accum_add_sub_simd']) {
    if (!(name in instance.exports)) throw new Error(`${file} did not export ${name}`);
  }
  return instance.exports;
}

function fillInputs(exports, seed) {
  const base = exports.scratch_ptr() >> 1;
  const accOffset = base;
  const addOffset = base + length;
  const subOffset = base + length * 2;
  const view = new Int16Array(exports.memory.buffer);
  let x = seed >>> 0;
  for (let lane = 0; lane < length * 3; lane += 1) {
    x = (Math.imul(x, 1664525) + 1013904223) >>> 0;
    view[base + lane] = ((x >>> 16) - 32768) | 0;
  }
  const initialAcc = view.slice(accOffset, accOffset + length);
  const add = view.slice(addOffset, addOffset + length);
  const sub = view.slice(subOffset, subOffset + length);
  return { view, accOffset, addOffset, subOffset, initialAcc, add, sub };
}

function installInputs(exports, vectors) {
  const base = exports.scratch_ptr() >> 1;
  const view = new Int16Array(exports.memory.buffer);
  const accOffset = base;
  const addOffset = base + length;
  const subOffset = base + length * 2;
  view.set(vectors.initialAcc, accOffset);
  view.set(vectors.add, addOffset);
  view.set(vectors.sub, subOffset);
  return { view, accOffset, addOffset, subOffset };
}

function callKernel(fn, offsets) {
  fn(offsets.accOffset << 1, offsets.addOffset << 1, offsets.subOffset << 1, length);
}

function resultVector(offsets) {
  return offsets.view.slice(offsets.accOffset, offsets.accOffset + length);
}

function checksum(values) {
  let hash = 0x811c9dc5;
  for (const value of values) {
    hash ^= value & 0xffff;
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash >>> 0;
}

function benchmark(exports, exportName, vectors) {
  const offsets = installInputs(exports, vectors);
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
    if (file) {
      current = file;
      continue;
    }
    const match = line.match(/simdOpcodeCount=(\d+)/);
    if (current && match) counts[current] = Number(match[1]);
  }
  return counts;
}

const workdir = mkdtempSync(join(tmpdir(), 'reckless-nnue-wasm-probe-'));
try {
  const source = join(workdir, 'accum_probe.rs');
  const scalarWasm = join(workdir, 'accum_scalar.wasm');
  const simdWasm = join(workdir, 'accum_simd.wasm');
  writeFileSync(source, rustSource);
  compileRust(source, scalarWasm, false);
  compileRust(source, simdWasm, true);

  const scalar = await instantiate(scalarWasm);
  const simd = await instantiate(simdWasm);
  const vectors = fillInputs(scalar, 0xdecafbad);

  const scalarOnce = installInputs(scalar, vectors);
  const simdOnce = installInputs(simd, vectors);
  callKernel(scalar.accum_add_sub_scalar, scalarOnce);
  callKernel(simd.accum_add_sub_simd, simdOnce);
  const scalarResult = resultVector(scalarOnce);
  const simdResult = resultVector(simdOnce);
  let mismatches = 0;
  for (let i = 0; i < length; i += 1) if (scalarResult[i] !== simdResult[i]) mismatches += 1;

  const scalarBench = benchmark(scalar, 'accum_add_sub_scalar', vectors);
  const simdBench = benchmark(simd, 'accum_add_sub_simd', vectors);
  const speedup = scalarBench.elapsedMs / Math.max(1e-9, simdBench.elapsedMs);
  const simdOpcodeCounts = inspectSimdOpcodeCounts([scalarWasm, simdWasm]);
  const report = {
    target: 'wasm32-wasip1',
    kernel: 'i16 accumulator acc += add - sub',
    length,
    iterations,
    parity: { mismatches, scalarChecksum: checksum(scalarResult), simdChecksum: checksum(simdResult) },
    simdOpcodeCounts: {
      scalar: simdOpcodeCounts[scalarWasm],
      simd: simdOpcodeCounts[simdWasm],
    },
    scalar: { elapsedMs: Number(scalarBench.elapsedMs.toFixed(3)), lanesPerSecond: Math.round(scalarBench.lanesPerSecond), checksum: scalarBench.checksum },
    simd: { elapsedMs: Number(simdBench.elapsedMs.toFixed(3)), lanesPerSecond: Math.round(simdBench.lanesPerSecond), checksum: simdBench.checksum },
    speedup: Number(speedup.toFixed(3)),
  };
  console.log(JSON.stringify(report, null, 2));
  if (mismatches !== 0 || scalarBench.checksum !== simdBench.checksum) process.exitCode = 1;
} finally {
  rmSync(workdir, { recursive: true, force: true });
}
