#!/usr/bin/env node
import { cpSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { execFileSync } from 'node:child_process';

const repo = process.env.RECKLESS_REPO ?? 'https://github.com/codedeliveryservice/Reckless.git';
// Pin the default upstream revision so release WASM artifacts can be rebuilt
// from a stable corresponding-source recipe. Override RECKLESS_REF to test a
// newer upstream branch/tag/commit.
const ref = process.env.RECKLESS_REF ?? '0010617448bdef4c8cd7d4f4825b7e42c8bc262a';
const workdir = resolve(process.env.RECKLESS_BUILD_DIR ?? '.local_engines/reckless-wasi-src');
const out = resolve(process.env.RECKLESS_WASM_OUT ?? 'public/reckless/reckless.wasm');
const evalfile = process.env.RECKLESS_EVALFILE ? resolve(process.env.RECKLESS_EVALFILE) : '';
const l1Size = process.env.RECKLESS_L1_SIZE ?? '';
const enableWasmSimdNnue = process.env.RECKLESS_WASM_SIMD_NNUE === '1';
if (l1Size && !/^\d+$/.test(l1Size)) throw new Error(`RECKLESS_L1_SIZE must be an integer, got ${l1Size}`);

function run(cmd, args, options = {}) {
  console.log(`$ ${cmd} ${args.join(' ')}`);
  execFileSync(cmd, args, { stdio: 'inherit', ...options });
}

function replace(path, oldText, newText) {
  const before = readFileSync(path, 'utf8');
  if (!before.includes(oldText)) throw new Error(`patch anchor not found in ${path}`);
  writeFileSync(path, before.replace(oldText, newText));
}

function patchRecklessForWasmSimdNnue(root) {
  replace(
    `${root}/src/nnue.rs`,
    `mod forward {\n    #[cfg(any(target_feature = "avx2", target_feature = "neon"))]\n    mod vectorized;\n    #[cfg(any(target_feature = "avx2", target_feature = "neon"))]\n    pub use vectorized::*;\n\n    #[cfg(not(any(target_feature = "avx2", target_feature = "neon")))]\n    mod scalar;\n    #[cfg(not(any(target_feature = "avx2", target_feature = "neon")))]\n    pub use scalar::*;\n}\n`,
    `mod forward {\n    #[cfg(any(target_feature = "avx2", target_feature = "neon", all(target_arch = "wasm32", target_feature = "simd128")))]\n    mod vectorized;\n    #[cfg(any(target_feature = "avx2", target_feature = "neon", all(target_arch = "wasm32", target_feature = "simd128")))]\n    pub use vectorized::*;\n\n    #[cfg(not(any(target_feature = "avx2", target_feature = "neon", all(target_arch = "wasm32", target_feature = "simd128"))))]\n    mod scalar;\n    #[cfg(not(any(target_feature = "avx2", target_feature = "neon", all(target_arch = "wasm32", target_feature = "simd128"))))]\n    pub use scalar::*;\n}\n`,
  );
  replace(
    `${root}/src/nnue.rs`,
    `    #[cfg(all(target_feature = "neon", not(any(target_feature = "avx2", target_feature = "avx512f"))))]\n    mod neon;\n    #[cfg(all(target_feature = "neon", not(any(target_feature = "avx2", target_feature = "avx512f"))))]\n    pub use neon::*;\n\n    #[cfg(not(any(target_feature = "avx512f", target_feature = "avx2", target_feature = "neon")))]\n    mod scalar;\n    #[cfg(not(any(target_feature = "avx512f", target_feature = "avx2", target_feature = "neon")))]\n    pub use scalar::*;\n}\n`,
    `    #[cfg(all(target_feature = "neon", not(any(target_feature = "avx2", target_feature = "avx512f"))))]\n    mod neon;\n    #[cfg(all(target_feature = "neon", not(any(target_feature = "avx2", target_feature = "avx512f"))))]\n    pub use neon::*;\n\n    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]\n    mod wasm32;\n    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]\n    pub use wasm32::*;\n\n    #[cfg(not(any(target_feature = "avx512f", target_feature = "avx2", target_feature = "neon", all(target_arch = "wasm32", target_feature = "simd128"))))]\n    mod scalar;\n    #[cfg(not(any(target_feature = "avx512f", target_feature = "avx2", target_feature = "neon", all(target_arch = "wasm32", target_feature = "simd128"))))]\n    pub use scalar::*;\n}\n`,
  );
  replace(
    `${root}/src/nnue/forward/vectorized.rs`,
    `#[cfg(all(not(target_feature = "neon"), not(target_feature = "avx512vbmi2")))]\npub unsafe fn find_nnz(\n`,
    `#[cfg(all(not(target_arch = "wasm32"), not(target_feature = "neon"), not(target_feature = "avx512vbmi2")))]\npub unsafe fn find_nnz(\n`,
  );
  writeFileSync(`${root}/src/nnue/simd/wasm32.rs`, String.raw`use std::{arch::wasm32::*, mem::size_of};

pub const F32_LANES: usize = size_of::<v128>() / size_of::<f32>();
pub const I32_LANES: usize = size_of::<v128>() / size_of::<i32>();
pub const I16_LANES: usize = size_of::<v128>() / size_of::<i16>();
pub const MUL_HI_SHIFT: i32 = 0;

pub fn add_i16(a: v128, b: v128) -> v128 { i16x8_add(a, b) }
pub fn sub_i16(a: v128, b: v128) -> v128 { i16x8_sub(a, b) }

pub unsafe fn zeroed() -> v128 { i32x4_splat(0) }
pub unsafe fn splat_i16(a: i16) -> v128 { i16x8_splat(a) }
pub unsafe fn clamp_i16(x: v128, min: v128, max: v128) -> v128 { i16x8_max(i16x8_min(x, max), min) }
pub unsafe fn min_i16(a: v128, b: v128) -> v128 { i16x8_min(a, b) }
pub unsafe fn shift_left_i16<const SHIFT: i32>(a: v128) -> v128 { i16x8_shl(a, SHIFT as u32) }

pub unsafe fn mul_high_i16(a: v128, b: v128) -> v128 {
    let lo = i32x4_shr(i32x4_extmul_low_i16x8(a, b), 16);
    let hi = i32x4_shr(i32x4_extmul_high_i16x8(a, b), 16);
    i16x8_narrow_i32x4(lo, hi)
}

pub unsafe fn convert_i8_i16(a: i64) -> v128 { i16x8_extend_low_i8x16(i64x2(a, 0)) }
pub unsafe fn packus(a: v128, b: v128) -> v128 { u8x16_narrow_i16x8(a, b) }
pub unsafe fn permute(a: v128) -> v128 { a }
pub unsafe fn splat_i32(a: i32) -> v128 { i32x4_splat(a) }
pub unsafe fn zero_f32() -> v128 { f32x4_splat(0.0) }
pub unsafe fn splat_f32(a: f32) -> v128 { f32x4_splat(a) }
pub unsafe fn mul_add_f32(a: v128, b: v128, c: v128) -> v128 { f32x4_add(f32x4_mul(a, b), c) }
pub unsafe fn convert_to_f32(a: v128) -> v128 { f32x4_convert_i32x4(a) }
pub unsafe fn clamp_f32(x: v128, min: v128, max: v128) -> v128 { f32x4_max(f32x4_min(x, max), min) }

unsafe fn dpbusd_once(i32s: v128, u8s: v128, i8s: v128) -> v128 {
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

pub unsafe fn dpbusd(i32s: v128, u8s: v128, i8s: v128) -> v128 { dpbusd_once(i32s, u8s, i8s) }
pub unsafe fn double_dpbusd(i32s: v128, u8s1: v128, i8s1: v128, u8s2: v128, i8s2: v128) -> v128 {
    dpbusd_once(dpbusd_once(i32s, u8s1, i8s1), u8s2, i8s2)
}

pub unsafe fn horizontal_sum(x: [v128; 4]) -> f32 {
    let mut sum = 0.0;
    for vector in x {
        sum += f32x4_extract_lane::<0>(vector);
        sum += f32x4_extract_lane::<1>(vector);
        sum += f32x4_extract_lane::<2>(vector);
        sum += f32x4_extract_lane::<3>(vector);
    }
    sum
}

pub unsafe fn nnz_bitmask(x: v128) -> u16 { u8x16_bitmask(i32x4_gt(x, i32x4_splat(0))) }
`);
  const vectorized = readFileSync(`${root}/src/nnue/forward/vectorized.rs`, 'utf8');
  writeFileSync(`${root}/src/nnue/forward/vectorized.rs`, `${vectorized}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub unsafe fn find_nnz(ft_out: &Aligned<[u8; L1_SIZE]>, _: &[SparseEntry]) -> (Aligned<[u16; L1_SIZE / 4]>, usize) {
    let mut indexes = Aligned::new([0; L1_SIZE / 4]);
    let mut count = 0;
    for i in 0..L1_SIZE / 4 {
        let mut nonzero = 0;
        for j in 0..4 {
            nonzero |= ft_out[i * 4 + j];
        }
        if nonzero != 0 {
            indexes[count] = i as u16;
            count += 1;
        }
    }
    (indexes, count)
}
`);
}

function patchRecklessForWasi(root) {
  if (l1Size) {
    replace(
      `${root}/src/nnue.rs`,
      `const L1_SIZE: usize = 768;`,
      `const L1_SIZE: usize = ${Number.parseInt(l1Size, 10)};`,
    );
  }

  replace(
    `${root}/src/uci.rs`,
    `    let rx = spawn_listener(shared.clone());\n\n    let mut mode = if buffer.is_empty() { Mode::Uci } else { Mode::Cli };`,
    `    let cli_mode = !buffer.is_empty();\n\n    #[cfg(target_arch = "wasm32")]\n    let rx: Option<std::sync::mpsc::Receiver<String>> = None;\n\n    #[cfg(not(target_arch = "wasm32"))]\n    let rx = if cli_mode { None } else { Some(spawn_listener(shared.clone())) };\n\n    let mut mode = if cli_mode { Mode::Cli } else { Mode::Uci };`,
  );
  replace(
    `${root}/src/uci.rs`,
    `        } else if mode == Mode::Uci {\n            match rx.recv() {\n                Ok(cmd) => cmd,\n                Err(_) => break,\n            }`,
    `        } else if mode == Mode::Uci {\n            #[cfg(target_arch = "wasm32")]\n            {\n                let _ = &rx;\n                let mut message = String::new();\n                match std::io::stdin().read_line(&mut message) {\n                    Ok(0) => break,\n                    Ok(_) => message,\n                    Err(_) => break,\n                }\n            }\n\n            #[cfg(not(target_arch = "wasm32"))]\n            {\n                match rx.as_ref().and_then(|rx| rx.recv().ok()) {\n                    Some(cmd) => cmd,\n                    None => break,\n                }\n            }`,
  );
  replace(
    `${root}/src/numa.rs`,
    `    pub fn execute_on_numa_node<F: FnOnce() + Send + 'static>(&self, n: NumaIndex, f: F) {\n        let cfg = self.clone();\n        let handle = thread::spawn(move || {\n            cfg.bind_current_thread_to_numa_node(n);\n            f();\n        });\n        handle.join().unwrap();\n    }`,
    `    pub fn execute_on_numa_node<F: FnOnce() + Send + 'static>(&self, n: NumaIndex, f: F) {\n        #[cfg(target_arch = "wasm32")]\n        {\n            let _ = n;\n            f();\n        }\n\n        #[cfg(not(target_arch = "wasm32"))]\n        {\n            let cfg = self.clone();\n            let handle = thread::spawn(move || {\n                cfg.bind_current_thread_to_numa_node(n);\n                f();\n            });\n            handle.join().unwrap();\n        }\n    }`,
  );
  replace(
    `${root}/src/threadpool.rs`,
    `pub struct WorkerThread {\n    handle: std::thread::JoinHandle<()>,\n    comms: WorkSender,\n}`,
    `pub struct WorkerThread {\n    handle: Option<std::thread::JoinHandle<()>>,\n    comms: WorkSender,\n}`,
  );
  replace(
    `${root}/src/threadpool.rs`,
    `    pub fn join(self) {\n        drop(self.comms); // Drop the sender to signal the worker thread to finish\n        self.handle.join().expect("Worker thread panicked");\n    }`,
    `    pub fn join(self) {\n        drop(self.comms); // Drop the sender to signal the worker thread to finish\n        if let Some(handle) = self.handle {\n            handle.join().expect("Worker thread panicked");\n        }\n    }`,
  );
  replace(
    `${root}/src/threadpool.rs`,
    `        // Safety: This file is structured such that threads never hold the data longer than is permissible.\n        let f = unsafe {\n            std::mem::transmute::<Box<dyn FnOnce() + Send + 'scope>, Box<dyn FnOnce() + Send + 'static>>(Box::new(f))\n        };\n\n        // Reset the completion flag before sending the task\n        {\n            let (lock, _) = &*thread.comms.completion_signal;\n            let mut completed = lock.lock().unwrap();\n            *completed = false;\n        }\n\n        thread.comms.sender.send(f).expect("Failed to send function to worker thread");\n\n        ReceiverHandle {\n            completion_signal: &thread.comms.completion_signal,\n            // Important: We start with \`received\` as false.\n            received: false,\n        }`,
    `        // Reset the completion flag before sending the task\n        {\n            let (lock, _) = &*thread.comms.completion_signal;\n            let mut completed = lock.lock().unwrap();\n            *completed = false;\n        }\n\n        #[cfg(target_arch = "wasm32")]\n        {\n            f();\n            let (lock, cvar) = &*thread.comms.completion_signal;\n            let mut completed = lock.lock().unwrap();\n            *completed = true;\n            drop(completed);\n            cvar.notify_one();\n        }\n\n        #[cfg(not(target_arch = "wasm32"))]\n        {\n            // Safety: This file is structured such that threads never hold the data longer than is permissible.\n            let f = unsafe {\n                std::mem::transmute::<Box<dyn FnOnce() + Send + 'scope>, Box<dyn FnOnce() + Send + 'static>>(Box::new(f))\n            };\n            thread.comms.sender.send(f).expect("Failed to send function to worker thread");\n        }\n\n        ReceiverHandle {\n            completion_signal: &thread.comms.completion_signal,\n            // Important: We start with \`received\` as false.\n            received: false,\n        }`,
  );
  replace(
    `${root}/src/threadpool.rs`,
    `fn make_worker_thread() -> WorkerThread {\n    let (sender, receiver) = make_work_channel();\n\n    let handle = std::thread::spawn(move || {\n        while let Ok(work) = receiver.receiver.recv() {\n            work();\n            let (lock, cvar) = &*receiver.completion_signal;\n            let mut completed = lock.lock().unwrap();\n            *completed = true;\n            drop(completed); // Release the lock before notifying\n            cvar.notify_one();\n        }\n    });\n\n    WorkerThread { handle, comms: sender }\n}`,
    `fn make_worker_thread() -> WorkerThread {\n    let (sender, receiver) = make_work_channel();\n\n    #[cfg(not(target_arch = "wasm32"))]\n    let handle = Some(std::thread::spawn(move || {\n        while let Ok(work) = receiver.receiver.recv() {\n            work();\n            let (lock, cvar) = &*receiver.completion_signal;\n            let mut completed = lock.lock().unwrap();\n            *completed = true;\n            drop(completed); // Release the lock before notifying\n            cvar.notify_one();\n        }\n    }));\n\n    #[cfg(target_arch = "wasm32")]\n    let handle = {\n        drop(receiver);\n        None\n    };\n\n    WorkerThread { handle, comms: sender }\n}`,
  );
  replace(
    `${root}/src/transposition.rs`,
    `unsafe fn parallel_clear<T: std::marker::Send>(threads: usize, ptr: *mut T, len: usize) {\n    std::thread::scope(|scope| {\n        let slice = std::slice::from_raw_parts_mut(ptr, len);\n\n        let chunk_size = len.div_ceil(threads);\n        for chunk in slice.chunks_mut(chunk_size) {\n            scope.spawn(|| chunk.as_mut_ptr().write_bytes(0, chunk.len()));\n        }\n    });\n}`,
    `unsafe fn parallel_clear<T: std::marker::Send>(threads: usize, ptr: *mut T, len: usize) {\n    #[cfg(target_arch = "wasm32")]\n    {\n        let _ = threads;\n        ptr.write_bytes(0, len);\n    }\n\n    #[cfg(not(target_arch = "wasm32"))]\n    std::thread::scope(|scope| {\n        let slice = std::slice::from_raw_parts_mut(ptr, len);\n        let chunk_size = len.div_ceil(threads);\n        for chunk in slice.chunks_mut(chunk_size) {\n            scope.spawn(|| chunk.as_mut_ptr().write_bytes(0, chunk.len()));\n        }\n    });\n}`,
  );
}

rmSync(workdir, { recursive: true, force: true });
mkdirSync(workdir, { recursive: true });
run('git', ['init'], { cwd: workdir });
run('git', ['remote', 'add', 'origin', repo], { cwd: workdir });
run('git', ['fetch', '--depth=1', 'origin', ref], { cwd: workdir });
run('git', ['checkout', '--detach', 'FETCH_HEAD'], { cwd: workdir });
patchRecklessForWasi(workdir);
if (enableWasmSimdNnue) patchRecklessForWasmSimdNnue(workdir);
run('rustup', ['target', 'add', 'wasm32-wasip1']);
run('cargo', ['build', '--release', '--no-default-features', '--target', 'wasm32-wasip1'], {
  cwd: workdir,
  env: evalfile ? { ...process.env, EVALFILE: evalfile } : process.env,
});
mkdirSync(dirname(out), { recursive: true });
const built = `${workdir}/target/wasm32-wasip1/release/reckless.wasm`;
if (!existsSync(built)) throw new Error(`expected build artifact missing: ${built}`);
cpSync(built, out);
console.log(`Wrote ${out}`);
