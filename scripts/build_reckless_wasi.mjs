#!/usr/bin/env node
import { cpSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

// Patched wasm32 Rust sources live as real files (rustfmt/clippy-visible) in
// engines/reckless/src; see docs/engine_packaging_reorg.md.
function engineSource(name) {
  return fileURLToPath(new URL(`../engines/reckless/src/${name}`, import.meta.url));
}
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
  writeFileSync(`${root}/src/nnue/simd/wasm32.rs`, readFileSync(engineSource('nnue_simd_wasm32.rs'), 'utf8'));
  const vectorized = readFileSync(`${root}/src/nnue/forward/vectorized.rs`, 'utf8');
  writeFileSync(`${root}/src/nnue/forward/vectorized.rs`, `${vectorized}\n${readFileSync(engineSource('find_nnz_wasm32.rs'), 'utf8')}`);
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
