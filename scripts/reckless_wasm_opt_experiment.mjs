#!/usr/bin/env node
import { gzipSync } from 'node:zlib';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { performance } from 'node:perf_hooks';
import { spawn } from 'node:child_process';

const ROOT = process.cwd();
const OUT_DIR = path.join(ROOT, '.local_engines', 'reckless-wasm-opt');
const DEFAULT_REPORT = path.join(ROOT, '.local-dev-artifacts', 'docs', 'reckless_wasm_opt_experiment_2026-06-04.json');

const INPUTS = [
  { key: 'scalar', path: path.join(ROOT, 'public', 'reckless', 'reckless.wasm'), simd: false },
  { key: 'simd128', path: path.join(ROOT, 'public', 'reckless', 'reckless-simd128.wasm'), simd: true },
];

const PASSES = [
  { key: 'O3', args: ['-O3'] },
  { key: 'O4', args: ['-O4'] },
  { key: 'O3-enable-simd', args: ['-O3', '--enable-simd'] },
];

function wasmOptBin() {
  const local = path.join(ROOT, 'node_modules', '.bin', process.platform === 'win32' ? 'wasm-opt.cmd' : 'wasm-opt');
  if (existsSync(local)) return local;
  return 'wasm-opt';
}

async function sizeSummary(file) {
  const bytes = await readFile(file);
  return {
    bytes: bytes.byteLength,
    gzipBytes: gzipSync(bytes, { level: 9 }).byteLength,
  };
}

async function compileMs(file) {
  const bytes = await readFile(file);
  const started = performance.now();
  await WebAssembly.compile(bytes);
  return performance.now() - started;
}

async function runChecked(command, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { cwd: ROOT, stdio: ['ignore', 'pipe', 'pipe'] });
    let stdout = '';
    let stderr = '';
    child.stdout.setEncoding('utf8');
    child.stderr.setEncoding('utf8');
    child.stdout.on('data', (chunk) => { stdout += chunk; });
    child.stderr.on('data', (chunk) => { stderr += chunk; });
    child.on('error', reject);
    child.on('close', (status) => {
      if (status !== 0) reject(new Error(`${command} ${args.join(' ')} failed with ${status}\n${stderr}`));
      else resolve({ stdout, stderr, status });
    });
  });
}

async function main() {
  const reportPath = process.argv[2] ? path.resolve(process.argv[2]) : DEFAULT_REPORT;
  await mkdir(OUT_DIR, { recursive: true });
  await mkdir(path.dirname(reportPath), { recursive: true });
  const wasmOpt = wasmOptBin();
  const rows = [];

  for (const input of INPUTS) {
    if (!existsSync(input.path)) {
      rows.push({ input: input.key, inputPath: input.path, status: 'missing' });
      continue;
    }
    const baseSize = await sizeSummary(input.path);
    const baseCompileMs = await compileMs(input.path);
    rows.push({
      input: input.key,
      pass: 'baseline',
      inputPath: path.relative(ROOT, input.path),
      outputPath: path.relative(ROOT, input.path),
      status: 'ok',
      ...baseSize,
      sizeRatio: 1,
      gzipRatio: 1,
      compileMs: baseCompileMs,
    });

    for (const pass of PASSES) {
      const outputPath = path.join(OUT_DIR, `${input.key}-${pass.key}.wasm`);
      const args = [input.path, ...pass.args, '-o', outputPath];
      if (input.simd && !args.includes('--enable-simd')) args.splice(1, 0, '--enable-simd');
      const started = performance.now();
      try {
        await runChecked(wasmOpt, args);
        const optimizedCompileMs = await compileMs(outputPath);
        const optimizedSize = await sizeSummary(outputPath);
        rows.push({
          input: input.key,
          pass: pass.key,
          inputPath: path.relative(ROOT, input.path),
          outputPath: path.relative(ROOT, outputPath),
          status: 'ok',
          ...optimizedSize,
          sizeRatio: optimizedSize.bytes / baseSize.bytes,
          gzipRatio: optimizedSize.gzipBytes / baseSize.gzipBytes,
          compileMs: optimizedCompileMs,
          optimizeMs: performance.now() - started,
          args: args.slice(1, -2),
        });
      } catch (error) {
        rows.push({
          input: input.key,
          pass: pass.key,
          inputPath: path.relative(ROOT, input.path),
          outputPath: path.relative(ROOT, outputPath),
          status: 'error',
          error: error instanceof Error ? error.message : String(error),
          args: args.slice(1, -2),
        });
      }
    }
  }

  const report = {
    generatedAt: new Date().toISOString(),
    wasmOpt,
    outputDir: path.relative(ROOT, OUT_DIR),
    rows,
  };
  await writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`);
  console.log(JSON.stringify(report, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
