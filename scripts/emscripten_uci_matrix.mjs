#!/usr/bin/env node
// Run under benchlock. Fresh subprocess per cell avoids retained heaps/code.
// --config matrix.json --rounds 5 --depth 11 --json results.json
// Each config cell: {engine,label,js,data,factory}; paths are repo-relative.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { createRequire } from 'node:module';
import { createHash } from 'node:crypto';
import { brotliCompressSync, constants } from 'node:zlib';
import { roundOrder, runSuite, summarizeCell } from './emscripten_uci_protocol.mjs';

const args = new Map();
for (let i = 2; i < process.argv.length; i += 2) args.set(process.argv[i].replace(/^--/, ''), process.argv[i + 1]);
const depth = Number(args.get('depth') ?? 11);
const hash = Number(args.get('hash') ?? 16);
const positions = Number(args.get('positions') ?? 20);
for (const [key, value] of Object.entries({ depth, hash, positions })) {
  if (!Number.isInteger(value) || value < 1 || (key === 'positions' && value > 20)) throw Error(`Invalid ${key}`);
}

if (args.has('cell')) {
  const cell = JSON.parse(args.get('cell'));
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'emscripten-matrix-'));
  try {
    const started = performance.now();
    const js = path.resolve(cell.js);
    const wasmPath = js.replace(/\.js$/, '.wasm');
    const bytes = fs.readFileSync(wasmPath);
    const data = fs.readFileSync(path.resolve(cell.data));
    const compileStart = performance.now();
    const compiled = await WebAssembly.compile(bytes);
    const compileMs = performance.now() - compileStart;
    const cjs = path.join(tmp, 'engine.cjs');
    fs.copyFileSync(js, cjs);
    const factory = createRequire(import.meta.url)(cjs);
    const stdout = [];
    let instantiateMs;
    const module = await factory({
      print: (line) => stdout.push(String(line)), printErr: () => {},
      locateFile: (file) => path.resolve(file.endsWith('.data') ? cell.data : wasmPath),
      getPreloadedPackage: () => data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength),
      instantiateWasm(imports, receive) {
        const t = performance.now();
        const instance = new WebAssembly.Instance(compiled, imports);
        instantiateMs = performance.now() - t;
        receive(instance, compiled);
        return instance.exports;
      },
    });
    const startupMs = performance.now() - started;
    const rows = runSuite(module, stdout, { depth, hash, positions });
    const result = { ...cell, runtime: process.version, v8: process.versions.v8, depth, hash, threads: 1,
      compileMs, instantiateMs, startupMs, rows, summary: summarizeCell(rows) };
    fs.writeFileSync(args.get('json'), `${JSON.stringify(result)}\n`);
  } finally { fs.rmSync(tmp, { recursive: true, force: true }); }
} else {
  const config = JSON.parse(fs.readFileSync(args.get('config'), 'utf8'));
  const rounds = Number(args.get('rounds') ?? 5);
  if (!Number.isInteger(rounds) || rounds < 1) throw Error('Invalid rounds');
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'emscripten-matrix-report-'));
  const report = { startedAt: new Date().toISOString(), depth, hash, threads: 1, positions, rounds, machine: { platform: process.platform,
    arch: process.arch, cpu: os.cpus()[0].model, os: os.release() }, artifacts: [], cells: [] };
  try {
    for (const cell of config) {
      const identity = { ...cell };
      for (const [key, file] of [['js', cell.js], ['wasm', cell.js.replace(/\.js$/, '.wasm')], ['data', cell.data]]) {
        const bytes = fs.readFileSync(file);
        identity[key] = { path: file, sha256: createHash('sha256').update(bytes).digest('hex'), bytes: bytes.length };
        if (key === 'wasm') identity[key].brotliBytes = brotliCompressSync(bytes, { params: { [constants.BROTLI_PARAM_QUALITY]: 11 } }).length;
      }
      report.artifacts.push(identity);
    }
    for (let round = 0; round < rounds; round += 1) {
      for (const engine of [...new Set(config.map((c) => c.engine))]) {
        for (const cell of roundOrder(config.filter((c) => c.engine === engine), round)) {
          const json = path.join(tmp, 'cell.json');
          const result = spawnSync(process.execPath, [import.meta.filename, '--cell', JSON.stringify(cell),
            '--depth', String(depth), '--hash', String(hash), '--positions', String(positions), '--json', json],
          { encoding: 'utf8', timeout: 300000 });
          if (result.status !== 0) throw Error(`${engine}/${cell.label}: ${result.stderr || result.error}`);
          const measured = { round, ...JSON.parse(fs.readFileSync(json, 'utf8')) };
          report.cells.push(measured);
          fs.writeFileSync(args.get('json'), `${JSON.stringify(report, null, 2)}\n`);
          console.log(`${round} ${engine}/${cell.label}: ${Math.round(measured.summary.wallNps)} wall NPS, ${measured.summary.wallMs.toFixed(1)} ms`);
        }
      }
    }
    report.finishedAt = new Date().toISOString();
    fs.writeFileSync(args.get('json'), `${JSON.stringify(report, null, 2)}\n`);
  } finally { fs.rmSync(tmp, { recursive: true, force: true }); }
}
