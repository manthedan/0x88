#!/usr/bin/env node
// Node bench/parity harness for Emscripten UCI engine artifacts (Berserk, PlentyChess, ...): drives the
// modularized UCI build over the same 20-position rotated FEN suite as the
// browser benchmark, comparing fixed-depth bestmove/score/nodes/PV across
// artifacts (scalar vs simd128 vs relaxed) and reporting engine NPS.
//
// Usage:
//   node scripts/emscripten_uci_bench.mjs \
//     --js public/plentychess/plentychess-emscripten.js,public/plentychess/plentychess-emscripten-sse41.js \
//     --depths 9,11 [--positions 20] [--json out.json]
import fs from 'node:fs';
import { createRequire } from 'node:module';
import os from 'node:os';
import path from 'node:path';

import { ROTATED_FEN_SUITE, exactSearch, parseInfo } from './emscripten_uci_protocol.mjs';

const args = new Map();
for (let i = 2; i < process.argv.length; i += 1) {
  const arg = process.argv[i];
  if (!arg.startsWith('--')) continue;
  const next = process.argv[i + 1];
  if (!next || next.startsWith('--')) args.set(arg.slice(2), 'true');
  else {
    args.set(arg.slice(2), next);
    i += 1;
  }
}

const jsPaths = (args.get('js') ?? '')
  .split(',')
  .map((s) => s.trim())
  .filter(Boolean);
if (!jsPaths.length) {
  console.error('Usage: node scripts/emscripten_uci_bench.mjs --js <baseline.js>[,<candidate.js>...] [--depths 9,11] [--positions 20] [--json out.json]');
  process.exit(2);
}
const depths = (args.get('depths') ?? '9').split(',').map((s) => Math.max(1, Math.floor(Number(s))));
const positionCount = Math.max(1, Math.min(ROTATED_FEN_SUITE.length, Math.floor(Number(args.get('positions') ?? ROTATED_FEN_SUITE.length))));
const timeoutMs = Math.max(1000, Math.floor(Number(args.get('timeout') ?? 120000)));
const jsonOut = args.get('json') ?? null;
const positions = ROTATED_FEN_SUITE.slice(0, positionCount);
const require = createRequire(import.meta.url);

async function loadModule(jsPath) {
  const base = path.basename(jsPath, '.js');
  const dir = path.dirname(path.resolve(jsPath));
  const perVariantData = path.join(dir, `${base}.data`);
  const canonicalData = path.join(dir, `${base.split('-emscripten')[0]}-emscripten.data`);
  const dataPath = fs.existsSync(perVariantData) ? perVariantData : canonicalData;
  if (!fs.existsSync(dataPath)) throw new Error(`Missing Emscripten preload: ${dataPath}`);
  for (const ext of ['js', 'wasm']) {
    const file = path.join(dir, `${base}.${ext}`);
    if (!fs.existsSync(file)) throw new Error(`Missing Emscripten engine artifact: ${file}`);
  }
  // Copy to .cjs so Node's createRequire treats the modularized output as CJS.
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'emscripten-uci-bench-'));
  fs.copyFileSync(path.join(dir, `${base}.js`), path.join(tmp, `${base}.cjs`));
  fs.copyFileSync(path.join(dir, `${base}.wasm`), path.join(tmp, `${base}.wasm`));
  fs.copyFileSync(dataPath, path.join(tmp, `${base}.data`));
  // SIMD glue may still request the canonical preload URL. Preserve both
  // names, using the selected tier's bytes even when it has a private .data.
  if (path.basename(canonicalData) !== `${base}.data`) fs.copyFileSync(dataPath, path.join(tmp, path.basename(canonicalData)));
  const factory = require(path.join(tmp, `${base}.cjs`));
  const stdout = [];
  let cursor = 0;
  const module = await factory({
    locateFile: (file) => path.join(tmp, file.replace(`${base}.js`, `${base}.cjs`)),
    print: (line) => stdout.push(String(line)),
    printErr: () => {},
  });
  const command = (text) => module.ccall('command', null, ['string'], [text]);
  const waitForLine = (predicate, label) => {
    const startIndex = cursor;
    const started = performance.now();
    return new Promise((resolve, reject) => {
      const tick = () => {
        for (let i = startIndex; i < stdout.length; i += 1) {
          if (predicate(stdout[i])) {
            cursor = i + 1;
            return resolve(stdout[i]);
          }
        }
        if (performance.now() - started > timeoutMs) return reject(new Error(`Timed out waiting for ${label}`));
        setTimeout(tick, 5);
      };
      tick();
    });
  };
  return { command, waitForLine, stdout, tmp, lineRange: () => stdout.length };
}

const rows = [];
for (const jsPath of jsPaths) {
  const engine = await loadModule(jsPath);
  engine.command('uci');
  await engine.waitForLine((line) => line === 'uciok', 'uciok');
  engine.command('isready');
  await engine.waitForLine((line) => line === 'readyok', 'readyok');
  engine.command('setoption name Threads value 1');
  engine.command(`setoption name Hash value ${Number(args.get('hash') ?? 16)}`);
  for (const depth of depths) {
    for (const [label, fen] of positions) {
      engine.command('ucinewgame');
      engine.command('isready');
      await engine.waitForLine((line) => line === 'readyok', 'pre-search readyok');
      const infoStart = engine.lineRange();
      engine.command(`position fen ${fen}`);
      engine.command(`go depth ${depth}`);
      const bestLine = await engine.waitForLine((line) => line.startsWith('bestmove '), `${label} bestmove`);
      const infos = engine.stdout.slice(infoStart).filter((line) => line.startsWith('info ') && line.includes(' pv '));
      const info = infos.length ? parseInfo(infos[infos.length - 1]) : {};
      rows.push({ artifact: jsPath, depth, position: label, bestmove: bestLine.split(/\s+/)[1] ?? null, ...info });
    }
  }
  engine.command('quit');
  fs.rmSync(engine.tmp, { recursive: true, force: true });
}

function median(values) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

const baseline = jsPaths[0];
const parity = {};
for (const jsPath of jsPaths.slice(1)) {
  let pairs = 0;
  let exact = 0;
  const mismatches = [];
  for (const row of rows.filter((r) => r.artifact === jsPath)) {
    const ref = rows.find((r) => r.artifact === baseline && r.depth === row.depth && r.position === row.position);
    if (!ref) continue;
    pairs += 1;
    const same = exactSearch(ref, row);
    if (same) exact += 1;
    else
      mismatches.push({
        position: row.position,
        depth: row.depth,
        baseline: { bestmove: ref.bestmove, score: ref.score, nodes: ref.nodes, pv: ref.pv },
        candidate: { bestmove: row.bestmove, score: row.score, nodes: row.nodes, pv: row.pv },
      });
  }
  parity[jsPath] = { pairs, exact, mismatches: mismatches.slice(0, 10) };
}

const summary = {};
for (const jsPath of jsPaths) {
  summary[jsPath] = {};
  for (const depth of depths) {
    const subset = rows.filter((r) => r.artifact === jsPath && r.depth === depth);
    summary[jsPath][`depth${depth}`] = {
      runs: subset.length,
      engineTimeMsMedian: median(subset.map((r) => r.timeMs ?? 0)),
      nodesTotal: subset.reduce((sum, r) => sum + (r.nodes ?? 0), 0),
      npsMedian: median(subset.map((r) => r.nps ?? 0)),
    };
  }
}

const report = { positions: positions.length, depths, baseline, parity, summary };
console.log(JSON.stringify(report, null, 2));
if (jsonOut) fs.writeFileSync(jsonOut, `${JSON.stringify({ ...report, rows }, null, 2)}\n`);
for (const data of Object.values(parity)) {
  if (data.pairs === 0 || data.exact !== data.pairs) process.exitCode = 1;
}
