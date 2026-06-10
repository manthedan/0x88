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
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';

const ROTATED_FEN_SUITE = [
  ['Start position', 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'],
  ['Ruy Lopez ply 2', 'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2'],
  ['Ruy Lopez ply 4', 'r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3'],
  ['Ruy Lopez ply 6', 'r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4'],
  ['Ruy Lopez ply 8', 'r1bqkb1r/1ppp1ppp/p1n2n2/4p3/B3P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 5'],
  ['Ruy Lopez ply 10', 'r1bqk2r/1pppbppp/p1n2n2/4p3/B3P3/5N2/PPPP1PPP/RNBQ1RK1 w kq - 4 6'],
  ['Ruy Lopez ply 12', 'r1bqk2r/2ppbppp/p1n2n2/1p2p3/B3P3/5N2/PPPP1PPP/RNBQR1K1 w kq b6 0 7'],
  ['Ruy Lopez ply 14', 'r1bqk2r/2p1bppp/p1np1n2/1p2p3/4P3/1B3N2/PPPP1PPP/RNBQR1K1 w kq - 0 8'],
  ['Ruy Lopez ply 16', 'r1bq1rk1/2p1bppp/p1np1n2/1p2p3/4P3/1BP2N2/PP1P1PPP/RNBQR1K1 w - - 1 9'],
  ['Ruy Lopez ply 18', 'r2q1rk1/1bp1bppp/p1np1n2/1p2p3/4P3/1BP2N1P/PP1P1PP1/RNBQR1K1 w - - 1 10'],
  ['Ruy Lopez ply 20', 'rn1q1rk1/1bp1bppp/p2p1n2/1p2p3/3PP3/1BP2N1P/PP3PP1/RNBQR1K1 w - - 1 11'],
  ['Ruy Lopez ply 22', 'r2q1rk1/1bpnbppp/p2p1n2/1p2p3/3PP3/1BP2N1P/PP1N1PP1/R1BQR1K1 w - - 3 12'],
  ['Ruy Lopez ply 24', 'r2q1rk1/1b1nbppp/p2p1n2/1pp1p3/P2PP3/1BP2N1P/1P1N1PP1/R1BQR1K1 w - c6 0 13'],
  ['Ruy Lopez ply 26', 'r2q1rk1/1b1nbppp/p2p1n2/1p1Pp3/P1p1P3/1BP2N1P/1P1N1PP1/R1BQR1K1 w - - 0 14'],
  ['Ruy Lopez ply 28', 'r2q1rk1/1b2bppp/p2p1n2/1pnPp3/P1p1P3/2P2N1P/1PBN1PP1/R1BQR1K1 w - - 2 15'],
  ['Ruy Lopez ply 30', 'r2q1rk1/1b1nbppp/p2p4/1pnPp3/P1p1P3/2P4P/1PBN1PPN/R1BQR1K1 w - - 4 16'],
  ['Ruy Lopez ply 32', 'r2q1rk1/1b1nbppp/p2p4/1pnPp3/P3P3/1pP4P/2BN1PPN/R1BQR1K1 w - - 0 17'],
  ['Ruy Lopez ply 34', 'r2q1rk1/1b1nbp1p/p2p2p1/1pnPp3/P3P3/1BP4P/3N1PPN/R1BQR1K1 w - - 0 18'],
  ['Ruy Lopez ply 36', 'r2qr1k1/1b1nbp1p/p2p2p1/1pnPp3/P3P3/1BP4P/1B1N1PPN/R2QR1K1 w - - 2 19'],
  ['Ruy Lopez ply 38', 'r1bqr1k1/3nbp1p/p2p2p1/1pnPp3/P3P1P1/1BP4P/1B1N1P1N/R2QR1K1 w - - 1 20'],
];

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

const jsPaths = (args.get('js') ?? '').split(',').map((s) => s.trim()).filter(Boolean);
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

function parseInfo(line) {
  const get = (key) => {
    const match = line.match(new RegExp(`\\b${key} (-?\\d+)`));
    return match ? Number(match[1]) : null;
  };
  const score = line.match(/score (cp|mate) (-?\d+)/);
  const pv = line.match(/\bpv (.+)$/);
  return {
    depth: get('depth'),
    nodes: get('nodes'),
    nps: get('nps'),
    timeMs: get('time'),
    score: score ? `${score[1]} ${score[2]}` : null,
    pv: pv ? pv[1].trim() : null,
  };
}

async function loadModule(jsPath) {
  const base = path.basename(jsPath, '.js');
  const dir = path.dirname(path.resolve(jsPath));
  for (const ext of ['js', 'wasm', 'data']) {
    const file = path.join(dir, `${base}.${ext}`);
    if (!fs.existsSync(file)) throw new Error(`Missing Emscripten engine artifact: ${file}`);
  }
  // Copy to .cjs so Node's createRequire treats the modularized output as CJS.
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'emscripten-uci-bench-'));
  fs.copyFileSync(path.join(dir, `${base}.js`), path.join(tmp, `${base}.cjs`));
  fs.copyFileSync(path.join(dir, `${base}.wasm`), path.join(tmp, `${base}.wasm`));
  fs.copyFileSync(path.join(dir, `${base}.data`), path.join(tmp, `${base}.data`));
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
    const same = ref.bestmove === row.bestmove && ref.score === row.score && ref.nodes === row.nodes && ref.pv === row.pv;
    if (same) exact += 1;
    else mismatches.push({ position: row.position, depth: row.depth, baseline: { bestmove: ref.bestmove, score: ref.score, nodes: ref.nodes }, candidate: { bestmove: row.bestmove, score: row.score, nodes: row.nodes } });
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
