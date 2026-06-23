#!/usr/bin/env node
// Node browser_wasi_shim harness for Reckless WASI artifacts: runs one-shot
// fixed-depth searches over the same 20-position rotated FEN suite as
// /lab/reckless-benchmark.html, checks cross-artifact fixed-depth parity
// (bestmove, score, nodes, PV), and reports wall/engine timing. One-shot
// process startup is included in wall ms, so engine-reported nodes/NPS is the
// compute-comparison signal; wall ms is still useful artifact-vs-artifact at
// equal depth because startup cost is shared.
//
// Usage:
//   node scripts/reckless_wasi_node_bench.mjs \
//     --wasm public/reckless/reckless.wasm,public/reckless/reckless-simd128.wasm \
//     --depths 7,8 --positions 20 --hash 16 [--json out.json]
import fs from 'node:fs/promises';
import { WASI, File, OpenFile, ConsoleStdout, PreopenDirectory } from '@bjorn3/browser_wasi_shim';

const ROTATED_FEN_SUITE = [
  ['Start position', 'startpos'],
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

const wasmPaths = (args.get('wasm') ?? 'public/reckless/reckless.wasm,public/reckless/reckless-simd128.wasm,public/reckless/reckless-relaxed-simd128.wasm').split(',').map((s) => s.trim()).filter(Boolean);
const depths = (args.get('depths') ?? '7,8').split(',').map((s) => Math.max(1, Math.floor(Number(s))));
const positionCount = Math.max(1, Math.min(ROTATED_FEN_SUITE.length, Math.floor(Number(args.get('positions') ?? ROTATED_FEN_SUITE.length))));
const hashMb = Math.max(1, Math.floor(Number(args.get('hash') ?? 16)));
const jsonOut = args.get('json') ?? null;
const positions = ROTATED_FEN_SUITE.slice(0, positionCount);

function lineCollector(lines) {
  const decoder = new TextDecoder('utf-8', { fatal: false });
  let pending = '';
  return new ConsoleStdout((chunk) => {
    pending += decoder.decode(chunk, { stream: true });
    const split = pending.split(/\r?\n/);
    pending = split.pop() ?? '';
    for (const line of split) lines.push(line);
  });
}

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

async function runOneShot(module, fen, depth) {
  const commands = [
    'uci',
    'isready',
    `setoption name Hash value ${hashMb}`,
    'setoption name Threads value 1',
    'ucinewgame',
    'isready',
    fen === 'startpos' ? 'position startpos' : `position fen ${fen}`,
    `go depth ${depth}`,
    'quit',
    '',
  ].join('\n');
  const stdout = [];
  const stderr = [];
  const wasi = new WASI(
    ['reckless'],
    [],
    [
      new OpenFile(new File(new TextEncoder().encode(commands))),
      lineCollector(stdout),
      lineCollector(stderr),
      new PreopenDirectory('.', new Map()),
    ],
    { debug: false },
  );
  const started = performance.now();
  const instance = await WebAssembly.instantiate(module, { wasi_snapshot_preview1: wasi.wasiImport });
  let exitCode = 0;
  try {
    exitCode = wasi.start(instance);
  } catch (error) {
    if (typeof error?.code === 'number') exitCode = error.code;
    else throw error;
  }
  const wallMs = performance.now() - started;
  const bestmove = stdout.findLast((line) => line.startsWith('bestmove'))?.split(/\s+/)[1] ?? null;
  const infoLines = stdout.filter((line) => line.startsWith('info ') && line.includes(' pv '));
  const info = infoLines.length ? parseInfo(infoLines[infoLines.length - 1]) : {};
  return { wallMs: Number(wallMs.toFixed(2)), exitCode, bestmove, ...info, stderr: stderr.slice(0, 5) };
}

function median(values) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

const modules = new Map();
for (const path of wasmPaths) {
  modules.set(path, await WebAssembly.compile(await fs.readFile(path)));
}

const rows = [];
for (const depth of depths) {
  for (const [label, fen] of positions) {
    for (const path of wasmPaths) {
      const result = await runOneShot(modules.get(path), fen, depth);
      rows.push({ artifact: path, depth, position: label, ...result });
      if (result.exitCode !== 0 || !result.bestmove) {
        console.error(`FAIL ${path} ${label} depth ${depth}: exit=${result.exitCode} bestmove=${result.bestmove}`, result.stderr);
      }
    }
  }
}

const baseline = wasmPaths[0];
const parity = {};
for (const path of wasmPaths.slice(1)) {
  let pairs = 0;
  let exact = 0;
  const mismatches = [];
  for (const row of rows.filter((r) => r.artifact === path)) {
    const ref = rows.find((r) => r.artifact === baseline && r.depth === row.depth && r.position === row.position);
    if (!ref) continue;
    pairs += 1;
    const same = ref.bestmove === row.bestmove && ref.score === row.score && ref.nodes === row.nodes && ref.pv === row.pv;
    if (same) exact += 1;
    else mismatches.push({ position: row.position, depth: row.depth, baseline: { bestmove: ref.bestmove, score: ref.score, nodes: ref.nodes }, candidate: { bestmove: row.bestmove, score: row.score, nodes: row.nodes } });
  }
  parity[path] = { pairs, exact, mismatches: mismatches.slice(0, 10) };
}

const summary = {};
for (const path of wasmPaths) {
  summary[path] = {};
  for (const depth of depths) {
    const subset = rows.filter((r) => r.artifact === path && r.depth === depth && r.exitCode === 0);
    summary[path][`depth${depth}`] = {
      runs: subset.length,
      wallMsMedian: Number((median(subset.map((r) => r.wallMs)) ?? 0).toFixed(2)),
      engineTimeMsMedian: median(subset.map((r) => r.timeMs ?? 0)),
      nodesTotal: subset.reduce((sum, r) => sum + (r.nodes ?? 0), 0),
      npsMedian: median(subset.map((r) => r.nps ?? 0)),
    };
  }
}

const report = { positions: positions.length, depths, hashMb, baseline, parity, summary };
console.log(JSON.stringify(report, null, 2));
if (jsonOut) await fs.writeFile(jsonOut, `${JSON.stringify({ ...report, rows }, null, 2)}\n`);
for (const data of Object.values(parity)) {
  if (data.pairs === 0 || data.exact !== data.pairs) process.exitCode = 1;
}
