#!/usr/bin/env node
// Smoke-test the Monty wasm32-wasip1 build through @bjorn3/browser_wasi_shim —
// the same WASI shim the browser worker uses — with the raw networks supplied
// as preopened in-memory files, exactly as the browser loader provides them.
//
//   node scripts/monty_wasi_smoke.mjs [--wasm public/monty/monty.wasm] \
//     [--net-dir ../models/monty] [--nodes 1000] [--contempt 0] [--fen <fen>]
//
// The nets are NOT embedded in the wasm. Monty opens them by canonical name
// (src/networks/{value,policy}.rs) from the preopened cwd:
//   nn-09da29a4b6ed.network  (value,  ~661MB raw)
//   nn-6e49a41bd7c0.network  (policy, ~286MB raw)
import fs from 'node:fs/promises';
import path from 'node:path';
import { WASI, File, OpenFile, ConsoleStdout, PreopenDirectory } from '@bjorn3/browser_wasi_shim';

const args = new Map();
for (let i = 2; i < process.argv.length; i += 1) {
  const arg = process.argv[i];
  if (!arg.startsWith('--')) continue;
  const key = arg.slice(2);
  const next = process.argv[i + 1];
  if (!next || next.startsWith('--')) args.set(key, 'true');
  else {
    args.set(key, next);
    i += 1;
  }
}

const VALUE_NET = 'nn-09da29a4b6ed.network';
const POLICY_NET = 'nn-6e49a41bd7c0.network';

const wasmPath = args.get('wasm') ?? 'public/monty/monty.wasm';
const netDir = args.get('net-dir') ?? '../models/monty';
const nodes = Math.max(1, Math.floor(Number(args.get('nodes') ?? 1000)));
const hashMb = Math.max(1, Math.floor(Number(args.get('hash') ?? 32)));
const contempt = Math.floor(Number(args.get('contempt') ?? 0));
const fen = args.get('fen') ?? 'startpos';
const commands = [
  'uci',
  'isready',
  `setoption name Hash value ${hashMb}`,
  ...(contempt !== 0 ? [`setoption name Contempt value ${contempt}`] : []),
  fen === 'startpos' ? 'position startpos' : `position fen ${fen}`,
  `go nodes ${nodes}`,
  'quit',
];

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

const startedAt = performance.now();
const netFiles = new Map();
for (const name of [VALUE_NET, POLICY_NET]) {
  netFiles.set(name, new File(await fs.readFile(path.join(netDir, name))));
}
const netLoadMs = performance.now() - startedAt;

const stdout = [];
const stderr = [];
const module = await WebAssembly.compile(await fs.readFile(wasmPath));
const wasiInstance = new WASI(
  ['monty', ...commands],
  [],
  [
    new OpenFile(new File([])),
    lineCollector(stdout),
    lineCollector(stderr),
    new PreopenDirectory('.', netFiles),
  ],
  { debug: false },
);
const instance = await WebAssembly.instantiate(module, { wasi_snapshot_preview1: wasiInstance.wasiImport });
let exitCode = 0;
try {
  exitCode = wasiInstance.start(instance);
} catch (error) {
  // `quit` calls std::process::exit(0) -> proc_exit(0), surfaced as an exception.
  if (!String(error).includes('exit with exit code 0')) throw error;
}
const elapsedMs = performance.now() - startedAt;
const bestmoves = stdout.filter((line) => line.startsWith('bestmove'));
const lastInfo = stdout.filter((line) => line.startsWith('info ')).at(-1) ?? null;

console.log(JSON.stringify({
  wasmPath,
  netDir,
  nodes,
  hashMb,
  contempt,
  fen,
  netLoadMs,
  elapsedMs,
  exitCode,
  uciok: stdout.includes('uciok'),
  bestmoves,
  lastInfo,
  stderr,
}, null, 2));

if (exitCode !== 0) process.exit(exitCode);
if (bestmoves.length !== 1) {
  console.error(`Expected 1 bestmove line, got ${bestmoves.length}`);
  process.exit(1);
}
if (!lastInfo) {
  console.error('Expected at least one info line');
  process.exit(1);
}
