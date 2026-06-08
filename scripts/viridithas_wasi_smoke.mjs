#!/usr/bin/env node
import fs from 'node:fs/promises';
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

const wasmPath = args.get('wasm') ?? 'public/viridithas/viridithas-simd128.wasm';
const depth = Math.max(1, Math.floor(Number(args.get('depth') ?? 2)));
const repeat = Math.max(1, Math.floor(Number(args.get('repeat') ?? 1)));
const hashMb = Math.max(1, Math.floor(Number(args.get('hash') ?? 16)));
const fen = args.get('fen') ?? 'startpos';
const commands = [
  'uci',
  'isready',
  `setoption name Hash value ${hashMb}`,
  'setoption name Threads value 1',
];
for (let i = 0; i < repeat; i += 1) {
  commands.push('ucinewgame', fen === 'startpos' ? 'position startpos' : `position fen ${fen}`, `go depth ${depth}`);
}

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

const stdout = [];
const stderr = [];
const startedAt = performance.now();
const module = await WebAssembly.compile(await fs.readFile(wasmPath));
const wasiInstance = new WASI(
  ['viridithas', ...commands],
  [],
  [
    new OpenFile(new File([])),
    lineCollector(stdout),
    lineCollector(stderr),
    new PreopenDirectory('.', new Map()),
  ],
  { debug: false },
);
const instance = await WebAssembly.instantiate(module, { wasi_snapshot_preview1: wasiInstance.wasiImport });
const exitCode = wasiInstance.start(instance);
const elapsedMs = performance.now() - startedAt;
const bestmoves = stdout.filter((line) => line.startsWith('bestmove'));
const infos = stdout.filter((line) => line.startsWith('info '));
const lastInfo = infos.at(-1) ?? null;

const report = {
  wasmPath,
  depth,
  repeat,
  hashMb,
  fen,
  elapsedMs,
  exitCode,
  bestmoves,
  lastInfo,
  stderr,
};
console.log(JSON.stringify(report, null, 2));

if (exitCode !== 0) process.exit(exitCode);
if (bestmoves.length !== repeat) {
  console.error(`Expected ${repeat} bestmove line(s), got ${bestmoves.length}`);
  process.exit(1);
}
if (!lastInfo) {
  console.error('Expected at least one info line');
  process.exit(1);
}
