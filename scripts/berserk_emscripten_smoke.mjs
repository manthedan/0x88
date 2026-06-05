#!/usr/bin/env node
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';

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

const root = process.cwd();
const jsPath = path.resolve(args.get('js') ?? path.join(root, 'public', 'berserk', 'berserk-emscripten.js'));
const depth = Math.max(1, Math.floor(Number(args.get('depth') ?? 1)));
const timeoutMs = Math.max(1000, Math.floor(Number(args.get('timeout') ?? 20000)));
const fen = args.get('fen') ?? 'r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 3';
const base = path.basename(jsPath, '.js');
const wasmPath = path.join(path.dirname(jsPath), `${base}.wasm`);
const dataPath = path.join(path.dirname(jsPath), `${base}.data`);

for (const file of [jsPath, wasmPath, dataPath]) {
  if (!fs.existsSync(file)) throw new Error(`Missing Berserk Emscripten artifact: ${file}`);
}

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'berserk-emscripten-smoke-'));
const tmpJs = path.join(tmp, `${base}.cjs`);
const tmpWasm = path.join(tmp, `${base}.wasm`);
const tmpData = path.join(tmp, `${base}.data`);
fs.copyFileSync(jsPath, tmpJs);
fs.copyFileSync(wasmPath, tmpWasm);
fs.copyFileSync(dataPath, tmpData);

const require = createRequire(import.meta.url);
const Berserk = require(tmpJs);
const stdout = [];
const stderr = [];
const startedAt = performance.now();

let stdoutCursor = 0;
function waitForLine(predicate, label) {
  const startIndex = stdoutCursor;
  const started = performance.now();
  return new Promise((resolve, reject) => {
    const tick = () => {
      for (let i = startIndex; i < stdout.length; i += 1) {
        if (predicate(stdout[i])) {
          stdoutCursor = i + 1;
          return resolve(stdout[i]);
        }
      }
      if (performance.now() - started > timeoutMs) {
        reject(new Error(`Timed out waiting for ${label}; last stdout lines:\n${stdout.slice(-12).join('\n')}`));
        return;
      }
      setTimeout(tick, 20);
    };
    tick();
  });
}

const module = await Berserk({
  locateFile(file) {
    return path.join(tmp, file.replace(`${base}.js`, `${base}.cjs`));
  },
  print(line) {
    stdout.push(String(line));
  },
  printErr(line) {
    stderr.push(String(line));
  },
});

function command(text) {
  module.ccall('command', null, ['string'], [text]);
}

command('uci');
await waitForLine((line) => line === 'uciok', 'uciok');
command('isready');
await waitForLine((line) => line === 'readyok', 'readyok');
command('setoption name Threads value 1');
command('ucinewgame');
command('position startpos');
command(`go depth ${depth}`);
const startBestmove = await waitForLine((line) => line.startsWith('bestmove '), 'startpos bestmove');
command('ucinewgame');
command(`position fen ${fen}`);
command(`go depth ${depth}`);
const fenBestmove = await waitForLine((line) => line.startsWith('bestmove '), 'FEN bestmove');
command('isready');
await waitForLine((line) => line === 'readyok', 'post-search readyok');
command('quit');

const infos = stdout.filter((line) => line.startsWith('info '));
const report = {
  jsPath,
  wasmPath,
  dataPath,
  depth,
  fen,
  elapsedMs: performance.now() - startedAt,
  startBestmove,
  fenBestmove,
  lastInfo: infos.at(-1) ?? null,
  stderr,
};
console.log(JSON.stringify(report, null, 2));

if (!infos.length) {
  console.error('Expected at least one info line');
  process.exit(1);
}
