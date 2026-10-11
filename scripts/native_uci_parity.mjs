#!/usr/bin/env node
// Compare a native SAME-COMMIT engine with rows from emscripten_uci_bench.mjs.
// Run under benchlock on Mac, or taskset within your assigned Yukon slot.
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { ROTATED_FEN_SUITE, exactSearch, parseInfo } from './emscripten_uci_protocol.mjs';

const args = new Map();
for (let i = 2; i < process.argv.length; i += 2) args.set(process.argv[i].replace(/^--/, ''), process.argv[i + 1]);
if (!args.has('engine') || !args.has('reference') || !args.has('json')) {
  throw Error('Usage: native_uci_parity.mjs --engine <executable> --reference <wasm-parity.json> --json <out.json> [--hash 16]');
}
const reference = JSON.parse(fs.readFileSync(args.get('reference'), 'utf8'));
const hash = Number(args.get('hash') ?? 16);
if (!Number.isInteger(hash) || hash < 1) throw Error('Invalid hash');
if (!reference.rows?.length) throw Error('Reference has no search rows');
const requests = [...new Map(reference.rows.map((r) => [`${r.position}/${r.depth}`, r])).values()];
for (const row of requests) {
  if (!ROTATED_FEN_SUITE.some(([label]) => label === row.position) || !Number.isInteger(row.depth) || row.depth < 1) {
    throw Error(`Invalid reference search: ${row.position}/${row.depth}`);
  }
}
const engine = path.resolve(args.get('engine'));
const child = spawn(engine, [], { stdio: ['pipe', 'pipe', 'pipe'] });
const lines = [];
let pending = '',
  cursor = 0,
  failure = null,
  stderr = '';
child.on('error', (error) => {
  failure = error;
});
child.on('exit', (code, signal) => {
  failure = Error(`Native engine exited: ${code ?? signal}\n${stderr}`);
});
child.stdin.on('error', (error) => {
  failure = error;
});
child.stderr.on('data', (chunk) => {
  stderr = (stderr + chunk).slice(-10000);
});
child.stdout.setEncoding('utf8');
child.stdout.on('data', (chunk) => {
  const parts = (pending + chunk).split('\n');
  pending = parts.pop();
  lines.push(...parts.map((line) => line.trim()));
});
async function wait(predicate, label) {
  const started = performance.now();
  while (performance.now() - started < 120000) {
    for (; cursor < lines.length; cursor++) {
      if (predicate(lines[cursor])) return lines[cursor++];
    }
    if (failure) throw failure;
    await delay(5);
  }
  throw Error(`Timed out waiting for ${label}\n${stderr}`);
}
const command = (text) => child.stdin.write(`${text}\n`);
const rows = [];
try {
  command('uci');
  await wait((s) => s === 'uciok', 'uciok');
  command('setoption name Threads value 1');
  command(`setoption name Hash value ${hash}`);
  command('isready');
  await wait((s) => s === 'readyok', 'readyok');
  for (const row of requests) {
    command('ucinewgame');
    command('isready');
    await wait((s) => s === 'readyok', 'pre-search readyok');
    const fen = ROTATED_FEN_SUITE.find(([label]) => label === row.position)[1];
    command(`position fen ${fen}`);
    const start = lines.length;
    command(`go depth ${row.depth}`);
    const best = await wait((s) => s.startsWith('bestmove '), 'bestmove');
    const info = lines
      .slice(start)
      .filter((s) => s.startsWith('info ') && s.includes(' pv '))
      .at(-1);
    const measured = { position: row.position, bestmove: best.split(/\s+/)[1], ...parseInfo(info ?? '') };
    if (measured.depth !== row.depth || !exactSearch(measured, measured)) throw Error(`Incomplete native search: ${JSON.stringify(measured)}`);
    rows.push(measured);
  }
  const parity = {};
  for (const artifact of new Set(reference.rows.map((r) => r.artifact))) {
    const compared = reference.rows.filter((r) => r.artifact === artifact);
    const mismatches = compared.flatMap((candidate) => {
      const native = rows.find((r) => r.position === candidate.position && r.depth === candidate.depth);
      return exactSearch(native, candidate) ? [] : [{ native, candidate }];
    });
    parity[artifact] = { pairs: compared.length, exact: compared.length - mismatches.length, mismatches };
  }
  const report = {
    engine,
    engineSha256: createHash('sha256').update(fs.readFileSync(engine)).digest('hex'),
    identity: lines.filter((s) => s.startsWith('id ')),
    hash,
    threads: 1,
    parity,
    rows,
  };
  fs.writeFileSync(args.get('json'), `${JSON.stringify(report, null, 2)}\n`);
  console.log(JSON.stringify(parity, null, 2));
  if (Object.values(parity).some((p) => p.exact !== p.pairs)) process.exitCode = 1;
} finally {
  child.stdin.end();
  child.kill();
}
