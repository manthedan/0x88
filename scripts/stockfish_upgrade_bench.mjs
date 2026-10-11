#!/usr/bin/env node
// Run Mac timings under benchlock; on Yukon taskset the entire process to 0-2.
// Cells: {label, command, args, parityGroup?}. The first cell in each group is
// its native SAME-SOURCE/SAME-NET reference, never the old-release speed cell.
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { ROTATED_FEN_SUITE, exactSearch, parseBestmove, parseUciSearch, roundOrder } from './emscripten_uci_protocol.mjs';

export function benchSignature(lines) {
  let nodes = null;
  for (const summary of lines.filter((line) => /^\s*Nodes\s+searched\b/.test(line))) {
    const match = summary.match(/^\s*Nodes searched\s*:\s*(\d+)\s*$/);
    if (!match || !Number.isSafeInteger(Number(match[1])) || Number(match[1]) <= 0) {
      throw new Error(`Invalid native bench node summary: ${summary}`);
    }
    nodes = match[1];
  }
  return nodes;
}

export function benchDefaults(source) {
  const block = source.match(/const std::vector<std::string> Defaults = \{([\s\S]*?)\n\};/)?.[1];
  if (!block) throw new Error('Upstream bench Defaults changed');
  const entries = [...block.matchAll(/^\s*"([^"]+)"/gm)].map((m) => m[1]);
  if (entries.filter((s) => !s.startsWith('setoption ')).length !== 51) throw new Error('Unexpected SF19 bench position count');
  return entries;
}

// Reviewed terminal boards in pinned Stockfish Defaults. New terminal corpus
// entries need adapter review; a null bestmove alone is never terminal proof.
const terminalBenchPositions = new Set([
  '8/8/8/8/8/6k1/6p1/6K1 w - -',
  '7k/7P/6K1/8/3B4/8/8/8 b - -',
]);
function isTerminalBenchPosition(entry) {
  const fields = entry.trim().split(/\s+/);
  return (fields.length === 4 || (fields.length === 6 && /^\d+$/.test(fields[4]) && /^\d+$/.test(fields[5])))
    && terminalBenchPositions.has(fields.slice(0, 4).join(' '));
}

function replaySearch(lines, best, entry) {
  const bestmove = parseBestmove(best);
  // Select node-bearing records, not just records whose numeric prefix parses.
  // A corrupt final count must never cause fallback to an earlier info line.
  const infos = lines.filter((line) => /^info\s/.test(line) && !/^info\s+string\b/.test(line)
    && line.trim().split(/\s+/).includes('nodes'));
  const terminal = isTerminalBenchPosition(entry);
  const noMove = ['0000', '(none)'].includes(bestmove);
  if (noMove !== terminal) throw new Error(`Invalid terminal bestmove for bench position: ${entry}`);
  if (!infos.length && !terminal) throw new Error(`Missing bench nodes: ${entry}`);
  let nodes = 0;
  for (const info of infos) {
    const tokens = info.trim().split(/\s+/);
    const token = tokens[tokens.indexOf('nodes') + 1];
    if (tokens.filter((part) => part === 'nodes').length !== 1 || !/^\d+$/.test(token ?? '')
      || !Number.isSafeInteger(Number(token))) throw new Error(`Malformed replay nodes record: ${info}`);
    nodes = Number(token);
  }
  if (nodes === 0 && !terminal) throw new Error(`Missing/zero bench nodes for nonterminal position: ${entry}`);
  return { nodes, bestmove };
}

function validateCompletions(lines, expected, phase) {
  const bestLines = lines.filter((line) => /^bestmove(?:\s|$)/.test(line));
  for (const line of bestLines) parseBestmove(line);
  if (bestLines.length !== expected) {
    throw new Error(`Unexpected ${phase} bestmove coverage: expected ${expected}, got ${bestLines.length}`);
  }
}

export async function runCell(cell, { depth = 9, benchDepth = 8, positions = 20, threads = 1, timeout = 180000, defaults } = {}) {
  const started = performance.now();
  const child = spawn(cell.command, cell.args || [], { stdio: ['pipe', 'pipe', 'pipe'] });
  const lines = [];
  let failure;
  let cursor = 0;
  const pending = new Set();
  let exited = false;
  let completed = false;
  let exitStatus;
  let expectedSearches = 0;
  let replayStart = 0;
  let nativeBenchStart;
  const closed = new Promise((done) => child.once('close', done));
  const fail = (err) => {
    failure = err;
    for (const notify of pending) notify();
  };
  child.on('error', (error) => { exited = true; fail(error); });
  child.stdin.on('error', fail);
  child.on('exit', (code, signal) => {
    exited = true;
    exitStatus = { code, signal };
    fail(new Error(`${cell.label} exited (${code}, ${signal}): ${lines.slice(-8).join('\n')}`));
  });
  for (const stream of [child.stdout, child.stderr]) {
    let buffer = '';
    stream.setEncoding('utf8');
    stream.on('data', (chunk) => {
      buffer += chunk;
      const parts = buffer.split(/\r?\n/);
      buffer = parts.pop();
      lines.push(...parts);
      for (const notify of pending) notify();
    });
    stream.on('end', () => {
      if (buffer) { lines.push(buffer); buffer = ''; }
      for (const notify of pending) notify();
    });
  }
  const command = (text) => child.stdin.write(`${text}\n`);
  const wait = (predicate) => new Promise((accept, reject) => {
    const timer = setTimeout(() => { pending.delete(check); reject(new Error(`${cell.label} timed out: ${lines.slice(-8).join('\n')}`)); }, timeout);
    const check = () => {
      for (; cursor < lines.length; cursor += 1) {
        if (predicate(lines[cursor])) {
          const line = lines[cursor++];
          clearTimeout(timer);
          pending.delete(check);
          accept(line);
          return;
        }
      }
      if (failure) { clearTimeout(timer); pending.delete(check); reject(failure); }
    };
    pending.add(check);
    check();
  });
  try {
    command('uci');
    await wait((l) => l === 'uciok');
    command(`setoption name Threads value ${threads}`);
    command('setoption name Hash value 16');
    command('isready');
    await wait((l) => l === 'readyok');
    const startupMs = performance.now() - started;
    let nativeSignature;
    if (cell.nativeBench) {
      nativeBenchStart = lines.length;
      command(`bench 16 1 ${benchDepth} default depth`);
      await wait((l) => /Nodes searched\s*:/.test(l));
      command('isready');
      await wait((l) => l === 'readyok');
      nativeSignature = benchSignature(lines.slice(nativeBenchStart));
      validateCompletions(lines.slice(nativeBenchStart), defaults.filter((entry) => !entry.startsWith('setoption ')).length, 'native built-in bench');
    }
    // Stockfish.js compiles bench out. Replay setup_bench's exact command
    // sequence, with ONE reset before the suite (not one reset per position).
    command('setoption name Threads value 1');
    command('setoption name Hash value 16');
    command('ucinewgame');
    command('isready');
    await wait((l) => l === 'readyok');
    // Built-in Stockfish bench also emits bestmoves. Its searches precede
    // this replay and must not count against replay completion coverage.
    replayStart = lines.length;
    let benchNodes = 0;
    const benchRows = [];
    for (const entry of defaults) {
      if (entry.startsWith('setoption ')) { command(entry); continue; }
      const from = lines.length;
      command(`position fen ${entry}`);
      expectedSearches += 1;
      command(`go depth ${benchDepth}`);
      const best = await wait((l) => /^bestmove(?:\s|$)/.test(l));
      const search = replaySearch(lines.slice(from, cursor - 1), best, entry);
      benchNodes += search.nodes;
      if (!Number.isSafeInteger(benchNodes)) throw new Error('Invalid replay bench node total');
      benchRows.push({ fen: entry, ...search });
    }
    const signature = String(benchNodes);
    if (!benchNodes || (cell.nativeBench && nativeSignature !== signature)) throw Error(`Bench replay mismatch: native=${nativeSignature}, replay=${signature}`);
    command(`setoption name Threads value ${threads}`);
    const rows = [];
    // Rotate each round's positions without changing the reference suite.
    const suite = ROTATED_FEN_SUITE.slice(0, positions);
    const offset = (cell.round || 0) % suite.length;
    for (const [label, fen] of [...suite.slice(offset), ...suite.slice(0, offset)]) {
      command('ucinewgame');
      command('isready');
      await wait((l) => l === 'readyok');
      const infoStart = lines.length;
      command(`position fen ${fen}`);
      const t = performance.now();
      expectedSearches += 1;
      command(`go depth ${depth}`);
      const best = await wait((l) => /^bestmove(?:\s|$)/.test(l));
      const wallMs = performance.now() - t;
      const info = parseUciSearch([...lines.slice(infoStart, cursor - 1), best], depth);
      rows.push({ position: label, ...info, wallMs });
    }
    completed = true;
    return { ...cell, startupMs, benchDepth, benchSignature: signature, nativeSignature, benchRows, depth, threads, hash: 16, rows,
      wallNps: rows.reduce((n, r) => n + r.nodes, 0) * 1000 / rows.reduce((n, r) => n + r.wallMs, 0) };
  } finally {
    if (!exited) {
      child.stdin.end('quit\n');
      const timer = setTimeout(() => child.kill('SIGKILL'), 2000);
      await closed;
      clearTimeout(timer);
    } else await closed;
    if (completed) {
      if (exitStatus?.code !== 0 || exitStatus?.signal) {
        throw new Error(`${cell.label} replay engine exited abnormally (${exitStatus?.code}, ${exitStatus?.signal})`);
      }
      // Drain the complete transcript, including output after the final search
      // and partial final lines. A first bestmove is not a license to ignore
      // malformed or surplus completion records before a clean shutdown.
      if (cell.nativeBench) {
        validateCompletions(lines.slice(nativeBenchStart, replayStart), defaults.filter((entry) => !entry.startsWith('setoption ')).length, 'native built-in bench');
      }
      validateCompletions(lines.slice(replayStart), expectedSearches, 'replay');
    }
  }
}

export function parityReport(cells) {
  const result = [];
  for (const group of [...new Set(cells.map((c) => c.parityGroup).filter(Boolean))]) {
    const subset = cells.filter((c) => c.parityGroup === group);
    const ref = subset[0];
    for (const cell of subset.slice(1)) {
      const mismatches = cell.rows.filter((row) => {
        const baseline = ref.rows.find((r) => r.position === row.position && r.depth === row.depth);
        return !baseline || !exactSearch(baseline, row);
      });
      result.push({ group, reference: ref.label, candidate: cell.label, round: cell.round,
        benchMatch: ref.benchSignature === cell.benchSignature, pairs: cell.rows.length,
        exact: cell.rows.length - mismatches.length, mismatches });
    }
  }
  return result;
}

async function main() {
  const args = new Map();
  for (let i = 2; i < process.argv.length; i += 2) args.set(process.argv[i].replace(/^--/, ''), process.argv[i + 1]);
  const config = JSON.parse(readFileSync(args.get('config'), 'utf8'));
  const rounds = Number(args.get('rounds') || 3);
  const defaults = benchDefaults(readFileSync(args.get('bench-source') || '.local_engines/stockfish-js-19-src/src/benchmark.cpp', 'utf8'));
  const options = { depth: Number(args.get('depth') || 9), benchDepth: Number(args.get('bench-depth') || 8),
    positions: Number(args.get('positions') || 20), threads: Number(args.get('threads') || 1) };
  for (const [key, n] of Object.entries({ rounds, ...options })) {
    if (!Number.isInteger(n) || n < 1 || (key === 'positions' && n > ROTATED_FEN_SUITE.length)) throw Error(`Invalid ${key}`);
  }
  const report = { runtime: process.version, platform: process.platform, arch: process.arch, options, rounds, cells: [], parity: [] };
  for (const cell of config) {
    if (cell.wasm) cell.wasmSha256 = createHash('sha256').update(readFileSync(cell.wasm)).digest('hex');
  }
  for (let round = 0; round < rounds; round += 1) {
    for (const cell of roundOrder(config, round)) {
      const measured = await runCell({ ...cell, round }, { ...options, defaults });
      report.cells.push(measured);
      console.log(`${round} ${cell.label}: bench=${measured.benchSignature}, NPS=${Math.round(measured.wallNps)}, startup=${Math.round(measured.startupMs)}ms`);
      writeFileSync(args.get('json'), `${JSON.stringify(report, null, 2)}\n`);
    }
  }
  report.parity = parityReport(report.cells);
  writeFileSync(args.get('json'), `${JSON.stringify(report, null, 2)}\n`);
  if (report.parity.some((p) => !p.benchMatch || p.exact !== p.pairs)) process.exitCode = 1;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) await main();
