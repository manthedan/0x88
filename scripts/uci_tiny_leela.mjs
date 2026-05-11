#!/usr/bin/env node
// UCI wrapper for Tiny Leela ONNX models.
//
// Scope: robust enough for local anchors/OpenBench smoke tests.  It supports the
// standard handshake, position/startpos/FEN, ucinewgame, isready, go
// depth|nodes|movetime, stop, and quit.  Search is Tiny Leela PUCT; depth is
// interpreted as visits when nodes is absent.

import fs from 'node:fs';
import readline from 'node:readline';
import { parseFen, START_FEN, boardToFen } from '../src/chess/board.ts';
import { legalMoves, makeMove } from '../src/chess/movegen.ts';
import { moveFromUci, moveToUci } from '../src/chess/moveCodec.ts';
import { chooseMove, classicPuctPolicy, auxPuctPolicy, actionValuePuctPolicy } from '../src/search/puct.ts';
import { OnnxEvaluator } from '../src/nn/onnxEvaluator.ts';
import { SquareFormerEvaluator } from '../src/nn/squareformerEvaluator.ts';

const argv = process.argv.slice(2);
function arg(name, fallback = undefined) {
  const i = argv.indexOf(name);
  return i >= 0 && i + 1 < argv.length ? argv[i + 1] : fallback;
}
function flag(name) { return argv.includes(name); }

const opts = {
  model: arg('--model', process.env.TINY_LEELA_MODEL || ''),
  meta: arg('--meta', process.env.TINY_LEELA_META || ''),
  visits: Number(arg('--visits', process.env.TINY_LEELA_VISITS || '64')),
  batchSize: Number(arg('--batch-size', process.env.TINY_LEELA_BATCH_SIZE || '8')),
  cpuct: Number(arg('--cpuct', process.env.TINY_LEELA_CPUCT || '1.5')),
  mode: arg('--mode', process.env.TINY_LEELA_SEARCH_MODE || 'puct'),
  avWeight: Number(arg('--av-weight', process.env.TINY_LEELA_AV_WEIGHT || '0')),
  rankWeight: Number(arg('--rank-weight', process.env.TINY_LEELA_RANK_WEIGHT || '0')),
  regretWeight: Number(arg('--regret-weight', process.env.TINY_LEELA_REGRET_WEIGHT || '0')),
  riskWeight: Number(arg('--risk-weight', process.env.TINY_LEELA_RISK_WEIGHT || '0')),
  uncertaintyWeight: Number(arg('--uncertainty-weight', process.env.TINY_LEELA_UNCERTAINTY_WEIGHT || '0')),
  name: arg('--name', 'TinyLeelaONNX'),
  debug: flag('--debug'),
};

let board = parseFen(START_FEN);
let historyFens = [];
let evaluatorPromise = null;
let searching = false;
let stopped = false;

function log(...xs) { if (opts.debug) console.error('[uci_tiny_leela]', ...xs); }
function send(s) { process.stdout.write(`${s}\n`); }

async function loadEvaluator() {
  if (evaluatorPromise) return evaluatorPromise;
  evaluatorPromise = (async () => {
    if (!opts.model || !opts.meta) throw new Error('Set --model and --meta or TINY_LEELA_MODEL/TINY_LEELA_META');
    if (!fs.existsSync(opts.model)) throw new Error(`model missing: ${opts.model}`);
    if (!fs.existsSync(opts.meta)) throw new Error(`meta missing: ${opts.meta}`);
    const meta = JSON.parse(fs.readFileSync(opts.meta, 'utf8'));
    log('loading', opts.model, opts.meta, meta.kind || meta.architecture || 'unknown');
    if (meta.kind === 'squareformer' || meta.kind === 'squareformer_v2') return SquareFormerEvaluator.create(opts.model, meta);
    return OnnxEvaluator.create(opts.model, meta);
  })();
  return evaluatorPromise;
}

function searchPolicy() {
  if (opts.mode === 'aux') return auxPuctPolicy;
  if (opts.mode === 'av') return actionValuePuctPolicy;
  return classicPuctPolicy;
}

function setOption(line) {
  const m = /^setoption\s+name\s+(.+?)(?:\s+value\s+(.*))?$/i.exec(line);
  if (!m) return;
  const name = m[1].trim().toLowerCase().replace(/[ _-]/g, '');
  const value = (m[2] ?? '').trim();
  if (name === 'model') { opts.model = value; evaluatorPromise = null; }
  else if (name === 'meta') { opts.meta = value; evaluatorPromise = null; }
  else if (name === 'visits' || name === 'nodes') opts.visits = Math.max(1, Number(value) || opts.visits);
  else if (name === 'batchsize') opts.batchSize = Math.max(1, Number(value) || opts.batchSize);
  else if (name === 'threads' || name === 'hash' || name === 'clearhash' || name === 'ponder' || name === 'ucianalysemode') { /* accepted for UCI GUI/OpenBench compatibility; search is controlled by JS/ORT env. */ }
  else if (name === 'cpuct') opts.cpuct = Number(value) || opts.cpuct;
  else if (name === 'searchmode') opts.mode = value || opts.mode;
  else if (name === 'avweight') opts.avWeight = Number(value) || 0;
  else if (name === 'rankweight') opts.rankWeight = Number(value) || 0;
  else if (name === 'regretweight') opts.regretWeight = Number(value) || 0;
  else if (name === 'riskweight') opts.riskWeight = Number(value) || 0;
  else if (name === 'uncertaintyweight') opts.uncertaintyWeight = Number(value) || 0;
  log('setoption', name, value);
}

function applyMoveUci(uci) {
  const legal = legalMoves(board);
  const move = legal.find((m) => moveToUci(m) === uci) ?? moveFromUci(uci);
  if (!legal.some((m) => moveToUci(m) === moveToUci(move))) throw new Error(`illegal move ${uci} in ${boardToFen(board)}`);
  historyFens = [boardToFen(board), ...historyFens].slice(0, 16);
  board = makeMove(board, move);
}

function setPosition(line) {
  const parts = line.trim().split(/\s+/);
  let i = 1;
  historyFens = [];
  if (parts[i] === 'startpos') {
    board = parseFen(START_FEN);
    i++;
  } else if (parts[i] === 'fen') {
    const fenParts = [];
    i++;
    while (i < parts.length && parts[i] !== 'moves' && fenParts.length < 6) fenParts.push(parts[i++]);
    board = parseFen(fenParts.join(' '));
  } else {
    throw new Error(`bad position command: ${line}`);
  }
  if (parts[i] === 'moves') {
    i++;
    for (; i < parts.length; i++) applyMoveUci(parts[i]);
  }
}

function visitsFromGo(line) {
  const parts = line.trim().split(/\s+/);
  const getNum = (key) => {
    const i = parts.indexOf(key);
    return i >= 0 && i + 1 < parts.length ? Number(parts[i + 1]) : undefined;
  };
  const nodes = getNum('nodes');
  if (Number.isFinite(nodes) && nodes > 0) return Math.max(1, Math.floor(nodes));
  const depth = getNum('depth');
  if (Number.isFinite(depth) && depth > 0) return Math.max(1, Math.floor(depth));
  const movetime = getNum('movetime');
  if (Number.isFinite(movetime) && movetime > 0) return Math.max(1, Math.floor(Math.min(1024, movetime / 10)));
  const sideTime = board.turn === 'w' ? getNum('wtime') : getNum('btime');
  const sideInc = board.turn === 'w' ? getNum('winc') : getNum('binc');
  if (Number.isFinite(sideTime) && sideTime > 0) {
    const softMs = Math.max(10, Math.min(1000, sideTime / 30 + (Number.isFinite(sideInc) ? sideInc * 0.5 : 0)));
    return Math.max(1, Math.floor(Math.min(1024, softMs / 10)));
  }
  return opts.visits;
}

async function go(line) {
  if (searching) return;
  searching = true;
  stopped = false;
  try {
    const evaluator = await loadEvaluator();
    const visits = visitsFromGo(line);
    const result = await chooseMove(board, evaluator, {
      visits,
      batchSize: opts.batchSize,
      cpuct: opts.cpuct,
      historyFens,
      searchPolicy: searchPolicy(),
      avWeight: opts.avWeight,
      rankWeight: opts.rankWeight,
      regretWeight: opts.regretWeight,
      riskWeight: opts.riskWeight,
      uncertaintyWeight: opts.uncertaintyWeight,
    });
    const best = result.move ? moveToUci(result.move) : '0000';
    if (!stopped) send(`info string visits ${visits} value ${result.value.toFixed(4)} completed ${result.stats?.completedVisits ?? 0}`);
    send(`bestmove ${best}`);
  } catch (e) {
    console.error(`info string tiny-leela error: ${e?.stack || e}`);
    send('bestmove 0000');
  } finally {
    searching = false;
  }
}

function uci() {
  send(`id name ${opts.name}`);
  send('id author tiny-leela');
  send('option name Model type string default');
  send('option name Meta type string default');
  send(`option name Visits type spin default ${opts.visits} min 1 max 100000`);
  send(`option name BatchSize type spin default ${opts.batchSize} min 1 max 1024`);
  send(`option name Cpuct type string default ${opts.cpuct}`);
  send('option name Threads type spin default 1 min 1 max 128');
  send('option name Hash type spin default 16 min 1 max 4096');
  send('option name Ponder type check default false');
  send('option name SearchMode type combo default puct var puct var av var aux');
  send('option name AvWeight type string default 0');
  send('option name RankWeight type string default 0');
  send('option name RegretWeight type string default 0');
  send('option name RiskWeight type string default 0');
  send('option name UncertaintyWeight type string default 0');
  send('uciok');
}

const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
rl.on('line', async (raw) => {
  const line = raw.trim();
  if (!line) return;
  try {
    if (line === 'uci') uci();
    else if (line === 'isready') { await loadEvaluator().catch((e) => { console.error(`info string load error: ${e.message || e}`); }); send('readyok'); }
    else if (line === 'ucinewgame') { board = parseFen(START_FEN); historyFens = []; }
    else if (line.startsWith('setoption ')) setOption(line);
    else if (line.startsWith('position ')) setPosition(line);
    else if (line.startsWith('go')) void go(line);
    else if (line === 'stop') stopped = true;
    else if (line === 'ponderhit') { /* no-op */ }
    else if (line === 'quit') process.exit(0);
    else if (line === 'd') send(`info string fen ${boardToFen(board)}`);
  } catch (e) {
    console.error(`info string command error: ${e?.message || e}`);
  }
});
