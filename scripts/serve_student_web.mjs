#!/usr/bin/env node
import { createServer } from 'node:http';
import { readFileSync } from 'node:fs';
import { join, normalize } from 'node:path';
import { parseFen, START_FEN, boardToFen, squareName } from '../src/chess/board.ts';
import { legalMoves, makeMove } from '../src/chess/movegen.ts';
import { moveToActionId, moveToUci } from '../src/chess/moveCodec.ts';
import { chooseMove } from '../src/search/puct.ts';
import { StudentEvaluator } from '../src/nn/studentEvaluator.ts';
import { rustChooseMove, rustPolicyForBoard } from './rust_engine.mjs';

function arg(name, fallback = undefined) {
  const prefix = `${name}=`;
  const inline = process.argv.find((value) => value.startsWith(prefix));
  if (inline) return inline.slice(prefix.length);
  const i = process.argv.indexOf(name);
  return i >= 0 ? process.argv[i + 1] : fallback;
}

const modelPath = arg('--model', 'artifacts/student_distill_benchmark.json');
const backend = arg('--backend', process.env.TINY_LEELA_BACKEND ?? 'rust');
const port = Number(arg('--port', process.env.PORT ?? '5173'));
const host = arg('--host', process.env.HOST ?? '127.0.0.1');
const evaluator = StudentEvaluator.fromJson(readFileSync(modelPath, 'utf8'));
const searchVisits = Number(arg('--visits', process.env.TINY_LEELA_SEARCH_VISITS ?? '8'));
const chessgroundCss = [
  readFileSync('node_modules/chessground/assets/chessground.base.css', 'utf8'),
  readFileSync('node_modules/chessground/assets/chessground.brown.css', 'utf8'),
  readFileSync('node_modules/chessground/assets/chessground.cburnett.css', 'utf8'),
].join('\n');
let board = parseFen(arg('--fen', START_FEN));
let lastEngine = null;
let lastMessage = 'Ready. Drag a piece on the Chessground board, or click a legal source/target.';

function currentLegalMoves() {
  return legalMoves(board);
}

function legalMoveByUci(uci) {
  return currentLegalMoves().find((move) => moveToUci(move) === uci) ?? null;
}

function boardPayload() {
  const evaluation = backend === 'ts' ? evaluator.evaluate(board) : rustPolicyForBoard(board, { model: modelPath, visits: searchVisits, temperature: 1 });
  const rustPrior = new Map((evaluation.policy ?? []).map((entry) => [moveToUci(entry.move), entry.prior ?? entry.probability ?? 0]));
  const legal = currentLegalMoves();
  const moves = legal.map((move) => ({
    uci: moveToUci(move),
    from: squareName(move.from),
    to: squareName(move.to),
    prior: backend === 'ts' ? (evaluation.policy.get(moveToActionId(move)) ?? 0) : (rustPrior.get(moveToUci(move)) ?? 0),
  })).sort((a, b) => b.prior - a.prior);
  return {
    fen: boardToFen(board),
    turn: board.turn,
    legalMoves: moves,
    backend,
    wdl: evaluation.wdl,
    value: evaluation.wdl[0] - evaluation.wdl[2],
    lastEngine,
    message: lastMessage,
  };
}

async function enginePly() {
  const result = backend === 'ts' ? await chooseMove(board, evaluator, { visits: searchVisits }) : rustChooseMove(board, { model: modelPath, visits: searchVisits });
  if (!result.move) {
    lastEngine = null;
    lastMessage = 'No engine move available.';
    return;
  }
  lastEngine = moveToUci(result.move);
  board = makeMove(board, result.move);
  lastMessage = `${backend.toUpperCase()} engine played ${lastEngine}.`;
}

async function handleApi(req, res, url) {
  try {
    if (url.pathname === '/api/state' && req.method === 'GET') return json(res, boardPayload());
    if (url.pathname === '/api/reset' && req.method === 'POST') {
      board = parseFen(START_FEN);
      lastEngine = null;
      lastMessage = 'Reset to start position.';
      return json(res, boardPayload());
    }
    if (url.pathname === '/api/fen' && req.method === 'POST') {
      const body = await readJson(req);
      board = parseFen(String(body.fen ?? START_FEN));
      lastEngine = null;
      lastMessage = 'Loaded FEN.';
      return json(res, boardPayload());
    }
    if (url.pathname === '/api/move' && req.method === 'POST') {
      const body = await readJson(req);
      const uci = String(body.uci ?? '');
      const move = legalMoveByUci(uci);
      if (!move) return json(res, { error: `Illegal move: ${uci}`, ...boardPayload() }, 400);
      board = makeMove(board, move);
      lastMessage = `You played ${uci}.`;
      if (body.engine !== false) await enginePly();
      return json(res, boardPayload());
    }
    if (url.pathname === '/api/engine' && req.method === 'POST') {
      await enginePly();
      return json(res, boardPayload());
    }
    json(res, { error: 'Not found' }, 404);
  } catch (err) {
    json(res, { error: err.message }, 400);
  }
}

function readJson(req) {
  return new Promise((resolve, reject) => {
    let data = '';
    req.on('data', (chunk) => { data += chunk; });
    req.on('end', () => {
      try { resolve(data ? JSON.parse(data) : {}); }
      catch (err) { reject(err); }
    });
    req.on('error', reject);
  });
}

function json(res, value, status = 200) {
  res.writeHead(status, { 'content-type': 'application/json; charset=utf-8' });
  res.end(JSON.stringify(value));
}

function serveChessgroundAsset(res, pathname) {
  const rel = pathname.replace('/vendor/chessground/', '');
  const root = normalize('node_modules/chessground');
  const target = normalize(join(root, rel));
  if (!target.startsWith(root) || !/^dist\/[\w.-]+\.js$/.test(rel)) {
    res.writeHead(404).end('not found');
    return;
  }
  res.writeHead(200, { 'content-type': 'text/javascript; charset=utf-8', 'cache-control': 'public, max-age=3600' });
  res.end(readFileSync(target));
}

const html = String.raw`<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Tiny Leela Student</title>
<style>
${chessgroundCss}
  :root { color-scheme: dark; --bg:#0b1020; --panel:#111827; --line:#334155; --text:#f8fafc; --muted:#94a3b8; --accent:#38bdf8; --good:#34d399; }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; background: radial-gradient(circle at top left, #2563eb55, transparent 34rem), radial-gradient(circle at bottom right, #0f766e55, transparent 30rem), var(--bg); color: var(--text); }
  main { max-width: 1180px; margin: 0 auto; padding: 24px; }
  h1 { margin: 0 0 6px; font-size: clamp(30px, 5vw, 54px); letter-spacing: -0.04em; }
  .sub { margin: 0 0 22px; color: var(--muted); }
  .layout { display: grid; grid-template-columns: minmax(320px, 620px) minmax(280px, 1fr); gap: 22px; align-items: start; }
  .board-shell { width: min(620px, calc(100vw - 48px)); }
  #ground { width: 100%; aspect-ratio: 1 / 1; border-radius: 18px; overflow: hidden; box-shadow: 0 24px 70px #0009; background: #312e2b; }
  .cg-wrap { width: 100%; height: 100%; }
  .cg-wrap coords { font-weight: 800; text-shadow: 0 1px 2px #0009; }
  .panel { background: #0f172acc; border: 1px solid var(--line); border-radius: 18px; padding: 16px; box-shadow: 0 12px 36px #0006; backdrop-filter: blur(8px); }
  .row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 12px; }
  button, input { border-radius: 12px; border: 1px solid var(--line); background: var(--panel); color: var(--text); padding: 10px 12px; font: inherit; }
  button { cursor: pointer; } button:hover { border-color: var(--accent); transform: translateY(-1px); }
  input { width: 100%; }
  code, .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  .fen { overflow-wrap: anywhere; color: var(--muted); }
  .message { color: var(--good); min-height: 1.4em; }
  .wdl { display: grid; gap: 8px; margin: 14px 0; }
  .bar { display: grid; grid-template-columns: 44px 1fr 54px; gap: 8px; align-items: center; color: var(--muted); }
  .track { height: 10px; background: #020617; border-radius: 999px; overflow: hidden; }
  .fill { height: 100%; background: linear-gradient(90deg, var(--accent), #a78bfa); }
  ol { margin: 8px 0 0; padding-left: 24px; max-height: 260px; overflow: auto; }
  li { padding: 3px 0; color: var(--muted); } li b { color: var(--text); }
  @media (max-width: 900px) { .layout { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<main>
  <h1>Tiny Leela Student</h1>
  <p class="sub">Now using lichess Chessground for proper drag/drop, coordinates, highlights, animations, and a non-cursed board.</p>
  <div class="layout">
    <div class="board-shell"><div id="ground" aria-label="Chessground board"></div></div>
    <section class="panel">
      <div class="row">
        <button id="engine">Engine move</button>
        <button id="reset">Reset</button>
        <button id="flip">Flip board</button>
      </div>
      <div class="message" id="message"></div>
      <p><b>Backend</b> <span class="mono" id="backend"></span></p>
      <p><b>FEN</b></p>
      <p class="fen mono" id="fen"></p>
      <div class="row">
        <input id="fenInput" placeholder="Paste FEN..." />
        <button id="loadFen">Load FEN</button>
      </div>
      <p><b>Value / WDL</b></p>
      <div class="wdl" id="wdl"></div>
      <p><b>Top legal policy moves</b></p>
      <ol id="moves"></ol>
    </section>
  </div>
</main>
<script type="module">
import { Chessground } from '/vendor/chessground/dist/chessground.js';
let state = null;
let orientation = 'white';
let ground = null;
const fenEl = document.getElementById('fen');
const msgEl = document.getElementById('message');
const movesEl = document.getElementById('moves');
const wdlEl = document.getElementById('wdl');

async function api(path, body) {
  const res = await fetch(path, { method: body === undefined ? 'GET' : 'POST', headers: { 'content-type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body) });
  const json = await res.json();
  if (!res.ok) msgEl.textContent = json.error || 'Request failed';
  state = json;
  render();
}
function boardFen() { return state.fen.split(' ')[0]; }
function dests() {
  const map = new Map();
  for (const move of state.legalMoves) {
    if (!map.has(move.from)) map.set(move.from, []);
    map.get(move.from).push(move.to);
  }
  return map;
}
function lastMove() {
  const move = state.lastEngine;
  return move && move.length >= 4 ? [move.slice(0, 2), move.slice(2, 4)] : undefined;
}
async function userMove(orig, dest) {
  const candidates = state.legalMoves.filter((m) => m.from === orig && m.to === dest);
  const chosen = candidates.find((m) => m.uci.endsWith('q')) ?? candidates[0];
  if (!chosen) { render(); return; }
  await api('/api/move', { uci: chosen.uci, engine: true });
}
function renderGround() {
  const config = {
    fen: boardFen(),
    orientation,
    coordinates: true,
    highlight: { lastMove: true, check: true },
    animation: { enabled: true, duration: 180 },
    movable: {
      free: false,
      color: state.turn === 'w' ? 'white' : 'black',
      dests: dests(),
      showDests: true,
      events: { after: userMove },
    },
    lastMove: lastMove(),
  };
  if (!ground) ground = Chessground(document.getElementById('ground'), config);
  else ground.set(config);
}
function renderWdl() {
  const labels = ['Win', 'Draw', 'Loss'];
  wdlEl.innerHTML = state.wdl.map((v, i) => '<div class="bar"><span>'+labels[i]+'</span><div class="track"><div class="fill" style="width:'+Math.round(v*100)+'%"></div></div><span>'+v.toFixed(3)+'</span></div>').join('') + '<div class="mono">value ' + state.value.toFixed(3) + '</div>';
}
function renderMoves() {
  movesEl.innerHTML = state.legalMoves.slice(0, 16).map(m => '<li><b>'+m.uci+'</b> <span class="mono">'+m.prior.toFixed(4)+'</span></li>').join('');
}
function render() {
  if (!state) return;
  fenEl.textContent = state.fen;
  document.getElementById('backend').textContent = state.backend || 'unknown';
  msgEl.textContent = state.message || '';
  document.getElementById('fenInput').value = state.fen;
  renderGround(); renderWdl(); renderMoves();
}
document.getElementById('engine').onclick = () => api('/api/engine', {});
document.getElementById('reset').onclick = () => api('/api/reset', {});
document.getElementById('flip').onclick = () => { orientation = orientation === 'white' ? 'black' : 'white'; render(); };
document.getElementById('loadFen').onclick = () => api('/api/fen', { fen: document.getElementById('fenInput').value });
api('/api/state');
</script>
</body>
</html>`;

const server = createServer((req, res) => {
  const url = new URL(req.url ?? '/', `http://${req.headers.host ?? 'localhost'}`);
  if (url.pathname.startsWith('/api/')) return void handleApi(req, res, url);
  if (url.pathname.startsWith('/vendor/chessground/')) return void serveChessgroundAsset(res, url.pathname);
  res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' });
  res.end(html);
});

server.listen(port, host, () => {
  const address = server.address();
  const shownHost = host === '0.0.0.0' ? '127.0.0.1' : host;
  console.log(`Tiny Leela web UI: http://${shownHost}:${address.port}`);
});
