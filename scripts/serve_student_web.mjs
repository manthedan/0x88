#!/usr/bin/env node
import { createServer } from 'node:http';
import { readFileSync } from 'node:fs';
import { parseFen, START_FEN, boardToFen, squareName } from '../src/chess/board.ts';
import { legalMoves, makeMove } from '../src/chess/movegen.ts';
import { moveToActionId, moveToUci } from '../src/chess/moveCodec.ts';
import { chooseMove } from '../src/search/puct.ts';
import { StudentEvaluator } from '../src/nn/studentEvaluator.ts';

function arg(name, fallback = undefined) {
  const prefix = `${name}=`;
  const inline = process.argv.find((value) => value.startsWith(prefix));
  if (inline) return inline.slice(prefix.length);
  const i = process.argv.indexOf(name);
  return i >= 0 ? process.argv[i + 1] : fallback;
}

const modelPath = arg('--model', 'artifacts/student_distill_benchmark.json');
const port = Number(arg('--port', process.env.PORT ?? '5173'));
const host = arg('--host', process.env.HOST ?? '127.0.0.1');
const evaluator = StudentEvaluator.fromJson(readFileSync(modelPath, 'utf8'));
const searchVisits = Number(arg('--visits', process.env.TINY_LEELA_SEARCH_VISITS ?? '8'));
let board = parseFen(arg('--fen', START_FEN));
let lastEngine = null;
let lastMessage = 'Ready. Click one of your pieces, then a highlighted target.';

function currentLegalMoves() {
  return legalMoves(board);
}

function legalMoveByUci(uci) {
  return currentLegalMoves().find((move) => moveToUci(move) === uci) ?? null;
}

function boardPayload() {
  const evaluation = evaluator.evaluate(board);
  const legal = currentLegalMoves();
  const moves = legal.map((move) => ({
    uci: moveToUci(move),
    from: squareName(move.from),
    to: squareName(move.to),
    prior: evaluation.policy.get(moveToActionId(move)) ?? 0,
  })).sort((a, b) => b.prior - a.prior);
  return {
    fen: boardToFen(board),
    turn: board.turn,
    squares: board.squares,
    legalMoves: moves,
    wdl: evaluation.wdl,
    value: evaluation.wdl[0] - evaluation.wdl[2],
    lastEngine,
    message: lastMessage,
  };
}

async function enginePly() {
  const result = await chooseMove(board, evaluator, { visits: searchVisits });
  if (!result.move) {
    lastEngine = null;
    lastMessage = 'No engine move available.';
    return;
  }
  lastEngine = moveToUci(result.move);
  board = makeMove(board, result.move);
  lastMessage = `Engine played ${lastEngine}.`;
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

const html = String.raw`<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Tiny Leela Student</title>
<style>
  :root { color-scheme: dark; --bg:#111827; --panel:#1f2937; --line:#374151; --text:#f9fafb; --muted:#9ca3af; --accent:#38bdf8; --good:#34d399; --bad:#fb7185; }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; background: radial-gradient(circle at top, #1e3a8a55, transparent 38rem), var(--bg); color: var(--text); }
  main { max-width: 1120px; margin: 0 auto; padding: 24px; }
  h1 { margin: 0 0 6px; font-size: clamp(28px, 5vw, 48px); }
  .sub { margin: 0 0 20px; color: var(--muted); }
  .layout { display: grid; grid-template-columns: minmax(300px, 560px) minmax(280px, 1fr); gap: 22px; align-items: start; }
  .board { width: min(560px, calc(100vw - 48px)); aspect-ratio: 1 / 1; display: grid; grid-template-columns: repeat(8, 1fr); grid-template-rows: repeat(8, 1fr); border: 2px solid #111; border-radius: 16px; overflow: hidden; box-shadow: 0 20px 60px #0008; }
  .sq { position: relative; width: 100%; height: 100%; min-width: 0; min-height: 0; padding: 0; border: 0; line-height: 1; font-size: clamp(24px, 7vw, 56px); cursor: pointer; display: grid; place-items: center; color: #111827; }
  .light { background: #f0d9b5; } .dark { background: #b58863; }
  .sq.selected { outline: 4px solid var(--accent); outline-offset: -4px; }
  .sq.target::after { content: ''; width: 28%; height: 28%; border-radius: 50%; background: #38bdf8bb; position: absolute; }
  .sq.capture::after { width: 72%; height: 72%; background: transparent; border: 5px solid #38bdf8bb; }
  .coord { position:absolute; left:5px; bottom:3px; font-size:11px; color:#11182799; font-weight:700; }
  .panel { background: #111827cc; border: 1px solid var(--line); border-radius: 16px; padding: 16px; box-shadow: 0 10px 30px #0005; }
  .row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 12px; }
  button, input { border-radius: 10px; border: 1px solid var(--line); background: var(--panel); color: var(--text); padding: 9px 11px; font: inherit; }
  button { cursor: pointer; } button:hover { border-color: var(--accent); }
  input { width: 100%; }
  code, .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  .fen { overflow-wrap: anywhere; color: var(--muted); }
  .message { color: var(--good); min-height: 1.4em; }
  .wdl { display: grid; gap: 8px; margin: 14px 0; }
  .bar { display: grid; grid-template-columns: 44px 1fr 54px; gap: 8px; align-items: center; color: var(--muted); }
  .track { height: 10px; background: #020617; border-radius: 999px; overflow: hidden; }
  .fill { height: 100%; background: var(--accent); }
  ol { margin: 8px 0 0; padding-left: 24px; max-height: 260px; overflow: auto; }
  li { padding: 3px 0; color: var(--muted); } li b { color: var(--text); }
  @media (max-width: 820px) { .layout { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<main>
  <h1>Tiny Leela Student</h1>
  <p class="sub">A tiny distilled policy/WDL student wired to the current move generator. Click a legal move; the engine replies automatically.</p>
  <div class="layout">
    <div id="board" class="board" aria-label="Chess board"></div>
    <section class="panel">
      <div class="row">
        <button id="engine">Engine move</button>
        <button id="reset">Reset</button>
        <button id="flip">Flip board</button>
      </div>
      <div class="message" id="message"></div>
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
<script>
const pieces = { wp:'♙', wn:'♘', wb:'♗', wr:'♖', wq:'♕', wk:'♔', bp:'♟', bn:'♞', bb:'♝', br:'♜', bq:'♛', bk:'♚' };
let state = null, selected = null, flipped = false;
const boardEl = document.getElementById('board');
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
function idx(file, rank) { return file + rank * 8; }
function sqName(i) { return String.fromCharCode(97 + (i % 8)) + (Math.floor(i / 8) + 1); }
function legalFrom(square) { return state.legalMoves.filter(m => m.from === square); }
function legalTo(square) { return selected ? legalFrom(selected).filter(m => m.to === square) : []; }
async function clickSquare(i) {
  const sq = sqName(i);
  if (selected) {
    const moves = legalTo(sq);
    if (moves.length) { selected = null; await api('/api/move', { uci: moves[0].uci, engine: true }); return; }
  }
  if (legalFrom(sq).length) selected = sq;
  else selected = null;
  render();
}
function renderBoard() {
  boardEl.innerHTML = '';
  const ranks = flipped ? [0,1,2,3,4,5,6,7] : [7,6,5,4,3,2,1,0];
  const files = flipped ? [7,6,5,4,3,2,1,0] : [0,1,2,3,4,5,6,7];
  for (const r of ranks) for (const f of files) {
    const i = idx(f, r), sq = sqName(i), piece = state.squares[i];
    const b = document.createElement('button');
    b.className = 'sq ' + ((f + r) % 2 ? 'dark' : 'light');
    if (selected === sq) b.classList.add('selected');
    const targets = legalTo(sq);
    if (targets.length) b.classList.add(piece ? 'capture' : 'target');
    b.innerHTML = (pieces[piece] || '') + '<span class="coord">' + sq + '</span>';
    b.onclick = () => clickSquare(i);
    boardEl.appendChild(b);
  }
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
  msgEl.textContent = state.message || '';
  document.getElementById('fenInput').value = state.fen;
  renderBoard(); renderWdl(); renderMoves();
}
document.getElementById('engine').onclick = () => api('/api/engine', {});
document.getElementById('reset').onclick = () => { selected = null; api('/api/reset', {}); };
document.getElementById('flip').onclick = () => { flipped = !flipped; render(); };
document.getElementById('loadFen').onclick = () => { selected = null; api('/api/fen', { fen: document.getElementById('fenInput').value }); };
api('/api/state');
</script>
</body>
</html>`;

const server = createServer((req, res) => {
  const url = new URL(req.url ?? '/', `http://${req.headers.host ?? 'localhost'}`);
  if (url.pathname.startsWith('/api/')) return void handleApi(req, res, url);
  res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' });
  res.end(html);
});

server.listen(port, host, () => {
  const address = server.address();
  const shownHost = host === '0.0.0.0' ? '127.0.0.1' : host;
  console.log(`Tiny Leela web UI: http://${shownHost}:${address.port}`);
});
