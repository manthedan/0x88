import { Chessground } from 'chessground';
import type { Key } from 'chessground/types';
import { parseFen, boardToFen, squareName, START_FEN, type BoardState } from './chess/board.ts';
import { legalMoves, makeMove } from './chess/movegen.ts';
import { moveFromUci, moveToActionId, moveToUci, type Move } from './chess/moveCodec.ts';
import { actionValuePuctPolicy, chooseMove } from './search/puct.ts';
import { OnnxEvaluator, type OnnxStudentMeta } from './nn/onnxEvaluator.ts';
import { SquareFormerEvaluator, type SquareFormerMeta } from './nn/squareformerEvaluator.ts';
import type { Evaluator } from './nn/evaluator.ts';

const MICRO_OPENING_BOOK = [
  ['e2e4', 'e7e5'], // Open Game
  ['e2e4', 'c7c5'], // Sicilian
  ['e2e4', 'c7c6'], // Caro-Kann
  ['e2e4', 'e7e6'], // French
  ['d2d4', 'd7d5'], // Queen's Pawn
  ['d2d4', 'g8f6'], // Indian setup
  ['c2c4', 'e7e5'], // English reversed Sicilian
  ['c2c4', 'c7c5'], // Symmetrical English
  ['g1f3', 'd7d5'], // Reti
  ['g2g3', 'd7d5'], // King's Fianchetto
  ['b2b3', 'e7e5'], // Larsen
  ['f2f4', 'd7d5'], // Bird
];

let board: BoardState = parseFen(START_FEN);
let historyFens: string[] = [];
let evaluator: Evaluator | null = null;
let ground: ReturnType<typeof Chessground> | null = null;
let orientation: 'white' | 'black' = 'white';
let lastMove: string | null = null;
let playedMoves: string[] = [];
let stockfish: Worker | null = null;
let stockfishReady = false;
let stockfishThinking = false;
let stockfishBest = '';
let stockfishScore = '';
let stockfishPv = '';
let stockfishSeq = 0;
const params = new URLSearchParams(location.search);
const visits = Number(params.get('visits') ?? '128');
const puctBatchSize = Math.max(1, Number(params.get('batch') ?? '16'));
const puctPolicy = params.get('puctPolicy') ?? 'classic';
const avWeight = Number(params.get('avWeight') ?? '0.25');
const requestedPlayMode = params.get('mode') ?? 'puct';
const temperature = Number(params.get('temperature') ?? '1');
const topK = Number(params.get('topk') ?? '0');
const topP = Number(params.get('topp') ?? '1');
const stockfishDepth = Number(params.get('sfdepth') ?? '10');
const openingMode = params.get('opening') ?? 'book';
const modelKey = params.get('model') ?? '32x4';
let busy = false;
let renderSeq = 0;
const models: Record<string, { onnx: string; meta: string; label: string; forcedMode?: string }> = {
  '32x4': { onnx: '/models/residual_32x4_history2.onnx', meta: '/models/residual_32x4_history2.meta.json', label: '32x4 supervised' },
  '48x5': { onnx: '/models/residual_48x5_history2_2026mix_best.onnx', meta: '/models/residual_48x5_history2_2026mix_best.meta.json', label: '48x5 supervised best' },
  'sfaux': { onnx: '/models/residual_48x5_history2_2026mix_sfauxfull_d8_mpv4.onnx', meta: '/models/residual_48x5_history2_2026mix_sfauxfull_d8_mpv4.meta.json', label: '48x5 Stockfish aux full' },
  '48x5-10m': { onnx: '/models/residual_48x5_10m_e6.onnx', meta: '/models/residual_48x5_10m_e6.meta.json', label: '48x5 10M e6' },
  '64x6-10m': { onnx: '/models/residual_64x6_10m_e9.onnx', meta: '/models/residual_64x6_10m_e9.meta.json', label: '64x6 10M e9' },
  '80x5hyb-10m': { onnx: '/models/residual_80x5_hybrid_10m_e9.onnx', meta: '/models/residual_80x5_hybrid_10m_e9.meta.json', label: '80x5 hybrid 10M e9' },
  '48x5-10m-e9': { onnx: '/models/residual_48x5_10m_e9.onnx', meta: '/models/residual_48x5_10m_e9.meta.json', label: '48x5 10M e9 guarded' },
  '64x6-10m-e12ema': { onnx: '/models/residual_64x6_10m_e12_ema.onnx', meta: '/models/residual_64x6_10m_e12_ema.meta.json', label: '64x6 10M e12 EMA' },
  '80x5hyb-10m-e12ema': { onnx: '/models/residual_80x5_hybrid_10m_e12_ema.onnx', meta: '/models/residual_80x5_hybrid_10m_e12_ema.meta.json', label: '80x5 hybrid 10M e12 EMA' },
  'square-v1-smoke-policy': { onnx: '/models/squareformer_v1_100k_e3_single.onnx', meta: '/models/squareformer_v1_100k_e3_single.meta.json', label: 'SquareFormer v1 100k policy-only', forcedMode: 'argmax' },
  'square-v1-smoke-puct': { onnx: '/models/squareformer_v1_100k_e3_single.onnx', meta: '/models/squareformer_v1_100k_e3_single.meta.json', label: 'SquareFormer v1 100k PUCT', forcedMode: 'puct' },
  'chessformer-v1-100m-e3-policy': { onnx: '/models/chessformer_v1_100m_e3_single.onnx', meta: '/models/chessformer_v1_100m_e3_single.meta.json', label: 'ChessFormer v1 100M e3 policy-only', forcedMode: 'argmax' },
  'chessformer-v1-100m-e3-puct': { onnx: '/models/chessformer_v1_100m_e3_single.onnx', meta: '/models/chessformer_v1_100m_e3_single.meta.json', label: 'ChessFormer v1 100M e3 PUCT', forcedMode: 'puct' },
  'cnn-64x6-100m-e3-puct': { onnx: '/models/cnn_64x6_100m_e3.onnx', meta: '/models/cnn_64x6_100m_e3.meta.json', label: 'CNN 64x6 100M e3 PUCT', forcedMode: 'puct' },
  'cnn-64x6-100m-e3-policy': { onnx: '/models/cnn_64x6_100m_e3.onnx', meta: '/models/cnn_64x6_100m_e3.meta.json', label: 'CNN 64x6 100M e3 policy-only', forcedMode: 'argmax' },
  'cnn-48x5-100m-e3-puct': { onnx: '/models/cnn_48x5_100m_e3.onnx', meta: '/models/cnn_48x5_100m_e3.meta.json', label: 'CNN 48x5 100M e3 PUCT', forcedMode: 'puct' },
  'cnn-48x5-100m-e3-policy': { onnx: '/models/cnn_48x5_100m_e3.onnx', meta: '/models/cnn_48x5_100m_e3.meta.json', label: 'CNN 48x5 100M e3 policy-only', forcedMode: 'argmax' },
  'cnn-32x4-100m-e3-puct': { onnx: '/models/cnn_32x4_100m_e3.onnx', meta: '/models/cnn_32x4_100m_e3.meta.json', label: 'CNN 32x4 100M e3 PUCT', forcedMode: 'puct' },
  'cnn-32x4-100m-e3-policy': { onnx: '/models/cnn_32x4_100m_e3.onnx', meta: '/models/cnn_32x4_100m_e3.meta.json', label: 'CNN 32x4 100M e3 policy-only', forcedMode: 'argmax' },
};
const selectedModel = models[modelKey] ?? models['32x4'];
const playMode = selectedModel.forcedMode ?? requestedPlayMode;

const $ = (id: string) => document.getElementById(id)!;
function legalDests() {
  const dests = new Map<Key, Key[]>();
  for (const m of legalMoves(board)) {
    const from = squareName(m.from) as Key, to = squareName(m.to) as Key;
    dests.set(from, [...(dests.get(from) ?? []), to]);
  }
  return dests;
}
function legalMoveByUci(uci: string) { return legalMoves(board).find((m) => moveToUci(m) === uci) ?? null; }
function boardFen() { return boardToFen(board).split(' ')[0]; }
function startFromMicroBook() {
  board = parseFen(START_FEN); historyFens = []; lastMove = null; playedMoves = [];
  if (openingMode === 'start') return 'Start position.';
  const line = MICRO_OPENING_BOOK[Math.floor(Math.random() * MICRO_OPENING_BOOK.length)];
  for (const uci of line) {
    const before = boardToFen(board);
    const move = legalMoves(board).find((m) => moveToUci(m) === uci) ?? moveFromUci(uci);
    historyFens = [before, ...historyFens];
    board = makeMove(board, move);
    lastMove = uci;
    playedMoves.push(uci);
  }
  return `Random 2-ply book: ${line.join(' ')}.`;
}
function initModelSelect() {
  const select = document.getElementById('modelSelect') as HTMLSelectElement | null;
  if (!select) return;
  select.innerHTML = Object.entries(models).map(([key, model]) => `<option value="${key}"${key === modelKey ? ' selected' : ''}>${model.label}</option>`).join('');
  select.onchange = () => {
    const url = new URL(location.href);
    url.searchParams.set('model', select.value);
    location.href = url.toString();
  };
}
function initRunConfigChips() {
  const visitsChip = document.getElementById('visitsChip');
  const batchChip = document.getElementById('batchChip');
  if (visitsChip) visitsChip.textContent = playMode === 'puct' ? `visits ${visits}` : `policy ${playMode}`;
  if (batchChip) batchChip.textContent = playMode === 'puct' ? `batch ${puctBatchSize}` : `top-k ${topK || 'all'}`;
}
type UiMode = 'play' | 'analysis';
let uiMode: UiMode = 'play';
function setUiMode(mode: UiMode) {
  uiMode = mode;
  document.body.classList.toggle('analysis-mode', mode === 'analysis');
  document.getElementById('playModeBtn')?.classList.toggle('active', mode === 'play');
  document.getElementById('analysisModeBtn')?.classList.toggle('active', mode === 'analysis');
}
function toggleUiMode() {
  setUiMode(uiMode === 'play' ? 'analysis' : 'play');
}
function initUiMode() {
  setUiMode(params.get('view') === 'analysis' ? 'analysis' : 'play');
  document.querySelector('.mode-pill')?.addEventListener('click', (event) => {
    event.preventDefault();
    toggleUiMode();
  });
}
function renderWdl(wdl: [number, number, number]) {
  const parts = [
    { name: 'win', value: wdl[0] ?? 0, cls: 'wdl-win' },
    { name: 'draw', value: wdl[1] ?? 0, cls: 'wdl-draw' },
    { name: 'loss', value: wdl[2] ?? 0, cls: 'wdl-loss' },
  ];
  $('wdl').innerHTML = `<div class="wdl-stack">${parts.map((p)=>`<div class="wdl-seg ${p.cls}" style="width:${Math.max(0, p.value * 100)}%">${(p.value * 100).toFixed(0)}%</div>`).join('')}</div><div class="wdl-labels"><span>win</span><span>draw</span><span>loss</span></div>`;
}
function renderMoves() {
  const cells: string[] = [];
  for (let i = 0; i < playedMoves.length; i += 2) {
    cells.push(`<span class="moveno">${Math.floor(i / 2) + 1}.</span><span class="moveuci">${playedMoves[i] ?? ''}</span><span class="moveuci">${playedMoves[i + 1] ?? ''}</span>`);
  }
  $('pgn').innerHTML = cells.join('') || '<span class="muted">No moves yet.</span>';
}
function renderMaterial() {
  const values: Record<string, number> = { p: 1, n: 3, b: 3, r: 5, q: 9 };
  const start: Record<string, number> = { p: 8, n: 2, b: 2, r: 2, q: 1 };
  const counts: Record<string, number> = { wp:0,wn:0,wb:0,wr:0,wq:0,wk:0,bp:0,bn:0,bb:0,br:0,bq:0,bk:0 };
  for (const pc of board.squares) if (pc) counts[pc] = (counts[pc] ?? 0) + 1;
  let whiteMissing = '', blackMissing = '', whiteMat = 0, blackMat = 0;
  const glyph: Record<string, string> = { p:'♟', n:'♞', b:'♝', r:'♜', q:'♛' };
  for (const p of ['q','r','b','n','p']) {
    whiteMat += (counts[`w${p}`] ?? 0) * values[p]; blackMat += (counts[`b${p}`] ?? 0) * values[p];
    whiteMissing += glyph[p].repeat(Math.max(0, start[p] - (counts[`w${p}`] ?? 0)));
    blackMissing += glyph[p].repeat(Math.max(0, start[p] - (counts[`b${p}`] ?? 0)));
  }
  const diff = whiteMat - blackMat;
  $('material').innerHTML = `<div class="score">${diff === 0 ? 'Even' : diff > 0 ? `White +${diff}` : `Black +${-diff}`}</div><div class="pill">White missing <span class="captured">${whiteMissing || '—'}</span></div><div class="pill">Black missing <span class="captured">${blackMissing || '—'}</span></div>`;
}
function renderStockfish() {
  const status = !stockfish ? 'Off. Click “Start Stockfish” for browser-side depth analysis.' : stockfishThinking ? `Thinking to depth ${stockfishDepth}…` : stockfishReady ? 'Ready.' : 'Loading…';
  $('stockfish').innerHTML = `<div>${status}</div><div class="score">${stockfishScore || '—'}</div><div>Best: <span class="mono">${stockfishBest || '—'}</span></div><div class="pv">PV: <span class="mono">${stockfishPv || '—'}</span></div>`;
}
function controlsEnabled(enabled: boolean) {
  for (const id of ['engine','engineAnalysis','reset','resetAnalysis','flip','flipAnalysis','loadFen']) {
    const el = document.getElementById(id) as HTMLButtonElement | null;
    if (el) el.disabled = !enabled;
  }
  for (const id of ['stockfishBtn','stockfishAnalysis']) {
    const el = document.getElementById(id) as HTMLButtonElement | null;
    if (el) el.disabled = !!stockfish && !stockfishReady;
  }
}
async function render(message = '') {
  const seq = ++renderSeq;
  $('fen').textContent = boardToFen(board);
  const statusText = evaluator ? `${selectedModel.label} · ${playMode === 'puct' ? `${visits} visits · batch ${puctBatchSize}${puctPolicy === 'av' ? ` · AV ${avWeight}` : ''}` : `policy ${playMode}`}${busy ? ' · thinking…' : ''}` : 'loading';
  $('status').textContent = statusText;
  const analysisStatus = document.getElementById('analysisStatus');
  if (analysisStatus) analysisStatus.textContent = statusText;
  controlsEnabled(!busy && !!evaluator);
  if (message) $('message').textContent = message;
  renderMoves();
  renderMaterial();
  renderStockfish();
  if (!ground) {
    ground = Chessground($('ground'), { orientation, fen: boardFen(), turnColor: board.turn === 'w' ? 'white' : 'black', coordinates: true, highlight: { lastMove: true, check: true }, animation: { enabled: true, duration: 180 }, premovable: { enabled: false }, predroppable: { enabled: false }, movable: { free: false, color: busy ? undefined : (board.turn === 'w' ? 'white' : 'black'), dests: busy ? new Map() : legalDests(), showDests: !busy, events: { after: onUserMove } } });
  } else {
    ground.set({ orientation, fen: boardFen(), turnColor: board.turn === 'w' ? 'white' : 'black', coordinates: true, highlight: { lastMove: true, check: true }, animation: { enabled: true, duration: 180 }, lastMove: lastMove ? [lastMove.slice(0,2) as Key, lastMove.slice(2,4) as Key] : undefined, premovable: { enabled: false }, predroppable: { enabled: false }, movable: { free: false, color: busy ? undefined : (board.turn === 'w' ? 'white' : 'black'), dests: busy ? new Map() : legalDests(), showDests: !busy, events: { after: onUserMove } } });
  }
  if (!evaluator) return;
  const ev = await evaluator.evaluate(board, { historyFens });
  if (seq !== renderSeq) return;
  renderWdl(ev.wdl);
  const rows = legalMoves(board).map((m: Move) => ({ uci: moveToUci(m), prior: ev.policy.get(moveToActionId(m)) ?? 0 })).sort((a,b)=>b.prior-a.prior).slice(0,16);
  const maxPrior = Math.max(1e-9, ...rows.map((r) => r.prior));
  $('moves').innerHTML = rows.map((r, i)=>`<li class="policy-row ${i === 0 ? 'best' : ''}"><span class="rank">${i + 1}</span><b>${r.uci}</b><span class="policy-meter"><span style="width:${Math.max(2, (r.prior / maxPrior) * 100)}%"></span></span><span class="pct">${(r.prior*100).toFixed(2)}%</span></li>`).join('');
  requestStockfishAnalysis();
}
function startStockfish() {
  if (stockfish) return;
  try {
    stockfish = new Worker('/stockfish/stockfish-18-lite-single.js');
    stockfish.onmessage = (ev) => handleStockfishLine(String(ev.data ?? ''));
    stockfish.onerror = () => { stockfishScore = 'Stockfish failed to load'; renderStockfish(); };
    sendStockfish('uci');
    renderStockfish();
  } catch (e) {
    stockfishScore = `Stockfish unavailable: ${(e as Error).message}`;
    renderStockfish();
  }
}
function sendStockfish(cmd: string) { stockfish?.postMessage(cmd); }
function handleStockfishLine(line: string) {
  if (line === 'uciok') { sendStockfish('isready'); return; }
  if (line === 'readyok') { stockfishReady = true; requestStockfishAnalysis(); renderStockfish(); return; }
  const best = line.match(/^bestmove\s+(\S+)/);
  if (best) { stockfishBest = best[1]; stockfishThinking = false; renderStockfish(); return; }
  if (!line.startsWith('info ')) return;
  const pv = line.match(/\spv\s+(.+)$/);
  if (pv) stockfishPv = pv[1].split(/\s+/).slice(0, 12).join(' ');
  const mate = line.match(/\sscore\s+mate\s+(-?\d+)/);
  const cp = line.match(/\sscore\s+cp\s+(-?\d+)/);
  const depth = line.match(/\sdepth\s+(\d+)/);
  if (mate) stockfishScore = `M${mate[1]}${depth ? ` d${depth[1]}` : ''}`;
  else if (cp) {
    const pawns = Number(cp[1]) / 100;
    stockfishScore = `${pawns >= 0 ? '+' : ''}${pawns.toFixed(2)}${depth ? ` d${depth[1]}` : ''}`;
  }
  renderStockfish();
}
function requestStockfishAnalysis() {
  if (!stockfish || !stockfishReady || stockfishThinking) return;
  stockfishSeq += 1;
  stockfishThinking = true;
  stockfishBest = ''; stockfishPv = '';
  sendStockfish('stop');
  sendStockfish(`position fen ${boardToFen(board)}`);
  sendStockfish(`go depth ${Math.max(1, stockfishDepth)}`);
  const seq = stockfishSeq;
  setTimeout(() => { if (seq === stockfishSeq && stockfishThinking) { sendStockfish('stop'); } }, 8000);
  renderStockfish();
}
async function playMove(move: Move, who: string) {
  const before = boardToFen(board);
  const uci = moveToUci(move);
  historyFens = [before, ...historyFens];
  board = makeMove(board, move);
  lastMove = uci;
  playedMoves.push(uci);
  await render(`${who} played ${uci}.`);
}
async function onUserMove(from: string, to: string) {
  if (busy) return;
  const candidates = legalMoves(board).filter((m) => squareName(m.from) === from && squareName(m.to) === to);
  const move = legalMoveByUci(from + to) ?? candidates.find((m) => moveToUci(m).endsWith('q')) ?? candidates[0];
  if (!move) { await render(`Illegal move ${from}${to}.`); return; }
  await playMove(move, 'You');
  await engineMove();
}
async function choosePolicyMove(): Promise<Move | null> {
  if (!evaluator) return null;
  const moves = legalMoves(board);
  if (!moves.length) return null;
  const ev = await evaluator.evaluate(board, { historyFens });
  let rows = moves.map((move) => ({ move, p: Math.max(0, ev.policy.get(moveToActionId(move)) ?? 0) })).sort((a,b)=>b.p-a.p);
  if (playMode === 'argmax' || temperature <= 0) return rows[0].move;
  if (topK > 0) rows = rows.slice(0, Math.max(1, topK));
  if (topP < 1) {
    const total = rows.reduce((s,r)=>s+r.p,0) || 1;
    let acc = 0, cut = rows.length;
    for (let i=0;i<rows.length;i++) { acc += rows[i].p / total; if (acc >= Math.max(0.01, topP)) { cut = i + 1; break; } }
    rows = rows.slice(0, Math.max(1, cut));
  }
  const weights = rows.map((r)=>Math.pow(Math.max(r.p, 1e-12), 1 / Math.max(1e-6, temperature)));
  const total = weights.reduce((a,b)=>a+b,0);
  let pick = Math.random() * total;
  for (let i=0;i<rows.length;i++) { pick -= weights[i]; if (pick <= 0) return rows[i].move; }
  return rows[0].move;
}
async function engineMove() {
  if (!evaluator || busy) return;
  busy = true;
  document.body.style.cursor = 'progress';
  await render('Engine thinking…');
  try {
    const move = playMode === 'puct' ? (await chooseMove(board, evaluator, { visits, batchSize: puctBatchSize, historyFens, searchPolicy: puctPolicy === 'av' ? actionValuePuctPolicy : undefined, avWeight })).move : await choosePolicyMove();
    if (move) await playMove(move, 'Engine');
    else await render('No legal engine move.');
  } catch (e) {
    console.error(e);
    await render(`Engine failed: ${(e as Error).message}`);
  } finally { busy = false; document.body.style.cursor = ''; await render(); }
}
const onReset = async () => { if (busy) return; await render(startFromMicroBook()); };
const onFlip = async () => { if (busy) return; orientation = orientation === 'white' ? 'black' : 'white'; await render(); };
$('engine').onclick = () => engineMove();
$('engineAnalysis').onclick = () => engineMove();
$('stockfishBtn').onclick = () => startStockfish();
$('stockfishAnalysis').onclick = () => startStockfish();
$('reset').onclick = onReset;
$('resetAnalysis').onclick = onReset;
$('flip').onclick = onFlip;
$('flipAnalysis').onclick = onFlip;
$('loadFen').onclick = async () => { if (busy) return; board = parseFen(($('fenInput') as HTMLInputElement).value || START_FEN); historyFens = []; lastMove = null; playedMoves = []; await render('Loaded FEN.'); };

async function main() {
  initUiMode();
  initModelSelect();
  initRunConfigChips();
  const initialMessage = startFromMicroBook();
  await render(initialMessage);
  const meta = await fetch(selectedModel.meta).then((r) => r.json()) as OnnxStudentMeta | SquareFormerMeta;
  evaluator = meta.kind === 'squareformer'
    ? await SquareFormerEvaluator.create(selectedModel.onnx, meta as SquareFormerMeta)
    : await OnnxEvaluator.create(selectedModel.onnx, meta as OnnxStudentMeta);
  await render(`Loaded ${selectedModel.label}. Mode: ${playMode === 'puct' ? `${visits} visits, batch ${puctBatchSize}${puctPolicy === 'av' ? `, AV ${avWeight}` : ''}` : `policy ${playMode}`}.`);
}
main().catch((e) => { console.error(e); $('message').textContent = `Failed: ${e.message}`; });
