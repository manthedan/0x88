import { Chessground } from 'chessground';
import type { DrawShape } from 'chessground/draw';
import type { Key } from 'chessground/types';
import { boardToFen, parseFen, squareName, START_FEN, type BoardState } from '../chess/board.ts';
import { legalMoves, makeMove } from '../chess/movegen.ts';
import { moveToUci, type Move } from '../chess/moveCodec.ts';
import { gameTreeToPgn, parsePgnGame } from '../chess/pgn.ts';
import { collectOrtRuntimeDiagnostics } from '../nn/ortRuntime.ts';
import { evalBarWhitePercent, lc0AnalysisLines, type AnalysisLine } from './analysisFormat.ts';
import { GameTree, type GameNode } from './gameTree.ts';
import { loadLc0ModelForOrt } from './modelCache.ts';
import { Lc0OnnxEvaluator } from './onnxEvaluator.ts';
import { Lc0PuctSearcher } from './search.ts';

type Ground = ReturnType<typeof Chessground>;

const params = new URLSearchParams(location.search);
const MODEL_URL = params.get('model') ?? '/models/lc0/t1-256x10-distilled-swa-2432500.batch1.f32.onnx';

let tree = new GameTree(params.get('fen') ?? START_FEN);
let searcher: Lc0PuctSearcher | null = null;
let ground: Ground | null = null;
let orientation: 'white' | 'black' = 'white';
let analysisAbort: AbortController | null = null;
let analyzing = false;
const lineCache = new Map<string, AnalysisLine[]>();
const nodeIndex = new Map<number, GameNode>();

function el(id: string): HTMLElement {
  const node = document.getElementById(id);
  if (!node) throw new Error(`Missing #${id}`);
  return node;
}
function inputEl(id: string): HTMLInputElement { return el(id) as HTMLInputElement; }
function htmlEscape(value: unknown): string {
  return String(value).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]!));
}
function setShapes(shapes: DrawShape[]) { ground?.setAutoShapes(shapes); }
function uciShape(uci: string, brush: string): DrawShape | null {
  return uci.length >= 4 ? { orig: uci.slice(0, 2) as Key, dest: uci.slice(2, 4) as Key, brush } : null;
}
function visits(): number { return Math.max(1, Math.floor(Number(inputEl('visitsInput').value) || 400)); }
function multiPv(): number { return Math.max(1, Math.floor(Number(inputEl('multiPvInput').value) || 3)); }

function legalDests(board: BoardState) {
  const dests = new Map<Key, Key[]>();
  for (const move of legalMoves(board)) {
    const from = squareName(move.from) as Key;
    dests.set(from, [...(dests.get(from) ?? []), squareName(move.to) as Key]);
  }
  return dests;
}
function legalMoveFromDrag(board: BoardState, from: Key, to: Key): Move | undefined {
  const base = `${from}${to}`;
  const all = legalMoves(board);
  return all.find((m) => moveToUci(m) === base)
    ?? all.find((m) => moveToUci(m) === `${base}q`)
    ?? all.find((m) => moveToUci(m) === `${base}r`)
    ?? all.find((m) => moveToUci(m) === `${base}b`)
    ?? all.find((m) => moveToUci(m) === `${base}n`);
}

function bestShapes(): DrawShape[] {
  const lines = lineCache.get(tree.current.fen) ?? [];
  const shapes: DrawShape[] = [];
  const best = lines[0]?.pvUci[0] ? uciShape(lines[0].pvUci[0], 'green') : null;
  if (best) shapes.push(best);
  for (const line of lines.slice(1)) {
    const alt = line.pvUci[0] ? uciShape(line.pvUci[0], 'blue') : null;
    if (alt && !(best && alt.orig === best.orig && alt.dest === best.dest)) shapes.push(alt);
  }
  return shapes;
}

function renderBoard() {
  const board = tree.current.fen ? parseFen(tree.current.fen) : parseFen(START_FEN);
  const lastUci = tree.current.move ? moveToUci(tree.current.move) : null;
  const config = {
    orientation,
    fen: tree.current.fen.split(' ')[0],
    turnColor: board.turn === 'w' ? 'white' as const : 'black' as const,
    coordinates: true,
    highlight: { lastMove: true, check: true },
    animation: { enabled: true, duration: 160 },
    movable: {
      free: false,
      color: board.turn === 'w' ? 'white' as const : 'black' as const,
      dests: legalDests(board),
      events: { after: onUserMove },
    },
    lastMove: lastUci ? [lastUci.slice(0, 2) as Key, lastUci.slice(2, 4) as Key] : undefined,
  };
  if (!ground) ground = Chessground(el('ground'), config);
  else ground.set(config);
  el('sideToMove').textContent = board.turn === 'w' ? 'White to move' : 'Black to move';
  renderEvalBar();
  setShapes(bestShapes());
}

function renderEvalBar() {
  const line = (lineCache.get(tree.current.fen) ?? [])[0];
  const board = parseFen(tree.current.fen);
  const pct = line ? evalBarWhitePercent(line.scoreCp, line.mateIn, board.turn) : 50;
  (el('evalWhite') as HTMLElement).style.height = `${pct}%`;
  el('posEval').textContent = line ? `${line.scoreText} (${line.engine})` : '—';
}

function renderLines() {
  const lines = lineCache.get(tree.current.fen) ?? [];
  el('lines').innerHTML = lines.map((line) => {
    const cls = line.scoreCp === undefined ? '' : line.scoreCp > 0 ? 'pos' : line.scoreCp < 0 ? 'neg' : '';
    return `<li data-uci="${htmlEscape(line.pvUci[0] ?? '')}" data-pv="${htmlEscape(line.pvUci.join(' '))}">`
      + `<span class="score ${cls}">${htmlEscape(line.scoreText)}<br><span class="eng">${htmlEscape(line.engine)} · ${htmlEscape(line.detail)}</span></span>`
      + `<span class="pv">${htmlEscape(line.pvSan)}</span></li>`;
  }).join('') || '<li class="small">no analysis yet</li>';
}

function moveNumberPrefix(node: GameNode, force: boolean): string {
  const parent = parseFen(node.parent!.fen);
  if (parent.turn === 'w') return `${parent.fullmove}. `;
  return force ? `${parent.fullmove}… ` : '';
}

function renderVariation(node: GameNode): string {
  let html = '';
  let cursor: GameNode | undefined = node;
  let force = true;
  while (cursor) {
    html += moveToken(cursor, force);
    force = false;
    cursor = cursor.children[0];
  }
  return html;
}

function moveToken(node: GameNode, force: boolean): string {
  const current = node === tree.current ? ' current' : '';
  return `<span class="mv${current}" data-node="${node.id}">${moveNumberPrefix(node, force)}${htmlEscape(node.san ?? '')}</span> `;
}

function renderMoveList() {
  nodeIndex.clear();
  const collect = (n: GameNode) => { nodeIndex.set(n.id, n); n.children.forEach(collect); };
  collect(tree.root);
  let html = '';
  let node: GameNode | undefined = tree.root.children[0];
  let force = true;
  while (node) {
    html += moveToken(node, force);
    force = false;
    const variations = node.parent!.children.slice(1);
    for (const variation of variations) {
      html += `<span class="var">(${renderVariation(variation)})</span> `;
      force = true;
    }
    node = node.children[0];
  }
  el('movelist').innerHTML = html || '<span class="small">no moves — drag a piece or load a PGN</span>';
}

function renderAll() {
  renderBoard();
  renderLines();
  renderMoveList();
}

async function analyzeCurrent() {
  if (!searcher) return;
  analysisAbort?.abort();
  const controller = new AbortController();
  analysisAbort = controller;
  analyzing = true;
  el('stop').toggleAttribute('disabled', false);
  el('analyze').toggleAttribute('disabled', true);
  const fen = tree.current.fen;
  const board = parseFen(fen);
  if (legalMoves(board).length === 0) { analyzing = false; el('stop').toggleAttribute('disabled', true); el('analyze').toggleAttribute('disabled', false); return; }
  el('message').textContent = `Analyzing (${visits()} visits, ${multiPv()} lines)…`;
  try {
    const result = await searcher.search({ positions: tree.historyBoards() }, { visits: visits(), multiPv: multiPv(), signal: controller.signal, yieldEveryMs: 16 });
    if (controller.signal.aborted) return;
    lineCache.set(fen, lc0AnalysisLines(result, fen, 'LC0'));
    if (tree.current.fen === fen) { renderLines(); renderEvalBar(); setShapes(bestShapes()); }
    el('message').textContent = `Analyzed: ${(lineCache.get(fen) ?? [])[0]?.scoreText ?? '—'}`;
  } catch (error) {
    if ((error as Error).name !== 'AbortError') el('message').textContent = `Analysis failed: ${(error as Error).message}`;
  } finally {
    if (analysisAbort === controller) {
      analyzing = false;
      analysisAbort = null;
      el('stop').toggleAttribute('disabled', true);
      el('analyze').toggleAttribute('disabled', false);
    }
  }
}

function afterNavigation() {
  renderAll();
  if (inputEl('autoAnalyze').checked && !lineCache.has(tree.current.fen)) void analyzeCurrent();
  else { renderEvalBar(); setShapes(bestShapes()); }
}

async function onUserMove(from: Key, to: Key) {
  const board = tree.current.fen ? parseFen(tree.current.fen) : parseFen(START_FEN);
  const move = legalMoveFromDrag(board, from, to);
  if (!move) { renderBoard(); return; }
  tree.addMove(move);
  afterNavigation();
}

function loadFen() {
  const raw = inputEl('fenInput').value.trim();
  if (!raw) return;
  try {
    parseFen(raw);
  } catch (error) {
    el('message').textContent = `Invalid FEN: ${(error as Error).message}`;
    return;
  }
  tree = new GameTree(raw);
  lineCache.clear();
  el('message').textContent = 'Loaded FEN.';
  afterNavigation();
}

function loadPgn() {
  const raw = inputEl('pgnInput').value.trim();
  if (!raw) return;
  try {
    const { tree: parsed, tags } = parsePgnGame(raw);
    tree = parsed;
    tree.toStart();
    lineCache.clear();
    el('message').textContent = `Loaded PGN${tags.White ? `: ${tags.White} – ${tags.Black}` : ''}.`;
    afterNavigation();
  } catch (error) {
    el('message').textContent = `PGN parse failed: ${(error as Error).message}`;
  }
}

function copyPgn() {
  const pgn = gameTreeToPgn(tree, {}, tree.root.children.length ? '*' : '*');
  inputEl('pgnInput').value = pgn;
  void navigator.clipboard?.writeText(pgn).catch(() => undefined);
  el('message').textContent = 'PGN copied to the box and clipboard.';
}

function hoverLine(pvUci: string[]) {
  const board = parseFen(tree.current.fen);
  const shapes: DrawShape[] = [];
  let stm = board.turn;
  for (const uci of pvUci.slice(0, 4)) {
    const shape = uciShape(uci, stm === board.turn ? 'green' : 'paleGreen');
    if (shape) shapes.push(shape);
    stm = stm === 'w' ? 'b' : 'w';
  }
  setShapes(shapes.length ? shapes : bestShapes());
}

function wireEvents() {
  el('navStart').addEventListener('click', () => { tree.toStart(); afterNavigation(); });
  el('navBack').addEventListener('click', () => { tree.back(); afterNavigation(); });
  el('navForward').addEventListener('click', () => { tree.forward(); afterNavigation(); });
  el('navEnd').addEventListener('click', () => { tree.toEnd(); afterNavigation(); });
  el('flip').addEventListener('click', () => { orientation = orientation === 'white' ? 'black' : 'white'; renderBoard(); });
  el('loadFen').addEventListener('click', loadFen);
  el('reset').addEventListener('click', () => { tree = new GameTree(); lineCache.clear(); el('message').textContent = 'Reset.'; afterNavigation(); });
  el('loadPgn').addEventListener('click', loadPgn);
  el('copyPgn').addEventListener('click', copyPgn);
  el('analyze').addEventListener('click', () => { void analyzeCurrent(); });
  el('stop').addEventListener('click', () => { analysisAbort?.abort(); });
  el('movelist').addEventListener('click', (event) => {
    const target = (event.target as HTMLElement).closest('[data-node]');
    if (!target) return;
    const node = nodeIndex.get(Number(target.getAttribute('data-node')));
    if (node) { tree.goTo(node); afterNavigation(); }
  });
  el('lines').addEventListener('click', (event) => {
    const li = (event.target as HTMLElement).closest('li[data-uci]');
    const uci = li?.getAttribute('data-uci');
    if (uci) { tree.addUci(uci); afterNavigation(); }
  });
  el('lines').addEventListener('mouseover', (event) => {
    const li = (event.target as HTMLElement).closest('li[data-pv]');
    const pv = li?.getAttribute('data-pv');
    if (pv) hoverLine(pv.split(' ').filter(Boolean));
  });
  el('lines').addEventListener('mouseout', () => setShapes(bestShapes()));
  document.addEventListener('keydown', (event) => {
    if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) return;
    if (event.key === 'ArrowLeft') { tree.back(); afterNavigation(); }
    else if (event.key === 'ArrowRight') { tree.forward(); afterNavigation(); }
    else if (event.key === 'ArrowUp') { tree.toStart(); afterNavigation(); }
    else if (event.key === 'ArrowDown') { tree.toEnd(); afterNavigation(); }
    else return;
    event.preventDefault();
  });
}

async function init() {
  renderAll();
  wireEvents();
  try {
    const modelLoad = await loadLc0ModelForOrt(MODEL_URL, { cache: false });
    const evaluator = await Lc0OnnxEvaluator.create(modelLoad.model);
    searcher = new Lc0PuctSearcher(evaluator);
    const diagnostics = await collectOrtRuntimeDiagnostics();
    el('backend').textContent = diagnostics.describe;
    el('analyze').toggleAttribute('disabled', false);
    el('message').textContent = 'Ready. Drag a move, load a PGN/FEN, or Analyze.';
    if (inputEl('autoAnalyze').checked) void analyzeCurrent();
  } catch (error) {
    el('message').textContent = `Model load failed: ${(error as Error).message}`;
  }
}

void init();
