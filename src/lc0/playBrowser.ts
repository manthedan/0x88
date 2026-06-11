// Play vs engine: a human-friendly game page on top of the same engine stack
// used by the arena and analysis pages. Engines load lazily on first move.
import { Chessground } from 'chessground';
import type { Key } from 'chessground/types';
import { boardToFen, parseFen, squareName, START_FEN, type BoardState, type Color } from '../chess/board.ts';
import { legalMoves, makeMove } from '../chess/movegen.ts';
import { moveToUci, type Move } from '../chess/moveCodec.ts';
import { moveToSan } from '../chess/san.ts';
import { gameTreeToPgn } from '../chess/pgn.ts';
import { GameTree } from './gameTree.ts';
import { gameOutcome, type GameResultCode } from './engineBattle.ts';
import { loadLc0ModelForOrt } from './modelCache.ts';
import { CachedLc0Evaluator, Lc0OnnxEvaluator } from './onnxEvaluator.ts';
import { Lc0PuctSearcher } from './search.ts';
import { BIG_NETS, Bt4WorkerSearcher, T3_NET, bigNetAssetStatusSync, bigNetMemoryCaution, checkBigNetAsset, probeBt4Support, bt4SupportedSync, type BigNetConfig } from './bt4Engine.ts';
import { StockfishEngine, stockfishFlavorUrl } from './stockfishEngine.ts';
import { RecklessEngine } from './recklessEngine.ts';
import { defaultRecklessVariantKey, recklessVariantByKey, resolveDefaultRecklessVariantAssetFallback } from './recklessVariants.ts';
import { ViridithasEngine } from './viridithasEngine.ts';
import { defaultViridithasVariantKey, resolveDefaultViridithasVariantAssetFallback, viridithasVariantByKey } from './viridithasVariants.ts';
import { BerserkEngine } from './berserkEngine.ts';
import { berserkVariantByKey, defaultBerserkVariantKey, resolveDefaultBerserkVariantAssetFallback } from './berserkVariants.ts';
import { PlentyChessEngine } from './plentychessEngine.ts';
import { defaultPlentyChessVariantKey, plentyChessVariantByKey, resolveDefaultPlentyChessVariantAssetFallback } from './plentychessVariants.ts';

const params = new URLSearchParams(location.search);
const DEFAULT_MODEL_URL = '/models/lc0/t1-256x10-distilled-swa-2432500.batch1.f32.onnx';
const MODEL_URL = params.get('model') ?? DEFAULT_MODEL_URL;

type PlayFamily = 'lc0' | 'sf' | 'reckless' | 'viridithas' | 'berserk' | 'plentychess';

interface PlayEngineOption {
  id: string;
  label: string;
  family: PlayFamily;
  /** lc0 net key ('small' | 't3' | 'bt4') or stockfish kind ('lite' | 'full'). */
  variant: string;
}

const ENGINE_OPTIONS: PlayEngineOption[] = [
  { id: 'lc0-small', label: 'Lc0 · Small net', family: 'lc0', variant: 'small' },
  { id: 'lc0-t3', label: 'Lc0 · t3-512 distill (WebGPU)', family: 'lc0', variant: 't3' },
  { id: 'lc0-bt4', label: 'Lc0 · BT4-it332 (WebGPU)', family: 'lc0', variant: 'bt4' },
  { id: 'sf-lite', label: 'Stockfish Lite', family: 'sf', variant: 'lite' },
  { id: 'sf-full', label: 'Stockfish', family: 'sf', variant: 'full' },
  { id: 'reckless', label: 'Reckless', family: 'reckless', variant: 'default' },
  { id: 'viridithas', label: 'Viridithas', family: 'viridithas', variant: 'default' },
  { id: 'berserk', label: 'Berserk', family: 'berserk', variant: 'default' },
  { id: 'plentychess', label: 'PlentyChess', family: 'plentychess', variant: 'default' },
];

const LEVEL_NAMES = ['Beginner', 'Casual', 'Club', 'Strong', 'Expert'] as const;

/** Per-family strength ladders indexed by level (0-4): visits for lc0, depth otherwise. */
const LEVELS: Record<PlayFamily, number[]> = {
  lc0: [8, 32, 100, 400, 1600],
  sf: [2, 4, 6, 10, 14],
  reckless: [2, 4, 6, 10, 14],
  viridithas: [2, 4, 6, 9, 12],
  berserk: [2, 4, 6, 9, 12],
  plentychess: [2, 4, 6, 9, 12],
};
/** Big nets are far slower per visit; keep upper levels playable. */
const BIG_NET_LEVELS = [4, 16, 64, 256, 800];

function el(id: string): HTMLElement {
  const found = document.getElementById(id);
  if (!found) throw new Error(`missing element #${id}`);
  return found;
}
function selectEl(id: string): HTMLSelectElement { return el(id) as HTMLSelectElement; }
function buttonEl(id: string): HTMLButtonElement { return el(id) as HTMLButtonElement; }

// ---------------------------------------------------------------------------
// Game state
// ---------------------------------------------------------------------------
let board: BoardState = parseFen(START_FEN);
let positions: BoardState[] = [board];
let moves: Move[] = [];
let sans: string[] = [];
let humanColor: Color = 'w';
let orientation: 'white' | 'black' = 'white';
let gameOver: { result: GameResultCode; reason: string } | null = null;
let engineThinking = false;
let gameSeq = 0;
let abort: AbortController | null = null;
let ground: ReturnType<typeof Chessground> | null = null;
let pendingPromotion: Move[] | null = null;

// ---------------------------------------------------------------------------
// Engines (all lazy; nothing downloads until the engine has to move)
// ---------------------------------------------------------------------------
let lc0Searcher: Lc0PuctSearcher | null = null;
let lc0LoadPromise: Promise<Lc0PuctSearcher> | null = null;
const bigNetSearchers: Record<'bt4' | 't3', Bt4WorkerSearcher> = {
  bt4: new Bt4WorkerSearcher(BIG_NETS.bt4),
  t3: new Bt4WorkerSearcher(T3_NET),
};
interface CpuEngine {
  setOptions(options: { depth?: number; movetimeMs?: number; threads?: number }): void;
  bestMove(fen: string, signal?: AbortSignal): Promise<string | null>;
}
const cpuEnginePromises = new Map<string, Promise<CpuEngine>>();

function selectedEngine(): PlayEngineOption {
  const id = selectEl('engineSelect').value;
  return ENGINE_OPTIONS.find((option) => option.id === id) ?? ENGINE_OPTIONS[0];
}
function selectedLevel(): number {
  return Math.max(0, Math.min(LEVEL_NAMES.length - 1, Number(selectEl('levelSelect').value) || 0));
}
function strengthFor(option: PlayEngineOption, level: number): number {
  if (option.family === 'lc0' && option.variant !== 'small') return BIG_NET_LEVELS[level];
  return LEVELS[option.family][level];
}
function strengthCaption(): string {
  const option = selectedEngine();
  const level = selectedLevel();
  const value = strengthFor(option, level);
  return option.family === 'lc0' ? `≈ ${value} visits per move` : `search depth ${value}`;
}

function setEngineNote(text: string, warn = false): void {
  const note = el('engineNote');
  note.textContent = text;
  note.hidden = !text;
  note.classList.toggle('warn', warn);
}

function ensureLc0Small(): Promise<Lc0PuctSearcher> {
  if (lc0Searcher) return Promise.resolve(lc0Searcher);
  if (!lc0LoadPromise) {
    setEngineNote('Loading Lc0 small net (first use downloads the model)…');
    lc0LoadPromise = (async () => {
      const modelLoad = await loadLc0ModelForOrt(MODEL_URL, { cache: false });
      const evaluator = await Lc0OnnxEvaluator.create(modelLoad.model);
      lc0Searcher = new Lc0PuctSearcher(new CachedLc0Evaluator(evaluator, { maxEntries: 2048 }));
      setEngineNote('');
      return lc0Searcher;
    })().catch((error: Error) => {
      lc0LoadPromise = null;
      setEngineNote(`Lc0 load failed: ${error.message}`, true);
      throw error;
    });
  }
  return lc0LoadPromise;
}

function cpuEngineFor(option: PlayEngineOption): Promise<CpuEngine> {
  const existing = cpuEnginePromises.get(option.id);
  if (existing) return existing;
  const label = option.label;
  const created = (async (): Promise<CpuEngine> => {
    setEngineNote(`Loading ${label} (first use downloads the engine)…`);
    try {
      switch (option.family) {
        case 'sf':
          return new StockfishEngine({ depth: 4, threads: 1 }, stockfishFlavorUrl(option.variant === 'lite' ? 'lite-single' : 'single'));
        case 'reckless': {
          const variant = await resolveDefaultRecklessVariantAssetFallback(recklessVariantByKey(defaultRecklessVariantKey()), false);
          return new RecklessEngine({ depth: 4, hashMb: 16 }, variant.wasmUrl, { backend: variant.backend ?? 'wasi', nnueUrl: variant.nnueUrl });
        }
        case 'viridithas': {
          const variant = await resolveDefaultViridithasVariantAssetFallback(viridithasVariantByKey(defaultViridithasVariantKey()), false);
          return new ViridithasEngine({ depth: 4, hashMb: 16 }, variant.wasmUrl);
        }
        case 'berserk': {
          const variant = await resolveDefaultBerserkVariantAssetFallback(berserkVariantByKey(defaultBerserkVariantKey()), false);
          return new BerserkEngine({ depth: 4, hashMb: 16, threads: 1 }, variant.jsUrl, variant.wasmUrl, variant.dataUrl);
        }
        case 'plentychess': {
          const variant = await resolveDefaultPlentyChessVariantAssetFallback(plentyChessVariantByKey(defaultPlentyChessVariantKey()), false);
          return new PlentyChessEngine({ depth: 4, hashMb: 16, threads: 1 }, variant.jsUrl, variant.wasmUrl, variant.dataUrl);
        }
        default:
          throw new Error(`unsupported engine family ${option.family}`);
      }
    } finally {
      setEngineNote('');
    }
  })().catch((error: Error) => {
    cpuEnginePromises.delete(option.id);
    setEngineNote(`${label} load failed: ${error.message}`, true);
    throw error;
  });
  cpuEnginePromises.set(option.id, created);
  return created;
}

async function requestEngineMove(signal: AbortSignal): Promise<string | null> {
  const option = selectedEngine();
  const level = selectedLevel();
  const visitsOrDepth = strengthFor(option, level);
  if (option.family === 'lc0' && option.variant === 'small') {
    const searcher = await ensureLc0Small();
    const result = await searcher.search({ positions }, { visits: visitsOrDepth, signal, yieldEveryMs: 16, reuseTree: true });
    return result.move ?? null;
  }
  if (option.family === 'lc0') {
    const searcher = bigNetSearchers[option.variant as 'bt4' | 't3'];
    if (!searcher.loaded) setEngineNote(`Loading Lc0 ${searcher.config.name} (~${searcher.config.approxMb}MB on first use)…`);
    const onAbort = () => searcher.cancel();
    signal.addEventListener('abort', onAbort, { once: true });
    try {
      const result = await searcher.search({ positions }, {
        visits: visitsOrDepth,
        reuseTree: true,
        batchSize: searcher.config.recommendedBatchSize,
        batchPipelineDepth: searcher.config.recommendedPipelineDepth,
        evalCacheEntries: 2048,
      });
      setEngineNote('');
      return result.cancelled ? null : result.move ?? null;
    } finally {
      signal.removeEventListener('abort', onAbort);
    }
  }
  const engine = await cpuEngineFor(option);
  engine.setOptions({ depth: visitsOrDepth, movetimeMs: undefined });
  return engine.bestMove(boardToFen(board), signal);
}

// ---------------------------------------------------------------------------
// Game flow
// ---------------------------------------------------------------------------
function priorFens(): string[] {
  return positions.slice(0, -1).map(boardToFen);
}

function checkGameOver(): void {
  const outcome = gameOutcome(board, priorFens());
  if (outcome) gameOver = outcome;
}

function applyMove(move: Move): void {
  sans.push(moveToSan(board, move));
  moves.push(move);
  board = makeMove(board, move);
  positions.push(board);
}

function cancelEngineTurn(): void {
  abort?.abort();
  abort = null;
  engineThinking = false;
}

async function engineTurn(): Promise<void> {
  if (gameOver || board.turn === humanColor) return;
  const seq = gameSeq;
  engineThinking = true;
  abort = new AbortController();
  const signal = abort.signal;
  render();
  let uci: string | null = null;
  try {
    uci = await requestEngineMove(signal);
  } catch (error) {
    if (seq !== gameSeq || signal.aborted) return;
    engineThinking = false;
    setEngineNote(`Engine error: ${(error as Error).message}`, true);
    render();
    return;
  }
  if (seq !== gameSeq || signal.aborted) return;
  engineThinking = false;
  abort = null;
  const move = uci ? legalMoves(board).find((m) => moveToUci(m) === uci) : undefined;
  if (!move) {
    // No/illegal move without cancellation counts as an engine forfeit.
    gameOver = { result: humanColor === 'w' ? '1-0' : '0-1', reason: uci ? `engine played illegal move ${uci}` : 'engine returned no move' };
    render();
    return;
  }
  applyMove(move);
  checkGameOver();
  render();
}

function legalDests(): Map<Key, Key[]> {
  const dests = new Map<Key, Key[]>();
  for (const move of legalMoves(board)) {
    const from = squareName(move.from) as Key;
    const to = squareName(move.to) as Key;
    dests.set(from, [...(dests.get(from) ?? []), to]);
  }
  return dests;
}

function onUserMove(from: Key, to: Key): void {
  if (engineThinking || gameOver || board.turn !== humanColor) { render(); return; }
  const matching = legalMoves(board).filter((move) => squareName(move.from) === from && squareName(move.to) === to);
  if (!matching.length) { render(); return; }
  if (matching.length > 1) {
    // Promotion: every matching move carries a promotion piece; let the user pick.
    pendingPromotion = matching;
    render();
    return;
  }
  applyHumanMove(matching[0]);
}

function applyHumanMove(move: Move): void {
  pendingPromotion = null;
  applyMove(move);
  checkGameOver();
  render();
  if (!gameOver) void engineTurn();
}

function newGame(): void {
  gameSeq += 1;
  cancelEngineTurn();
  const colorChoice = selectEl('colorSelect').value;
  humanColor = colorChoice === 'random' ? (Math.random() < 0.5 ? 'w' : 'b') : colorChoice === 'black' ? 'b' : 'w';
  orientation = humanColor === 'w' ? 'white' : 'black';
  board = parseFen(START_FEN);
  positions = [board];
  moves = [];
  sans = [];
  gameOver = null;
  pendingPromotion = null;
  lc0Searcher?.resetTree();
  for (const searcher of Object.values(bigNetSearchers)) {
    if (searcher.loaded) void searcher.resetTree();
  }
  setEngineNote('');
  render();
  if (humanColor === 'b') void engineTurn();
}

function takeback(): void {
  if (!moves.length) return;
  gameSeq += 1;
  cancelEngineTurn();
  // Undo plies until it is the human's move again (one ply if the engine had
  // not replied yet, two plies after an engine reply).
  do {
    moves.pop();
    sans.pop();
    positions.pop();
    board = positions[positions.length - 1];
  } while (moves.length && board.turn !== humanColor);
  gameOver = null;
  pendingPromotion = null;
  render();
  if (board.turn !== humanColor) void engineTurn();
}

function resign(): void {
  if (gameOver || !moves.length) return;
  gameSeq += 1;
  cancelEngineTurn();
  gameOver = { result: humanColor === 'w' ? '0-1' : '1-0', reason: 'resignation' };
  render();
}

function exportPgn(): string {
  const tree = new GameTree();
  let replay = parseFen(START_FEN);
  for (const move of moves) {
    tree.addMove(move);
    replay = makeMove(replay, move);
  }
  const engineName = selectedEngine().label;
  const date = new Date().toISOString().slice(0, 10).replaceAll('-', '.');
  return gameTreeToPgn(tree, {
    Event: 'Casual browser game',
    Site: location.host || 'local',
    Date: date,
    White: humanColor === 'w' ? 'You' : engineName,
    Black: humanColor === 'b' ? 'You' : engineName,
  }, gameOver?.result ?? '*');
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------
function verdictText(): string {
  if (!gameOver) return '';
  const { result, reason } = gameOver;
  if (result === '1/2-1/2') return `Draw — ${reason}`;
  const humanWon = (result === '1-0') === (humanColor === 'w');
  return humanWon ? `You win — ${reason}` : `${selectedEngine().label} wins — ${reason}`;
}

function statusText(): string {
  if (gameOver) return verdictText();
  if (pendingPromotion) return 'Choose a promotion piece';
  if (engineThinking) return `${selectedEngine().label} is thinking…`;
  if (!moves.length && board.turn === humanColor) return `Your move — you play ${humanColor === 'w' ? 'White' : 'Black'}. Moving a piece starts the game.`;
  return board.turn === humanColor ? 'Your move' : `${selectedEngine().label} to move`;
}

function renderMoveList(): void {
  const list = el('moveList');
  if (!sans.length) { list.innerHTML = '<span class="placeholder">No moves yet</span>'; return; }
  const parts: string[] = [];
  for (let i = 0; i < sans.length; i += 2) {
    const number = i / 2 + 1;
    const white = sans[i];
    const black = sans[i + 1];
    parts.push(`<span class="num">${number}.</span> <span class="san">${white}</span>${black ? ` <span class="san">${black}</span>` : ''}`);
  }
  list.innerHTML = parts.join(' ');
  list.scrollTop = list.scrollHeight;
}

function renderPromotionPicker(): void {
  const picker = el('promoPicker');
  picker.hidden = !pendingPromotion;
  if (!pendingPromotion) return;
  const glyphs: Record<string, string> = { q: '♛ Queen', r: '♜ Rook', b: '♝ Bishop', n: '♞ Knight' };
  picker.innerHTML = pendingPromotion
    .map((move) => `<button data-promo="${move.promotion}">${glyphs[move.promotion ?? 'q']}</button>`)
    .join('') + '<button data-promo="">Cancel</button>';
  for (const button of picker.querySelectorAll('button')) {
    button.addEventListener('click', () => {
      const promo = (button as HTMLButtonElement).dataset.promo;
      const choice = pendingPromotion?.find((move) => move.promotion === promo);
      if (choice) applyHumanMove(choice);
      else { pendingPromotion = null; render(); }
    });
  }
}

function refreshEngineOptions(): void {
  const select = selectEl('engineSelect');
  const selected = select.value;
  select.innerHTML = ENGINE_OPTIONS.map((option) => {
    let disabled = false;
    let suffix = '';
    if (option.family === 'lc0' && option.variant !== 'small') {
      const config: BigNetConfig = option.variant === 'bt4' ? BIG_NETS.bt4 : BIG_NETS.t3;
      const asset = bigNetAssetStatusSync(config);
      if (!bt4SupportedSync()) { disabled = true; suffix = ' (needs WebGPU)'; }
      else if (asset === 'missing') { disabled = true; suffix = ' (net not hosted here)'; }
    }
    return `<option value="${option.id}"${disabled ? ' disabled' : ''}>${option.label}${suffix}</option>`;
  }).join('');
  if (selected && ENGINE_OPTIONS.some((option) => option.id === selected)) select.value = selected;
  renderEngineCaution();
}

function renderEngineCaution(): void {
  const option = selectedEngine();
  const caution = el('engineCaution');
  if (option.family === 'lc0' && option.variant !== 'small') {
    const config = option.variant === 'bt4' ? BIG_NETS.bt4 : BIG_NETS.t3;
    const memory = bigNetMemoryCaution(config);
    caution.textContent = `First move downloads the ~${config.approxMb}MB net.${memory ? ` ${memory}` : ''}`;
    caution.hidden = false;
  } else {
    caution.hidden = true;
  }
}

function render(): void {
  el('status').textContent = statusText();
  el('status').classList.toggle('over', !!gameOver);
  el('levelCaption').textContent = strengthCaption();
  buttonEl('takeback').disabled = !moves.length || !!pendingPromotion;
  buttonEl('resign').disabled = !!gameOver || !moves.length;
  renderMoveList();
  renderPromotionPicker();
  el('pgnOut').textContent = '';
  const humanCanMove = !engineThinking && !gameOver && !pendingPromotion && board.turn === humanColor;
  const lastUci = moves.length ? moveToUci(moves[moves.length - 1]) : undefined;
  const config = {
    orientation,
    fen: boardToFen(board).split(' ')[0],
    turnColor: board.turn === 'w' ? 'white' as const : 'black' as const,
    coordinates: true,
    // Allows synthetic pointer events so automated browser checks can move pieces.
    trustAllEvents: true,
    highlight: { lastMove: true, check: true },
    animation: { enabled: true, duration: 160 },
    movable: {
      free: false,
      color: humanCanMove ? (humanColor === 'w' ? 'white' as const : 'black' as const) : undefined,
      dests: humanCanMove ? legalDests() : new Map<Key, Key[]>(),
      showDests: humanCanMove,
      events: { after: onUserMove },
    },
    lastMove: lastUci ? [lastUci.slice(0, 2) as Key, lastUci.slice(2, 4) as Key] : undefined,
  };
  if (!ground) ground = Chessground(el('ground'), config);
  else ground.set(config);
}

function init(): void {
  selectEl('engineSelect').innerHTML = ENGINE_OPTIONS.map((option) => `<option value="${option.id}">${option.label}</option>`).join('');
  selectEl('engineSelect').value = 'sf-lite';
  selectEl('levelSelect').innerHTML = LEVEL_NAMES.map((name, i) => `<option value="${i}">${i + 1} · ${name}</option>`).join('');
  selectEl('levelSelect').value = '2';
  el('newGame').addEventListener('click', newGame);
  el('takeback').addEventListener('click', takeback);
  el('resign').addEventListener('click', resign);
  el('flip').addEventListener('click', () => { orientation = orientation === 'white' ? 'black' : 'white'; render(); });
  el('exportPgn').addEventListener('click', () => { el('pgnOut').textContent = exportPgn(); });
  el('copyPgn').addEventListener('click', () => { void navigator.clipboard.writeText(exportPgn()); });
  selectEl('engineSelect').addEventListener('change', () => { renderEngineCaution(); render(); });
  selectEl('levelSelect').addEventListener('change', render);
  void probeBt4Support().then(refreshEngineOptions);
  void checkBigNetAsset(BIG_NETS.bt4, refreshEngineOptions);
  void checkBigNetAsset(T3_NET, refreshEngineOptions);
  refreshEngineOptions();
  render();
}

init();
