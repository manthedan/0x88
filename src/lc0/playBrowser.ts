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
import { Lc0PolicyOnlyPlayer } from './policyOnlyPlayer.ts';
import { Lc0PuctSearcher } from './search.ts';
import { BIG_NETS, Bt4WorkerSearcher, LQO_NET, T3_NET, bigNetAssetStatusSync, bigNetMemoryCaution, checkBigNetAsset, probeBt4Support, bt4SupportedSync, type BigNetConfig } from './bt4Engine.ts';
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

type PlayFamily = 'maia' | 'lc0' | 'sf' | 'reckless' | 'viridithas' | 'berserk' | 'plentychess';

interface PlayEngineOption {
  id: string;
  label: string;
  family: PlayFamily;
  /** Family-specific: Maia Elo, lc0 net key ('small' | 't3' | 'bt4' | 'lqo'), or stockfish kind. */
  variant: string;
  group: 'human' | 'odds' | 'engine';
}

const ENGINE_OPTIONS: PlayEngineOption[] = [
  { id: 'maia-1100', label: 'Maia 1100', family: 'maia', variant: '1100', group: 'human' },
  { id: 'maia-1300', label: 'Maia 1300', family: 'maia', variant: '1300', group: 'human' },
  { id: 'maia-1500', label: 'Maia 1500', family: 'maia', variant: '1500', group: 'human' },
  { id: 'maia-1700', label: 'Maia 1700', family: 'maia', variant: '1700', group: 'human' },
  { id: 'maia-1900', label: 'Maia 1900', family: 'maia', variant: '1900', group: 'human' },
  { id: 'leela-queen-odds', label: 'Leela Queen Odds (WebGPU)', family: 'lc0', variant: 'lqo', group: 'odds' },
  { id: 'sf-lite', label: 'Stockfish Lite', family: 'sf', variant: 'lite', group: 'engine' },
  { id: 'sf-full', label: 'Stockfish', family: 'sf', variant: 'full', group: 'engine' },
  { id: 'lc0-small', label: 'Lc0 · Small net', family: 'lc0', variant: 'small', group: 'engine' },
  { id: 'lc0-t3', label: 'Lc0 · t3-512 distill (WebGPU)', family: 'lc0', variant: 't3', group: 'engine' },
  { id: 'lc0-bt4', label: 'Lc0 · BT4-it332 (WebGPU)', family: 'lc0', variant: 'bt4', group: 'engine' },
  { id: 'reckless', label: 'Reckless', family: 'reckless', variant: 'default', group: 'engine' },
  { id: 'viridithas', label: 'Viridithas', family: 'viridithas', variant: 'default', group: 'engine' },
  { id: 'berserk', label: 'Berserk', family: 'berserk', variant: 'default', group: 'engine' },
  { id: 'plentychess', label: 'PlentyChess', family: 'plentychess', variant: 'default', group: 'engine' },
];

const LEVEL_COUNT = 5;
/** Search-effort labels. Deliberately not "Beginner": engine play is strong at any setting. */
const EFFORT_LEVEL_NAMES = ['Fastest', 'Quick', 'Standard', 'Strong', 'Deep'] as const;
/** Stockfish handicap ladder: UCI Skill Level plus a depth cap per level. */
const SF_LEVELS = [
  { skill: 0, depth: 5, label: '1 · Weakest (skill 0)' },
  { skill: 5, depth: 6, label: '2 · Casual (skill 5)' },
  { skill: 10, depth: 8, label: '3 · Club (skill 10)' },
  { skill: 15, depth: 12, label: '4 · Strong (skill 15)' },
  { skill: 20, depth: 18, label: '5 · Full strength' },
] as const;

/** Per-family strength ladders indexed by level (0-4): visits for lc0, depth otherwise. */
const LEVELS: Record<Exclude<PlayFamily, 'maia' | 'sf'>, number[]> = {
  lc0: [8, 32, 100, 400, 1600],
  reckless: [2, 4, 6, 10, 14],
  viridithas: [2, 4, 6, 9, 12],
  berserk: [2, 4, 6, 9, 12],
  plentychess: [2, 4, 6, 9, 12],
};
/** Big nets are far slower per visit; keep upper levels playable. */
const BIG_NET_LEVELS = [4, 16, 64, 256, 800];

/**
 * WDL draw contempt vs the human (negative = avoid draws and press for the
 * win). The odds bot presses hardest — its Lichess incarnation runs DrawScore
 * around ±0.4-0.6 — while regular Lc0 opponents get a milder anti-draw lean.
 */
const PLAY_DRAW_SCORE = -0.25;
const LQO_DRAW_SCORE = -0.5;
/** LeelaQueenOdds README search settings (CPuct 1.5, ScLimit scaled to browser visit budgets). */
const LQO_CPUCT = 1.5;
const LQO_SEARCH_CONTEMPT_LIMIT = 24;

function maiaModelUrl(elo: string): string {
  return `/models/lc0/maia-${elo}.f32.onnx`;
}

/**
 * Sample a move in proportion to Maia's human-move distribution instead of
 * always playing the argmax, so games vary the way human opponents do. The
 * deep tail (moves under 10% of the top prior) is dropped: the policy gives
 * rare-blunder moves small but nonzero mass, and over a long game those
 * one-in-twenty picks would dominate the experience.
 */
function sampleHumanMove(legalPriors: { uci: string; prior: number }[]): string | undefined {
  if (!legalPriors.length) return undefined;
  const floor = legalPriors[0].prior * 0.1;
  const pool = legalPriors.filter((entry) => entry.prior >= floor);
  const total = pool.reduce((sum, entry) => sum + entry.prior, 0);
  if (!(total > 0)) return legalPriors[0].uci;
  let r = Math.random() * total;
  for (const entry of pool) {
    r -= entry.prior;
    if (r <= 0) return entry.uci;
  }
  return pool[pool.length - 1].uci;
}

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
let startFen: string = START_FEN;
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
type BigNetKey = 'bt4' | 't3' | 'lqo';
const bigNetSearchers: Record<BigNetKey, Bt4WorkerSearcher> = {
  bt4: new Bt4WorkerSearcher(BIG_NETS.bt4),
  t3: new Bt4WorkerSearcher(T3_NET),
  lqo: new Bt4WorkerSearcher(LQO_NET),
};

/** Start FEN for the current game; odds opponents remove their own queen. */
function startFenFor(option: PlayEngineOption, human: Color): string {
  if (option.variant !== 'lqo') return START_FEN;
  return human === 'w'
    ? 'rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
    : 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1';
}
interface CpuEngine {
  setOptions(options: { depth?: number; movetimeMs?: number; threads?: number; skillLevel?: number }): void;
  bestMove(fen: string, signal?: AbortSignal): Promise<string | null>;
}
const cpuEnginePromises = new Map<string, Promise<CpuEngine>>();
const maiaPlayerPromises = new Map<string, Promise<Lc0PolicyOnlyPlayer>>();

function ensureMaia(elo: string): Promise<Lc0PolicyOnlyPlayer> {
  const existing = maiaPlayerPromises.get(elo);
  if (existing) return existing;
  const created = (async () => {
    setEngineNote(`Loading Maia ${elo}…`);
    const modelLoad = await loadLc0ModelForOrt(maiaModelUrl(elo), {
      cache: true,
      onProgress: (loaded, total) => showDownloadProgress(`Maia ${elo}`, loaded, total),
    });
    hideDownloadProgress();
    const evaluator = await Lc0OnnxEvaluator.create(modelLoad.model);
    setEngineNote('');
    return new Lc0PolicyOnlyPlayer(evaluator);
  })().catch((error: Error) => {
    maiaPlayerPromises.delete(elo);
    hideDownloadProgress();
    setEngineNote(`Maia ${elo} load failed: ${error.message}`, true);
    throw error;
  });
  maiaPlayerPromises.set(elo, created);
  return created;
}

function selectedEngine(): PlayEngineOption {
  const id = selectEl('engineSelect').value;
  return ENGINE_OPTIONS.find((option) => option.id === id) ?? ENGINE_OPTIONS[0];
}
function selectedLevel(): number {
  return Math.max(0, Math.min(LEVEL_COUNT - 1, Number(selectEl('levelSelect').value) || 0));
}
function strengthFor(option: PlayEngineOption, level: number): number {
  if (option.family === 'maia') return 1;
  if (option.family === 'sf') return SF_LEVELS[level].depth;
  if (option.family === 'lc0' && option.variant !== 'small') return BIG_NET_LEVELS[level];
  return LEVELS[option.family][level];
}
function strengthCaption(): string {
  const option = selectedEngine();
  const level = selectedLevel();
  if (option.family === 'maia') {
    return `Plays like a ~${option.variant}-rated human — moves are sampled from its human-move predictions, so games vary.`;
  }
  if (option.family === 'sf') {
    const sf = SF_LEVELS[level];
    return sf.skill >= 20 ? `full strength · depth ${sf.depth}` : `UCI skill ${sf.skill} · depth ${sf.depth}`;
  }
  const value = strengthFor(option, level);
  if (option.variant === 'lqo') return `≈ ${value} visits per move — higher levels press harder for tricks`;
  const base = option.family === 'lc0' ? `≈ ${value} visits per move` : `search depth ${value}`;
  return option.family === 'lc0' ? `${base} — strong even on Fastest; pick a Maia for a human-level opponent` : base;
}

function renderLevelOptions(): void {
  const select = selectEl('levelSelect');
  const previous = select.value;
  const option = selectedEngine();
  const field = select.closest('.field') as HTMLElement;
  if (option.family === 'maia') {
    field.hidden = true;
    return;
  }
  field.hidden = false;
  const labels = option.family === 'sf'
    ? SF_LEVELS.map((sf) => sf.label)
    : EFFORT_LEVEL_NAMES.map((name, i) => `${i + 1} · ${name}`);
  select.innerHTML = labels.map((label, i) => `<option value="${i}">${label}</option>`).join('');
  select.value = previous && Number(previous) < LEVEL_COUNT ? previous : '2';
}

function setEngineNote(text: string, warn = false): void {
  const note = el('engineNote');
  note.textContent = text;
  note.hidden = !text;
  note.classList.toggle('warn', warn);
}

function mb(bytes: number): string {
  return `${(bytes / 1_000_000).toFixed(0)} MB`;
}

function showDownloadProgress(label: string, loadedBytes: number, totalBytes?: number): void {
  const wrap = el('dlProgress');
  wrap.hidden = false;
  const bar = wrap.querySelector('progress') as HTMLProgressElement;
  const text = wrap.querySelector('.dl-label') as HTMLElement;
  if (totalBytes && totalBytes > 0) {
    bar.max = totalBytes;
    bar.value = Math.min(loadedBytes, totalBytes);
    text.textContent = `${label} · ${mb(loadedBytes)} / ${mb(totalBytes)}`;
  } else {
    bar.removeAttribute('value');
    text.textContent = `${label} · ${mb(loadedBytes)}`;
  }
}

function hideDownloadProgress(): void {
  el('dlProgress').hidden = true;
}

function ensureLc0Small(): Promise<Lc0PuctSearcher> {
  if (lc0Searcher) return Promise.resolve(lc0Searcher);
  if (!lc0LoadPromise) {
    setEngineNote('Loading Lc0 small net…');
    lc0LoadPromise = (async () => {
      // cache:true persists the validated net in Cache Storage, so the
      // download (with progress) happens once per browser.
      const modelLoad = await loadLc0ModelForOrt(MODEL_URL, {
        cache: true,
        onProgress: (loaded, total) => showDownloadProgress('Lc0 small net', loaded, total),
      });
      hideDownloadProgress();
      const evaluator = await Lc0OnnxEvaluator.create(modelLoad.model);
      lc0Searcher = new Lc0PuctSearcher(new CachedLc0Evaluator(evaluator, { maxEntries: 2048 }));
      setEngineNote('');
      return lc0Searcher;
    })().catch((error: Error) => {
      lc0LoadPromise = null;
      hideDownloadProgress();
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
  if (option.family === 'maia') {
    const player = await ensureMaia(option.variant);
    if (signal.aborted) return null;
    const choice = await player.chooseMove({ positions });
    return sampleHumanMove(choice.evaluation.legalPriors) ?? choice.move ?? null;
  }
  if (option.family === 'lc0' && option.variant === 'small') {
    const searcher = await ensureLc0Small();
    const result = await searcher.search({ positions }, { visits: visitsOrDepth, signal, yieldEveryMs: 16, reuseTree: true, drawScore: PLAY_DRAW_SCORE });
    return result.move ?? null;
  }
  if (option.family === 'lc0') {
    const searcher = bigNetSearchers[option.variant as BigNetKey];
    if (!searcher.loaded) {
      setEngineNote(`Loading Lc0 ${searcher.config.name} (~${searcher.config.approxMb}MB on first use)…`);
      searcher.onDownloadProgress = (loaded, total) => showDownloadProgress(`Lc0 ${searcher.config.name}`, loaded, total);
    }
    const onAbort = () => searcher.cancel();
    signal.addEventListener('abort', onAbort, { once: true });
    try {
      const result = await searcher.search({ positions }, {
        visits: visitsOrDepth,
        reuseTree: true,
        batchSize: searcher.config.recommendedBatchSize,
        batchPipelineDepth: searcher.config.recommendedPipelineDepth,
        evalCacheEntries: 2048,
        drawScore: option.variant === 'lqo' ? LQO_DRAW_SCORE : PLAY_DRAW_SCORE,
        ...(option.variant === 'lqo' ? { cpuct: LQO_CPUCT, searchContemptLimit: LQO_SEARCH_CONTEMPT_LIMIT } : {}),
      });
      hideDownloadProgress();
      setEngineNote('');
      return result.cancelled ? null : result.move ?? null;
    } finally {
      signal.removeEventListener('abort', onAbort);
      hideDownloadProgress();
    }
  }
  const engine = await cpuEngineFor(option);
  if (option.family === 'sf') {
    engine.setOptions({ depth: visitsOrDepth, movetimeMs: undefined, skillLevel: SF_LEVELS[level].skill });
  } else {
    engine.setOptions({ depth: visitsOrDepth, movetimeMs: undefined });
  }
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
  startFen = startFenFor(selectedEngine(), humanColor);
  board = parseFen(startFen);
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
  const tree = new GameTree(startFen);
  let replay = parseFen(startFen);
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
    ...(startFen === START_FEN ? {} : { SetUp: '1', FEN: startFen }),
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

function engineOptionHtml(option: PlayEngineOption): string {
  let disabled = false;
  let suffix = '';
  if (option.family === 'lc0' && option.variant !== 'small') {
    const config: BigNetConfig = BIG_NETS[option.variant as BigNetKey];
    const asset = bigNetAssetStatusSync(config);
    if (!bt4SupportedSync()) { disabled = true; suffix = ' (needs WebGPU)'; }
    else if (asset === 'missing') { disabled = true; suffix = ' (net not hosted here)'; }
  }
  return `<option value="${option.id}"${disabled ? ' disabled' : ''}>${option.label}${suffix}</option>`;
}

function refreshEngineOptions(): void {
  const select = selectEl('engineSelect');
  const selected = select.value;
  const group = (key: PlayEngineOption['group']) => ENGINE_OPTIONS.filter((option) => option.group === key).map(engineOptionHtml).join('');
  select.innerHTML = `<optgroup label="Human-like (Maia, plays like a rated human)">${group('human')}</optgroup>`
    + `<optgroup label="Odds bots (give you material, then hunt for tricks)">${group('odds')}</optgroup>`
    + `<optgroup label="Engines (strong at any level)">${group('engine')}</optgroup>`;
  if (selected && ENGINE_OPTIONS.some((option) => option.id === selected)) select.value = selected;
  renderEngineCaution();
}

function renderEngineCaution(): void {
  const option = selectedEngine();
  const caution = el('engineCaution');
  if (option.family === 'lc0' && option.variant !== 'small') {
    const config = BIG_NETS[option.variant as BigNetKey];
    const memory = bigNetMemoryCaution(config);
    const odds = option.variant === 'lqo'
      ? ' The bot starts without its queen and plays for traps — the Lichess LeelaQueenOdds net. ' : ' ';
    caution.textContent = `First move downloads the ~${config.approxMb}MB net.${odds}${memory ?? ''}`.trimEnd();
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
  refreshEngineOptions();
  selectEl('engineSelect').value = 'maia-1500';
  renderLevelOptions();
  el('newGame').addEventListener('click', newGame);
  el('takeback').addEventListener('click', takeback);
  el('resign').addEventListener('click', resign);
  el('flip').addEventListener('click', () => { orientation = orientation === 'white' ? 'black' : 'white'; render(); });
  el('exportPgn').addEventListener('click', () => { el('pgnOut').textContent = exportPgn(); });
  el('copyPgn').addEventListener('click', () => { void navigator.clipboard.writeText(exportPgn()); });
  selectEl('engineSelect').addEventListener('change', () => {
    renderLevelOptions();
    renderEngineCaution();
    // Before any move is played, apply the opponent's start position (odds
    // bots remove their queen) without starting the engine's clock.
    if (!moves.length && !engineThinking) {
      startFen = startFenFor(selectedEngine(), humanColor);
      board = parseFen(startFen);
      positions = [board];
      gameOver = null;
    }
    render();
  });
  selectEl('levelSelect').addEventListener('change', render);
  void probeBt4Support().then(refreshEngineOptions);
  void checkBigNetAsset(BIG_NETS.bt4, refreshEngineOptions);
  void checkBigNetAsset(T3_NET, refreshEngineOptions);
  void checkBigNetAsset(LQO_NET, refreshEngineOptions);
  render();
}

init();
