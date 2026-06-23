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
import { boardCheck, hidePromotionOverlay, legalDests, matchUserMoves, showPromotionOverlay } from './boardUx.ts';
import { loadLc0ModelForOrt } from './modelCache.ts';
import { CachedLc0Evaluator, Lc0OnnxEvaluator } from './onnxEvaluator.ts';
import { Maia3BrowserEvaluator, MAIA3_DEFAULT_ELO, MAIA3_MAX_ELO, MAIA3_MIN_ELO, type Maia3MoveStyle } from './maia3.ts';
import { Lc0PuctSearcher } from './search.ts';
import { BIG_NETS, Bt4WorkerSearcher, LQO_NET, T3_NET, bigNetMemoryCaution, bigNetOptionState, checkBigNetAsset, probeBt4Support, bt4SupportedSync, type BigNetConfig } from './bt4Engine.ts';
import { StockfishEngine, stockfishFlavorUrl } from './stockfishEngine.ts';
import type { RecklessEngine } from './recklessEngine.ts';
import { defaultRecklessVariantKey, recklessVariantByKey, resolveDefaultRecklessVariantAssetFallback } from './recklessVariants.ts';
import type { ViridithasEngine } from './viridithasEngine.ts';
import { defaultViridithasVariantKey, resolveDefaultViridithasVariantAssetFallback, viridithasVariantByKey } from './viridithasVariants.ts';
import type { BerserkEngine } from './berserkEngine.ts';
import { berserkVariantByKey, defaultBerserkVariantKey, resolveDefaultBerserkVariantAssetFallback } from './berserkVariants.ts';
import type { PlentyChessEngine } from './plentychessEngine.ts';
import { defaultPlentyChessVariantKey, plentyChessVariantByKey, resolveDefaultPlentyChessVariantAssetFallback } from './plentychessVariants.ts';
import { createBerserkEngine, createPlentyChessEngine, createRecklessEngine, createViridithasEngine } from './engineProvision.ts';
import { resolvePublicAssetUrl } from './assetUrls.ts';
import { isV0DeployProfile } from './engineCatalog.ts';
import { hideLoadingProgress, renderLoadingProgress } from './loadingProgress.ts';

const params = new URLSearchParams(location.search);
const DEFAULT_MODEL_URL = resolvePublicAssetUrl('/models/lc0/t1-256x10-distilled-swa-2432500.batch1.f16.qdq8.onnx');
const MODEL_URL = isV0DeployProfile() ? DEFAULT_MODEL_URL : resolvePublicAssetUrl(params.get('model') ?? DEFAULT_MODEL_URL);

type PlayFamily = 'maia3' | 'lc0' | 'sf' | 'reckless' | 'viridithas' | 'berserk' | 'plentychess';

interface PlayEngineOption {
  id: string;
  label: string;
  family: PlayFamily;
  /** Family-specific: Maia Elo, lc0 net key ('small' | 't3' | 'bt4' | 'lqo'), or stockfish kind. */
  variant: string;
  group: 'human' | 'odds' | 'engine';
}

const ALL_ENGINE_OPTIONS: PlayEngineOption[] = [
  { id: 'maia3', label: 'Maia3 · Elo-conditioned human model', family: 'maia3', variant: 'maia3', group: 'human' },
  { id: 'leela-queen-odds', label: 'Leela Queen Odds', family: 'lc0', variant: 'lqo', group: 'odds' },
  { id: 'sf-lite', label: 'Stockfish Lite', family: 'sf', variant: 'lite', group: 'engine' },
  { id: 'sf-full', label: 'Stockfish', family: 'sf', variant: 'full', group: 'engine' },
  { id: 'lc0-small', label: 'Lc0 · Small net', family: 'lc0', variant: 'small', group: 'engine' },
  { id: 'lc0-t3', label: 'Lc0 · t3-512 distill', family: 'lc0', variant: 't3', group: 'engine' },
  { id: 'lc0-bt4', label: 'Lc0 · BT4-it332', family: 'lc0', variant: 'bt4', group: 'engine' },
  { id: 'reckless', label: 'Reckless', family: 'reckless', variant: 'default', group: 'engine' },
  { id: 'viridithas', label: 'Viridithas', family: 'viridithas', variant: 'default', group: 'engine' },
  { id: 'berserk', label: 'Berserk', family: 'berserk', variant: 'default', group: 'engine' },
  { id: 'plentychess', label: 'PlentyChess', family: 'plentychess', variant: 'default', group: 'engine' },
];
const ENGINE_OPTIONS: PlayEngineOption[] = isV0DeployProfile()
  ? ALL_ENGINE_OPTIONS.filter((option) => option.family === 'maia3' || option.id === 'leela-queen-odds' || option.id === 'sf-lite' || option.id === 'lc0-small' || option.id === 'reckless' || option.id === 'berserk' || option.id === 'viridithas' || option.id === 'plentychess')
  : ALL_ENGINE_OPTIONS;

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
const LEVELS: Record<Exclude<PlayFamily, 'maia3' | 'sf'>, number[]> = {
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
/**
 * Model the human opponent as a budget-limited searcher (see
 * docs/search_contempt_design.md). The limit reflects the opponent's search
 * ability, not ours, so it stays fixed across the visit ladder; 16 is the
 * A/B-validated point (92% vs 58% baseline at queen odds vs Maia 1900).
 */
const PLAY_SEARCH_CONTEMPT_LIMIT = 16;
/** LeelaQueenOdds README search settings (CPuct 1.5, ScLimit scaled to browser visit budgets). */
const LQO_CPUCT = 1.5;
const LQO_SEARCH_CONTEMPT_LIMIT = 24;

// ---------------------------------------------------------------------------
// Module-level engine caches (intentionally persistent across mount/unmount
// to avoid re-downloading engines on navigation)
// ---------------------------------------------------------------------------
type BigNetKey = 'bt4' | 't3' | 'lqo';
const bigNetSearchers: Record<BigNetKey, Bt4WorkerSearcher> = {
  bt4: new Bt4WorkerSearcher(BIG_NETS.bt4),
  t3: new Bt4WorkerSearcher(T3_NET),
  lqo: new Bt4WorkerSearcher(LQO_NET),
};
let lc0Searcher: Lc0PuctSearcher | null = null;
let lc0LoadPromise: Promise<Lc0PuctSearcher> | null = null;
const cpuEnginePromises = new Map<string, Promise<CpuEngine>>();
let maia3Promise: Promise<Maia3BrowserEvaluator> | null = null;
/** One-line model/cache status shown in the caption once Maia3 has loaded. */
let maia3Status: string | null = null;

interface CpuEngine {
  setOptions(options: { depth?: number; movetimeMs?: number; threads?: number; skillLevel?: number }): void;
  bestMove(fen: string, signal?: AbortSignal): Promise<string | null>;
}

// ---------------------------------------------------------------------------
// Screen wake lock (best-effort): keeps the screen on while the engine is
// thinking so a sleeping laptop does not abandon the move.
// ---------------------------------------------------------------------------
type WakeLockSentinelLike = { release: () => Promise<void> };
let wakeLock: WakeLockSentinelLike | null = null;
async function requestWakeLock(): Promise<void> {
  try {
    const nav = navigator as Navigator & { wakeLock?: { request: (type: 'screen') => Promise<WakeLockSentinelLike> } };
    wakeLock = await nav.wakeLock?.request('screen');
  } catch {
    // Surface API missing or denied; play continues without the lock.
  }
}
function releaseWakeLock(): void {
  if (!wakeLock) return;
  void wakeLock.release().catch(() => {});
  wakeLock = null;
}

// ---------------------------------------------------------------------------
// DOM helpers
// ---------------------------------------------------------------------------
function el(id: string): HTMLElement {
  const found = document.getElementById(id);
  if (!found) throw new Error(`missing element #${id}`);
  return found;
}
function selectEl(id: string): HTMLSelectElement { return el(id) as HTMLSelectElement; }
function buttonEl(id: string): HTMLButtonElement { return el(id) as HTMLButtonElement; }

/** Start FEN for the current game; odds opponents remove their own queen. */
function startFenFor(option: PlayEngineOption, human: Color): string {
  if (option.variant !== 'lqo') return START_FEN;
  return human === 'w'
    ? 'rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
    : 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1';
}

// ---------------------------------------------------------------------------
// Factory: scoped per-mount play context. All game/UI state lives here so
// re-entrant mounts (SvelteKit preloading, error boundaries) cannot clobber
// each other. The abort signal replaces the old gameSeq counter as the
// stale-write guard for async engine callbacks.
// ---------------------------------------------------------------------------
interface PlayContext {
  abort: AbortController;
  ground: ReturnType<typeof Chessground> | null;
  startFen: string;
  board: BoardState;
  positions: BoardState[];
  moves: Move[];
  sans: string[];
  humanColor: Color;
  orientation: 'white' | 'black';
  gameOver: { result: GameResultCode; reason: string } | null;
  engineThinking: boolean;
  pendingPromotion: Move[] | null;
  pendingRestart: { engine: string; color: 'white' | 'black' | 'random' } | null;
  activeEngineId: string;
  activeColor: 'white' | 'black' | 'random';
  lastEngineId: string;
  resignArmed: boolean;
  resignArmTimer: ReturnType<typeof setTimeout> | null;
  // Bound listeners for cleanup
  listeners: Array<{ type: string; target: EventTarget; fn: EventListenerOrEventListenerObject }>;
}

function createPlayContext(): PlayContext {
  return {
    abort: new AbortController(),
    ground: null,
    startFen: START_FEN,
    board: parseFen(START_FEN),
    positions: [parseFen(START_FEN)],
    moves: [],
    sans: [],
    humanColor: 'w',
    orientation: 'white',
    gameOver: null,
    engineThinking: false,
    pendingPromotion: null,
    pendingRestart: null,
    activeEngineId: 'maia3',
    activeColor: 'white',
    lastEngineId: 'maia3',
    resignArmed: false,
    resignArmTimer: null,
    listeners: [],
  };
}

function trackListener(ctx: PlayContext, target: EventTarget, type: string, fn: EventListenerOrEventListenerObject): void {
  target.addEventListener(type, fn);
  ctx.listeners.push({ type, target, fn });
}

// ---------------------------------------------------------------------------
// Context-scoped functions (all game state and UI logic)
// ---------------------------------------------------------------------------

function ctxSelectedEngine(ctx: PlayContext): PlayEngineOption {
  const id = selectEl('engineSelect').value;
  return ENGINE_OPTIONS.find((option) => option.id === id) ?? ENGINE_OPTIONS[0];
}
function ctxSelectedLevel(): number {
  return Math.max(0, Math.min(LEVEL_COUNT - 1, Number(selectEl('levelSelect').value) || 0));
}
function ctxSelectedMaia3Elo(): number {
  const input = el('maia3Elo') as HTMLInputElement;
  return Math.max(MAIA3_MIN_ELO, Math.min(MAIA3_MAX_ELO, Number(input.value) || MAIA3_DEFAULT_ELO));
}
function ctxSelectedMaia3Style(): Maia3MoveStyle {
  return selectEl('maia3Style').value === 'argmax' ? 'argmax' : 'sample';
}
function ctxSelectedMaia3Temperature(): number {
  const input = el('maia3Temperature') as HTMLInputElement;
  return Math.max(0.01, Math.min(5, Number(input.value) || 1));
}
function ctxSelectedMaia3TopP(): number {
  const input = el('maia3TopP') as HTMLInputElement;
  return Math.max(0.01, Math.min(1, Number(input.value) || 1));
}

function ctxStrengthFor(option: PlayEngineOption, level: number): number {
  if (option.family === 'maia3') return 1;
  if (option.family === 'sf') return SF_LEVELS[level].depth;
  if (option.family === 'lc0' && option.variant !== 'small') {
    if (!bt4SupportedSync()) return BIG_NETS[option.variant as BigNetKey].wasmLevels[level];
    return BIG_NET_LEVELS[level];
  }
  return LEVELS[option.family][level];
}

function ctxStrengthCaption(ctx: PlayContext): string {
  const option = ctxSelectedEngine(ctx);
  const level = ctxSelectedLevel();
  if (option.family === 'maia3') {
    const style = ctxSelectedMaia3Style();
    const suffix = style === 'argmax' ? 'deterministic top human move' : `sampled, temperature ${ctxSelectedMaia3Temperature().toFixed(2)}, top-p ${ctxSelectedMaia3TopP().toFixed(2)}`;
    const status = maia3Status ? ` ${maia3Status}.` : '';
    return `Maia3 predicts human moves at Elo ${ctxSelectedMaia3Elo()} — ${suffix}. No LC0/PUCT search; its WDL output predicts the human game outcome, not an engine eval.${status}`;
  }
  if (option.family === 'sf') {
    const sf = SF_LEVELS[level];
    return sf.skill >= 20 ? `full strength · depth ${sf.depth}` : `UCI skill ${sf.skill} · depth ${sf.depth}`;
  }
  const value = ctxStrengthFor(option, level);
  const cpuNote = option.family === 'lc0' && option.variant !== 'small' && !bt4SupportedSync()
    ? ' · CPU fallback (no WebGPU): expect several seconds per move'
    : '';
  if (option.variant === 'lqo') return `≈ ${value} visits per move — higher levels press harder for tricks${cpuNote}`;
  const base = option.family === 'lc0' ? `≈ ${value} visits per move` : `search depth ${value}`;
  return option.family === 'lc0' ? `${base} — strong even on Fastest; pick a Maia for a human-level opponent${cpuNote}` : base;
}

function ctxRenderLevelOptions(ctx: PlayContext): void {
  const select = selectEl('levelSelect');
  const previous = select.value;
  const option = ctxSelectedEngine(ctx);
  const field = select.closest('.field') as HTMLElement;
  if (option.family === 'maia3') {
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

function ctxRenderMaia3Controls(ctx: PlayContext): void {
  const isMaia3 = ctxSelectedEngine(ctx).family === 'maia3';
  el('maia3Controls').hidden = !isMaia3;
  const elo = ctxSelectedMaia3Elo();
  el('maia3EloValue').textContent = String(elo);
  const sample = ctxSelectedMaia3Style() === 'sample';
  (el('maia3Temperature') as HTMLInputElement).disabled = !sample;
  (el('maia3TopP') as HTMLInputElement).disabled = !sample;
  (el('maia3TemperatureField') as HTMLElement).style.opacity = sample ? '' : '0.5';
  (el('maia3TopPField') as HTMLElement).style.opacity = sample ? '' : '0.5';
}

function ctxSetEngineNote(text: string, warn = false): void {
  const note = el('engineNote');
  note.textContent = text;
  note.hidden = !text;
  note.classList.toggle('warn', warn);
}

function ctxShowDownloadProgress(label: string, loadedBytes?: number, totalBytes?: number, phase = 'Downloading'): void {
  renderLoadingProgress(el('dlProgress'), { label, phase, loadedBytes, totalBytes });
}

function ctxHideDownloadProgress(): void {
  hideLoadingProgress(el('dlProgress'));
}

function ctxEnsureLc0Small(ctx: PlayContext): Promise<Lc0PuctSearcher> {
  if (lc0Searcher) return Promise.resolve(lc0Searcher);
  if (!lc0LoadPromise) {
    ctxSetEngineNote('Loading Lc0 small net…');
    lc0LoadPromise = (async () => {
      const modelLoad = await loadLc0ModelForOrt(MODEL_URL, {
        cache: true,
        onProgress: (loaded, total) => ctxShowDownloadProgress('Lc0 small net', loaded, total),
      });
      ctxHideDownloadProgress();
      const evaluator = await Lc0OnnxEvaluator.create(modelLoad.model);
      lc0Searcher = new Lc0PuctSearcher(new CachedLc0Evaluator(evaluator, { maxEntries: 2048 }));
      ctxSetEngineNote('');
      return lc0Searcher;
    })().catch((error: Error) => {
      lc0LoadPromise = null;
      ctxHideDownloadProgress();
      ctxSetEngineNote(`Lc0 load failed: ${error.message}`, true);
      throw error;
    });
  }
  return lc0LoadPromise;
}

function ctxEnsureMaia3(ctx: PlayContext): Promise<Maia3BrowserEvaluator> {
  if (maia3Promise) return maia3Promise;
  maia3Promise = (async () => {
    ctxSetEngineNote('Loading Maia3 human model…');
    const evaluator = await Maia3BrowserEvaluator.create({
      selfElo: ctxSelectedMaia3Elo(),
      oppoElo: ctxSelectedMaia3Elo(),
      onProgress: (loaded, total) => ctxShowDownloadProgress('Maia3', loaded, total),
    });
    ctxHideDownloadProgress();
    const load = evaluator.modelLoad;
    const origin = load.cacheStatus === 'hit' ? 'from cache' : load.cacheStatus === 'miss' ? 'downloaded' : 'loaded';
    const integrity = load.sha256Valid === true ? ', sha256 ✓' : load.sha256Valid === false ? ', sha256 MISMATCH' : '';
    maia3Status = `Model ${origin} (${((load.bytes ?? 0) / 1e6).toFixed(0)}MB${integrity})`;
    ctxSetEngineNote('');
    return evaluator;
  })().catch((error: Error) => {
    maia3Promise = null;
    ctxHideDownloadProgress();
    ctxSetEngineNote(`Maia3 load failed: ${error.message}`, true);
    throw error;
  });
  return maia3Promise;
}

function ctxCpuEngineFor(ctx: PlayContext, option: PlayEngineOption): Promise<CpuEngine> {
  const existing = cpuEnginePromises.get(option.id);
  if (existing) return existing;
  const label = option.label;
  const created = (async (): Promise<CpuEngine> => {
    ctxSetEngineNote(`Loading ${label} (first use downloads the engine)…`);
    try {
      switch (option.family) {
        case 'sf':
          return new StockfishEngine({ depth: 4, threads: 1 }, stockfishFlavorUrl(option.variant === 'lite' ? 'lite-single' : 'single'));
        case 'reckless':
          return createRecklessEngine(await resolveDefaultRecklessVariantAssetFallback(recklessVariantByKey(defaultRecklessVariantKey()), false));
        case 'viridithas':
          return createViridithasEngine(await resolveDefaultViridithasVariantAssetFallback(viridithasVariantByKey(defaultViridithasVariantKey()), false));
        case 'berserk':
          return createBerserkEngine(await resolveDefaultBerserkVariantAssetFallback(berserkVariantByKey(defaultBerserkVariantKey()), false));
        case 'plentychess':
          return createPlentyChessEngine(await resolveDefaultPlentyChessVariantAssetFallback(plentyChessVariantByKey(defaultPlentyChessVariantKey()), false));
        default:
          throw new Error(`unsupported engine family ${option.family}`);
      }
    } finally {
      ctxSetEngineNote('');
    }
  })().catch((error: Error) => {
    cpuEnginePromises.delete(option.id);
    ctxSetEngineNote(`${label} load failed: ${error.message}`, true);
    throw error;
  });
  cpuEnginePromises.set(option.id, created);
  return created;
}

async function ctxRequestEngineMove(ctx: PlayContext, signal: AbortSignal): Promise<string | null> {
  const option = ctxSelectedEngine(ctx);
  const level = ctxSelectedLevel();
  const visitsOrDepth = ctxStrengthFor(option, level);
  if (option.family === 'maia3') {
    const player = await ctxEnsureMaia3(ctx);
    if (signal.aborted) return null;
    const choice = await player.chooseMove({ positions: ctx.positions }, {
      selfElo: ctxSelectedMaia3Elo(),
      oppoElo: ctxSelectedMaia3Elo(),
      style: ctxSelectedMaia3Style(),
      temperature: ctxSelectedMaia3Temperature(),
      topP: ctxSelectedMaia3TopP(),
    });
    return choice.move;
  }
  if (option.family === 'lc0' && option.variant === 'small') {
    const searcher = await ctxEnsureLc0Small(ctx);
    const result = await searcher.search({ positions: ctx.positions }, { visits: visitsOrDepth, signal, yieldEveryMs: 16, reuseTree: true, drawScore: PLAY_DRAW_SCORE, searchContemptLimit: PLAY_SEARCH_CONTEMPT_LIMIT });
    return result.move ?? null;
  }
  if (option.family === 'lc0') {
    const searcher = bigNetSearchers[option.variant as BigNetKey];
    if (!searcher.loaded) {
      ctxSetEngineNote(`Loading Lc0 ${searcher.config.name} (~${searcher.config.approxMb}MB on first use)…`);
      ctxShowDownloadProgress(`Lc0 ${searcher.config.name}`, undefined, undefined, 'Preparing');
      searcher.onDownloadProgress = (loaded, total) => ctxShowDownloadProgress(`Lc0 ${searcher.config.name}`, loaded, total);
    }
    const onAbort = () => searcher.cancel();
    signal.addEventListener('abort', onAbort, { once: true });
    try {
      const result = await searcher.search({ positions: ctx.positions }, {
        visits: visitsOrDepth,
        reuseTree: true,
        batchSize: searcher.config.recommendedBatchSize,
        batchPipelineDepth: searcher.config.recommendedPipelineDepth,
        evalCacheEntries: 2048,
        drawScore: option.variant === 'lqo' ? LQO_DRAW_SCORE : PLAY_DRAW_SCORE,
        searchContemptLimit: option.variant === 'lqo' ? LQO_SEARCH_CONTEMPT_LIMIT : PLAY_SEARCH_CONTEMPT_LIMIT,
        ...(option.variant === 'lqo' ? { cpuct: LQO_CPUCT } : {}),
      });
      ctxHideDownloadProgress();
      ctxSetEngineNote('');
      return result.cancelled ? null : result.move ?? null;
    } finally {
      signal.removeEventListener('abort', onAbort);
      ctxHideDownloadProgress();
    }
  }
  const engine = await ctxCpuEngineFor(ctx, option);
  if (option.family === 'sf') {
    engine.setOptions({ depth: visitsOrDepth, movetimeMs: undefined, skillLevel: SF_LEVELS[level].skill });
  } else {
    engine.setOptions({ depth: visitsOrDepth, movetimeMs: undefined });
  }
  return engine.bestMove(boardToFen(ctx.board), signal);
}

// ---------------------------------------------------------------------------
// Game flow (context-scoped)
// ---------------------------------------------------------------------------

function ctxCancelEngineTurn(ctx: PlayContext): void {
  ctx.abort.abort();
  ctx.abort = new AbortController();
  ctx.engineThinking = false;
  releaseWakeLock();
}

function ctxPriorFens(ctx: PlayContext): string[] {
  return ctx.positions.slice(0, -1).map(boardToFen);
}

function ctxCheckGameOver(ctx: PlayContext): void {
  const outcome = gameOutcome(ctx.board, ctxPriorFens(ctx));
  if (outcome) ctx.gameOver = outcome;
}

function ctxApplyMove(ctx: PlayContext, move: Move): void {
  if (!ctx.moves.length) {
    ctx.activeEngineId = selectEl('engineSelect').value;
    ctx.activeColor = selectEl('colorSelect').value as 'white' | 'black' | 'random';
  }
  ctx.sans.push(moveToSan(ctx.board, move));
  ctx.moves.push(move);
  ctx.board = makeMove(ctx.board, move);
  ctx.positions.push(ctx.board);
}

async function ctxEngineTurn(ctx: PlayContext): Promise<void> {
  if (ctx.gameOver || ctx.board.turn === ctx.humanColor) return;
  const signal = ctx.abort.signal;
  ctx.engineThinking = true;
  await requestWakeLock();
  ctxRender(ctx);
  let uci: string | null = null;
  try {
    uci = await ctxRequestEngineMove(ctx, signal);
  } catch (error) {
    if (signal.aborted) return;
    ctx.engineThinking = false;
    releaseWakeLock();
    ctxSetEngineNote(`Engine error: ${(error as Error).message}`, true);
    ctxRender(ctx);
    return;
  } finally {
    releaseWakeLock();
  }
  if (signal.aborted) return;
  ctx.engineThinking = false;
  const move = uci ? legalMoves(ctx.board).find((m) => moveToUci(m) === uci) : undefined;
  if (!move) {
    ctx.gameOver = { result: ctx.humanColor === 'w' ? '1-0' : '0-1', reason: uci ? `engine played illegal move ${uci}` : 'engine returned no move' };
    ctxRender(ctx);
    return;
  }
  ctxApplyMove(ctx, move);
  ctxCheckGameOver(ctx);
  ctxRender(ctx);
}

function ctxOnUserMove(ctx: PlayContext, from: Key, to: Key): void {
  if (ctx.engineThinking || ctx.gameOver || ctx.board.turn !== ctx.humanColor) { ctxRender(ctx); return; }
  const matching = matchUserMoves(ctx.board, from, to);
  if (!matching.length) { ctxRender(ctx); return; }
  if (matching.length > 1) {
    ctx.pendingPromotion = matching;
    ctxRender(ctx);
    return;
  }
  ctxApplyHumanMove(ctx, matching[0]);
}

function ctxApplyHumanMove(ctx: PlayContext, move: Move): void {
  ctx.pendingPromotion = null;
  ctxApplyMove(ctx, move);
  ctxCheckGameOver(ctx);
  ctxRender(ctx);
  if (!ctx.gameOver) void ctxEngineTurn(ctx);
}

function ctxNewGame(ctx: PlayContext): void {
  ctxCancelEngineTurn(ctx);
  const colorChoice = selectEl('colorSelect').value;
  ctx.humanColor = colorChoice === 'random' ? (Math.random() < 0.5 ? 'w' : 'b') : colorChoice === 'black' ? 'b' : 'w';
  ctx.orientation = ctx.humanColor === 'w' ? 'white' : 'black';
  ctx.startFen = startFenFor(ctxSelectedEngine(ctx), ctx.humanColor);
  ctx.board = parseFen(ctx.startFen);
  ctx.positions = [ctx.board];
  ctx.moves = [];
  ctx.sans = [];
  ctx.gameOver = null;
  ctx.pendingPromotion = null;
  lc0Searcher?.resetTree();
  for (const searcher of Object.values(bigNetSearchers)) {
    if (searcher.loaded) void searcher.resetTree();
  }
  ctxSetEngineNote('');
  ctxDisarmResign(ctx);
  ctx.activeEngineId = selectEl('engineSelect').value;
  ctx.activeColor = selectEl('colorSelect').value as 'white' | 'black' | 'random';
  ctxRender(ctx);
  if (ctx.humanColor === 'b') void ctxEngineTurn(ctx);
}

function ctxTakeback(ctx: PlayContext): void {
  if (!ctx.moves.length) return;
  ctxCancelEngineTurn(ctx);
  ctxDisarmResign(ctx);
  do {
    ctx.moves.pop();
    ctx.sans.pop();
    ctx.positions.pop();
    ctx.board = ctx.positions[ctx.positions.length - 1];
  } while (ctx.moves.length && ctx.board.turn !== ctx.humanColor);
  ctx.gameOver = null;
  ctx.pendingPromotion = null;
  ctxRender(ctx);
  if (ctx.board.turn !== ctx.humanColor) void ctxEngineTurn(ctx);
}

function ctxResign(ctx: PlayContext): void {
  if (ctx.gameOver || !ctx.moves.length) return;
  if (!ctx.resignArmed) {
    ctx.resignArmed = true;
    const btn = buttonEl('resign');
    btn.textContent = 'Click again to resign';
    btn.classList.add('danger');
    if (ctx.resignArmTimer) clearTimeout(ctx.resignArmTimer);
    ctx.resignArmTimer = setTimeout(() => ctxDisarmResign(ctx), 4000);
    return;
  }
  ctxDisarmResign(ctx);
  ctxCancelEngineTurn(ctx);
  ctx.gameOver = { result: ctx.humanColor === 'w' ? '0-1' : '1-0', reason: 'resignation' };
  ctxRender(ctx);
}

function ctxDisarmResign(ctx: PlayContext): void {
  ctx.resignArmed = false;
  if (ctx.resignArmTimer) { clearTimeout(ctx.resignArmTimer); ctx.resignArmTimer = null; }
  const btn = buttonEl('resign');
  if (btn.classList.contains('danger')) {
    btn.classList.remove('danger');
    btn.textContent = 'Resign';
  }
}

function ctxExportPgn(ctx: PlayContext): string {
  const tree = new GameTree(ctx.startFen);
  let replay = parseFen(ctx.startFen);
  for (const move of ctx.moves) {
    tree.addMove(move);
    replay = makeMove(replay, move);
  }
  const engineName = ctxSelectedEngine(ctx).label;
  const date = new Date().toISOString().slice(0, 10).replaceAll('-', '.');
  return gameTreeToPgn(tree, {
    Event: 'Casual browser game',
    Site: location.host || 'local',
    Date: date,
    White: ctx.humanColor === 'w' ? 'You' : engineName,
    Black: ctx.humanColor === 'b' ? 'You' : engineName,
    ...(ctx.startFen === START_FEN ? {} : { SetUp: '1', FEN: ctx.startFen }),
  }, ctx.gameOver?.result ?? '*');
}

// ---------------------------------------------------------------------------
// Rendering (context-scoped)
// ---------------------------------------------------------------------------

function ctxVerdictText(ctx: PlayContext): string {
  if (!ctx.gameOver) return '';
  const { result, reason } = ctx.gameOver;
  if (result === '1/2-1/2') return `Draw — ${reason}`;
  const humanWon = (result === '1-0') === (ctx.humanColor === 'w');
  return humanWon ? `You win — ${reason}` : `${ctxSelectedEngine(ctx).label} wins — ${reason}`;
}

function ctxStatusText(ctx: PlayContext): string {
  if (ctx.gameOver) return ctxVerdictText(ctx);
  if (ctx.pendingPromotion) return 'Choose a promotion piece';
  if (ctx.engineThinking) return `${ctxSelectedEngine(ctx).label} is thinking…`;
  if (!ctx.moves.length && ctx.board.turn === ctx.humanColor) return `Your move — you play ${ctx.humanColor === 'w' ? 'White' : 'Black'}. Moving a piece starts the game.`;
  return ctx.board.turn === ctx.humanColor ? 'Your move' : `${ctxSelectedEngine(ctx).label} to move`;
}

function ctxRenderMoveList(ctx: PlayContext): void {
  const list = el('moveList');
  if (!ctx.sans.length) { list.innerHTML = '<span class="placeholder">No moves yet</span>'; return; }
  const parts: string[] = [];
  for (let i = 0; i < ctx.sans.length; i += 2) {
    const number = i / 2 + 1;
    const white = ctx.sans[i];
    const black = ctx.sans[i + 1];
    parts.push(`<span class="num">${number}.</span> <span class="san">${white}</span>${black ? ` <span class="san">${black}</span>` : ''}`);
  }
  list.innerHTML = parts.join(' ');
  list.scrollTop = list.scrollHeight;
}

function ctxRenderRestartBanner(ctx: PlayContext): void {
  const banner = el('restartBanner');
  if (!ctx.pendingRestart) { banner.hidden = true; return; }
  banner.hidden = false;
  const engineLabel = ENGINE_OPTIONS.find((o) => o.id === ctx.pendingRestart?.engine)?.label ?? 'the new engine';
  const colorLabel = ctx.pendingRestart.color === 'random' ? 'random side' : `as ${ctx.pendingRestart.color}`;
  el('restartMessage').textContent = `Start a new game vs ${engineLabel}, ${colorLabel}? The current game will be discarded.`;
}

function ctxRenderPromotionPicker(ctx: PlayContext): void {
  el('promoPicker').hidden = true;
  if (!ctx.pendingPromotion) {
    hidePromotionOverlay(el('ground'));
    return;
  }
  showPromotionOverlay({
    boardContainer: el('ground'),
    orientation: ctx.orientation,
    color: ctx.humanColor,
    choices: ctx.pendingPromotion,
    onPick: (move) => ctxApplyHumanMove(ctx, move),
    onCancel: () => { ctx.pendingPromotion = null; ctxRender(ctx); },
  });
}

function ctxEngineOptionState(option: PlayEngineOption): { disabled: boolean; suffix: string } {
  if (option.family !== 'lc0' || option.variant === 'small') return { disabled: false, suffix: '' };
  return bigNetOptionState(BIG_NETS[option.variant as BigNetKey]);
}

function ctxEngineOptionHtml(option: PlayEngineOption): string {
  const { disabled, suffix } = ctxEngineOptionState(option);
  return `<option value="${option.id}"${disabled ? ' disabled' : ''}>${option.label}${suffix}</option>`;
}

function ctxRefreshEngineOptions(ctx: PlayContext): void {
  const select = selectEl('engineSelect');
  const selected = select.value;
  const group = (key: PlayEngineOption['group']) => ENGINE_OPTIONS.filter((option) => option.group === key).map(ctxEngineOptionHtml).join('');
  select.innerHTML = `<optgroup label="Human-like (Maia, plays like a rated human)">${group('human')}</optgroup>`
    + `<optgroup label="Odds bots (give you material, then hunt for tricks)">${group('odds')}</optgroup>`
    + `<optgroup label="Engines (strong at any level)">${group('engine')}</optgroup>`;
  const selectedOption = ENGINE_OPTIONS.find((option) => option.id === selected);
  if (selectedOption && !ctxEngineOptionState(selectedOption).disabled) select.value = selected;
  else select.value = 'maia3';
  ctxRenderEngineCaution(ctx);
}

function ctxRenderEngineCaution(ctx: PlayContext): void {
  const option = ctxSelectedEngine(ctx);
  const caution = el('engineCaution');
  if (option.family === 'lc0' && option.variant !== 'small') {
    const config = BIG_NETS[option.variant as BigNetKey];
    const memory = bigNetMemoryCaution(config);
    const odds = option.variant === 'lqo'
      ? ' The bot starts without its queen and plays for traps — the Lichess LeelaQueenOdds net. ' : ' ';
    const cpu = bt4SupportedSync() ? '' : ' No WebGPU here: runs on the CPU (wasm) fallback with reduced visit budgets — expect several seconds per move.';
    caution.textContent = `First move downloads the ~${config.approxMb}MB net.${odds}${memory ?? ''}${cpu}`.trimEnd();
    caution.hidden = false;
  } else {
    caution.hidden = true;
  }
}

function ctxRender(ctx: PlayContext): void {
  el('status').textContent = ctxStatusText(ctx);
  el('status').classList.toggle('over', !!ctx.gameOver);
  ctxRenderMaia3Controls(ctx);
  el('levelCaption').textContent = ctxStrengthCaption(ctx);
  buttonEl('takeback').disabled = !ctx.moves.length || !!ctx.pendingPromotion;
  buttonEl('resign').disabled = !!ctx.gameOver || !ctx.moves.length;
  ctxRenderMoveList(ctx);
  ctxRenderPromotionPicker(ctx);
  ctxRenderRestartBanner(ctx);
  el('pgnOut').textContent = '';
  const humanCanMove = !ctx.engineThinking && !ctx.gameOver && !ctx.pendingPromotion && ctx.board.turn === ctx.humanColor;
  const lastUci = ctx.moves.length ? moveToUci(ctx.moves[ctx.moves.length - 1]) : undefined;
  const config = {
    orientation: ctx.orientation,
    fen: boardToFen(ctx.board).split(' ')[0],
    turnColor: ctx.board.turn === 'w' ? 'white' as const : 'black' as const,
    coordinates: true,
    trustAllEvents: true,
    check: boardCheck(ctx.board),
    highlight: { lastMove: true, check: true },
    animation: { enabled: true, duration: 160 },
    movable: {
      free: false,
      color: humanCanMove ? (ctx.humanColor === 'w' ? 'white' as const : 'black' as const) : undefined,
      dests: humanCanMove ? legalDests(ctx.board) : new Map<Key, Key[]>(),
      showDests: humanCanMove,
      events: { after: (from: Key, to: Key) => ctxOnUserMove(ctx, from, to) },
    },
    lastMove: lastUci ? [lastUci.slice(0, 2) as Key, lastUci.slice(2, 4) as Key] : undefined,
  };
  if (!ctx.ground) ctx.ground = Chessground(el('ground'), config);
  else ctx.ground.set(config);
}

// ---------------------------------------------------------------------------
// Restart banner logic
// ---------------------------------------------------------------------------

function ctxMaybeQueueRestart(ctx: PlayContext): void {
  if (!ctx.moves.length || ctx.gameOver || ctx.engineThinking) {
    ctx.pendingRestart = null;
    ctxRender(ctx);
    return;
  }
  ctx.pendingRestart = {
    engine: selectEl('engineSelect').value,
    color: selectEl('colorSelect').value as 'white' | 'black' | 'random',
  };
  ctxRender(ctx);
}

function ctxConfirmRestart(ctx: PlayContext): void {
  ctx.pendingRestart = null;
  void ctxNewGame(ctx);
}

function ctxDismissRestart(ctx: PlayContext): void {
  selectEl('engineSelect').value = ctx.activeEngineId;
  selectEl('colorSelect').value = ctx.activeColor;
  ctx.lastEngineId = ctx.activeEngineId;
  ctx.pendingRestart = null;
  ctxRenderLevelOptions(ctx);
  ctxRenderMaia3Controls(ctx);
  ctxRenderEngineCaution(ctx);
  ctxRender(ctx);
}

// ---------------------------------------------------------------------------
// Mount/init
// ---------------------------------------------------------------------------

function ctxResetGameStateForMount(ctx: PlayContext): void {
  ctxCancelEngineTurn(ctx);
  selectEl('colorSelect').value = 'white';
  ctx.humanColor = 'w';
  ctx.orientation = 'white';
  ctx.startFen = startFenFor(ctxSelectedEngine(ctx), ctx.humanColor);
  ctx.board = parseFen(ctx.startFen);
  ctx.positions = [ctx.board];
  ctx.moves = [];
  ctx.sans = [];
  ctx.gameOver = null;
  ctx.pendingPromotion = null;
  ctx.pendingRestart = null;
  ctx.activeEngineId = selectEl('engineSelect').value;
  ctx.activeColor = selectEl('colorSelect').value as 'white' | 'black' | 'random';
  ctxSetEngineNote('');
  ctxDisarmResign(ctx);
}

function ctxInit(ctx: PlayContext): void {
  ctxRefreshEngineOptions(ctx);
  selectEl('engineSelect').value = 'maia3';
  ctx.activeEngineId = 'maia3';
  ctx.activeColor = 'white';
  ctxResetGameStateForMount(ctx);
  ctxRenderLevelOptions(ctx);

  trackListener(ctx, el('newGame'), 'click', () => ctxNewGame(ctx));
  trackListener(ctx, el('takeback'), 'click', () => ctxTakeback(ctx));
  trackListener(ctx, el('resign'), 'click', () => ctxResign(ctx));
  trackListener(ctx, el('flip'), 'click', () => { ctx.orientation = ctx.orientation === 'white' ? 'black' : 'white'; ctxRender(ctx); });
  trackListener(ctx, el('exportPgn'), 'click', () => { el('pgnOut').textContent = ctxExportPgn(ctx); });
  trackListener(ctx, el('copyPgn'), 'click', () => {
    void navigator.clipboard.writeText(ctxExportPgn(ctx)).then(() => {
      const btn = buttonEl('copyPgn');
      const original = btn.textContent;
      btn.textContent = 'Copied';
      btn.disabled = true;
      setTimeout(() => { btn.textContent = original; btn.disabled = false; }, 1500);
    });
  });
  trackListener(ctx, el('confirmRestart'), 'click', () => ctxConfirmRestart(ctx));
  trackListener(ctx, el('dismissRestart'), 'click', () => ctxDismissRestart(ctx));
  trackListener(ctx, selectEl('colorSelect'), 'change', () => {
    if (!ctx.moves.length) {
      ctxCancelEngineTurn(ctx);
      const choice = selectEl('colorSelect').value;
      ctx.humanColor = choice === 'random' ? (Math.random() < 0.5 ? 'w' : 'b') : (choice === 'black' ? 'b' : 'w');
      ctx.orientation = ctx.humanColor === 'w' ? 'white' : 'black';
      ctx.startFen = startFenFor(ctxSelectedEngine(ctx), ctx.humanColor);
      ctx.board = parseFen(ctx.startFen);
      ctx.positions = [ctx.board];
      ctx.gameOver = null;
      ctx.activeEngineId = selectEl('engineSelect').value;
      ctx.activeColor = selectEl('colorSelect').value as 'white' | 'black' | 'random';
      ctxRender(ctx);
      if (ctx.humanColor === 'b') void ctxEngineTurn(ctx);
    } else {
      ctxMaybeQueueRestart(ctx);
    }
  });
  trackListener(ctx, selectEl('engineSelect'), 'change', () => {
    const option = ctxSelectedEngine(ctx);
    if (option.family === 'maia3' && ctx.lastEngineId.startsWith('maia-')) {
      const carried = Number(ctx.lastEngineId.slice('maia-'.length));
      if (Number.isFinite(carried)) {
        (el('maia3Elo') as HTMLInputElement).value = String(Math.max(MAIA3_MIN_ELO, Math.min(MAIA3_MAX_ELO, carried)));
      }
    }
    ctx.lastEngineId = option.id;
    ctxRenderLevelOptions(ctx);
    ctxRenderMaia3Controls(ctx);
    ctxRenderEngineCaution(ctx);
    if (!ctx.moves.length) {
      ctxCancelEngineTurn(ctx);
      ctx.orientation = ctx.humanColor === 'w' ? 'white' : 'black';
      ctx.startFen = startFenFor(ctxSelectedEngine(ctx), ctx.humanColor);
      ctx.board = parseFen(ctx.startFen);
      ctx.positions = [ctx.board];
      ctx.gameOver = null;
      ctx.activeEngineId = selectEl('engineSelect').value;
      ctx.activeColor = selectEl('colorSelect').value as 'white' | 'black' | 'random';
      ctxRender(ctx);
      if (ctx.humanColor === 'b') void ctxEngineTurn(ctx);
    } else if (!ctx.gameOver && !ctx.engineThinking) {
      ctxMaybeQueueRestart(ctx);
      ctxRender(ctx);
    } else {
      ctxRender(ctx);
    }
  });
  trackListener(ctx, selectEl('levelSelect'), 'change', () => ctxRender(ctx));
  trackListener(ctx, selectEl('maia3Style'), 'change', () => ctxRender(ctx));
  trackListener(ctx, el('maia3Elo'), 'input', () => ctxRender(ctx));
  trackListener(ctx, el('maia3Temperature'), 'input', () => ctxRender(ctx));
  trackListener(ctx, el('maia3TopP'), 'input', () => ctxRender(ctx));
  if (!isV0DeployProfile()) {
    void probeBt4Support().then(() => ctxRefreshEngineOptions(ctx));
    void checkBigNetAsset(BIG_NETS.bt4, () => ctxRefreshEngineOptions(ctx));
    void checkBigNetAsset(T3_NET, () => ctxRefreshEngineOptions(ctx));
    void checkBigNetAsset(LQO_NET, () => ctxRefreshEngineOptions(ctx));
  } else {
    void probeBt4Support().then(() => ctxRefreshEngineOptions(ctx));
    void checkBigNetAsset(LQO_NET, () => ctxRefreshEngineOptions(ctx));
  }
  ctxRender(ctx);
}

export function mountPlayBrowser(): () => void {
  const ctx = createPlayContext();
  ctxInit(ctx);

  // Test hook for automated browser checks: synthetic chessground drags are
  // unreliable, so smokes call this to route through the real user-move path.
  const hook = (from: string, to: string) => ctxOnUserMove(ctx, from as Key, to as Key);
  (globalThis as unknown as { __playUserMove?: (from: string, to: string) => void }).__playUserMove = hook;

  return () => {
    ctxCancelEngineTurn(ctx);
    ctxDisarmResign(ctx);
    (ctx.ground as { destroy?: () => void } | null)?.destroy?.();
    ctx.ground = null;
    for (const { target, type, fn } of ctx.listeners) {
      target.removeEventListener(type, fn);
    }
    ctx.listeners = [];
    // Only clear the test hook if it is still ours (a newer re-entrant mount
    // may have installed its own).
    const g = globalThis as unknown as { __playUserMove?: (from: string, to: string) => void };
    if (g.__playUserMove === hook) delete g.__playUserMove;
  };
}
