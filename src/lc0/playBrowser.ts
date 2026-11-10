// Play vs engine: a human-friendly game page on top of the same engine stack
// used by the arena and analysis pages. Engines load lazily on first move.
import { Chessground } from 'chessground';
import type { Key } from 'chessground/types';
import { type BoardState, boardToFen, type Color, parseFen, START_FEN } from '../chess/board.ts';
import { type Move, moveToUci } from '../chess/moveCodec.ts';
import { legalMoves, makeMove } from '../chess/movegen.ts';
import { sanToMove } from '../chess/pgn.ts';
import { moveToSan } from '../chess/san.ts';
import { createBrowserSquareformerRuntimeEvaluator } from '../nn/browserRuntimeEvaluator.ts';
import { CachedEvaluator, type Evaluator } from '../nn/evaluator.ts';
import {
  BROWSER_RUNTIME_AUDIT_EVENT,
  type BrowserRuntimeAuditDetail,
  browserRuntimeAuditIdentity,
  clearRuntimeFallbackWarning,
  updateRuntimeFallbackWarning,
} from '../nn/runtimeAudit.ts';
import { chooseMove, montyLitePuctPolicy } from '../search/puct.ts';
import { takePlayHandoff, writeAnalysisHandoff } from './analysisHandoff.ts';
import { createAnalysisKeyboardBoard } from './analysisKeyboardBoard.ts';
import { resolvePublicAssetUrl } from './assetUrls.ts';
import { acquireBigNetSearcher, type BigNetKey, peekBigNetSearcher, releaseUnusedBigNetSearchers } from './bigNetSessionPool.ts';
import { boardCheck, hidePromotionOverlay, legalDests, matchUserMoves, showPromotionOverlay } from './boardUx.ts';
import { buttonEl, el, htmlEscape, maybeEl, selectEl } from './browserDom.ts';
import {
  BIG_NETS,
  type Bt4SearchOptions,
  bigNetAssetStatusSync,
  bigNetMemoryCaution,
  bigNetOptionState,
  bt4SupportedSync,
  checkBigNetAsset,
  LQO_NET,
  probeBt4Support,
  T3_NET,
} from './bt4Engine.ts';
import { inactiveCpuEnginesToEvict } from './cpuEngineResidency.ts';
import { type GameResultCode, gameOutcome } from './engineBattle.ts';
import { type EngineFamily, enginePlayLevels, enginePlayOptions, isV0DeployProfile } from './engineCatalog.ts';
import { cpuEngineHashMbForSurface, createDefaultBrowserUciEngine, isDefaultBrowserUciFamily } from './engineProvision.ts';
import { GameTree } from './gameTree.ts';
import { createWorkspaceAutosave, decodePlaySnapshot, hasExplicitWorkspaceInput, readWorkspace } from './workspaceAutosave.ts';
import { hideLoadingProgress, renderLoadingProgress } from './loadingProgress.ts';
import { lqoBlackBookMove, lqoWhiteBookMove, lqoWhiteFirstPolicyBookMove } from './lqoOpeningBook.ts';
import { MAIA3_DEFAULT_ELO, MAIA3_MAX_ELO, MAIA3_MIN_ELO, Maia3BrowserEvaluator, type Maia3MoveStyle } from './maia3.ts';
import { loadLc0ModelForOrt } from './modelCache.ts';
import {
  displayedPly,
  type MoveNavAction,
  moveNavActionForKey,
  moveNavAvailability,
  navigateViewPly,
  normalizeViewPly,
  searchProgressLabel,
  viewPlyFor,
} from './moveNavigation.ts';
import { CachedLc0Evaluator, Lc0OnnxEvaluator } from './onnxEvaluator.ts';
import { type OpeningNameIndex, openingPositionKey } from './openingNames.ts';
import { playMaterial } from './playMaterial.ts';
import { playGameToPgn, playMoveLabel, playMoveRows, type PlayPosition, playTakebackPly, replayPlayPosition } from './playPosition.ts';
import { DEFAULT_PLAY_PREFERENCES, loadPlayPreferences, type PlayPreferences, type PreferenceStorage, savePlayPreferences } from './playPreferences.ts';
import { committedPlayOpponent, type PlayOpponentState, transitionPlayOpponent } from './playOpponentState.ts';
import { Lc0PuctSearcher } from './search.ts';
import { StockfishEngine, stockfishFlavorUrl } from './stockfishEngine.ts';
import { defaultStormphraxVariantKey, stormphraxVariantByKey, stormphraxVariantUnsupportedReason } from './stormphraxVariants.ts';

const params = new URLSearchParams(location.search);
const DEFAULT_MODEL_URL = resolvePublicAssetUrl('/models/lc0/t1-256x10-distilled-swa-2432500.batch1.f16.qdq8.onnx');
const MODEL_URL = isV0DeployProfile() ? DEFAULT_MODEL_URL : resolvePublicAssetUrl(params.get('model') ?? DEFAULT_MODEL_URL);
const CENTIPAWN_MODEL_URL = resolvePublicAssetUrl('/models/bt4_soap_rem_c19000_final.onnx');
const CENTIPAWN_META_URL = resolvePublicAssetUrl('/models/bt4_soap_rem_c19000_final.meta.json');
const CENTIPAWN_TVMJS_MANIFEST_URL = resolvePublicAssetUrl('/runtimes/centipawn-tvmjs-webgpu/bt4-soap-rem-c19000-final/f32/v2-shape-k16/manifest.json');

type PlayFamily = 'maia3' | EngineFamily;

interface PlayEngineOption {
  id: string;
  label: string;
  family: PlayFamily;
  /** Family-specific: Maia Elo, lc0 net key ('small' | 't3' | 'bt4' | 'lqo'), or stockfish kind. */
  variant: string;
  group: 'human' | 'odds' | 'engine';
}

const ENGINE_OPTIONS: PlayEngineOption[] = [
  { id: 'maia3', label: 'Maia3 · Human-like opponent', family: 'maia3', variant: 'maia3', group: 'human' },
  ...enginePlayOptions().map(({ id, label, family, variant, group }) => ({ id, label, family, variant, group })),
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

/** Big nets are far slower per visit; keep upper levels playable. */
const BIG_NET_LEVELS = [4, 16, 64, 256, 800];

/**
 * WDL draw contempt against the human opponent. Negative values reduce the
 * value assigned to draws. The LQO constants below follow its color-specific
 * upstream settings; regular Lc0 opponents use a smaller adjustment.
 */
const PLAY_DRAW_SCORE = -0.25;
/**
 * Model the human opponent as a budget-limited searcher (see
 * docs/search_contempt_design.md). The limit reflects the opponent's search
 * ability, not ours, so it stays fixed across the visit ladder; 16 is the
 * A/B-validated point (92% vs 58% baseline at queen odds vs Maia 1900).
 */
const PLAY_SEARCH_CONTEMPT_LIMIT = 16;
/** LeelaQueenOdds README search settings, including SwapColors when LQO plays Black. */
const LQO_CPUCT = 1.5;
const LQO_FPU_VALUE = 0.4;
const LQO_WHITE_DRAW_SCORE = -0.4;
const LQO_BLACK_DRAW_SCORE = -0.6;
const LQO_WHITE_SEARCH_CONTEMPT_LIMIT = 40;
const LQO_BLACK_SEARCH_CONTEMPT_LIMIT = 32;

// ---------------------------------------------------------------------------
// Module-level engine caches (intentionally persistent across mount/unmount
// to avoid re-downloading engines on navigation)
// ---------------------------------------------------------------------------
const BIG_NET_KEYS: readonly BigNetKey[] = ['bt4', 't3', 'lqo'];
let lc0Searcher: Lc0PuctSearcher | null = null;
let lc0LoadPromise: Promise<Lc0PuctSearcher> | null = null;
let centipawnEvaluator: CachedEvaluator | null = null;
let centipawnLoadPromise: Promise<Evaluator> | null = null;
let centipawnEvaluatorGeneration = 0;
const playRuntimeFallbackAudits = new Map<string, BrowserRuntimeAuditDetail>();
const cpuEnginePromises = new Map<string, Promise<CpuEngine>>();
let maia3Promise: Promise<Maia3BrowserEvaluator> | null = null;
/** One-line model/cache status shown in the caption once Maia3 has loaded. */
let maia3Status: string | null = null;

interface CpuEngine {
  setOptions(options: { depth?: number; movetimeMs?: number; threads?: number; skillLevel?: number }): void;
  bestMove(fen: string, signal?: AbortSignal): Promise<string | null>;
  dispose?(): void;
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

/** Start FEN for the current game; odds opponents remove their own queen. */
function startFenFor(option: PlayEngineOption, human: Color): string {
  if (option.variant !== 'lqo') return START_FEN;
  return human === 'w' ? 'rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1' : 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1';
}

// ---------------------------------------------------------------------------
// Factory: scoped per-mount play context. All game/UI state lives here so
// re-entrant mounts (SvelteKit preloading, error boundaries) cannot clobber
// each other. The abort signal replaces the old gameSeq counter as the
// stale-write guard for async engine callbacks.
// ---------------------------------------------------------------------------
interface PlayContext {
  autosave: ReturnType<typeof createWorkspaceAutosave> | null;
  restoredGame: boolean;
  abort: AbortController;
  disposed: boolean;
  ground: ReturnType<typeof Chessground> | null;
  startFen: string;
  /** Imported path is immutable; resets replay it and takebacks cannot cross it. */
  practicePosition: PlayPosition | null;
  practiceBoundary: number;
  handoffNotice: string;
  board: BoardState;
  positions: BoardState[];
  moves: Move[];
  sans: string[];
  humanColor: Color;
  orientation: 'white' | 'black';
  gameOver: { result: GameResultCode; reason: string } | null;
  engineThinking: boolean;
  pendingPromotion: Move[] | null;
  /**
   * Browsed ply (position after N half-moves) or null when following the live
   * game. View-only: the engine, takeback and the human-move path all keep
   * operating on the live `board`/`positions`. A browsing user stays where
   * they are when the engine replies (lichess behaviour).
   */
  viewPly: number | null;
  /** Latest search progress rendered into #dlProgress, so its live-position label can follow view changes. */
  searchProgress: { label: string; progress: SearchProgress } | null;
  /** Last view key the move list was scrolled for, so engine replies do not yank a manually scrolled list. */
  moveListScrollKey: string;
  opponent: PlayOpponentState;
  lastEngineId: string;
  pendingPreferredEngineId: string | null;
  pendingPreferredEngineTimer: ReturnType<typeof setTimeout> | null;
  resignArmed: boolean;
  resignArmTimer: ReturnType<typeof setTimeout> | null;
  /** Two-step confirm for abandoning a game in progress via New game. */
  newGameArmed: boolean;
  newGameArmTimer: ReturnType<typeof setTimeout> | null;
  /** UI-only: the player pressed Play or made a move, so the sidebar shows the game, not the setup. */
  started: boolean;
  /** UI-only: the opponent settings are expanded during a game ("Change"). */
  configOpen: boolean;
  keyboardBoard: ReturnType<typeof createAnalysisKeyboardBoard> | null;
  lastProgressAnnouncement: number;
  engineReady: boolean;
  preparingEngine: boolean;
  enginePreparationGeneration: number;
  // Bound listeners for cleanup
  listeners: Array<{ type: string; target: EventTarget; fn: EventListenerOrEventListenerObject }>;
}

function createPlayContext(): PlayContext {
  return {
    autosave: null,
    restoredGame: false,
    abort: new AbortController(),
    disposed: false,
    ground: null,
    startFen: START_FEN,
    practicePosition: null,
    practiceBoundary: 0,
    handoffNotice: '',
    board: parseFen(START_FEN),
    positions: [parseFen(START_FEN)],
    moves: [],
    sans: [],
    humanColor: 'w',
    orientation: 'white',
    gameOver: null,
    engineThinking: false,
    pendingPromotion: null,
    viewPly: null,
    searchProgress: null,
    moveListScrollKey: '',
    opponent: committedPlayOpponent(DEFAULT_PLAY_PREFERENCES),
    lastEngineId: 'maia3',
    pendingPreferredEngineId: null,
    pendingPreferredEngineTimer: null,
    resignArmed: false,
    resignArmTimer: null,
    newGameArmed: false,
    newGameArmTimer: null,
    started: false,
    configOpen: false,
    keyboardBoard: null,
    lastProgressAnnouncement: -1,
    engineReady: false,
    preparingEngine: false,
    enginePreparationGeneration: 0,
    listeners: [],
  };
}

function trackListener(ctx: PlayContext, target: EventTarget, type: string, fn: EventListenerOrEventListenerObject): void {
  target.addEventListener(type, fn);
  ctx.listeners.push({ type, target, fn });
}

function ctxIsDisposed(ctx: PlayContext): boolean {
  return ctx.disposed;
}

function ctxAsyncUiIsCurrent(ctx: PlayContext, generation: number): boolean {
  return !ctxIsDisposed(ctx) && generation === ctx.enginePreparationGeneration;
}

// ---------------------------------------------------------------------------
// Context-scoped functions (all game state and UI logic)
// ---------------------------------------------------------------------------

function ctxSelectedEngine(ctx: PlayContext): PlayEngineOption {
  return ENGINE_OPTIONS.find((option) => option.id === ctx.opponent.active.engineId) ?? ENGINE_OPTIONS[0];
}
function ctxDraftEngine(): PlayEngineOption {
  return ENGINE_OPTIONS.find((option) => option.id === selectEl('engineSelect').value) ?? ENGINE_OPTIONS[0];
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

function playPreferenceStorage(): PreferenceStorage | undefined {
  try {
    return globalThis.localStorage;
  } catch {
    return undefined;
  }
}

function ctxCurrentPlayPreferences(ctx: PlayContext): PlayPreferences {
  return {
    engineId: ctxDraftEngine().id,
    level: ctxSelectedLevel(),
    color: selectEl('colorSelect').value as PlayPreferences['color'],
    maiaElo: ctxSelectedMaia3Elo(),
    maiaStyle: ctxSelectedMaia3Style(),
    maiaTemperature: ctxSelectedMaia3Temperature(),
    maiaTopP: ctxSelectedMaia3TopP(),
  };
}

function ctxPersistPlayPreferences(ctx: PlayContext): void {
  savePlayPreferences(playPreferenceStorage(), ctx.opponent.active);
}

function ctxStrengthFor(option: PlayEngineOption, level: number): number {
  if (option.family === 'maia3') return 1;
  if (option.family === 'sf') return SF_LEVELS[level].depth;
  if (option.family === 'lc0' && option.variant !== 'small') {
    if (!bt4SupportedSync()) return BIG_NETS[option.variant as BigNetKey].wasmLevels[level];
    return BIG_NET_LEVELS[level];
  }
  return enginePlayLevels(option.family)[level];
}

function ctxStrengthCaption(ctx: PlayContext): string {
  const option = ctxDraftEngine();
  const level = ctxSelectedLevel();
  if (option.family === 'maia3') {
    const style = ctxSelectedMaia3Style();
    const suffix =
      style === 'argmax'
        ? 'deterministic top human move'
        : `sampled, temperature ${ctxSelectedMaia3Temperature().toFixed(2)}, top-p ${ctxSelectedMaia3TopP().toFixed(2)}`;
    const status = maia3Status ? ` ${maia3Status}.` : '';
    return `Maia3 predicts human moves at Elo ${ctxSelectedMaia3Elo()} — ${suffix}. No LC0/PUCT search; its WDL output predicts the human game outcome, not an engine eval.${status}`;
  }
  if (option.family === 'sf') {
    const sf = SF_LEVELS[level];
    return sf.skill >= 20 ? `full strength · depth ${sf.depth}` : `UCI skill ${sf.skill} · depth ${sf.depth}`;
  }
  const value = ctxStrengthFor(option, level);
  const cpuNote =
    option.family === 'lc0' && option.variant !== 'small' && !bt4SupportedSync() ? ' · CPU fallback (no WebGPU): expect several seconds per move' : '';
  if (option.variant === 'lqo') return `≈ ${value} visits per move · specialized queen-odds search${cpuNote}`;
  const isNeuralSearch = option.family === 'lc0' || option.family === 'centipawn';
  const base = isNeuralSearch ? `≈ ${value} visits per move` : `search depth ${value}`;
  return isNeuralSearch ? `${base} — strong even on Fastest; pick a Maia for a human-level opponent${cpuNote}` : base;
}

function ctxLqoSearchOptions(ctx: PlayContext): Pick<Bt4SearchOptions, 'drawScore' | 'searchContemptLimit' | 'cpuct' | 'fpu' | 'swapColors'> {
  const lqoPlaysBlack = ctx.humanColor === 'w';
  return {
    drawScore: lqoPlaysBlack ? LQO_BLACK_DRAW_SCORE : LQO_WHITE_DRAW_SCORE,
    searchContemptLimit: lqoPlaysBlack ? LQO_BLACK_SEARCH_CONTEMPT_LIMIT : LQO_WHITE_SEARCH_CONTEMPT_LIMIT,
    cpuct: LQO_CPUCT,
    fpu: LQO_FPU_VALUE,
    swapColors: lqoPlaysBlack,
  };
}

function ctxRenderLevelOptions(ctx: PlayContext): void {
  const select = selectEl('levelSelect');
  const previous = select.value;
  const option = ctxDraftEngine();
  const field = select.closest('.field') as HTMLElement;
  field.hidden = option.family === 'maia3';
  const labels = option.family === 'sf' ? SF_LEVELS.map((sf) => sf.label) : EFFORT_LEVEL_NAMES.map((name, i) => `${i + 1} · ${name}`);
  select.innerHTML = labels.map((label, i) => `<option value="${i}">${label}</option>`).join('');
  select.value = previous && Number(previous) < LEVEL_COUNT ? previous : '2';
}

function ctxRenderMaia3Controls(ctx: PlayContext): void {
  const isMaia3 = ctxDraftEngine().family === 'maia3';
  el('maia3Controls').hidden = !isMaia3;
  el('maia3Advanced').hidden = !isMaia3;
  const elo = ctxSelectedMaia3Elo();
  el('maia3EloValue').textContent = String(elo);
  el('maia3Elo').setAttribute('aria-valuetext', `${elo} rating · ${maiaRatingBand(elo)}`);
  const sample = ctxSelectedMaia3Style() === 'sample';
  (el('maia3Temperature') as HTMLInputElement).disabled = !sample;
  (el('maia3TopP') as HTMLInputElement).disabled = !sample;
  (el('maia3TemperatureField') as HTMLElement).style.opacity = sample ? '' : '0.5';
  (el('maia3TopPField') as HTMLElement).style.opacity = sample ? '' : '0.5';
}

/**
 * Engine notes are visible detail, not a second live region: #gameAnnouncement
 * already speaks readiness and turn changes. Only failures interrupt (role=alert);
 * routine readiness is hidden from sight and from the accessibility tree.
 */
function ctxSetEngineNote(text: string, warn = false, routine = false): void {
  const note = el('engineNote');
  note.textContent = text;
  note.hidden = !text;
  note.classList.toggle('warn', warn);
  note.dataset.routine = String(routine && !warn);
  if (warn && text) note.setAttribute('role', 'alert');
  else note.removeAttribute('role');
  if (routine && !warn) note.setAttribute('aria-hidden', 'true');
  else note.removeAttribute('aria-hidden');
  if (!text) buttonEl('retryEngine').hidden = true;
}

function ctxSetRuntimeWarning(text: string): void {
  const warning = el('runtimeWarning');
  clearRuntimeFallbackWarning(warning);
  if (!text) return;
  warning.textContent = text;
  warning.hidden = false;
}

function playOptionMatchesRuntimeAudit(option: PlayEngineOption, detail: BrowserRuntimeAuditDetail): boolean {
  return (
    option.id === 'centipawn' &&
    detail.family === 'centipawn' &&
    detail.engineLabel === 'Centipawn' &&
    detail.modelId === 'bt4-soap-rem-c19000-final' &&
    detail.requestedRuntime === 'auto'
  );
}

function ctxRestoreSelectedRuntimeWarning(ctx: PlayContext): void {
  const warning = el('runtimeWarning');
  clearRuntimeFallbackWarning(warning);
  const option = ctxSelectedEngine(ctx);
  for (const detail of playRuntimeFallbackAudits.values()) {
    if (playOptionMatchesRuntimeAudit(option, detail)) updateRuntimeFallbackWarning(warning, detail);
  }
}

function ctxActionableEngineError(ctx: PlayContext, error: unknown): string {
  const detail = error instanceof Error ? error.message : String(error);
  const option = ctxSelectedEngine(ctx);
  if (/fetch|network|load failed|failed to load|404/i.test(detail)) {
    return `${option.label} could not download a required asset. Check your connection, then retry; choose another engine if the asset remains unavailable. Details: ${detail}`;
  }
  if (/webgpu|gpu|backend/i.test(detail)) {
    return `${option.label} could not start its GPU runtime. Retry to use its supported fallback, or choose a CPU engine. Details: ${detail}`;
  }
  if (/wasm|worker|sharedarraybuffer|cross-origin/i.test(detail)) {
    return `${option.label} could not start its browser runtime. Retry, or choose another engine supported by the capability panel. Details: ${detail}`;
  }
  return `${option.label} stopped before making a move. Retry the engine or choose another opponent. Details: ${detail}`;
}

function ctxSetEngineFailure(ctx: PlayContext, error: unknown): void {
  ctxSetEngineNote(ctxActionableEngineError(ctx, error), true);
  buttonEl('retryEngine').hidden = false;
}

type SearchProgress = { completedVisits?: number; requestedVisits?: number; visits: number; move?: string | null; value: number; elapsedMs?: number };

function ctxSearchProgressText(label: string, progress: SearchProgress): string {
  const completed = progress.completedVisits ?? progress.visits;
  const requested = progress.requestedVisits ?? progress.visits;
  const pct = requested > 0 ? ` ${((100 * completed) / requested).toFixed(0)}%` : '';
  const speed = progress.elapsedMs && progress.elapsedMs > 0 ? ` · ${(completed / Math.max(1e-9, progress.elapsedMs / 1000)).toFixed(1)} v/s` : '';
  return `${label}: ${completed}/${requested} visits${pct} · best ${progress.move ?? '—'} · Q ${progress.value.toFixed(3)}${speed}`;
}

function ctxShowSearchProgress(ctx: PlayContext, label: string, progress: SearchProgress): void {
  const container = maybeEl('dlProgress');
  if (!container) return;
  const completed = progress.completedVisits ?? progress.visits;
  const requested = Math.max(1, progress.requestedVisits ?? progress.visits);
  const row = document.createElement('div');
  row.className = 'loading-progress-row';
  const bar = document.createElement('progress');
  bar.max = requested;
  bar.value = Math.max(0, Math.min(requested, completed));
  bar.setAttribute('aria-label', `${label} search progress`);
  bar.setAttribute('aria-valuetext', `${completed} of ${requested} visits`);
  const text = document.createElement('div');
  text.className = 'dl-label small';
  text.dataset.searchProgress = '';
  ctx.searchProgress = { label, progress };
  text.textContent = ctxSearchProgressText(searchProgressLabel(label, ctx.viewPly), progress);
  row.append(bar, text);
  container.replaceChildren(row);
  container.hidden = false;
  const milestone = Math.min(4, Math.floor((4 * completed) / requested));
  if (milestone > ctx.lastProgressAnnouncement) {
    ctx.lastProgressAnnouncement = milestone;
    el('progressAnnouncement').textContent = milestone === 4 ? `${label} search complete.` : `${label} search ${milestone * 25}% complete.`;
  }
}

/**
 * Re-label the visible search progress after a view change: the search always
 * runs on the live position, so it is tagged as such only while browsing.
 * Only touches a row ctxShowSearchProgress rendered (download progress and
 * hidden containers have no [data-search-progress] node).
 */
function ctxRefreshSearchProgressLabel(ctx: PlayContext): void {
  const text = maybeEl('dlProgress')?.querySelector<HTMLElement>('[data-search-progress]');
  if (!text || !ctx.searchProgress) return;
  text.textContent = ctxSearchProgressText(searchProgressLabel(ctx.searchProgress.label, ctx.viewPly), ctx.searchProgress.progress);
}

function ctxShowDownloadProgress(label: string, loadedBytes?: number, totalBytes?: number, phase = 'Downloading'): void {
  const container = maybeEl('dlProgress');
  if (!container) return;
  renderLoadingProgress(container, { label, phase, loadedBytes, totalBytes });
}

function ctxHideDownloadProgress(): void {
  const container = maybeEl('dlProgress');
  if (!container) return;
  hideLoadingProgress(container);
}

function ctxEnsureLc0Small(ctx: PlayContext): Promise<Lc0PuctSearcher> {
  if (lc0Searcher) return Promise.resolve(lc0Searcher);
  if (!lc0LoadPromise) {
    const uiGeneration = ctx.enginePreparationGeneration;
    if (ctxAsyncUiIsCurrent(ctx, uiGeneration)) ctxSetEngineNote('Loading Lc0 small net…');
    lc0LoadPromise = (async () => {
      const modelLoad = await loadLc0ModelForOrt(MODEL_URL, {
        cache: true,
        requestPersistentStorage: true,
        onProgress: (loaded, total) => {
          if (ctxAsyncUiIsCurrent(ctx, uiGeneration)) ctxShowDownloadProgress('Lc0 small net', loaded, total);
        },
      });
      if (ctxAsyncUiIsCurrent(ctx, uiGeneration)) ctxHideDownloadProgress();
      const evaluator = await Lc0OnnxEvaluator.create(modelLoad.model);
      lc0Searcher = new Lc0PuctSearcher(new CachedLc0Evaluator(evaluator, { maxEntries: 2048 }));
      if (ctxAsyncUiIsCurrent(ctx, uiGeneration)) ctxSetEngineNote('');
      return lc0Searcher;
    })().catch((error: Error) => {
      lc0LoadPromise = null;
      if (ctxAsyncUiIsCurrent(ctx, uiGeneration)) {
        ctxHideDownloadProgress();
        ctxSetEngineNote(`Lc0 load failed: ${error.message}`, true);
      }
      throw error;
    });
  }
  return lc0LoadPromise;
}

function ctxEnsureCentipawn(ctx: PlayContext): Promise<Evaluator> {
  if (centipawnEvaluator) return Promise.resolve(centipawnEvaluator);
  if (!centipawnLoadPromise) {
    const evaluatorGeneration = centipawnEvaluatorGeneration;
    const uiGeneration = ctx.enginePreparationGeneration;
    if (ctxAsyncUiIsCurrent(ctx, uiGeneration)) ctxSetEngineNote('Loading Centipawn…');
    centipawnLoadPromise = createBrowserSquareformerRuntimeEvaluator(
      {
        id: 'bt4-soap-rem-c19000-final',
        modelId: 'bt4-soap-rem-c19000-final',
        label: 'Centipawn',
        onnx: CENTIPAWN_MODEL_URL,
        meta: CENTIPAWN_META_URL,
        runtime: 'auto',
        manifestUrl: CENTIPAWN_TVMJS_MANIFEST_URL,
      },
      {
        runtime: 'auto',
        manifestUrl: CENTIPAWN_TVMJS_MANIFEST_URL,
        fallback: true,
        audit: { surface: 'play' },
      },
    )
      .then((loaded) => {
        if (ctxIsDisposed(ctx) || evaluatorGeneration !== centipawnEvaluatorGeneration) {
          (loaded.evaluator as Evaluator & { destroy?: () => void }).destroy?.();
          const error = new Error('Centipawn load discarded after Play teardown');
          error.name = 'AbortError';
          throw error;
        }
        centipawnEvaluator = new CachedEvaluator(loaded.evaluator, {
          maxEntries: 4096,
          includeHistory: true,
          includeLegalMoves: true,
          label: `centipawn-play:${loaded.resolvedRuntime}`,
        });
        if (ctxAsyncUiIsCurrent(ctx, uiGeneration)) ctxSetEngineNote('');
        return centipawnEvaluator;
      })
      .catch((error: Error) => {
        if (evaluatorGeneration === centipawnEvaluatorGeneration) centipawnLoadPromise = null;
        if (ctxAsyncUiIsCurrent(ctx, uiGeneration)) ctxSetEngineNote(`Centipawn load failed: ${error.message}`, true);
        throw error;
      });
  }
  return centipawnLoadPromise;
}

async function ctxEnsureMaia3(ctx: PlayContext): Promise<Maia3BrowserEvaluator> {
  if (maia3Promise) {
    const cached = maia3Promise;
    const evaluator = await cached;
    if (!evaluator.failure) return evaluator;
    if (maia3Promise === cached) maia3Promise = null;
    return ctxEnsureMaia3(ctx);
  }
  const uiGeneration = ctx.enginePreparationGeneration;
  maia3Promise = (async () => {
    if (ctxAsyncUiIsCurrent(ctx, uiGeneration)) ctxSetEngineNote('Loading Maia3 human model…');
    const evaluator = await Maia3BrowserEvaluator.create({
      selfElo: ctx.opponent.active.maiaElo,
      oppoElo: ctx.opponent.active.maiaElo,
      onProgress: (loaded, total) => {
        if (ctxAsyncUiIsCurrent(ctx, uiGeneration)) ctxShowDownloadProgress('Maia3', loaded, total);
      },
    });
    if (ctxAsyncUiIsCurrent(ctx, uiGeneration)) ctxHideDownloadProgress();
    const load = evaluator.modelLoad;
    const origin = load.cacheStatus === 'hit' ? 'from cache' : load.cacheStatus === 'miss' ? 'downloaded' : 'loaded';
    const integrity = load.sha256Valid === true ? ', sha256 ✓' : load.sha256Valid === false ? ', sha256 MISMATCH' : '';
    maia3Status = `Model ${origin} (${((load.bytes ?? 0) / 1e6).toFixed(0)}MB${integrity})`;
    if (ctxAsyncUiIsCurrent(ctx, uiGeneration)) ctxSetEngineNote('');
    return evaluator;
  })().catch((error: Error) => {
    maia3Promise = null;
    if (ctxAsyncUiIsCurrent(ctx, uiGeneration)) {
      ctxHideDownloadProgress();
      ctxSetEngineNote(`Maia3 load failed: ${error.message}`, true);
    }
    throw error;
  });
  return maia3Promise;
}

function ctxCpuEngineFor(ctx: PlayContext, option: PlayEngineOption): Promise<CpuEngine> {
  const existing = cpuEnginePromises.get(option.id);
  if (existing) {
    retainCpuEngine(option.id, existing);
    return existing;
  }
  const label = option.label;
  const uiGeneration = ctx.enginePreparationGeneration;
  const slot: { current: Promise<CpuEngine> | null } = { current: null };
  const created = (async (): Promise<CpuEngine> => {
    if (ctxAsyncUiIsCurrent(ctx, uiGeneration)) ctxSetEngineNote(`Loading ${label} (first use downloads the engine)…`);
    try {
      switch (option.family) {
        case 'sf':
          return new StockfishEngine(
            { depth: 4, threads: 1, hashMb: cpuEngineHashMbForSurface('play') },
            stockfishFlavorUrl(option.variant === 'lite' ? 'lite-single' : 'single'),
          );
        default:
          if (isDefaultBrowserUciFamily(option.family)) return createDefaultBrowserUciEngine(option.family);
          throw new Error(`unsupported engine family ${option.family}`);
      }
    } finally {
      if (ctxAsyncUiIsCurrent(ctx, uiGeneration)) ctxSetEngineNote('');
    }
  })().catch((error: Error) => {
    if (slot.current && cpuEnginePromises.get(option.id) === slot.current) cpuEnginePromises.delete(option.id);
    if (ctxAsyncUiIsCurrent(ctx, uiGeneration)) ctxSetEngineNote(`${label} load failed: ${error.message}`, true);
    throw error;
  });
  slot.current = created;
  retainCpuEngine(option.id, created);
  return created;
}

/** Keep `id` as the newest resident engine and dispose inactive engines past the cap. */
function retainCpuEngine(id: string, pending: Promise<CpuEngine>): void {
  cpuEnginePromises.delete(id);
  cpuEnginePromises.set(id, pending);
  evictInactiveCpuEngines(id);
}

function evictInactiveCpuEngines(activeId: string): void {
  for (const evictId of inactiveCpuEnginesToEvict([...cpuEnginePromises.keys()], activeId)) disposeCachedCpuEngine(evictId);
}

function disposeCachedCpuEngine(id: string): void {
  const pending = cpuEnginePromises.get(id);
  if (!pending) return;
  cpuEnginePromises.delete(id);
  void pending.then((engine) => engine.dispose?.()).catch(() => undefined);
}

function ctxPreloadLqoAfterBookMove(ctx: PlayContext, signal: AbortSignal): boolean {
  const searcher = acquireBigNetSearcher('lqo');
  if (searcher.loaded) return false;
  const label = `Lc0 ${searcher.config.name}`;
  ctxShowDownloadProgress(label, undefined, undefined, 'Loading in background');
  searcher.onDownloadProgress = (loaded, total) => {
    if (!signal.aborted && !ctxIsDisposed(ctx)) ctxShowDownloadProgress(label, loaded, total);
  };
  void searcher
    .init({ evalCacheEntries: 2048 })
    .then(() => {
      if (signal.aborted || ctxIsDisposed(ctx)) return;
      if (ctx.engineThinking) return;
      ctxHideDownloadProgress();
      ctxSetEngineNote('LQO model ready for post-book search.');
    })
    .catch((error: Error) => {
      if (signal.aborted || ctxIsDisposed(ctx)) return;
      if (ctx.engineThinking) return;
      ctxHideDownloadProgress();
      ctxSetEngineNote(`LQO background load failed: ${error.message}`, true);
    });
  return true;
}

async function ctxRequestEngineMove(ctx: PlayContext, signal: AbortSignal): Promise<string | null> {
  const option = ctxSelectedEngine(ctx);
  const preferences = ctx.opponent.active;
  const level = preferences.level;
  const visitsOrDepth = ctxStrengthFor(option, level);
  if (option.family === 'lc0' && option.variant === 'lqo') {
    const historyUcis = ctx.moves.map(moveToUci);
    if (ctx.humanColor === 'w') {
      const bookMove = lqoBlackBookMove(ctx.board, ctx.moves.length, historyUcis);
      if (bookMove) {
        const preloading = ctxPreloadLqoAfterBookMove(ctx, signal);
        ctxSetEngineNote(`LQO opening book: ${bookMove}${preloading ? ' · loading model in background' : ''}`);
        return bookMove;
      }
    } else if (ctx.moves.length === 0) {
      const searcher = acquireBigNetSearcher('lqo');
      let progressShown = false;
      if (!searcher.loaded) {
        ctxSetEngineNote(`Loading Lc0 ${searcher.config.name} (~${searcher.config.approxMb}MB on first use)…`);
        ctxShowDownloadProgress(`Lc0 ${searcher.config.name}`, undefined, undefined, 'Preparing');
        progressShown = true;
        searcher.onDownloadProgress = (loaded, total) => ctxShowDownloadProgress(`Lc0 ${searcher.config.name}`, loaded, total);
      }
      const evaluation = await (async () => {
        try {
          return await searcher.evaluate({ positions: ctx.positions }, { evalCacheEntries: 2048 });
        } finally {
          if (progressShown) ctxHideDownloadProgress();
        }
      })();
      if (signal.aborted) return null;
      const bookMove = lqoWhiteFirstPolicyBookMove(ctx.board, ctx.moves.length, historyUcis, evaluation.legalPriors, Math.random, 8);
      if (bookMove) {
        ctxSetEngineNote(`LQO policy book: ${bookMove}`);
        return bookMove;
      }
    } else {
      const bookMove = lqoWhiteBookMove(ctx.board, ctx.moves.length, historyUcis);
      if (bookMove) {
        const preloading = ctxPreloadLqoAfterBookMove(ctx, signal);
        ctxSetEngineNote(`LQO opening book: ${bookMove}${preloading ? ' · loading model in background' : ''}`);
        return bookMove;
      }
    }
  }
  if (option.family === 'maia3') {
    const player = await ctxEnsureMaia3(ctx);
    if (signal.aborted) return null;
    const choice = await player.chooseMove(
      { positions: ctx.positions },
      {
        selfElo: preferences.maiaElo,
        oppoElo: preferences.maiaElo,
        style: preferences.maiaStyle,
        temperature: preferences.maiaTemperature,
        topP: preferences.maiaTopP,
      },
    );
    return choice.move;
  }
  if (option.family === 'centipawn') {
    const evaluator = await ctxEnsureCentipawn(ctx);
    if (signal.aborted) return null;
    try {
      const result = await chooseMove(ctx.board, evaluator, {
        visits: visitsOrDepth,
        // Same visit-argmax move as the full budget once the leader cannot be caught.
        // docs/play_early_stop_2026-10.md. Big-net Play searches stay fixed-visit.
        earlyStop: 'root-dominance',
        batchSize: Math.max(1, Math.min(32, visitsOrDepth)),
        signal,
        historyFens: ctx.positions.slice(0, -1).map(boardToFen).reverse().slice(0, 16),
        searchPolicy: montyLitePuctPolicy,
        includePv: true,
        onProgress: (progress) => {
          if (!signal.aborted)
            ctxShowSearchProgress(ctx, 'Centipawn search', {
              ...progress,
              move: progress.move ? moveToUci(progress.move) : null,
            });
        },
      });
      return result.move ? moveToUci(result.move) : null;
    } finally {
      if (!signal.aborted && !ctxIsDisposed(ctx)) {
        ctxHideDownloadProgress();
        ctxSetEngineNote('');
      }
    }
  }
  if (option.family === 'lc0' && option.variant === 'small') {
    const searcher = await ctxEnsureLc0Small(ctx);
    try {
      const result = await searcher.search(
        { positions: ctx.positions },
        {
          visits: visitsOrDepth,
          // Same visit-argmax move as the full budget once the leader cannot be caught.
          // docs/play_early_stop_2026-10.md. Big-net Play searches stay fixed-visit.
          earlyStop: 'root-dominance',
          signal,
          yieldEveryMs: 16,
          reuseTree: true,
          drawScore: PLAY_DRAW_SCORE,
          searchContemptLimit: PLAY_SEARCH_CONTEMPT_LIMIT,
          onProgress: (progress) => {
            if (!signal.aborted) ctxShowSearchProgress(ctx, 'Lc0 search', progress);
          },
        },
      );
      return result.move ?? null;
    } finally {
      if (!signal.aborted && !ctxIsDisposed(ctx)) ctxHideDownloadProgress();
    }
  }
  if (option.family === 'lc0') {
    const searcher = acquireBigNetSearcher(option.variant as BigNetKey);
    if (!searcher.loaded && !signal.aborted && !ctxIsDisposed(ctx)) {
      ctxSetEngineNote(`Loading Lc0 ${searcher.config.name} (~${searcher.config.approxMb}MB on first use)…`);
      ctxShowDownloadProgress(`Lc0 ${searcher.config.name}`, undefined, undefined, 'Preparing');
      searcher.onDownloadProgress = (loaded, total) => {
        if (!signal.aborted && !ctxIsDisposed(ctx)) ctxShowDownloadProgress(`Lc0 ${searcher.config.name}`, loaded, total);
      };
    }
    try {
      const lqoOptions = option.variant === 'lqo' ? ctxLqoSearchOptions(ctx) : null;
      const result = await searcher.search(
        { positions: ctx.positions },
        {
          visits: visitsOrDepth,
          reuseTree: true,
          batchSize: searcher.config.recommendedBatchSize,
          batchPipelineDepth: searcher.config.recommendedPipelineDepth,
          evalCacheEntries: 2048,
          drawScore: lqoOptions?.drawScore ?? PLAY_DRAW_SCORE,
          searchContemptLimit: lqoOptions?.searchContemptLimit ?? PLAY_SEARCH_CONTEMPT_LIMIT,
          signal,
          onProgress: (progress) => {
            if (!signal.aborted) ctxShowSearchProgress(ctx, `Lc0 ${searcher.config.name} search`, progress);
          },
          ...(lqoOptions ?? {}),
        },
      );
      if (!signal.aborted && !ctxIsDisposed(ctx)) {
        ctxHideDownloadProgress();
        ctxSetEngineNote('');
      }
      return result.cancelled ? null : (result.move ?? null);
    } finally {
      if (!signal.aborted && !ctxIsDisposed(ctx)) ctxHideDownloadProgress();
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
async function ctxVerifySelectedEngineReady(ctx: PlayContext): Promise<void> {
  if (ctx.disposed || ctx.board.turn !== ctx.humanColor) return;
  const resumingGame = ctx.moves.length > 0;
  const option = ctxSelectedEngine(ctx);
  const preferences = ctx.opponent.active;
  releaseUnusedBigNetSearchers(option.family === 'lc0' && option.variant !== 'small' ? [option.variant as BigNetKey] : [], 0);
  const signal = ctx.abort.signal;
  const generation = ++ctx.enginePreparationGeneration;
  ctx.engineReady = false;
  ctx.preparingEngine = true;
  buttonEl('retryEngine').hidden = true;
  ctxSetEngineNote(resumingGame ? `Checking ${option.label} before play resumes…` : `Preparing ${option.label} before the first move…`);
  ctxRender(ctx);
  try {
    if (option.family === 'maia3') {
      const player = await ctxEnsureMaia3(ctx);
      await player.chooseMove(
        { positions: ctx.positions },
        {
          selfElo: preferences.maiaElo,
          oppoElo: preferences.maiaElo,
          style: 'argmax',
        },
      );
    } else if (option.family === 'centipawn') {
      const evaluator = await ctxEnsureCentipawn(ctx);
      await evaluator.evaluate(ctx.board, {
        historyFens: ctx.positions.slice(0, -1).map(boardToFen).reverse().slice(0, 16),
        legalMoves: legalMoves(ctx.board),
      });
    } else if (option.family === 'lc0' && option.variant === 'small') {
      const searcher = await ctxEnsureLc0Small(ctx);
      await searcher.search(
        { positions: ctx.positions },
        { visits: 1, signal, reuseTree: false, drawScore: PLAY_DRAW_SCORE, searchContemptLimit: PLAY_SEARCH_CONTEMPT_LIMIT },
      );
      if (signal.aborted || ctx.disposed || generation !== ctx.enginePreparationGeneration || option.id !== ctxSelectedEngine(ctx).id) return;
      searcher.resetTree();
    } else if (option.family === 'lc0') {
      const searcher = acquireBigNetSearcher(option.variant as BigNetKey);
      const lqoOptions = option.variant === 'lqo' ? ctxLqoSearchOptions(ctx) : null;
      await searcher.search(
        { positions: ctx.positions },
        {
          visits: 1,
          signal,
          reuseTree: false,
          batchSize: 1,
          batchPipelineDepth: 1,
          evalCacheEntries: 2048,
          drawScore: lqoOptions?.drawScore ?? PLAY_DRAW_SCORE,
          searchContemptLimit: lqoOptions?.searchContemptLimit ?? PLAY_SEARCH_CONTEMPT_LIMIT,
          ...(lqoOptions ?? {}),
        },
      );
      if (signal.aborted || ctx.disposed || generation !== ctx.enginePreparationGeneration || option.id !== ctxSelectedEngine(ctx).id) return;
      await searcher.resetTree();
    } else {
      const engine = await ctxCpuEngineFor(ctx, option);
      engine.setOptions({ depth: 1, movetimeMs: undefined });
      await engine.bestMove(boardToFen(ctx.board), signal);
    }
    if (signal.aborted || ctx.disposed || generation !== ctx.enginePreparationGeneration || option.id !== ctxSelectedEngine(ctx).id) return;
    ctx.engineReady = true;
    ctxSetEngineNote(resumingGame ? `${option.label} is ready. You can continue the game.` : `${option.label} is ready.`, false, true);
  } catch (error) {
    if (!signal.aborted && !ctx.disposed && generation === ctx.enginePreparationGeneration) ctxSetEngineFailure(ctx, error);
  } finally {
    if (!ctx.disposed && generation === ctx.enginePreparationGeneration) {
      ctx.preparingEngine = false;
      ctxHideDownloadProgress();
      ctxRender(ctx);
    }
  }
}

// ---------------------------------------------------------------------------
// Game flow (context-scoped)
// ---------------------------------------------------------------------------

function ctxCancelEngineTurn(ctx: PlayContext): void {
  ctx.abort.abort();
  ctx.abort = new AbortController();
  ctx.engineThinking = false;
  ctx.preparingEngine = false;
  ctx.enginePreparationGeneration += 1;
  ctxHideDownloadProgress();
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
  if (ctx.moves.length === ctx.practiceBoundary) {
    ctxSetEngineNote('');
  }
  ctx.sans.push(moveToSan(ctx.board, move));
  ctx.moves.push(move);
  ctx.board = makeMove(ctx.board, move);
  ctx.positions.push(ctx.board);
}

function ctxHumanMoveCount(ctx: PlayContext): number {
  let count = 0;
  for (let i = ctx.practiceBoundary; i < ctx.moves.length; i++) if (ctx.positions[i]?.turn === ctx.humanColor) count++;
  return count;
}

/** Nothing the player did yet: settings changes still restart silently and the sidebar shows setup. */
function ctxGameIsFresh(ctx: PlayContext): boolean {
  return !ctx.started && !ctx.gameOver && ctxHumanMoveCount(ctx) === 0;
}

type PlayPhase = 'setup' | 'playing' | 'over';
function ctxPhase(ctx: PlayContext): PlayPhase {
  if (ctx.gameOver) return 'over';
  return ctxGameIsFresh(ctx) ? 'setup' : 'playing';
}

async function ctxEngineTurn(ctx: PlayContext): Promise<void> {
  if (ctx.gameOver || ctx.board.turn === ctx.humanColor) return;
  const signal = ctx.abort.signal;
  ctx.engineThinking = true;
  ctx.lastProgressAnnouncement = -1;
  el('progressAnnouncement').textContent = '';
  await requestWakeLock();
  ctxRender(ctx);
  let uci: string | null = null;
  try {
    uci = await ctxRequestEngineMove(ctx, signal);
  } catch (error) {
    if (signal.aborted) return;
    ctx.engineThinking = false;
    ctx.engineReady = false;
    releaseWakeLock();
    ctxSetEngineFailure(ctx, error);
    ctxRender(ctx);
    return;
  } finally {
    releaseWakeLock();
  }
  if (signal.aborted || ctx.disposed) return;
  ctx.engineReady = true;
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
  if (ctx.viewPly !== null) {
    // Browsing an earlier ply is view-only; snap the board back to the browsed position.
    ctxRender(ctx);
    return;
  }
  if (!ctx.engineReady && !ctx.preparingEngine && !ctx.gameOver && ctx.board.turn === ctx.humanColor) void ctxVerifySelectedEngineReady(ctx);
  if (ctx.preparingEngine || !ctx.engineReady || ctx.engineThinking || ctx.gameOver || ctx.board.turn !== ctx.humanColor) {
    ctxRender(ctx);
    return;
  }
  const matching = matchUserMoves(ctx.board, from, to);
  if (!matching.length) {
    ctxRender(ctx);
    return;
  }
  if (matching.length > 1) {
    ctx.pendingPromotion = matching;
    ctxRender(ctx);
    return;
  }
  ctxApplyHumanMove(ctx, matching[0]);
}

function ctxApplyHumanMove(ctx: PlayContext, move: Move): void {
  ctx.pendingPromotion = null;
  ctx.started = true;
  ctx.configOpen = false;
  ctxApplyMove(ctx, move);
  ctxCheckGameOver(ctx);
  ctxRender(ctx);
  if (!ctx.gameOver) void ctxEngineTurn(ctx);
}

/** Why the player cannot move right now, or null when a move would be accepted. */
function ctxHumanMoveBlockedReason(ctx: PlayContext): string | null {
  if (ctx.viewPly !== null) return 'You are viewing an earlier move. Press End to return to the game.';
  if (ctx.gameOver) return 'The game is over.';
  if (ctx.engineThinking) return `${ctxOpponentName(ctxSelectedEngine(ctx))} is thinking.`;
  if (ctx.board.turn !== ctx.humanColor) return 'It is not your move.';
  if (ctx.preparingEngine || !ctx.engineReady) return `${ctxOpponentName(ctxSelectedEngine(ctx))} is still loading.`;
  return null;
}

/** Keyboard move entry: SAN (Nf3, exd5, e8=Q) or UCI (g1f3, e7e8q). */
function ctxSubmitTypedMove(ctx: PlayContext): void {
  const input = el('moveInput') as HTMLInputElement;
  const raw = input.value.trim();
  if (!raw) return;
  const fail = (message: string) => {
    input.classList.add('invalid');
    input.setCustomValidity(message);
    input.reportValidity();
    el('boardAnnouncement').textContent = message;
  };
  if (!ctx.engineReady && !ctx.preparingEngine && !ctx.gameOver && ctx.board.turn === ctx.humanColor) void ctxVerifySelectedEngineReady(ctx);
  const blocked = ctxHumanMoveBlockedReason(ctx);
  if (blocked) {
    fail(blocked);
    return;
  }
  const uci = raw.toLowerCase();
  const isUci = /^[a-h][1-8][a-h][1-8][qrbn]?$/.test(uci);
  const move = isUci ? legalMoves(ctx.board).find((candidate) => moveToUci(candidate) === uci) : sanToMove(ctx.board, raw);
  if (!move) {
    const needsPiece = isUci && uci.length === 4 && legalMoves(ctx.board).some((candidate) => moveToUci(candidate).startsWith(uci));
    fail(needsPiece ? `Add the promotion piece, for example ${uci}q.` : `${raw} is not a legal move here.`);
    return;
  }
  input.value = '';
  input.setCustomValidity('');
  input.classList.remove('invalid');
  el('boardAnnouncement').textContent = `You played ${moveToSan(ctx.board, move)}.`;
  ctxApplyHumanMove(ctx, move);
}

function ctxNewGame(ctx: PlayContext): void {
  ctx.autosave?.clear();
  ctx.restoredGame = false;
  el('playRestoreNotice').hidden = true;
  ctxCancelEngineTurn(ctx);
  ctx.engineReady = false;
  ctxSetOpponent(ctx, committedPlayOpponent(ctxCurrentPlayPreferences(ctx)));
  ctxPersistPlayPreferences(ctx);
  const colorChoice = ctx.opponent.active.color;
  ctx.humanColor = colorChoice === 'random' ? (Math.random() < 0.5 ? 'w' : 'b') : colorChoice === 'black' ? 'b' : 'w';
  ctx.orientation = ctx.humanColor === 'w' ? 'white' : 'black';
  ctxResetStartPosition(ctx);
  ctx.gameOver = null;
  ctx.pendingPromotion = null;
  ctx.viewPly = null;
  lc0Searcher?.resetTree();
  for (const key of BIG_NET_KEYS) {
    const searcher = peekBigNetSearcher(key);
    if (!searcher) continue;
    if (searcher.loaded) void searcher.resetTree();
  }
  ctxSetEngineNote('');
  ctxDisarmResign(ctx);
  ctxDisarmNewGame(ctx);
  ctxRestoreSelectedRuntimeWarning(ctx);
  evictInactiveCpuEngines(ctx.opponent.active.engineId);
  ctxRender(ctx);
  ctxPrepareOrStartEngine(ctx);
}

function ctxTakeback(ctx: PlayContext): void {
  if (ctx.moves.length <= ctx.practiceBoundary) return;
  ctxCancelEngineTurn(ctx);
  ctxDisarmResign(ctx);
  const ply = playTakebackPly(ctx.positions, ctx.humanColor, ctx.practiceBoundary);
  ctx.moves.length = ply;
  ctx.sans.length = ply;
  ctx.positions.length = ply + 1;
  ctx.board = ctx.positions[ply];
  ctx.gameOver = null;
  ctx.pendingPromotion = null;
  ctx.viewPly = null;
  ctxRender(ctx);
  if (ctx.board.turn !== ctx.humanColor) void ctxEngineTurn(ctx);
  else if (!ctx.engineReady) void ctxVerifySelectedEngineReady(ctx);
}

function ctxResign(ctx: PlayContext): void {
  if (ctx.gameOver || (!ctx.moves.length && !ctx.started)) return;
  if (!ctx.resignArmed) {
    ctx.resignArmed = true;
    const btn = buttonEl('resign');
    setButtonLabel(btn, 'Confirm resign');
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
  if (ctx.resignArmTimer) {
    clearTimeout(ctx.resignArmTimer);
    ctx.resignArmTimer = null;
  }
  const btn = buttonEl('resign');
  if (btn.classList.contains('danger')) {
    btn.classList.remove('danger');
    setButtonLabel(btn, 'Resign');
  }
}

/** Icon buttons keep their SVG; only the text label changes. */
function setButtonLabel(button: HTMLElement, text: string): void {
  const label = button.querySelector<HTMLElement>('.lbl');
  if (label) label.textContent = text;
  else button.textContent = text;
}

function ctxArmNewGame(ctx: PlayContext): void {
  ctx.newGameArmed = true;
  if (ctx.newGameArmTimer) clearTimeout(ctx.newGameArmTimer);
  ctx.newGameArmTimer = setTimeout(() => {
    ctxDisarmNewGame(ctx);
    ctxRender(ctx);
  }, 4000);
  ctxRender(ctx);
}

function ctxDisarmNewGame(ctx: PlayContext): void {
  ctx.newGameArmed = false;
  if (ctx.newGameArmTimer) {
    clearTimeout(ctx.newGameArmTimer);
    ctx.newGameArmTimer = null;
  }
}

function ctxExportPgn(ctx: PlayContext): string {
  const engineName = ctxSelectedEngine(ctx).label;
  const date = new Date().toISOString().slice(0, 10).replaceAll('-', '.');
  return playGameToPgn(
    ctx.startFen,
    ctx.moves,
    {
      Event: 'Casual browser game',
      Site: location.host || 'local',
      Date: date,
      White: ctx.humanColor === 'w' ? 'You' : engineName,
      Black: ctx.humanColor === 'b' ? 'You' : engineName,
    },
    ctx.gameOver?.result ?? '*',
  );
}

// ---------------------------------------------------------------------------
// Rendering (context-scoped)
// ---------------------------------------------------------------------------

function ctxVerdictText(ctx: PlayContext): string {
  if (!ctx.gameOver) return '';
  const { result, reason } = ctx.gameOver;
  if (result === '1/2-1/2') return `Draw — ${reason}`;
  const humanWon = (result === '1-0') === (ctx.humanColor === 'w');
  return humanWon ? `You win — ${reason}` : `${ctxOpponentName(ctxSelectedEngine(ctx))} wins — ${reason}`;
}

function ctxStatusText(ctx: PlayContext): string {
  if (ctx.viewPly === null) return ctxLiveStatusText(ctx);
  const ply = ctx.viewPly;
  const live = ctxLiveStatusText(ctx);
  return `Viewing ${ply > 0 ? playMoveLabel(ctx.positions[ply - 1], ctx.sans[ply - 1]) : 'the start position'} — press → or End to return. Live game: ${/[.…!?]$/.test(live) ? live : `${live}.`}`;
}

function ctxLiveStatusText(ctx: PlayContext): string {
  if (ctx.gameOver) return ctxVerdictText(ctx);
  if (ctx.pendingPromotion) return 'Choose a promotion piece';
  if (ctx.preparingEngine) return `Preparing ${ctxSelectedEngine(ctx).label} before the first move…`;
  if (ctx.engineThinking) return `${ctxOpponentName(ctxSelectedEngine(ctx))} is thinking…`;
  if (ctx.practicePosition && !ctx.started) return 'Choose your side and opponent, then press Play.';
  if (!ctx.engineReady) return `${ctxSelectedEngine(ctx).label} is not ready. Retry the engine or choose another opponent.`;
  if (ctxPhase(ctx) === 'setup') return 'Press Play or make a move.';
  if (boardCheck(ctx.board)) return ctx.board.turn === ctx.humanColor ? 'You are in check' : `${ctxOpponentName(ctxSelectedEngine(ctx))} is in check`;
  return ctx.board.turn === ctx.humanColor ? 'Your move' : `${ctxOpponentName(ctxSelectedEngine(ctx))} to move`;
}

function ctxRenderMoveList(ctx: PlayContext): void {
  const list = el('moveList');
  if (!ctx.sans.length) {
    list.innerHTML = '<span class="placeholder">Your game starts here. Move a piece, or type a move below (e.g. Nf3).</span>';
    return;
  }
  const rows: string[] = [];
  const shown = displayedPly(ctx.viewPly, ctx.sans.length);
  // Button i is the move that produces ply i + 1; clicking it shows that position.
  const san = (i: number) =>
    `<button type="button" class="san${i + 1 === shown ? ' current' : ''}" data-ply="${i + 1}"${i + 1 === shown ? ' aria-current="step"' : ''}>${ctx.sans[i]}</button>`;
  for (const row of playMoveRows(ctx.positions, ctx.sans)) {
    rows.push(`<div class="move-row"><span class="num">${row.number}.</span>${row.white === null ? '<span></span>' : san(row.white)}${row.black === null ? '<span></span>' : san(row.black)}</div>`);
  }
  list.innerHTML = rows.join('');
  if (ctx.viewPly === null) {
    ctx.moveListScrollKey = 'live';
    list.scrollTop = list.scrollHeight;
    return;
  }
  // Only scroll when the browsed ply changes, so an engine reply arriving
  // mid-browse does not yank a list the user scrolled by hand.
  const scrollKey = `ply:${shown}`;
  if (ctx.moveListScrollKey === scrollKey) return;
  ctx.moveListScrollKey = scrollKey;
  const current = list.querySelector<HTMLElement>('.san.current');
  if (!current) {
    list.scrollTop = 0;
    return;
  }
  const listRect = list.getBoundingClientRect();
  const rect = current.getBoundingClientRect();
  if (rect.top < listRect.top) list.scrollTop -= listRect.top - rect.top + 4;
  else if (rect.bottom > listRect.bottom) list.scrollTop += rect.bottom - listRect.bottom + 4;
}

function ctxRenderMoveNav(ctx: PlayContext): void {
  const available = moveNavAvailability(ctx.viewPly, ctx.moves.length);
  buttonEl('navFirst').disabled = !available.first;
  buttonEl('navPrev').disabled = !available.prev;
  buttonEl('navNext').disabled = !available.next;
  const last = buttonEl('navLast');
  last.disabled = !available.last;
  last.classList.toggle('live-pending', ctx.viewPly !== null);
  last.title = ctx.viewPly === null ? 'Last move (End / ↓)' : 'Back to the live game (End / ↓)';
}

function ctxNavigate(ctx: PlayContext, action: MoveNavAction): void {
  ctxSetViewPly(ctx, navigateViewPly(ctx.viewPly, ctx.moves.length, action));
}

function ctxSetViewPly(ctx: PlayContext, viewPly: number | null): void {
  const next = normalizeViewPly(viewPly, ctx.moves.length);
  if (next === ctx.viewPly) return;
  // A promotion choice belongs to the live position; leaving it cancels the pick.
  if (ctx.pendingPromotion) ctx.pendingPromotion = null;
  // Chessground's set() keeps an active selection or drag; drop it so it does
  // not leak into the browsed position or back into live play.
  ctx.ground?.cancelMove();
  ctx.viewPly = next;
  ctxRender(ctx);
}

function ctxRenderRestartBanner(ctx: PlayContext): void {
  const banner = el('restartBanner');
  if (!ctx.opponent.pending) {
    banner.hidden = true;
    return;
  }
  banner.hidden = false;
  const engineLabel = ENGINE_OPTIONS.find((o) => o.id === ctx.opponent.pending?.engineId)?.label ?? 'the new engine';
  const colorLabel = ctx.opponent.pending.color === 'random' ? 'random side' : `as ${ctx.opponent.pending.color}`;
  el('restartMessage').textContent = `Start a new game vs ${engineLabel}, ${colorLabel}? The current game will be discarded.`;
}

function ctxRenderPromotionPicker(ctx: PlayContext): void {
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
    onCancel: () => {
      ctx.pendingPromotion = null;
      ctxRender(ctx);
    },
  });
}

function ctxEngineOptionState(option: PlayEngineOption): { disabled: boolean; suffix: string } {
  if (option.family === 'stormphrax') {
    const reason = stormphraxVariantUnsupportedReason(stormphraxVariantByKey(defaultStormphraxVariantKey()));
    return reason ? { disabled: true, suffix: ` (${reason})` } : { disabled: false, suffix: '' };
  }
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
  const group = (key: PlayEngineOption['group']) =>
    ENGINE_OPTIONS.filter((option) => option.group === key && (!ctx.practicePosition || option.group !== 'odds'))
      .map(ctxEngineOptionHtml)
      .join('');
  select.innerHTML =
    `<optgroup label="Human-like · 600–2600 rating">${group('human')}</optgroup>` +
    (ctx.practicePosition ? '' : `<optgroup label="Handicap · plays without its queen">${group('odds')}</optgroup>`) +
    `<optgroup label="Full-strength engines">${group('engine')}</optgroup>`;
  const selectedOption = ENGINE_OPTIONS.find((option) => option.id === selected);
  // A restored game must not silently switch opponents when an asset probe refreshes the options.
  if (selectedOption && (!ctx.practicePosition || selectedOption.group !== 'odds') && (ctx.restoredGame || !ctxEngineOptionState(selectedOption).disabled)) select.value = selected;
  else select.value = 'maia3';
  ctxRenderEngineCaution(ctx);
}

function ctxResolvePendingPreferredEngine(ctx: PlayContext): void {
  const id = ctx.pendingPreferredEngineId;
  if (!id || ctx.disposed) return;
  const option = ENGINE_OPTIONS.find((candidate) => candidate.id === id);
  if (!option || option.family !== 'lc0' || option.variant === 'small') {
    ctx.pendingPreferredEngineId = null;
    return;
  }
  const status = bigNetAssetStatusSync(BIG_NETS[option.variant as BigNetKey]);
  if (status === 'unknown') return;
  ctx.pendingPreferredEngineId = null;
  if (ctx.pendingPreferredEngineTimer) {
    clearTimeout(ctx.pendingPreferredEngineTimer);
    ctx.pendingPreferredEngineTimer = null;
  }
  const restored = status === 'present';
  if (restored) selectEl('engineSelect').value = option.id;
  ctxRenderLevelOptions(ctx);
  ctx.engineReady = false;
  ctxResetGameStateForMount(ctx);
  ctxPersistPlayPreferences(ctx);
  ctxRenderEngineCaution(ctx);
  if (!restored) ctxSetRuntimeWarning('Your saved engine asset is unavailable, so Play is using Maia instead.');
  ctxRender(ctx);
  ctxPrepareOrStartEngine(ctx);
}

function ctxRenderEngineCaution(ctx: PlayContext): void {
  const option = ctxDraftEngine();
  const caution = el('engineCaution');
  if (option.family === 'lc0' && option.variant !== 'small') {
    const config = BIG_NETS[option.variant as BigNetKey];
    const memory = bigNetMemoryCaution(config);
    const odds = option.variant === 'lqo' ? ' The bot starts without its queen and uses the LeelaQueenOdds v2 network. ' : ' ';
    const cpu = bt4SupportedSync() ? '' : ' No WebGPU here: runs on the CPU (wasm) fallback with reduced visit budgets — expect several seconds per move.';
    caution.textContent = `First move downloads the ~${config.approxMb}MB net.${odds}${memory ?? ''}${cpu}`.trimEnd();
    caution.hidden = false;
  } else if (option.family === 'centipawn') {
    caution.textContent = 'First use downloads about 26MB for the Centipawn TVMJS runtime. ORT downloads only if the WebGPU runtime needs to fall back.';
    caution.hidden = false;
  } else {
    caution.hidden = true;
  }
}

// ---------------------------------------------------------------------------
// Game chrome: player strips, setup summary, game-over card, opening name.
// Presentation only; live identity is committed, while setup fields show the draft.
// ---------------------------------------------------------------------------

const SF_LEVEL_NAMES = ['Weakest', 'Casual', 'Club', 'Strong', 'Full strength'] as const;
const ENGINE_MONOGRAMS: Partial<Record<PlayFamily, string>> = {
  maia3: 'M3',
  lc0: 'Lc0',
  sf: 'SF',
  centipawn: 'CP',
  reckless: 'Rk',
  viridithas: 'Vi',
  berserk: 'Bk',
  plentychess: 'PC',
  stormphrax: 'Sx',
};

let openingIndex: OpeningNameIndex | null = null;
let openingIndexPromise: Promise<void> | null = null;

function ctxEnsureOpeningNames(ctx: PlayContext): void {
  if (openingIndex || openingIndexPromise || !ctx.moves.length || ctx.startFen !== START_FEN) return;
  openingIndexPromise = fetch(resolvePublicAssetUrl('/openings/eco.json'))
    .then((response) => (response.ok ? response.json() : null))
    .then((index: unknown) => {
      if (index && typeof index === 'object' && !Array.isArray(index)) openingIndex = index as OpeningNameIndex;
      if (!ctxIsDisposed(ctx)) ctxRender(ctx);
    })
    .catch(() => undefined);
}

function ctxOpeningAt(ctx: PlayContext, ply: number): readonly [string, string] | null {
  if (!openingIndex || ctx.startFen !== START_FEN) return null;
  for (let i = Math.min(ply, ctx.positions.length - 1); i > 0; i--) {
    const hit = openingIndex[openingPositionKey(boardToFen(ctx.positions[i]))];
    if (hit) return hit;
  }
  return null;
}

function ctxOpponentName(option: PlayEngineOption): string {
  return option.family === 'maia3' ? 'Maia3' : option.label;
}

function maiaRatingBand(elo: number): string {
  if (elo < 1000) return 'Beginner';
  if (elo < 1400) return 'Casual player';
  if (elo < 1800) return 'Club player';
  if (elo < 2200) return 'Strong club player';
  return 'Expert';
}

function ctxOpponentMeta(ctx: PlayContext): string {
  const option = ctxSelectedEngine(ctx);
  if (option.family === 'maia3') return `${ctx.opponent.active.maiaElo} · ${ctx.opponent.active.maiaStyle === 'argmax' ? 'top human move' : 'human-like'}`;
  const level = ctx.opponent.active.level;
  const name = option.family === 'sf' ? SF_LEVEL_NAMES[level] : EFFORT_LEVEL_NAMES[level];
  return `${option.variant === 'lqo' ? 'Queen odds · ' : ''}Level ${level + 1} · ${name}`;
}

function ctxOpponentBlurb(ctx: PlayContext): string {
  const option = ctxDraftEngine();
  if (option.family === 'maia3') {
    const elo = ctxSelectedMaia3Elo();
    return ctxSelectedMaia3Style() === 'argmax'
      ? `Always plays the most typical move of a ${elo}-rated human.`
      : `Plays like a ${elo}-rated human, mistakes included.`;
  }
  if (option.variant === 'lqo') return 'Starts without its queen, then plays tricky, aggressive chess to win it back.';
  if (option.family === 'sf') return 'Level 1 is beatable; level 5 is full-strength Stockfish.';
  return `A full-strength engine, strong even at level 1. For an even game, pick Maia3.`;
}

function playCtaLabel(): string {
  const choice = selectEl('colorSelect').value;
  if (choice === 'random') return 'Play · random colour';
  return `Play as ${choice === 'black' ? 'Black' : 'White'}`;
}

function ctxRenderStrip(ctx: PlayContext, id: string, color: Color, shown: BoardState, phase: PlayPhase): void {
  const strip = el(id);
  const option = ctxSelectedEngine(ctx);
  const isHuman = color === ctx.humanColor;
  const side = color === 'w' ? 'White' : 'Black';
  const q = <T extends HTMLElement>(selector: string) => strip.querySelector<T>(selector)!;
  strip.dataset.player = isHuman ? 'you' : 'engine';
  const avatar = q('[data-avatar]');
  avatar.textContent = isHuman ? 'You' : (ENGINE_MONOGRAMS[option.family] ?? option.label.slice(0, 2));
  avatar.classList.toggle('you', isHuman);
  q('[data-name]').textContent = isHuman ? 'You' : ctxOpponentName(option);
  q('[data-meta]').textContent = isHuman ? side : `${ctxOpponentMeta(ctx)} · ${side}`;
  const material = playMaterial(parseFen(ctx.startFen), shown, color);
  const materialSlot = q('[data-material]');
  materialSlot.innerHTML = material.html;
  materialSlot.title = material.label;
  materialSlot.setAttribute('role', 'img');
  materialSlot.setAttribute('aria-label', material.label);
  const state = q('[data-state]');
  let text = '';
  let kind = '';
  if (!isHuman && ctx.engineThinking) [text, kind] = ['Thinking…', 'thinking'];
  else if (!isHuman && ctx.preparingEngine) [text, kind] = ['Loading…', 'thinking'];
  else if (!isHuman && ctx.practicePosition && !ctx.started && !ctx.engineReady) [text, kind] = ['Loads on Play', 'ready'];
  else if (!isHuman && !ctx.engineReady && !ctx.gameOver)
    [text, kind] = buttonEl('retryEngine').hidden ? ['Loading…', 'thinking'] : ['Error · retry', 'error'];
  // Readiness only matters before the first move; during play the strips carry the turn.
  else if (!isHuman && phase === 'setup') [text, kind] = ['Ready', 'ready'];
  else if (isHuman && phase === 'playing' && ctx.viewPly === null && ctx.board.turn === color && ctx.engineReady && !ctx.pendingPromotion) [text, kind] = ['Your move', 'turn'];
  state.textContent = text;
  state.dataset.kind = kind;
  strip.dataset.turn = String(!ctx.gameOver && ctx.viewPly === null && phase !== 'setup' && ctx.board.turn === color);
  strip.setAttribute('aria-label', `${isHuman ? 'You' : ctxOpponentName(option)}, ${side}${text ? `, ${text.replace('…', '')}` : ''}`);
}

function ctxRenderChrome(ctx: PlayContext, shownBoard: BoardState, shownPly: number): void {
  const phase = ctxPhase(ctx);
  const main = el('main');
  main.dataset.phase = phase;
  main.dataset.opponentState = ctx.opponent.tag;
  main.dataset.opponent = ctx.opponent.active.engineId;
  main.dataset.pendingOpponent = ctx.opponent.pending?.engineId ?? '';
  const hasNotice = !buttonEl('retryEngine').hidden || !el('runtimeWarning').hidden || !el('dlProgress').hidden;
  main.dataset.quiet = String(!hasNotice && ctx.engineReady && !ctx.preparingEngine && !ctx.engineThinking && !ctx.pendingPromotion && ctx.viewPly === null && !boardCheck(ctx.board));
  main.dataset.config = ctx.configOpen && phase !== 'setup' ? 'open' : 'closed';
  const option = ctxSelectedEngine(ctx);
  const opponent = ctxOpponentName(option);
  const topColor: Color = ctx.orientation === 'white' ? 'b' : 'w';
  ctxRenderStrip(ctx, 'topStrip', topColor, shownBoard, phase);
  ctxRenderStrip(ctx, 'bottomStrip', topColor === 'w' ? 'b' : 'w', shownBoard, phase);

  // Setup controls mirror the hidden selects so preferences and listeners stay single-sourced.
  const colorChoice = selectEl('colorSelect').value;
  for (const button of document.querySelectorAll<HTMLButtonElement>('.color-choice [data-color]')) {
    button.setAttribute('aria-pressed', String(button.dataset.color === colorChoice));
  }
  const elo = ctxSelectedMaia3Elo();
  el('maia3EloBand').textContent = `· ${maiaRatingBand(elo)}`;
  for (const chip of document.querySelectorAll<HTMLButtonElement>('.rating-presets [data-elo]')) {
    chip.setAttribute('aria-pressed', String(Number(chip.dataset.elo) === elo));
  }
  el('opponentBlurb').textContent = ctxOpponentBlurb(ctx);

  const change = buttonEl('changeOpponent');
  change.setAttribute('aria-expanded', String(ctx.configOpen));
  change.textContent = ctx.configOpen ? 'Done' : 'Change opponent';
  change.title = ctx.configOpen ? '' : 'Change opponent (asks before restarting the game)';
  const detailsLabel = maybeEl('detailsLabel');
  if (detailsLabel) detailsLabel.textContent = option.family === 'maia3' ? 'Exact rating & opponent details' : 'Opponent details';

  const newGame = buttonEl('newGame');
  newGame.classList.toggle('primary', phase === 'setup');
  newGame.classList.toggle('armed', ctx.newGameArmed);
  setButtonLabel(newGame, phase === 'setup' ? playCtaLabel() : ctx.newGameArmed ? 'Confirm new game' : 'New game');
  newGame.title = phase === 'playing' && !ctx.newGameArmed ? 'Abandon this game and start another (asks to confirm)' : '';

  const card = el('gameOverCard');
  card.hidden = !ctx.gameOver;
  if (ctx.gameOver) {
    const { result, reason } = ctx.gameOver;
    const humanWon = result !== '1/2-1/2' && (result === '1-0') === (ctx.humanColor === 'w');
    el('resultHeadline').textContent = result === '1/2-1/2' ? 'Draw' : humanWon ? 'You won' : `${opponent} won`;
    el('resultScore').textContent = result === '1/2-1/2' ? '½–½' : result.replace('-', '–');
    el('resultReason').textContent =
      `${reason.charAt(0).toUpperCase()}${reason.slice(1)} · ${ctx.sans.length ? `${Math.ceil(ctx.sans.length / 2)} moves` : 'no moves'}`;
  }

  const hasRecord = ctx.moves.length > 0 || !!ctx.practicePosition;
  buttonEl('copyPgn').disabled = !hasRecord;
  buttonEl('downloadPgn').disabled = !hasRecord;
  for (const link of document.querySelectorAll<HTMLAnchorElement>('[data-analyze]')) {
    link.setAttribute('aria-disabled', String(!hasRecord));
    // Mid-game, analysis opens beside the game instead of replacing it.
    if (phase === 'playing') {
      link.target = '_blank';
      link.rel = 'noopener';
    } else {
      link.removeAttribute('target');
      link.removeAttribute('rel');
    }
  }

  ctxEnsureOpeningNames(ctx);
  const opening = el('openingName');
  const name = ctx.moves.length ? ctxOpeningAt(ctx, shownPly) : null;
  opening.hidden = !name;
  opening.innerHTML = name ? `<strong>${htmlEscape(name[0])}</strong>${htmlEscape(name[1])}` : '';
  opening.title = name ? `${name[0]} ${name[1]} · Lichess opening names (CC0)` : '';
}

function ctxRender(ctx: PlayContext): void {
  if (ctx.disposed) return;
  ctx.autosave?.schedule(ctx.started || ctx.moves.length || ctx.gameOver || ctx.practicePosition ? {
    startFen: ctx.startFen,
    moves: ctx.moves.map(moveToUci),
    humanColor: ctx.humanColor,
    orientation: ctx.orientation,
    // Queued opponent/colour changes are not the opponent of this game until confirmed.
    preferences: ctx.opponent.active,
    gameOver: ctx.gameOver,
    started: ctx.started,
    ...(ctx.practiceBoundary ? { practiceBoundary: ctx.practiceBoundary } : {}),
    ...(ctx.practicePosition ? { practicePosition: ctx.practicePosition } : {}),
  } : null);
  ctx.viewPly = normalizeViewPly(ctx.viewPly, ctx.moves.length);
  const browsing = ctx.viewPly !== null;
  const statusText = ctxStatusText(ctx);
  const setupReady = ctxPhase(ctx) === 'setup' && ctx.engineReady && !ctx.engineThinking && !ctx.preparingEngine && !browsing;
  // Setup reads as a heading; the "Press Play or make a move" hint lives under the Play button.
  el('status').textContent = setupReady ? `New game vs ${ctxOpponentName(ctxSelectedEngine(ctx))}` : statusText;
  // The one live region: turn and readiness changes are announced here and nowhere else.
  const announcement = el('gameAnnouncement');
  const spokenStatus = setupReady ? `${ctxOpponentName(ctxSelectedEngine(ctx))} is ready. ${statusText}` : statusText;
  if (announcement.textContent !== spokenStatus) announcement.textContent = spokenStatus;
  el('status').classList.toggle('over', !!ctx.gameOver);
  el('status').classList.toggle('browsing', browsing);
  ctxRefreshSearchProgressLabel(ctx);
  el('status').setAttribute('aria-busy', ctx.engineThinking ? 'true' : 'false');
  ctxRenderMaia3Controls(ctx);
  el('levelCaption').textContent = ctxStrengthCaption(ctx);
  buttonEl('takeback').disabled = ctx.moves.length <= ctx.practiceBoundary || !!ctx.pendingPromotion;
  buttonEl('takeback').title = ctx.practicePosition ? 'Take back to your previous turn, no earlier than the imported position' : '';
  buttonEl('resign').disabled = !!ctx.gameOver || (!ctx.moves.length && !ctx.started);
  selectEl('engineSelect').disabled = ctx.engineThinking;
  ctxRenderMoveList(ctx);
  ctxRenderMoveNav(ctx);
  ctxRenderPromotionPicker(ctx);
  ctxRenderRestartBanner(ctx);
  const humanCanMove =
    !browsing &&
    !ctx.pendingPreferredEngineId &&
    !ctx.preparingEngine &&
    ctx.engineReady &&
    !ctx.engineThinking &&
    !ctx.gameOver &&
    !ctx.pendingPromotion &&
    ctx.board.turn === ctx.humanColor;
  const shownPly = displayedPly(ctx.viewPly, ctx.moves.length);
  const shownBoard = ctx.positions[shownPly] ?? ctx.board;
  const lastUci = shownPly > 0 ? moveToUci(ctx.moves[shownPly - 1]) : undefined;
  const config = {
    orientation: ctx.orientation,
    fen: boardToFen(shownBoard).split(' ')[0],
    turnColor: shownBoard.turn === 'w' ? ('white' as const) : ('black' as const),
    coordinates: true,
    trustAllEvents: true,
    check: boardCheck(shownBoard),
    highlight: { lastMove: true, check: true },
    animation: { enabled: true, duration: 160 },
    movable: {
      free: false,
      color: humanCanMove ? (ctx.humanColor === 'w' ? ('white' as const) : ('black' as const)) : undefined,
      dests: humanCanMove ? legalDests(ctx.board) : new Map<Key, Key[]>(),
      showDests: humanCanMove,
      events: { after: (from: Key, to: Key) => ctxOnUserMove(ctx, from, to) },
    },
    lastMove: lastUci ? [lastUci.slice(0, 2) as Key, lastUci.slice(2, 4) as Key] : undefined,
  };
  if (!ctx.ground) ctx.ground = Chessground(el('ground'), config);
  else ctx.ground.set(config);
  // Screen readers and keyboard users get the square grid; the drawn board is decorative to them.
  el('ground').querySelector('cg-board')?.setAttribute('aria-hidden', 'true');
  ctx.keyboardBoard ??= createAnalysisKeyboardBoard(el('keyboardBoard'), {
    onSelect: (key) => ctx.ground?.selectSquare(key),
    onMove: (from, to) => ctxOnUserMove(ctx, from, to),
    announce: (text) => {
      el('boardAnnouncement').textContent = text;
    },
  });
  ctx.keyboardBoard.render(shownBoard, ctx.orientation);
  el('positionNotice').hidden = !ctx.practicePosition && !ctx.handoffNotice;
  el('positionNoticeText').textContent = ctx.practicePosition
    ? 'Continue this position. Queen-odds opponents are unavailable because they need their own starting position.'
    : ctx.handoffNotice;
  // Switching roots is only offered before play or after the game is finished.
  buttonEl('useStartingPosition').hidden = !ctx.practicePosition || (ctx.started && !ctx.gameOver);
  ctxRenderChrome(ctx, shownBoard, shownPly);
}

// ---------------------------------------------------------------------------
// Restart banner logic
// ---------------------------------------------------------------------------

function ctxSetOpponent(ctx: PlayContext, next: PlayOpponentState): void {
  const previousEngineId = ctx.opponent.active.engineId;
  ctx.opponent = next;
  // Stormphrax's large WASM heap must not share residency with the next model.
  // A pending restart still belongs to the current game, so retain it until commit.
  if (previousEngineId === 'stormphrax' && next.active.engineId !== previousEngineId) disposeCachedCpuEngine(previousEngineId);
}

function ctxRestoreOpponentControls(ctx: PlayContext): void {
  const preferences = ctx.opponent.pending ?? ctx.opponent.active;
  selectEl('engineSelect').value = preferences.engineId;
  selectEl('colorSelect').value = preferences.color;
  ctx.lastEngineId = preferences.engineId;
  ctxRenderLevelOptions(ctx);
  selectEl('levelSelect').value = String(preferences.level);
  (el('maia3Elo') as HTMLInputElement).value = String(preferences.maiaElo);
  selectEl('maia3Style').value = preferences.maiaStyle;
  (el('maia3Temperature') as HTMLInputElement).value = String(preferences.maiaTemperature);
  (el('maia3TopP') as HTMLInputElement).value = String(preferences.maiaTopP);
  ctxRenderMaia3Controls(ctx);
  ctxRenderEngineCaution(ctx);
}

function ctxMaybeQueueRestart(ctx: PlayContext): void {
  const gameInProgress = !ctxGameIsFresh(ctx) && !ctx.gameOver;
  ctxSetOpponent(ctx, transitionPlayOpponent(ctx.opponent, {
    type: 'edit', preferences: ctxCurrentPlayPreferences(ctx), gameInProgress,
  }));
  ctxRestoreOpponentControls(ctx);
  ctxPersistPlayPreferences(ctx);
  if (!gameInProgress) {
    ctxNewGame(ctx);
    return;
  }
  ctxRender(ctx);
}

function ctxConfirmRestart(ctx: PlayContext): void {
  ctxSetOpponent(ctx, transitionPlayOpponent(ctx.opponent, { type: 'confirm' }));
  ctxRestoreOpponentControls(ctx);
  ctxNewGame(ctx);
}

function ctxDismissRestart(ctx: PlayContext): void {
  ctxSetOpponent(ctx, transitionPlayOpponent(ctx.opponent, { type: 'dismiss' }));
  ctxRestoreOpponentControls(ctx);
  ctxPersistPlayPreferences(ctx);
  ctxRender(ctx);
}

function ctxTuneOpponent(ctx: PlayContext): void {
  ctxSetOpponent(ctx, transitionPlayOpponent(ctx.opponent, { type: 'tune', preferences: ctxCurrentPlayPreferences(ctx) }));
  ctxPersistPlayPreferences(ctx);
  ctxRender(ctx);
}

// ---------------------------------------------------------------------------
// Mount/init
// ---------------------------------------------------------------------------

/** Import before any autosaved-game restore; callers must skip restore when this returns true. */
function ctxApplyPlayHandoffForMount(ctx: PlayContext): boolean {
  const handoff = takePlayHandoff();
  if (handoff.status === 'ready') {
    ctx.practicePosition = handoff.position;
    return true;
  }
  if (handoff.status === 'invalid') {
    ctx.handoffNotice = 'That position could not be loaded or is finished. Set up a normal game instead.';
    return true;
  }
  return false;
}

function ctxResetStartPosition(ctx: PlayContext): void {
  if (ctx.practicePosition) {
    Object.assign(ctx, replayPlayPosition(ctx.practicePosition));
    ctx.practiceBoundary = ctx.moves.length;
  } else {
    ctx.startFen = startFenFor(ctxSelectedEngine(ctx), ctx.humanColor);
    ctx.board = parseFen(ctx.startFen);
    ctx.positions = [ctx.board];
    ctx.moves = [];
    ctx.sans = [];
    ctx.practiceBoundary = 0;
  }
}

/** Imported setup must not make an opponent move before the player presses Play. */
function ctxPrepareOrStartEngine(ctx: PlayContext): void {
  if (ctx.practicePosition && !ctx.started) return;
  if (ctx.board.turn !== ctx.humanColor) void ctxEngineTurn(ctx);
  else if (!ctx.engineReady) void ctxVerifySelectedEngineReady(ctx);
}

function ctxResetGameStateForMount(ctx: PlayContext): void {
  ctxCancelEngineTurn(ctx);
  ctxSetOpponent(ctx, committedPlayOpponent(ctxCurrentPlayPreferences(ctx)));
  const colorChoice = ctx.opponent.active.color;
  ctx.humanColor = colorChoice === 'random' ? (Math.random() < 0.5 ? 'w' : 'b') : colorChoice === 'black' ? 'b' : 'w';
  ctx.orientation = ctx.humanColor === 'w' ? 'white' : 'black';
  ctxResetStartPosition(ctx);
  ctx.gameOver = null;
  ctx.pendingPromotion = null;
  ctx.viewPly = null;
  ctx.lastEngineId = ctx.opponent.active.engineId;
  ctxSetEngineNote('');
  ctxDisarmResign(ctx);
}

function ctxInit(ctx: PlayContext): void {
  // Handoff (including rejected transfers) takes precedence over saved-game restore; explicit URLs also skip restore.
  const handedOff = ctxApplyPlayHandoffForMount(ctx);
  const validEngineIds = new Set(ENGINE_OPTIONS.map((option) => option.id));
  const saved = handedOff || hasExplicitWorkspaceInput(location.search, location.hash) ? null
    : readWorkspace('play', (raw) => decodePlaySnapshot(raw, validEngineIds));
  ctx.restoredGame = !!saved;
  if (saved) ctx.practicePosition = saved.practicePosition;
  ctxRefreshEngineOptions(ctx);
  const preferences = saved?.preferences ?? loadPlayPreferences(playPreferenceStorage(), validEngineIds);
  const preferredOption = ENGINE_OPTIONS.find((option) => option.id === preferences.engineId && (!ctx.practicePosition || option.group !== 'odds'));
  const preferredBigNetStatus =
    preferredOption?.family === 'lc0' && preferredOption.variant !== 'small' ? bigNetAssetStatusSync(BIG_NETS[preferredOption.variant as BigNetKey]) : null;
  const preferredBigNetCanProbe =
    preferredOption?.family === 'lc0' && preferredOption.variant !== 'small' && (!isV0DeployProfile() || preferredOption.variant === 'lqo');
  ctx.pendingPreferredEngineId = !saved && preferredOption && preferredBigNetCanProbe && preferredBigNetStatus === 'unknown' ? preferredOption.id : null;
  selectEl('engineSelect').value = preferredOption && (saved || !ctxEngineOptionState(preferredOption).disabled) ? preferredOption.id : 'maia3';
  selectEl('colorSelect').value = ctx.practicePosition && !saved
    ? (replayPlayPosition(ctx.practicePosition).board.turn === 'w' ? 'white' : 'black')
    : preferences.color;
  (el('maia3Elo') as HTMLInputElement).value = String(preferences.maiaElo);
  selectEl('maia3Style').value = preferences.maiaStyle;
  (el('maia3Temperature') as HTMLInputElement).value = String(preferences.maiaTemperature);
  (el('maia3TopP') as HTMLInputElement).value = String(preferences.maiaTopP);
  ctxRenderLevelOptions(ctx);
  selectEl('levelSelect').value = String(preferences.level);
  ctxResetGameStateForMount(ctx);
  if (saved) {
    const { preferences: _preferences, ...game } = saved;
    Object.assign(ctx, game);
    el('playRestoreNotice').hidden = false;
  }
  ctx.autosave = createWorkspaceAutosave('play');
  if (handedOff || hasExplicitWorkspaceInput(location.search, location.hash)) ctx.autosave.clear();
  trackListener(ctx, window, 'pagehide', () => ctx.autosave?.flush());
  trackListener(ctx, document, 'visibilitychange', () => { if (document.hidden) ctx.autosave?.flush(); });
  trackListener(ctx, el('playRestoreDismiss'), 'click', () => { el('playRestoreNotice').hidden = true; });
  trackListener(ctx, el('playRestoreNew'), 'click', () => {
    ctx.started = false;
    ctx.configOpen = false;
    ctxNewGame(ctx);
  });

  trackListener(ctx, el('newGame'), 'click', () => {
    const resolvingSavedEngine = !!ctx.pendingPreferredEngineId;
    ctx.pendingPreferredEngineId = null;
    if (ctx.pendingPreferredEngineTimer) {
      clearTimeout(ctx.pendingPreferredEngineTimer);
      ctx.pendingPreferredEngineTimer = null;
    }
    const phase = ctxPhase(ctx);
    if (phase === 'setup' && !resolvingSavedEngine) {
      // Play: the board is already set up with these settings, so start without a reset.
      ctx.started = true;
      ctx.configOpen = false;
      if (!ctx.preparingEngine && !ctx.engineThinking) ctxPrepareOrStartEngine(ctx);
      ctxRender(ctx);
      if (matchMedia('(max-width: 900px)').matches) document.querySelector('.board-panel')?.scrollIntoView({ block: 'start' });
      return;
    }
    if (phase === 'playing' && ctxHumanMoveCount(ctx) > 0 && !ctx.newGameArmed) {
      ctxArmNewGame(ctx);
      return;
    }
    ctxDisarmNewGame(ctx);
    ctx.started = true;
    ctxNewGame(ctx);
  });
  trackListener(ctx, el('useStartingPosition'), 'click', () => {
    ctx.practicePosition = null;
    ctx.handoffNotice = '';
    ctx.started = false;
    ctx.configOpen = false;
    ctxRefreshEngineOptions(ctx);
    ctxResetGameStateForMount(ctx);
    ctx.engineReady = false;
    ctxRender(ctx);
    ctxPrepareOrStartEngine(ctx);
  });
  trackListener(ctx, el('changeOpponent'), 'click', () => {
    ctx.configOpen = !ctx.configOpen;
    ctxRender(ctx);
    if (ctx.configOpen) selectEl('engineSelect').focus();
  });
  trackListener(ctx, document.querySelector('.color-choice') ?? document, 'click', (event: Event) => {
    const button = (event.target as HTMLElement | null)?.closest?.<HTMLButtonElement>('[data-color]');
    const select = selectEl('colorSelect');
    if (!button?.dataset.color || button.dataset.color === select.value) return;
    select.value = button.dataset.color;
    select.dispatchEvent(new Event('change', { bubbles: true }));
  });
  trackListener(ctx, document.querySelector('.rating-presets') ?? document, 'click', (event: Event) => {
    const chip = (event.target as HTMLElement | null)?.closest?.<HTMLButtonElement>('[data-elo]');
    const input = el('maia3Elo') as HTMLInputElement;
    if (!chip?.dataset.elo) return;
    input.value = chip.dataset.elo;
    input.dispatchEvent(new Event('input', { bubbles: true }));
  });
  trackListener(ctx, el('moveForm'), 'submit', (event: Event) => {
    event.preventDefault();
    ctxSubmitTypedMove(ctx);
  });
  trackListener(ctx, el('moveInput'), 'input', () => {
    const input = el('moveInput') as HTMLInputElement;
    input.setCustomValidity('');
    input.classList.remove('invalid');
  });
  for (const link of document.querySelectorAll<HTMLAnchorElement>('[data-analyze]')) {
    trackListener(ctx, link, 'click', (event: Event) => {
      if (!ctx.moves.length && !ctx.practicePosition) {
        event.preventDefault();
        return;
      }
      writeAnalysisHandoff(ctxExportPgn(ctx));
    });
  }
  trackListener(ctx, el('downloadPgn'), 'click', () => {
    if (!ctx.moves.length && !ctx.practicePosition) return;
    const pgn = ctxExportPgn(ctx);
    const url = URL.createObjectURL(new Blob([pgn], { type: 'application/x-chess-pgn' }));
    const link = document.createElement('a');
    const opponent = ctxOpponentName(ctxSelectedEngine(ctx))
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-|-$/g, '');
    link.href = url;
    link.download = `0x88-${new Date().toISOString().slice(0, 10)}-vs-${opponent || 'engine'}.pgn`;
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  trackListener(ctx, el('retryEngine'), 'click', () => {
    if (ctx.engineThinking || ctx.preparingEngine || ctx.gameOver) return;
    buttonEl('retryEngine').hidden = true;
    ctxSetEngineNote('Retrying engine…');
    if (ctx.board.turn === ctx.humanColor) void ctxVerifySelectedEngineReady(ctx);
    else void ctxEngineTurn(ctx);
  });
  trackListener(ctx, window, BROWSER_RUNTIME_AUDIT_EVENT, (event: Event) => {
    const detail = (event as CustomEvent<BrowserRuntimeAuditDetail>).detail;
    if (detail.surface && detail.surface !== 'play') return;
    const identity = browserRuntimeAuditIdentity(detail);
    if (detail.fallbackReason) playRuntimeFallbackAudits.set(identity, detail);
    else playRuntimeFallbackAudits.delete(identity);
    if (playOptionMatchesRuntimeAudit(ctxSelectedEngine(ctx), detail)) ctxRestoreSelectedRuntimeWarning(ctx);
  });
  trackListener(ctx, el('takeback'), 'click', () => ctxTakeback(ctx));
  trackListener(ctx, el('navFirst'), 'click', () => ctxNavigate(ctx, 'first'));
  trackListener(ctx, el('navPrev'), 'click', () => ctxNavigate(ctx, 'prev'));
  trackListener(ctx, el('navNext'), 'click', () => ctxNavigate(ctx, 'next'));
  trackListener(ctx, el('navLast'), 'click', () => ctxNavigate(ctx, 'last'));
  trackListener(ctx, el('moveList'), 'click', (event: Event) => {
    const span = (event.target as HTMLElement | null)?.closest?.('.san[data-ply]') as HTMLElement | null;
    if (!span) return;
    ctxSetViewPly(ctx, viewPlyFor(Number(span.dataset.ply), ctx.moves.length));
  });
  trackListener(ctx, document, 'keydown', (event: Event) => {
    const keyEvent = event as KeyboardEvent;
    // The promotion overlay owns the arrow keys (and Escape) while it is open.
    if (keyEvent.defaultPrevented || ctx.pendingPromotion) return;
    const action = moveNavActionForKey(keyEvent);
    if (!action) return;
    keyEvent.preventDefault();
    ctxNavigate(ctx, action);
  });
  trackListener(ctx, el('resign'), 'click', () => ctxResign(ctx));
  trackListener(ctx, el('flip'), 'click', () => {
    ctx.orientation = ctx.orientation === 'white' ? 'black' : 'white';
    ctxRender(ctx);
  });
  trackListener(ctx, el('copyPgn'), 'click', () => {
    void navigator.clipboard.writeText(ctxExportPgn(ctx)).then(() => {
      const btn = buttonEl('copyPgn');
      setButtonLabel(btn, 'Copied');
      btn.dataset.copied = 'true';
      setTimeout(() => {
        setButtonLabel(btn, 'Copy PGN');
        btn.dataset.copied = 'false';
      }, 1500);
    });
  });
  trackListener(ctx, el('confirmRestart'), 'click', () => ctxConfirmRestart(ctx));
  trackListener(ctx, el('dismissRestart'), 'click', () => ctxDismissRestart(ctx));
  trackListener(ctx, selectEl('colorSelect'), 'change', () => ctxMaybeQueueRestart(ctx));
  trackListener(ctx, selectEl('engineSelect'), 'change', () => {
    ctx.pendingPreferredEngineId = null;
    if (ctx.pendingPreferredEngineTimer) {
      clearTimeout(ctx.pendingPreferredEngineTimer);
      ctx.pendingPreferredEngineTimer = null;
    }
    const previousEngineId = ctx.lastEngineId;
    if (ctx.engineThinking) {
      selectEl('engineSelect').value = previousEngineId;
      ctxRender(ctx);
      return;
    }
    const option = ctxDraftEngine();
    if (option.family === 'maia3' && previousEngineId.startsWith('maia-')) {
      const carried = Number(ctx.lastEngineId.slice('maia-'.length));
      if (Number.isFinite(carried)) {
        (el('maia3Elo') as HTMLInputElement).value = String(Math.max(MAIA3_MIN_ELO, Math.min(MAIA3_MAX_ELO, carried)));
      }
    }
    ctx.lastEngineId = option.id;
    ctxRenderLevelOptions(ctx);
    ctxMaybeQueueRestart(ctx);
  });
  trackListener(ctx, selectEl('levelSelect'), 'change', () => ctxTuneOpponent(ctx));
  trackListener(ctx, selectEl('maia3Style'), 'change', () => ctxTuneOpponent(ctx));
  trackListener(ctx, el('maia3Elo'), 'input', () => ctxTuneOpponent(ctx));
  trackListener(ctx, el('maia3Temperature'), 'input', () => ctxTuneOpponent(ctx));
  trackListener(ctx, el('maia3TopP'), 'input', () => ctxTuneOpponent(ctx));
  const refreshIfMounted = () => {
    if (!ctxIsDisposed(ctx)) {
      ctxRefreshEngineOptions(ctx);
      ctxResolvePendingPreferredEngine(ctx);
    }
  };
  if (!isV0DeployProfile()) {
    void probeBt4Support().then(refreshIfMounted);
    void checkBigNetAsset(BIG_NETS.bt4).then(refreshIfMounted);
    void checkBigNetAsset(T3_NET).then(refreshIfMounted);
    void checkBigNetAsset(LQO_NET).then(refreshIfMounted);
  } else {
    void probeBt4Support().then(refreshIfMounted);
    void checkBigNetAsset(LQO_NET).then(refreshIfMounted);
  }
  ctxRender(ctx);
  if (ctx.pendingPreferredEngineId) {
    ctxSetEngineNote('Checking your saved engine assets…');
    ctx.pendingPreferredEngineTimer = setTimeout(() => {
      if (ctx.disposed || !ctx.pendingPreferredEngineId) return;
      ctx.pendingPreferredEngineId = null;
      ctx.pendingPreferredEngineTimer = null;
      ctxPersistPlayPreferences(ctx);
      ctx.engineReady = false;
      ctxSetRuntimeWarning('The saved-engine check timed out, so Play is using Maia. You can retry the engine from the selector.');
      ctxRender(ctx);
      ctxPrepareOrStartEngine(ctx);
    }, 10_000);
  } else if (!ctx.gameOver) ctxPrepareOrStartEngine(ctx);
}

function hasPlayBrowserDom(): boolean {
  const required = [
    'positionNotice',
    'positionNoticeText',
    'useStartingPosition',
    'ground',
    'status',
    'restartBanner',
    'restartMessage',
    'confirmRestart',
    'dismissRestart',
    'engineSelect',
    'levelSelect',
    'maia3Controls',
    'maia3Elo',
    'maia3EloValue',
    'maia3Style',
    'maia3TemperatureField',
    'maia3Temperature',
    'maia3TopPField',
    'maia3TopP',
    'levelCaption',
    'engineCaution',
    'engineNote',
    'runtimeWarning',
    'retryEngine',
    'dlProgress',
    'progressAnnouncement',
    'colorSelect',
    'newGame',
    'takeback',
    'resign',
    'flip',
    'moveList',
    'navFirst',
    'navPrev',
    'navNext',
    'navLast',
    'copyPgn',
    'downloadPgn',
    'analyzeGame',
    'main',
    'topStrip',
    'bottomStrip',
    'keyboardBoard',
    'boardAnnouncement',
    'gameAnnouncement',
    'opponentBlurb',
    'changeOpponent',
    'maia3EloBand',
    'maia3Advanced',
    'gameOverCard',
    'resultHeadline',
    'resultScore',
    'resultReason',
    'openingName',
    'moveForm',
    'moveInput',
  ];
  const missing = required.filter((id) => !document.getElementById(id));
  if (!missing.length) return true;
  console.warn(`Play browser mount skipped; missing required DOM element(s): ${missing.map((id) => `#${id}`).join(', ')}`);
  return false;
}

let lastPlayMountHref: string | null = null;

export function mountPlayBrowser(): () => void {
  if (!hasPlayBrowserDom()) return () => undefined;
  const href = location.href;
  if (lastPlayMountHref !== null && lastPlayMountHref !== href) {
    location.reload();
    return () => undefined;
  }
  lastPlayMountHref = href;
  const ctx = createPlayContext();
  ctxInit(ctx);

  // Test hook for automated browser checks: synthetic chessground drags are
  // unreliable, so smokes call this to route through the real user-move path.
  const hook = (from: string, to: string) => ctxOnUserMove(ctx, from as Key, to as Key);
  (globalThis as unknown as { __playUserMove?: (from: string, to: string) => void }).__playUserMove = hook;

  return () => {
    ctx.autosave?.flush();
    ctx.autosave = null;
    ctx.disposed = true;
    ctxCancelEngineTurn(ctx);
    ctx.abort.abort();
    releaseUnusedBigNetSearchers([], 0);
    centipawnEvaluatorGeneration += 1;
    centipawnEvaluator?.clear();
    const destroyCentipawn = (centipawnEvaluator?.inner as (Evaluator & { destroy?: () => void }) | undefined)?.destroy;
    if (typeof destroyCentipawn === 'function') destroyCentipawn.call(centipawnEvaluator?.inner);
    centipawnEvaluator = null;
    centipawnLoadPromise = null;
    playRuntimeFallbackAudits.clear();
    // Stormphrax reserves a 512 MiB WASM heap, so unlike the smaller CPU
    // engines it must not remain cached after this Play route is gone.
    disposeCachedCpuEngine('stormphrax');
    if (ctx.resignArmTimer) {
      clearTimeout(ctx.resignArmTimer);
      ctx.resignArmTimer = null;
    }
    ctxDisarmNewGame(ctx);
    ctx.keyboardBoard?.destroy();
    ctx.keyboardBoard = null;
    if (ctx.pendingPreferredEngineTimer) {
      clearTimeout(ctx.pendingPreferredEngineTimer);
      ctx.pendingPreferredEngineTimer = null;
    }
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
