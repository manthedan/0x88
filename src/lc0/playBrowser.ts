// Play vs engine: a human-friendly game page on top of the same engine stack
// used by the arena and analysis pages. Engines load lazily on first move.
import { Chessground } from 'chessground';
import type { Key } from 'chessground/types';
import { type BoardState, boardToFen, type Color, parseFen, START_FEN } from '../chess/board.ts';
import { searchHistoryFens } from '../chess/drawRules.ts';
import { type Move, moveToUci } from '../chess/moveCodec.ts';
import { legalMoves, makeMove } from '../chess/movegen.ts';
import { sanToMove } from '../chess/pgn.ts';
import { moveToSan } from '../chess/san.ts';
import type { RouteDom } from '../features/dom.ts';
import type { RouteId } from '../features/registry.ts';
import {
  BROWSER_RUNTIME_AUDIT_EVENT,
  type BrowserRuntimeAuditDetail,
  browserRuntimeAuditIdentity,
  clearRuntimeFallbackWarning,
  updateRuntimeFallbackWarning,
} from '../nn/runtimeAudit.ts';
import { centipawnSearchOptions } from '../search/centipawn.ts';
import { chooseMove } from '../search/puct.ts';
import { acknowledgeInputTransfer, takePlayHandoff, writeAnalysisHandoff } from './analysisHandoff.ts';
import { createAnalysisKeyboardBoard } from './analysisKeyboardBoard.ts';
import { resolvePublicAssetUrl } from './assetUrls.ts';
import { type BigNetKey, peekBigNetSearcher, releaseUnusedBigNetSearchers } from './bigNetSessionPool.ts';
import { boardCheck, hidePromotionOverlay, legalDests, matchUserMoves, showPromotionOverlay } from './boardUx.ts';
import { createRootDom, htmlEscape, type RootDom } from './browserDom.ts';
import {
  BIG_NETS,
  bigNetAssetStatusSync,
  bigNetBackendUsesWasm,
  bigNetCpuWarning,
  bigNetMemoryCaution,
  bigNetOptionState,
  bt4SupportedSync,
  checkBigNetAsset,
  phoneBlockedBigNetModelUrl,
  probeBt4Support,
  renderBigNetCpuWarning,
} from './bt4Engine.ts';
import { writeClipboard } from './clipboard.ts';
import { isMemoryConstrainedDevice, phoneReclaimSettings } from './deviceMemoryClass.ts';
import { type GameResultCode, gameOutcome } from './engineBattle.ts';
import { enginePlayLevels, enginePlayOptions, isV0DeployProfile } from './engineCatalog.ts';
import { requestLargeNetConsent, showPhoneLargeNetFallback } from './largeNetConsent.ts';
import { createBrowserSquareformerRuntimeEvaluator, EvaluatorImportError } from './lazyOrtEvaluators.ts';
import { LC0_SMALL_BATCH1_QDQ8_PATH, lc0SmallModelRequest } from './lc0ModelUrls.ts';
import { lqoBlackBookMove, lqoWhiteBookMove, lqoWhiteFirstPolicyBookMove } from './lqoOpeningBook.ts';
import { LQO_BLACK_MISSING_QUEEN_FEN, LQO_WHITE_MISSING_QUEEN_FEN } from './lqoStartingPositions.ts';
import { MAIA3_DEFAULT_ELO, MAIA3_MAX_ELO, MAIA3_MIN_ELO, type Maia3MoveStyle, maia3ModelFallbackUrls } from './maia3.ts';
import type { MountScope } from './mountScope.ts';
import { rememberMoveFocus } from './moveListFocus.ts';
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
import { type OpeningNameIndex, openingPositionKey } from './openingNames.ts';
import { type PendingInput, type PendingInputEvent, transitionPendingInput } from './pendingInput.ts';
import { PHONE_LARGE_NET_CHANGE_EVENT, phoneLargeNetsBlocked, phoneSafeEngineVariant } from './phoneLargeNetPolicy.ts';
import {
  type CentipawnAssets,
  createPlayEngines,
  type EngineUiPort,
  type Lc0SmallModel,
  type PlayEngineSpec,
  type PlayFamily,
  playCentipawnAssets,
} from './play/playEngines.ts';
import {
  createEngineUiPort,
  hideDownloadProgress as ctxHideDownloadProgress,
  setEngineNote as ctxSetEngineNote,
  showDownloadProgress as ctxShowDownloadProgress,
  type PlayEngineUiState,
} from './play/playEngineUi.ts';
import { reloadForChangedOverrides } from './play/playOverrides.ts';
import { BIG_NET_LEVELS, lqoSearchOptions, PLAY_DRAW_SCORE, PLAY_SEARCH_CONTEMPT_LIMIT } from './play/playSearchSettings.ts';
import { playMaterial } from './playMaterial.ts';
import { committedPlayOpponent, type PlayOpponentState, transitionPlayOpponent } from './playOpponentState.ts';
import { type PlayPosition, playGameToPgn, playMoveLabel, playMoveRows, playTakebackPly, replayPlayPosition } from './playPosition.ts';
import { DEFAULT_PLAY_PREFERENCES, loadPlayPreferences, type PlayPreferences, type PreferenceStorage, savePlayPreferences } from './playPreferences.ts';
import { defaultStormphraxVariantKey, stormphraxVariantByKey, stormphraxVariantUnsupportedReason } from './stormphraxVariants.ts';
import { uciGameHistory } from './uciGameHistory.ts';
import { createWakeLockManager } from './wakeLock.ts';
import { createWorkspaceAutosave, decodePlaySnapshot, readWorkspace } from './workspaceAutosave.ts';

/** Read per mount: client-side navigation can change ?model=, ?assetBase= and ?modelBase= without reloading this module. */
function playModelRequest(url: URL): Pick<PlayContext, 'blockedModelOverride' | 'lc0SmallModel' | 'maia3ModelUrls' | 'centipawnAssets'> {
  const requested = url.searchParams.get('model');
  // Asset URLs resolve against the live page URL, which is this mount's.
  const defaultModelUrl = resolvePublicAssetUrl(LC0_SMALL_BATCH1_QDQ8_PATH);
  const blockedModelOverride = phoneBlockedBigNetModelUrl(requested);
  return {
    blockedModelOverride,
    lc0SmallModel: lc0SmallModelRequest(defaultModelUrl, isV0DeployProfile() || blockedModelOverride ? null : requested, resolvePublicAssetUrl),
    maia3ModelUrls: maia3ModelFallbackUrls(),
    centipawnAssets: playCentipawnAssets(),
  };
}

interface PlayEngineOption extends PlayEngineSpec {
  group: 'human' | 'odds' | 'engine';
}

const ENGINE_OPTIONS: PlayEngineOption[] = [
  { id: 'maia3', label: 'Maia3 · Human-like opponent', family: 'maia3', variant: 'maia3', group: 'human' },
  ...enginePlayOptions().map(({ id, label, family, variant, group }) => ({ id, label, family, variant, group })),
];

const LEVEL_COUNT = 5;
/** Search-effort labels. Deliberately not "Beginner": engine play is strong at any setting. */
const EFFORT_LEVEL_NAMES = ['Fastest', 'Quick', 'Standard', 'Strong', 'Deep'] as const;
const SF_LEVELS = [
  { skill: 0, depth: 5, label: '1 · Weakest (skill 0)' },
  { skill: 5, depth: 6, label: '2 · Casual (skill 5)' },
  { skill: 10, depth: 8, label: '3 · Club (skill 10)' },
  { skill: 15, depth: 12, label: '4 · Strong (skill 15)' },
  { skill: 20, depth: 18, label: '5 · Full strength' },
] as const;

const engines = createPlayEngines({ loadCentipawn: createBrowserSquareformerRuntimeEvaluator });

const wakeLocks = createWakeLockManager(() => navigator.wakeLock?.request('screen'));

/** Start FEN for the current game; odds opponents remove their own queen. */
function startFenFor(option: PlayEngineOption, human: Color): string {
  if (option.variant !== 'lqo') return START_FEN;
  return human === 'w' ? LQO_BLACK_MISSING_QUEEN_FEN : LQO_WHITE_MISSING_QUEEN_FEN;
}

// ---------------------------------------------------------------------------
// Factory: scoped per-mount play context. All game/UI state lives here so a
// remount (client navigation back to Play, a new ?model=) starts fresh. The
// abort signal replaces the old gameSeq counter as the stale-write guard for
// async engine callbacks.
// ---------------------------------------------------------------------------
/** A mount's root-scoped lookups with literal ids checked against the Play feature registry (src/features/dom.ts). */
type PlayDom = Omit<RootDom, keyof RouteDom<string>> & RouteDom<RouteId<'play'>>;

interface PlayContext extends PlayEngineUiState {
  scope: MountScope;
  /** Lookups confined to this mount's root; they keep working on the detached root during teardown. Ids are typed against the feature registry. */
  dom: PlayDom;
  blockedModelOverride: boolean;
  lc0SmallModel: Lc0SmallModel;
  maia3ModelUrls: string[];
  centipawnAssets: CentipawnAssets;
  runtimeFallbackAudits: Map<string, BrowserRuntimeAuditDetail>;
  autosave: ReturnType<typeof createWorkspaceAutosave> | null;
  pendingInput: PendingInput | null;
  restoredGame: boolean;
  consentAbort: AbortController;
  pickerConsent: { controller: AbortController; resumeFallback: boolean } | null;
  restoredDownloadEngineId: string | null;
  restoredDownloadConsent: Promise<boolean> | null;
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
  /**
   * UI-only: the thinking indicator is showing. It trails engineThinking by
   * THINKING_SHOW_DELAY_MS so fast replies do not flash "Thinking…" and the
   * status block; gating (move blocking, aria-busy) still uses engineThinking.
   */
  thinkingShown: boolean;
  thinkingShowTimer: ReturnType<typeof setTimeout> | null;
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
  /** This mount asked for the shared opening-name index (at most once). */
  openingNamesRequested: boolean;
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
  /** Where engine loads report notes and progress; stale loads see a superseded generation. */
  engineUi: EngineUiPort;
}

function createPlayContext(scope: MountScope): PlayContext {
  const ctx: PlayContext = {
    scope,
    dom: createRootDom(scope.root),
    ...playModelRequest(scope.url),
    runtimeFallbackAudits: new Map(),
    engineNote: '',
    autosave: null,
    pendingInput: null,
    restoredGame: false,
    consentAbort: new AbortController(),
    pickerConsent: null,
    restoredDownloadEngineId: null,
    restoredDownloadConsent: null,
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
    thinkingShown: false,
    thinkingShowTimer: null,
    pendingPromotion: null,
    viewPly: null,
    searchProgress: null,
    moveListScrollKey: '',
    openingNamesRequested: false,
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
    engineUi: null as unknown as EngineUiPort,
  };
  ctx.engineUi = createEngineUiPort(ctx);
  return ctx;
}

function ctxIsDisposed(ctx: PlayContext): boolean {
  return ctx.disposed;
}

function ctxSelectedEngine(ctx: PlayContext): PlayEngineOption {
  return ENGINE_OPTIONS.find((option) => option.id === ctx.opponent.active.engineId) ?? ENGINE_OPTIONS[0];
}
function ctxDraftEngine(ctx: PlayContext): PlayEngineOption {
  return ENGINE_OPTIONS.find((option) => option.id === ctx.dom.selectEl('engineSelect').value) ?? ENGINE_OPTIONS[0];
}
function ctxSelectedLevel(ctx: PlayContext): number {
  return Math.max(0, Math.min(LEVEL_COUNT - 1, Number(ctx.dom.selectEl('levelSelect').value) || 0));
}
function ctxSelectedMaia3Elo(ctx: PlayContext): number {
  const input = ctx.dom.el('maia3Elo') as HTMLInputElement;
  return Math.max(MAIA3_MIN_ELO, Math.min(MAIA3_MAX_ELO, Number(input.value) || MAIA3_DEFAULT_ELO));
}
function ctxSelectedMaia3Style(ctx: PlayContext): Maia3MoveStyle {
  return ctx.dom.selectEl('maia3Style').value === 'argmax' ? 'argmax' : 'sample';
}
function ctxSelectedMaia3Temperature(ctx: PlayContext): number {
  const input = ctx.dom.el('maia3Temperature') as HTMLInputElement;
  return Math.max(0.01, Math.min(5, Number(input.value) || 1));
}
function ctxSelectedMaia3TopP(ctx: PlayContext): number {
  const input = ctx.dom.el('maia3TopP') as HTMLInputElement;
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
    engineId: ctxDraftEngine(ctx).id,
    level: ctxSelectedLevel(ctx),
    color: ctx.dom.selectEl('colorSelect').value as PlayPreferences['color'],
    maiaElo: ctxSelectedMaia3Elo(ctx),
    maiaStyle: ctxSelectedMaia3Style(ctx),
    maiaTemperature: ctxSelectedMaia3Temperature(ctx),
    maiaTopP: ctxSelectedMaia3TopP(ctx),
  };
}

function ctxPersistPlayPreferences(ctx: PlayContext): void {
  savePlayPreferences(playPreferenceStorage(), ctx.opponent.active);
}

function ctxStrengthFor(option: PlayEngineOption, level: number, backend?: string): number {
  if (option.family === 'maia3') return 1;
  if (option.family === 'sf') return SF_LEVELS[level].depth;
  if (option.family === 'lc0' && option.variant !== 'small') {
    const usesWasm = backend === undefined ? !bt4SupportedSync() : bigNetBackendUsesWasm(backend);
    return (usesWasm ? BIG_NETS[option.variant as BigNetKey].wasmLevels : BIG_NET_LEVELS)[level];
  }
  return enginePlayLevels(option.family)[level];
}

function ctxStrengthCaption(ctx: PlayContext): string {
  const option = ctxDraftEngine(ctx);
  const level = ctxSelectedLevel(ctx);
  if (option.family === 'maia3') {
    const style = ctxSelectedMaia3Style(ctx);
    const suffix =
      style === 'argmax'
        ? 'deterministic top human move'
        : `sampled, temperature ${ctxSelectedMaia3Temperature(ctx).toFixed(2)}, top-p ${ctxSelectedMaia3TopP(ctx).toFixed(2)}`;
    const maia3Status = engines.maia3Status();
    const status = maia3Status ? ` ${maia3Status}.` : '';
    return `Maia3 predicts human moves at Elo ${ctxSelectedMaia3Elo(ctx)} — ${suffix}. No LC0/PUCT search; its WDL output predicts the human game outcome, not an engine eval.${status}`;
  }
  if (option.family === 'sf') {
    const sf = SF_LEVELS[level];
    return sf.skill >= 20 ? `full strength · depth ${sf.depth}` : `UCI skill ${sf.skill} · depth ${sf.depth}`;
  }
  const backend = option.family === 'lc0' && option.variant !== 'small' ? peekBigNetSearcher(option.variant as BigNetKey)?.backend : undefined;
  const value = ctxStrengthFor(option, level, backend || undefined);
  if (option.variant === 'lqo') return `≈ ${value} visits per move · specialized queen-odds search`;
  const isNeuralSearch = option.family === 'lc0' || option.family === 'centipawn';
  const base = isNeuralSearch ? `≈ ${value} visits per move` : `search depth ${value}`;
  return isNeuralSearch ? `${base} — strong even on Fastest; pick a Maia for a human-level opponent` : base;
}

function ctxRenderLevelOptions(ctx: PlayContext): void {
  const select = ctx.dom.selectEl('levelSelect');
  const previous = select.value;
  const option = ctxDraftEngine(ctx);
  const field = select.closest('.field') as HTMLElement;
  field.hidden = option.family === 'maia3';
  const labels = option.family === 'sf' ? SF_LEVELS.map((sf) => sf.label) : EFFORT_LEVEL_NAMES.map((name, i) => `${i + 1} · ${name}`);
  select.innerHTML = labels.map((label, i) => `<option value="${i}">${label}</option>`).join('');
  select.value = previous && Number(previous) < LEVEL_COUNT ? previous : '2';
}

function ctxRenderMaia3Controls(ctx: PlayContext): void {
  const isMaia3 = ctxDraftEngine(ctx).family === 'maia3';
  ctx.dom.el('maia3Controls').hidden = !isMaia3;
  ctx.dom.el('maia3Advanced').hidden = !isMaia3;
  const elo = ctxSelectedMaia3Elo(ctx);
  ctx.dom.el('maia3EloValue').textContent = String(elo);
  ctx.dom.el('maia3Elo').setAttribute('aria-valuetext', `${elo} rating · ${maiaRatingBand(elo)}`);
  const sample = ctxSelectedMaia3Style(ctx) === 'sample';
  (ctx.dom.el('maia3Temperature') as HTMLInputElement).disabled = !sample;
  (ctx.dom.el('maia3TopP') as HTMLInputElement).disabled = !sample;
  (ctx.dom.el('maia3TemperatureField') as HTMLElement).style.opacity = sample ? '' : '0.5';
  (ctx.dom.el('maia3TopPField') as HTMLElement).style.opacity = sample ? '' : '0.5';
}

function ctxSetRuntimeWarning(ctx: PlayContext, text: string): void {
  const warning = ctx.dom.el('runtimeWarning');
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
    detail.modelId === 's4fix-v-ema' &&
    detail.requestedRuntime === 'ort'
  );
}

function ctxRestoreSelectedRuntimeWarning(ctx: PlayContext): void {
  ctx.engineUi.deviceWarning?.('');
  const warning = ctx.dom.el('runtimeWarning');
  clearRuntimeFallbackWarning(warning);
  const option = ctxSelectedEngine(ctx);
  for (const detail of ctx.runtimeFallbackAudits.values()) {
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

function ctxShowEngineRetry(ctx: PlayContext, action: 'retry' | 'reload'): void {
  const button = ctx.dom.buttonEl('retryEngine');
  button.disabled = action === 'reload';
  button.textContent = action === 'reload' ? 'Reload required' : 'Retry engine';
  button.hidden = false;
}

function ctxSetEngineFailure(ctx: PlayContext, error: unknown): void {
  const reload = error instanceof EvaluatorImportError;
  ctxSetEngineNote(ctx, reload ? error.message : ctxActionableEngineError(ctx, error), true);
  ctxShowEngineRetry(ctx, reload ? 'reload' : 'retry');
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
  const container = ctx.dom.maybeEl('dlProgress');
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
    ctx.dom.el('progressAnnouncement').textContent = milestone === 4 ? `${label} search complete.` : `${label} search ${milestone * 25}% complete.`;
  }
}

/**
 * Re-label the visible search progress after a view change: the search always
 * runs on the live position, so it is tagged as such only while browsing.
 * Only touches a row ctxShowSearchProgress rendered (download progress and
 * hidden containers have no [data-search-progress] node).
 */
function ctxRefreshSearchProgressLabel(ctx: PlayContext): void {
  const text = ctx.dom.maybeEl('dlProgress')?.querySelector<HTMLElement>('[data-search-progress]');
  if (!text || !ctx.searchProgress) return;
  text.textContent = ctxSearchProgressText(searchProgressLabel(ctx.searchProgress.label, ctx.viewPly), ctx.searchProgress.progress);
}

async function ctxPreloadLqoAfterBookMove(ctx: PlayContext, signal: AbortSignal): Promise<boolean> {
  const searcher = await engines.acquireBigNet('lqo', signal, ctx.engineUi);
  if (searcher.loaded) return false;
  const label = `Lc0 ${searcher.config.name}`;
  ctxShowDownloadProgress(ctx, label, undefined, undefined, 'Loading in background');
  searcher.onDownloadProgress = (loaded, total) => {
    if (!signal.aborted && !ctxIsDisposed(ctx)) ctxShowDownloadProgress(ctx, label, loaded, total);
  };
  void searcher
    .init({ evalCacheEntries: 2048 })
    .then(() => {
      if (signal.aborted || ctxIsDisposed(ctx)) return;
      if (ctx.engineThinking) return;
      ctxHideDownloadProgress(ctx);
      ctxSetEngineNote(ctx, 'LQO model ready for post-book search.');
    })
    .catch((error: Error) => {
      if (signal.aborted || ctxIsDisposed(ctx)) return;
      if (ctx.engineThinking) return;
      ctxHideDownloadProgress(ctx);
      ctxSetEngineNote(ctx, `LQO background load failed: ${error.message}`, true);
    });
  return true;
}

function ctxPreloadLqoInBackground(ctx: PlayContext, signal: AbortSignal): boolean {
  if (peekBigNetSearcher('lqo')?.isInitializedFor({ evalCacheEntries: 2048 })) return false;
  // The book has already supplied the answer; reclaim and init need not hold the move.
  void Promise.resolve()
    .then(() => {
      if (!signal.aborted && !ctxIsDisposed(ctx)) return ctxPreloadLqoAfterBookMove(ctx, signal);
    })
    .catch((error: Error) => {
      if (!signal.aborted && !ctxIsDisposed(ctx) && !ctx.engineThinking) ctxSetEngineNote(ctx, `LQO background load failed: ${error.message}`, true);
    });
  return true;
}

async function ctxRequestEngineMove(ctx: PlayContext, signal: AbortSignal): Promise<string | null> {
  const option = ctxSelectedEngine(ctx);
  await engines.retireUnusedEngines(option);
  if (signal.aborted || ctx.disposed) return null;
  const preferences = ctx.opponent.active;
  const level = preferences.level;
  if (option.family === 'lc0' && option.variant === 'lqo') {
    const historyUcis = ctx.moves.map(moveToUci);
    if (ctx.humanColor === 'w') {
      const bookMove = lqoBlackBookMove(ctx.board, ctx.moves.length, historyUcis);
      if (bookMove) {
        const preloading = isMemoryConstrainedDevice() ? ctxPreloadLqoInBackground(ctx, signal) : await ctxPreloadLqoAfterBookMove(ctx, signal);
        ctxSetEngineNote(ctx, `LQO opening book: ${bookMove}${preloading ? ' · loading model in background' : ''}`);
        return bookMove;
      }
    } else if (ctx.moves.length === 0) {
      const searcher = await engines.acquireBigNet('lqo', signal, ctx.engineUi);
      let progressShown = false;
      if (!searcher.loaded) {
        ctxSetEngineNote(ctx, `Loading Lc0 ${searcher.config.name} (~${searcher.config.approxMb}MB on first use)…`);
        ctxShowDownloadProgress(ctx, `Lc0 ${searcher.config.name}`, undefined, undefined, 'Preparing');
        progressShown = true;
        searcher.onDownloadProgress = (loaded, total) => {
          if (!signal.aborted && !ctxIsDisposed(ctx)) ctxShowDownloadProgress(ctx, `Lc0 ${searcher.config.name}`, loaded, total);
        };
      }
      const evaluation = await (async () => {
        try {
          return await searcher.evaluate({ positions: ctx.positions }, { evalCacheEntries: 2048, signal });
        } finally {
          if (progressShown && !signal.aborted && !ctxIsDisposed(ctx)) ctxHideDownloadProgress(ctx);
        }
      })();
      if (signal.aborted) return null;
      const bookMove = lqoWhiteFirstPolicyBookMove(ctx.board, ctx.moves.length, historyUcis, evaluation.legalPriors, Math.random, 8);
      if (bookMove) {
        ctxSetEngineNote(ctx, `LQO policy book: ${bookMove}`);
        return bookMove;
      }
    } else {
      const bookMove = lqoWhiteBookMove(ctx.board, ctx.moves.length, historyUcis);
      if (bookMove) {
        const preloading = isMemoryConstrainedDevice() ? ctxPreloadLqoInBackground(ctx, signal) : await ctxPreloadLqoAfterBookMove(ctx, signal);
        ctxSetEngineNote(ctx, `LQO opening book: ${bookMove}${preloading ? ' · loading model in background' : ''}`);
        return bookMove;
      }
    }
  }
  if (option.family === 'maia3') {
    const choice = await engines.maia3Move(
      preferences.maiaElo,
      ctx.engineUi,
      ctx.maia3ModelUrls,
      signal,
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
    const evaluator = await engines.ensureCentipawn(ctx.engineUi, ctx.centipawnAssets);
    if (signal.aborted) return null;
    const visits = ctxStrengthFor(option, level);
    try {
      const result = await chooseMove(ctx.board, evaluator, {
        visits,
        // Same visit-argmax move as the full budget once the leader cannot be caught.
        // docs/play_early_stop_2026-10.md. Big-net Play searches stay fixed-visit.
        earlyStop: 'root-dominance',
        batchSize: Math.max(1, Math.min(32, visits)),
        signal,
        historyFens: searchHistoryFens(ctx.positions, ctx.positions.length),
        ...centipawnSearchOptions,
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
        ctxHideDownloadProgress(ctx);
        ctxSetEngineNote(ctx, '');
      }
    }
  }
  if (option.family === 'lc0' && option.variant === 'small') {
    const searcher = await engines.ensureLc0Small(ctx.lc0SmallModel, ctx.engineUi);
    try {
      const result = await searcher.search(
        { positions: ctx.positions },
        {
          visits: ctxStrengthFor(option, level),
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
      if (!signal.aborted && !ctxIsDisposed(ctx)) ctxHideDownloadProgress(ctx);
    }
  }
  if (option.family === 'lc0') {
    const searcher = await engines.acquireBigNet(option.variant as BigNetKey, signal, ctx.engineUi);
    if (!searcher.loaded && !signal.aborted && !ctxIsDisposed(ctx)) {
      ctxSetEngineNote(ctx, `Loading Lc0 ${searcher.config.name} (~${searcher.config.approxMb}MB on first use)…`);
      ctxShowDownloadProgress(ctx, `Lc0 ${searcher.config.name}`, undefined, undefined, 'Preparing');
      searcher.onDownloadProgress = (loaded, total) => {
        if (!signal.aborted && !ctxIsDisposed(ctx)) ctxShowDownloadProgress(ctx, `Lc0 ${searcher.config.name}`, loaded, total);
      };
    }
    try {
      const lqoOptions = option.variant === 'lqo' ? lqoSearchOptions(ctx.humanColor) : null;
      const result = await searcher.search(
        { positions: ctx.positions },
        {
          visits: (backend) => ctxStrengthFor(option, level, backend),
          reuseTree: true,
          batchSize: searcher.config.recommendedBatchSize,
          batchPipelineDepth: searcher.config.recommendedPipelineDepth,
          evalCacheEntries: 2048,
          drawScore: lqoOptions?.drawScore ?? PLAY_DRAW_SCORE,
          searchContemptLimit: lqoOptions?.searchContemptLimit ?? PLAY_SEARCH_CONTEMPT_LIMIT,
          signal,
          onSessionReady: () => {
            if (!ctxIsDisposed(ctx)) ctxRender(ctx);
          },
          onProgress: (progress) => {
            if (!signal.aborted) ctxShowSearchProgress(ctx, `Lc0 ${searcher.config.name} search`, progress);
          },
          ...(lqoOptions ?? {}),
        },
      );
      if (!signal.aborted && !ctxIsDisposed(ctx)) {
        ctxHideDownloadProgress(ctx);
        ctxSetEngineNote(ctx, '');
      }
      return result.cancelled ? null : (result.move ?? null);
    } finally {
      if (!signal.aborted && !ctxIsDisposed(ctx)) ctxHideDownloadProgress(ctx);
    }
  }
  const engine = await engines.cpuEngineForMove(option, signal, ctx.engineUi);
  const depth = ctxStrengthFor(option, level);
  if (option.family === 'sf') {
    engine.setOptions({ depth, movetimeMs: undefined, skillLevel: SF_LEVELS[level].skill });
  } else {
    engine.setOptions({ depth, movetimeMs: undefined });
  }
  // ctx.positions runs from the game's start to ctx.board; sending it as moves
  // lets the engine see repetitions and the 50-move clock.
  const history = uciGameHistory(ctx.positions);
  return engine.bestMove(history.fen, signal, { moves: history.moves });
}
async function ctxVerifySelectedEngineReady(ctx: PlayContext): Promise<void> {
  if (ctx.disposed || ctx.board.turn !== ctx.humanColor || ctx.restoredDownloadConsent) return;
  if (ctx.pickerConsent) ctxCancelPickerConsent(ctx);
  const resumingGame = ctx.moves.length > 0;
  const option = ctxSelectedEngine(ctx);
  const preferences = ctx.opponent.active;
  const signal = ctx.abort.signal;
  if (ctx.restoredDownloadEngineId && !(await ctxCheckRestoredDownloadConsent(ctx))) return;
  if (signal.aborted || ctx.disposed) return;
  const retirement = engines.retireUnusedEngines(option);
  const generation = ++ctx.enginePreparationGeneration;
  ctx.engineReady = false;
  ctx.preparingEngine = true;
  ctx.dom.buttonEl('retryEngine').hidden = true;
  ctxSetEngineNote(ctx, resumingGame ? `Checking ${option.label} before play resumes…` : `Preparing ${option.label} before the first move…`);
  ctxRender(ctx);
  try {
    await retirement;
    if (signal.aborted || ctx.disposed || generation !== ctx.enginePreparationGeneration) return;
    if (option.family === 'maia3') {
      await engines.maia3Move(
        preferences.maiaElo,
        ctx.engineUi,
        ctx.maia3ModelUrls,
        signal,
        { positions: ctx.positions },
        {
          selfElo: preferences.maiaElo,
          oppoElo: preferences.maiaElo,
          style: 'argmax',
        },
      );
    } else if (option.family === 'centipawn') {
      const evaluator = await engines.ensureCentipawn(ctx.engineUi, ctx.centipawnAssets);
      await evaluator.evaluate(ctx.board, {
        historyFens: searchHistoryFens(ctx.positions, ctx.positions.length),
        legalMoves: legalMoves(ctx.board),
      });
    } else if (option.family === 'lc0' && option.variant === 'small') {
      const searcher = await engines.ensureLc0Small(ctx.lc0SmallModel, ctx.engineUi);
      await searcher.search(
        { positions: ctx.positions },
        { visits: 1, signal, reuseTree: false, drawScore: PLAY_DRAW_SCORE, searchContemptLimit: PLAY_SEARCH_CONTEMPT_LIMIT },
      );
      if (signal.aborted || ctx.disposed || generation !== ctx.enginePreparationGeneration || option.id !== ctxSelectedEngine(ctx).id) return;
      searcher.resetTree();
    } else if (option.family === 'lc0') {
      const searcher = await engines.acquireBigNet(option.variant as BigNetKey, signal, ctx.engineUi);
      const lqoOptions = option.variant === 'lqo' ? lqoSearchOptions(ctx.humanColor) : null;
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
      const engine = await engines.cpuEngineFor(option, ctx.engineUi);
      engine.setOptions({ depth: 1, movetimeMs: undefined });
      const uci = await engine.bestMove(boardToFen(ctx.board), signal);
      if (!signal.aborted && (!uci || !legalMoves(ctx.board).some((move) => moveToUci(move) === uci))) {
        throw new Error(`${option.label} readiness check returned ${uci ? `illegal move ${uci}` : 'no move'}.`);
      }
    }
    if (signal.aborted || ctx.disposed || generation !== ctx.enginePreparationGeneration || option.id !== ctxSelectedEngine(ctx).id) return;
    ctx.engineReady = true;
    ctxSetEngineNote(ctx, resumingGame ? `${option.label} is ready. You can continue the game.` : `${option.label} is ready.`, false, true);
  } catch (error) {
    if (!signal.aborted && !ctx.disposed && generation === ctx.enginePreparationGeneration) ctxSetEngineFailure(ctx, error);
  } finally {
    if (!ctx.disposed && generation === ctx.enginePreparationGeneration) {
      ctx.preparingEngine = false;
      ctxHideDownloadProgress(ctx);
      ctxRender(ctx);
    }
  }
}

const THINKING_SHOW_DELAY_MS = 400;

function ctxSetEngineThinking(ctx: PlayContext, thinking: boolean): void {
  ctx.engineThinking = thinking;
  if (ctx.thinkingShowTimer) clearTimeout(ctx.thinkingShowTimer);
  ctx.thinkingShowTimer = null;
  ctx.thinkingShown = false;
  if (!thinking) return;
  ctx.thinkingShowTimer = setTimeout(() => {
    ctx.thinkingShowTimer = null;
    if (ctx.disposed || !ctx.engineThinking) return;
    ctx.thinkingShown = true;
    ctxRender(ctx);
  }, THINKING_SHOW_DELAY_MS);
}

function ctxCancelPickerConsent(ctx: PlayContext): void {
  ctx.pickerConsent?.controller.abort();
  ctx.pickerConsent = null;
}

function ctxCancelEngineTurn(ctx: PlayContext): void {
  ctxCancelPickerConsent(ctx);
  ctx.abort.abort();
  ctx.abort = new AbortController();
  ctx.restoredDownloadConsent = null;
  ctxSetEngineThinking(ctx, false);
  ctx.preparingEngine = false;
  ctx.enginePreparationGeneration += 1;
  ctxHideDownloadProgress(ctx);
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
    ctxSetEngineNote(ctx, '');
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
  if (ctx.gameOver || ctx.board.turn === ctx.humanColor || ctx.restoredDownloadConsent) return;
  if (ctx.pickerConsent) ctxCancelPickerConsent(ctx);
  const signal = ctx.abort.signal;
  if (ctx.restoredDownloadEngineId && !(await ctxCheckRestoredDownloadConsent(ctx))) return;
  if (signal.aborted || ctx.disposed) return;
  // Engine-first games skip readiness preparation, so retire the previous net
  // here too. Async pool acquisition awaits its bounded cleanup before loading.
  const option = ctxSelectedEngine(ctx);
  releaseUnusedBigNetSearchers(option.family === 'lc0' && option.variant !== 'small' ? [option.variant as BigNetKey] : [], 0);
  ctxSetEngineThinking(ctx, true);
  ctx.lastProgressAnnouncement = -1;
  ctx.dom.el('progressAnnouncement').textContent = '';
  const releaseWakeLock = await wakeLocks.acquire(signal);
  if (signal.aborted || ctx.disposed) {
    releaseWakeLock();
    return;
  }
  ctxRender(ctx);
  let uci: string | null = null;
  try {
    uci = await ctxRequestEngineMove(ctx, signal);
  } catch (error) {
    if (signal.aborted) return;
    ctxSetEngineThinking(ctx, false);
    ctx.engineReady = false;
    ctxSetEngineFailure(ctx, error);
    ctxRender(ctx);
    return;
  } finally {
    releaseWakeLock();
  }
  if (signal.aborted || ctx.disposed) return;
  ctx.engineReady = true;
  ctxSetEngineThinking(ctx, false);
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
  const input = ctx.dom.el('moveInput') as HTMLInputElement;
  const raw = input.value.trim();
  if (!raw) return;
  const fail = (message: string) => {
    input.classList.add('invalid');
    input.setCustomValidity(message);
    input.reportValidity();
    ctx.dom.el('boardAnnouncement').textContent = message;
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
  ctx.dom.el('boardAnnouncement').textContent = `You played ${moveToSan(ctx.board, move)}.`;
  ctxApplyHumanMove(ctx, move);
}

function ctxNewGame(ctx: PlayContext): void {
  ctxUpdatePendingInput(ctx, { type: 'discarded' });
  ctx.autosave?.clear();
  ctx.restoredGame = false;
  ctx.dom.el('playRestoreNotice').hidden = true;
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
  engines.startNewGame(ctx.opponent.active.engineId);
  ctxSetEngineNote(ctx, '');
  ctxDisarmResign(ctx);
  ctxDisarmNewGame(ctx);
  ctxRestoreSelectedRuntimeWarning(ctx);
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
    const btn = ctx.dom.buttonEl('resign');
    setButtonLabel(btn, 'Confirm resign');
    btn.classList.add('danger');
    if (ctx.resignArmTimer) ctx.scope.clearTimeout(ctx.resignArmTimer);
    ctx.resignArmTimer = ctx.scope.timeout(() => ctxDisarmResign(ctx), 4000);
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
    ctx.scope.clearTimeout(ctx.resignArmTimer);
    ctx.resignArmTimer = null;
  }
  const btn = ctx.dom.buttonEl('resign');
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
  if (ctx.newGameArmTimer) ctx.scope.clearTimeout(ctx.newGameArmTimer);
  ctx.newGameArmTimer = ctx.scope.timeout(() => {
    ctxDisarmNewGame(ctx);
    ctxRender(ctx);
  }, 4000);
  ctxRender(ctx);
}

function ctxDisarmNewGame(ctx: PlayContext): void {
  ctx.newGameArmed = false;
  if (ctx.newGameArmTimer) {
    ctx.scope.clearTimeout(ctx.newGameArmTimer);
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
  if (ctx.thinkingShown) return `${ctxOpponentName(ctxSelectedEngine(ctx))} is thinking…`;
  if (ctx.engineThinking) return `${ctxOpponentName(ctxSelectedEngine(ctx))} to move`;
  if (ctx.practicePosition && !ctx.started) return 'Choose your side and opponent, then press Play.';
  if (!ctx.engineReady) return `${ctxSelectedEngine(ctx).label} is not ready. Retry the engine or choose another opponent.`;
  if (ctxPhase(ctx) === 'setup') return 'Press Play or make a move.';
  if (boardCheck(ctx.board)) return ctx.board.turn === ctx.humanColor ? 'You are in check' : `${ctxOpponentName(ctxSelectedEngine(ctx))} is in check`;
  return ctx.board.turn === ctx.humanColor ? 'Your move' : `${ctxOpponentName(ctxSelectedEngine(ctx))} to move`;
}

function ctxRenderMoveList(ctx: PlayContext): void {
  const list = ctx.dom.el('moveList');
  const restoreFocus = rememberMoveFocus(list, 'data-ply');
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
    rows.push(
      `<div class="move-row"><span class="num">${row.number}.</span>${row.white === null ? '<span></span>' : san(row.white)}${row.black === null ? '<span></span>' : san(row.black)}</div>`,
    );
  }
  list.innerHTML = rows.join('');
  restoreFocus();
  if (ctx.viewPly === null) {
    ctx.moveListScrollKey = 'live';
    list.scrollTop = list.scrollHeight;
    return;
  }
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
  ctx.dom.buttonEl('navFirst').disabled = !available.first;
  ctx.dom.buttonEl('navPrev').disabled = !available.prev;
  ctx.dom.buttonEl('navNext').disabled = !available.next;
  const last = ctx.dom.buttonEl('navLast');
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
  if (ctx.pendingPromotion) ctx.pendingPromotion = null;
  // Chessground's set() keeps an active selection or drag; drop it so it does
  // not leak into the browsed position or back into live play.
  ctx.ground?.cancelMove();
  ctx.viewPly = next;
  ctxRender(ctx);
}

function ctxRenderRestartBanner(ctx: PlayContext): void {
  const banner = ctx.dom.el('restartBanner');
  if (!ctx.opponent.pending) {
    banner.hidden = true;
    return;
  }
  banner.hidden = false;
  const engineLabel = ENGINE_OPTIONS.find((o) => o.id === ctx.opponent.pending?.engineId)?.label ?? 'the new engine';
  const colorLabel = ctx.opponent.pending.color === 'random' ? 'random side' : `as ${ctx.opponent.pending.color}`;
  ctx.dom.el('restartMessage').textContent = `Start a new game vs ${engineLabel}, ${colorLabel}? The current game will be discarded.`;
}

function ctxRenderPromotionPicker(ctx: PlayContext): void {
  if (!ctx.pendingPromotion) {
    hidePromotionOverlay(ctx.dom.el('ground'));
    return;
  }
  showPromotionOverlay({
    boardContainer: ctx.dom.el('ground'),
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
  const key = option.variant as BigNetKey;
  return bigNetOptionState(BIG_NETS[key], peekBigNetSearcher(key)?.backend);
}

function ctxEngineOptionHtml(option: PlayEngineOption): string {
  const { disabled, suffix } = ctxEngineOptionState(option);
  return `<option value="${option.id}"${disabled ? ' disabled' : ''}>${option.label}${suffix}</option>`;
}

function ctxRefreshEngineOptions(ctx: PlayContext): void {
  const select = ctx.dom.selectEl('engineSelect');
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
  if (selectedOption && (!ctx.practicePosition || selectedOption.group !== 'odds') && (ctx.restoredGame || !ctxEngineOptionState(selectedOption).disabled))
    select.value = selected;
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
    ctx.scope.clearTimeout(ctx.pendingPreferredEngineTimer);
    ctx.pendingPreferredEngineTimer = null;
  }
  const restored = status === 'present';
  if (restored) ctx.dom.selectEl('engineSelect').value = option.id;
  ctxRenderLevelOptions(ctx);
  ctx.engineReady = false;
  ctxResetGameStateForMount(ctx);
  ctxPersistPlayPreferences(ctx);
  ctxRenderEngineCaution(ctx);
  if (!restored) ctxSetRuntimeWarning(ctx, 'Your saved engine asset is unavailable, so Play is using Maia instead.');
  ctxRender(ctx);
  ctxPrepareOrStartEngine(ctx);
}

function ctxRenderEngineCaution(ctx: PlayContext): void {
  const option = ctxDraftEngine(ctx);
  const caution = ctx.dom.el('engineCaution');
  if (option.family === 'lc0' && option.variant !== 'small') {
    const config = BIG_NETS[option.variant as BigNetKey];
    const memory = bigNetMemoryCaution(config);
    const odds = option.variant === 'lqo' ? ' The bot starts without its queen and uses the LeelaQueenOdds v2 network. ' : ' ';
    const cpu = bigNetCpuWarning(config, peekBigNetSearcher(config.key)?.backend);
    caution.textContent = `First move downloads the ~${config.approxMb}MB net.${odds}${memory ?? ''} ${cpu}`.trimEnd();
    caution.hidden = false;
  } else if (option.family === 'centipawn') {
    caution.textContent = 'First use downloads the 6.2MB Centipawn model and ONNX Runtime Web.';
    caution.hidden = false;
  } else {
    caution.hidden = true;
  }
}

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
  if (openingIndex || ctx.openingNamesRequested || !ctx.moves.length || ctx.startFen !== START_FEN) return;
  ctx.openingNamesRequested = true;
  // The index is shared by every mount; each mount re-renders itself when it lands.
  openingIndexPromise ??= fetch(resolvePublicAssetUrl('/openings/eco.json'))
    .then((response) => (response.ok ? response.json() : null))
    .then((index: unknown) => {
      if (index && typeof index === 'object' && !Array.isArray(index)) openingIndex = index as OpeningNameIndex;
    })
    .catch(() => undefined)
    .then(() => {
      // A failed fetch must not stick for the document: the next mount retries.
      if (!openingIndex) openingIndexPromise = null;
    });
  void openingIndexPromise.then(() => {
    if (!ctxIsDisposed(ctx) && openingIndex) ctxRender(ctx);
  });
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
  const option = ctxDraftEngine(ctx);
  if (option.family === 'maia3') {
    const elo = ctxSelectedMaia3Elo(ctx);
    return ctxSelectedMaia3Style(ctx) === 'argmax'
      ? `Always plays the most typical move of a ${elo}-rated human.`
      : `Plays like a ${elo}-rated human, mistakes included.`;
  }
  if (option.variant === 'lqo') return 'Starts without its queen, then plays tricky, aggressive chess to win it back.';
  if (option.family === 'sf') return 'Level 1 is beatable; level 5 is full-strength Stockfish.';
  return `A full-strength engine, strong even at level 1. For an even game, pick Maia3.`;
}

function playCtaLabel(ctx: PlayContext): string {
  const choice = ctx.dom.selectEl('colorSelect').value;
  if (choice === 'random') return 'Play · random colour';
  return `Play as ${choice === 'black' ? 'Black' : 'White'}`;
}

function ctxRenderStrip(ctx: PlayContext, id: string, color: Color, shownPly: number, phase: PlayPhase): void {
  const strip = ctx.dom.el(id);
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
  const material = playMaterial(ctx.positions, color, shownPly);
  const materialSlot = q('[data-material]');
  materialSlot.innerHTML = material.html;
  materialSlot.title = material.label;
  materialSlot.setAttribute('role', 'img');
  materialSlot.setAttribute('aria-label', material.label);
  const state = q('[data-state]');
  let text = '';
  let kind = '';
  if (!isHuman && ctx.thinkingShown) [text, kind] = ['Thinking…', 'thinking'];
  else if (!isHuman && ctx.engineThinking) [text, kind] = ['', ''];
  else if (!isHuman && ctx.preparingEngine) [text, kind] = ['Loading…', 'thinking'];
  else if (!isHuman && ctx.practicePosition && !ctx.started && !ctx.engineReady) [text, kind] = ['Loads on Play', 'ready'];
  else if (!isHuman && !ctx.engineReady && !ctx.gameOver) {
    const retry = ctx.dom.buttonEl('retryEngine');
    [text, kind] = retry.hidden ? ['Loading…', 'thinking'] : [`Error · ${retry.disabled ? 'reload' : 'retry'}`, 'error'];
  } else if (!isHuman && phase === 'setup') [text, kind] = ['Ready', 'ready'];
  else if (isHuman && phase === 'playing' && ctx.viewPly === null && ctx.board.turn === color && ctx.engineReady && !ctx.pendingPromotion)
    [text, kind] = ['Your move', 'turn'];
  state.textContent = text;
  state.dataset.kind = kind;
  strip.dataset.turn = String(!ctx.gameOver && ctx.viewPly === null && phase !== 'setup' && ctx.board.turn === color);
  strip.setAttribute('aria-label', `${isHuman ? 'You' : ctxOpponentName(option)}, ${side}${text ? `, ${text.replace('…', '')}` : ''}`);
}

function ctxRenderChrome(ctx: PlayContext, shownPly: number): void {
  const phase = ctxPhase(ctx);
  const main = ctx.dom.el('main');
  main.dataset.phase = phase;
  main.dataset.opponentState = ctx.opponent.tag;
  main.dataset.opponent = ctx.opponent.active.engineId;
  main.dataset.pendingOpponent = ctx.opponent.pending?.engineId ?? '';
  const option = ctxSelectedEngine(ctx);
  const key = option.variant as BigNetKey;
  const bigNetWarning = ctx.dom.el('bigNetCpuWarning');
  renderBigNetCpuWarning(bigNetWarning, [
    option.family === 'lc0' && option.variant !== 'small' ? bigNetCpuWarning(BIG_NETS[key], peekBigNetSearcher(key)?.backend) : '',
  ]);
  const hasNotice =
    !bigNetWarning.hidden ||
    !ctx.dom.buttonEl('retryEngine').hidden ||
    !ctx.dom.el('runtimeWarning').hidden ||
    !ctx.dom.el('deviceLossWarning').hidden ||
    !ctx.dom.el('dlProgress').hidden;
  main.dataset.quiet = String(
    !hasNotice && ctx.engineReady && !ctx.preparingEngine && !ctx.thinkingShown && !ctx.pendingPromotion && ctx.viewPly === null && !boardCheck(ctx.board),
  );
  main.dataset.config = ctx.configOpen && phase !== 'setup' ? 'open' : 'closed';
  const opponent = ctxOpponentName(option);
  const topColor: Color = ctx.orientation === 'white' ? 'b' : 'w';
  ctxRenderStrip(ctx, 'topStrip', topColor, shownPly, phase);
  ctxRenderStrip(ctx, 'bottomStrip', topColor === 'w' ? 'b' : 'w', shownPly, phase);

  // Setup controls mirror the hidden selects so preferences and listeners stay single-sourced.
  const colorChoice = ctx.dom.selectEl('colorSelect').value;
  for (const button of ctx.dom.queryAll<HTMLButtonElement>('.color-choice [data-color]')) {
    button.setAttribute('aria-pressed', String(button.dataset.color === colorChoice));
  }
  const elo = ctxSelectedMaia3Elo(ctx);
  ctx.dom.el('maia3EloBand').textContent = `· ${maiaRatingBand(elo)}`;
  for (const chip of ctx.dom.queryAll<HTMLButtonElement>('.rating-presets [data-elo]')) {
    chip.setAttribute('aria-pressed', String(Number(chip.dataset.elo) === elo));
  }
  ctx.dom.el('opponentBlurb').textContent = ctxOpponentBlurb(ctx);

  const change = ctx.dom.buttonEl('changeOpponent');
  change.setAttribute('aria-expanded', String(ctx.configOpen));
  change.textContent = ctx.configOpen ? 'Done' : 'Change opponent';
  change.title = ctx.configOpen ? '' : 'Change opponent (asks before restarting the game)';
  const detailsLabel = ctx.dom.maybeEl('detailsLabel');
  if (detailsLabel) detailsLabel.textContent = option.family === 'maia3' ? 'Exact rating & opponent details' : 'Opponent details';

  const newGame = ctx.dom.buttonEl('newGame');
  newGame.classList.toggle('primary', phase === 'setup');
  newGame.classList.toggle('armed', ctx.newGameArmed);
  setButtonLabel(newGame, phase === 'setup' ? playCtaLabel(ctx) : ctx.newGameArmed ? 'Confirm new game' : 'New game');
  newGame.title = phase === 'playing' && !ctx.newGameArmed ? 'Abandon this game and start another (asks to confirm)' : '';

  const card = ctx.dom.el('gameOverCard');
  card.hidden = !ctx.gameOver;
  if (ctx.gameOver) {
    const { result, reason } = ctx.gameOver;
    const humanWon = result !== '1/2-1/2' && (result === '1-0') === (ctx.humanColor === 'w');
    ctx.dom.el('resultHeadline').textContent = result === '1/2-1/2' ? 'Draw' : humanWon ? 'You won' : `${opponent} won`;
    ctx.dom.el('resultScore').textContent = result === '1/2-1/2' ? '½–½' : result.replace('-', '–');
    ctx.dom.el('resultReason').textContent =
      `${reason.charAt(0).toUpperCase()}${reason.slice(1)} · ${ctx.sans.length ? `${Math.ceil(ctx.sans.length / 2)} moves` : 'no moves'}`;
  }

  const hasRecord = ctx.moves.length > 0 || !!ctx.practicePosition;
  ctx.dom.buttonEl('copyPgn').disabled = !hasRecord;
  ctx.dom.buttonEl('downloadPgn').disabled = !hasRecord;
  for (const link of ctx.dom.queryAll<HTMLAnchorElement>('[data-analyze]')) {
    link.setAttribute('aria-disabled', String(!hasRecord));
    if (phase === 'playing') {
      link.target = '_blank';
      link.rel = 'noopener';
    } else {
      link.removeAttribute('target');
      link.removeAttribute('rel');
    }
  }

  ctxEnsureOpeningNames(ctx);
  const opening = ctx.dom.el('openingName');
  const name = ctx.moves.length ? ctxOpeningAt(ctx, shownPly) : null;
  opening.hidden = !name;
  opening.innerHTML = name ? `<strong>${htmlEscape(name[0])}</strong>${htmlEscape(name[1])}` : '';
  opening.title = name ? `${name[0]} ${name[1]} · Lichess opening names (CC0)` : '';
}

function ctxRender(ctx: PlayContext): void {
  if (ctx.disposed) return;
  ctx.autosave?.schedule(
    ctx.started || ctx.moves.length || ctx.gameOver || ctx.practicePosition
      ? {
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
        }
      : null,
  );
  ctx.viewPly = normalizeViewPly(ctx.viewPly, ctx.moves.length);
  const browsing = ctx.viewPly !== null;
  const statusText = ctxStatusText(ctx);
  const setupReady = ctxPhase(ctx) === 'setup' && ctx.engineReady && !ctx.engineThinking && !ctx.preparingEngine && !browsing;
  ctx.dom.el('status').textContent = setupReady ? `New game vs ${ctxOpponentName(ctxSelectedEngine(ctx))}` : statusText;
  // The one live region: turn and readiness changes are announced here and nowhere else.
  const announcement = ctx.dom.el('gameAnnouncement');
  const spokenStatus = setupReady ? `${ctxOpponentName(ctxSelectedEngine(ctx))} is ready. ${statusText}` : statusText;
  if (announcement.textContent !== spokenStatus) announcement.textContent = spokenStatus;
  ctx.dom.el('status').classList.toggle('over', !!ctx.gameOver);
  ctx.dom.el('status').classList.toggle('browsing', browsing);
  ctxRefreshSearchProgressLabel(ctx);
  ctx.dom.el('status').setAttribute('aria-busy', ctx.engineThinking ? 'true' : 'false');
  ctxRenderMaia3Controls(ctx);
  ctx.dom.el('levelCaption').textContent = ctxStrengthCaption(ctx);
  ctx.dom.buttonEl('takeback').disabled = ctx.moves.length <= ctx.practiceBoundary || !!ctx.pendingPromotion;
  ctx.dom.buttonEl('takeback').title = ctx.practicePosition ? 'Take back to your previous turn, no earlier than the imported position' : '';
  ctx.dom.buttonEl('resign').disabled = !!ctx.gameOver || (!ctx.moves.length && !ctx.started);
  ctx.dom.selectEl('engineSelect').disabled = ctx.engineThinking;
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
  if (!ctx.ground) ctx.ground = Chessground(ctx.dom.el('ground'), config);
  else ctx.ground.set(config);
  // Screen readers and keyboard users get the square grid; the drawn board is decorative to them.
  ctx.dom.el('ground').querySelector('cg-board')?.setAttribute('aria-hidden', 'true');
  ctx.keyboardBoard ??= createAnalysisKeyboardBoard(ctx.dom.el('keyboardBoard'), {
    onSelect: (key) => ctx.ground?.selectSquare(key),
    onMove: (from, to) => ctxOnUserMove(ctx, from, to),
    announce: (text) => {
      ctx.dom.el('boardAnnouncement').textContent = text;
    },
  });
  ctx.keyboardBoard.render(shownBoard, ctx.orientation);
  ctx.dom.el('positionNotice').hidden = !ctx.practicePosition && !ctx.handoffNotice;
  ctx.dom.el('positionNoticeText').textContent = ctx.practicePosition
    ? 'Continue this position. Queen-odds opponents are unavailable because they need their own starting position.'
    : ctx.handoffNotice;
  ctx.dom.buttonEl('useStartingPosition').hidden = !ctx.practicePosition || (ctx.started && !ctx.gameOver);
  ctxRenderChrome(ctx, shownPly);
}

function ctxSetOpponent(ctx: PlayContext, next: PlayOpponentState): void {
  const previousEngineId = ctx.opponent.active.engineId;
  ctx.opponent = next;
  // A draft approval is not permission to download the still-live opponent.
  // Only a committed replacement retires that live model's outstanding gate.
  if (next.active.engineId !== previousEngineId && ctx.restoredDownloadEngineId === previousEngineId) ctx.restoredDownloadEngineId = null;
  // `active` changes only on commit; a pending restart still belongs to the current game.
  engines.opponentCommitted(previousEngineId, next.active.engineId);
}

function ctxRestoreOpponentControls(ctx: PlayContext): void {
  const preferences = ctx.opponent.pending ?? ctx.opponent.active;
  ctx.dom.selectEl('engineSelect').value = preferences.engineId;
  ctx.dom.selectEl('colorSelect').value = preferences.color;
  ctx.lastEngineId = preferences.engineId;
  ctxRenderLevelOptions(ctx);
  ctx.dom.selectEl('levelSelect').value = String(preferences.level);
  (ctx.dom.el('maia3Elo') as HTMLInputElement).value = String(preferences.maiaElo);
  ctx.dom.selectEl('maia3Style').value = preferences.maiaStyle;
  (ctx.dom.el('maia3Temperature') as HTMLInputElement).value = String(preferences.maiaTemperature);
  (ctx.dom.el('maia3TopP') as HTMLInputElement).value = String(preferences.maiaTopP);
  ctxRenderMaia3Controls(ctx);
  ctxRenderEngineCaution(ctx);
}

function ctxMaybeQueueRestart(ctx: PlayContext): void {
  const gameInProgress = !ctxGameIsFresh(ctx) && !ctx.gameOver;
  ctxSetOpponent(
    ctx,
    transitionPlayOpponent(ctx.opponent, {
      type: 'edit',
      preferences: ctxCurrentPlayPreferences(ctx),
      gameInProgress,
    }),
  );
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

function ctxUpdatePendingInput(ctx: PlayContext, event: PendingInputEvent): void {
  const { pending, acknowledged } = transitionPendingInput(ctx.pendingInput, event);
  ctx.pendingInput = pending;
  for (const transfer of acknowledged?.transfers ?? []) acknowledgeInputTransfer(transfer);
}

/** Import before any autosaved-game restore; callers must skip restore when this returns true. */
function ctxApplyPlayHandoffForMount(ctx: PlayContext): boolean {
  const handoff = takePlayHandoff(Date.now(), (_consume, transfer) => {
    ctxUpdatePendingInput(ctx, { type: 'applied', input: { urlKeys: [], transfers: [transfer] } });
  });
  if (handoff.status === 'ready') {
    ctx.practicePosition = handoff.position;
    return true;
  }
  if (handoff.status === 'invalid') {
    ctx.handoffNotice = 'That position could not be loaded or is finished. Set up a normal game instead.';
    return false;
  }
  return false;
}

function ctxResetStartPosition(ctx: PlayContext): void {
  // Engines stay resident across games and page mounts; clear their hash
  // before their next move in this game.
  engines.markCpuNewGame();
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

/** Restored preferences are not download approval: Cache Storage may have been
 * evicted since the previous visit. Both readiness and engine-first moves enter
 * here before any acquisition; one owner shares the prompt during this mount.
 */
async function ctxCheckRestoredDownloadConsent(ctx: PlayContext): Promise<boolean> {
  const option = ctxSelectedEngine(ctx);
  if (ctx.restoredDownloadEngineId !== option.id) return true;
  if (ctx.restoredDownloadConsent) return ctx.restoredDownloadConsent;
  const signal = AbortSignal.any([ctx.abort.signal, ctx.consentAbort.signal]);
  const pending = requestLargeNetConsent(BIG_NETS[option.variant as BigNetKey], signal, false)
    .then((accepted) => {
      if (signal.aborted || ctx.disposed || ctx.opponent.active.engineId !== option.id) return false;
      if (accepted) ctx.restoredDownloadEngineId = null;
      else {
        ctx.engineReady = false;
        ctxSetEngineNote(ctx, 'Download cancelled. Use Retry when you want to load this opponent.');
        ctxShowEngineRetry(ctx, 'retry');
        ctxRender(ctx);
      }
      return accepted;
    })
    .catch((error) => {
      if (!signal.aborted && !ctx.disposed) {
        ctxSetEngineFailure(ctx, error);
        ctxRender(ctx);
      }
      return false;
    })
    .finally(() => {
      if (ctx.restoredDownloadConsent === pending) ctx.restoredDownloadConsent = null;
    });
  ctx.restoredDownloadConsent = pending;
  return pending;
}

function ctxPrepareOrStartEngine(ctx: PlayContext): void {
  ctxCancelPickerConsent(ctx);
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
  ctxSetEngineNote(ctx, '');
  ctxDisarmResign(ctx);
}

function ctxRevokePhoneLargeNets(ctx: PlayContext): void {
  if (!phoneLargeNetsBlocked()) return;
  if (ctx.pickerConsent) ctxCancelPickerConsent(ctx);
  const pending = ENGINE_OPTIONS.find((option) => option.id === ctx.pendingPreferredEngineId);
  const cancellingRestore = !!pending && phoneSafeEngineVariant(pending.family, pending.variant) !== pending.variant;
  if (cancellingRestore) ctx.pendingPreferredEngineId = null;
  if (cancellingRestore && ctx.pendingPreferredEngineTimer) {
    ctx.scope.clearTimeout(ctx.pendingPreferredEngineTimer);
    ctx.pendingPreferredEngineTimer = null;
  }
  const active = ctxSelectedEngine(ctx);
  const draft = ctxDraftEngine(ctx);
  // An earlier game's gated net may still be retained behind an unaffected opponent.
  // Queen Odds is exempt and may own live work, so never evict it on revocation.
  releaseUnusedBigNetSearchers(['lqo'], 0);
  if (cancellingRestore || phoneSafeEngineVariant(active.family, active.variant) !== active.variant) {
    ctxCancelEngineTurn(ctx);
    ctxSetEngineNote(ctx, '');
    const preferences = { ...ctx.opponent.active, engineId: 'lc0-small' };
    ctxSetOpponent(ctx, committedPlayOpponent(preferences));
    ctxRestoreOpponentControls(ctx);
    ctx.engineReady = false;
    showPhoneLargeNetFallback();
    ctxPersistPlayPreferences(ctx);
    ctxRender(ctx);
    if (!ctx.gameOver) ctxPrepareOrStartEngine(ctx);
  } else if (phoneSafeEngineVariant(draft.family, draft.variant) !== draft.variant) {
    // Controls are only a restart proposal; the unaffected live opponent owns its game.
    ctxDismissRestart(ctx);
  }
  ctxRefreshEngineOptions(ctx);
}

function ctxInit(ctx: PlayContext): void {
  if (ctx.blockedModelOverride) showPhoneLargeNetFallback(true);
  // Only a decoded storage position handoff replaces a saved game. URL options never contain a Play game.
  const handedOff = ctxApplyPlayHandoffForMount(ctx);
  const validEngineIds = new Set(ENGINE_OPTIONS.map((option) => option.id));
  const saved = handedOff ? null : readWorkspace('play', (raw) => decodePlaySnapshot(raw, validEngineIds));
  ctx.restoredGame = !!saved;
  if (saved) ctx.practicePosition = saved.practicePosition;
  ctxRefreshEngineOptions(ctx);
  let preferences = saved?.preferences ?? loadPlayPreferences(playPreferenceStorage(), validEngineIds);
  const requestedOption = ENGINE_OPTIONS.find((option) => option.id === preferences.engineId);
  if (requestedOption && phoneSafeEngineVariant(requestedOption.family, requestedOption.variant) !== requestedOption.variant) {
    preferences = { ...preferences, engineId: 'lc0-small' };
    showPhoneLargeNetFallback();
  }
  const preferredOption = ENGINE_OPTIONS.find((option) => option.id === preferences.engineId && (!ctx.practicePosition || option.group !== 'odds'));
  ctx.pendingPreferredEngineId =
    !saved &&
    preferredOption?.family === 'lc0' &&
    preferredOption.variant !== 'small' &&
    bigNetAssetStatusSync(BIG_NETS[preferredOption.variant as BigNetKey]) === 'unknown'
      ? preferredOption.id
      : null;
  ctx.dom.selectEl('engineSelect').value = preferredOption && (saved || !ctxEngineOptionState(preferredOption).disabled) ? preferredOption.id : 'maia3';
  ctx.dom.selectEl('colorSelect').value =
    ctx.practicePosition && !saved ? (replayPlayPosition(ctx.practicePosition).board.turn === 'w' ? 'white' : 'black') : preferences.color;
  (ctx.dom.el('maia3Elo') as HTMLInputElement).value = String(preferences.maiaElo);
  ctx.dom.selectEl('maia3Style').value = preferences.maiaStyle;
  (ctx.dom.el('maia3Temperature') as HTMLInputElement).value = String(preferences.maiaTemperature);
  (ctx.dom.el('maia3TopP') as HTMLInputElement).value = String(preferences.maiaTopP);
  ctxRenderLevelOptions(ctx);
  ctx.dom.selectEl('levelSelect').value = String(preferences.level);
  ctxResetGameStateForMount(ctx);
  if (saved) {
    const { preferences: _preferences, ...game } = saved;
    Object.assign(ctx, game);
    ctx.dom.el('playRestoreNotice').hidden = false;
  }
  ctx.restoredDownloadEngineId = preferredOption?.family === 'lc0' && preferredOption.variant !== 'small' ? preferredOption.id : null;
  ctx.autosave = createWorkspaceAutosave('play', undefined, 400, undefined, undefined, (type) => ctxUpdatePendingInput(ctx, { type }));
  ctx.scope.listen(window, 'pagehide', () => ctx.autosave?.flush());
  ctx.scope.listen(window, 'pageshow', (event: Event) => {
    if (!(event as PageTransitionEvent).persisted) return;
    ctxDisarmResign(ctx);
    ctxDisarmNewGame(ctx);
    ctxRender(ctx);
  });
  ctx.scope.listen(document, 'visibilitychange', () => {
    if (document.hidden) ctx.autosave?.flush();
  });
  ctx.scope.listen(ctx.dom.el('playRestoreDismiss'), 'click', () => {
    ctx.dom.el('playRestoreNotice').hidden = true;
  });
  ctx.scope.listen(ctx.dom.el('playRestoreNew'), 'click', () => {
    ctx.started = false;
    ctx.configOpen = false;
    ctxNewGame(ctx);
  });

  ctx.scope.listen(ctx.dom.el('newGame'), 'click', () => {
    const resolvingSavedEngine = !!ctx.pendingPreferredEngineId;
    ctx.pendingPreferredEngineId = null;
    if (ctx.pendingPreferredEngineTimer) {
      ctx.scope.clearTimeout(ctx.pendingPreferredEngineTimer);
      ctx.pendingPreferredEngineTimer = null;
    }
    const phase = ctxPhase(ctx);
    if (phase === 'setup' && !resolvingSavedEngine) {
      ctx.started = true;
      ctx.configOpen = false;
      if (!ctx.preparingEngine && !ctx.engineThinking) ctxPrepareOrStartEngine(ctx);
      ctxRender(ctx);
      if (matchMedia('(max-width: 900px)').matches) ctx.dom.query('.board-panel')?.scrollIntoView({ block: 'start' });
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
  ctx.scope.listen(ctx.dom.el('useStartingPosition'), 'click', () => {
    ctxUpdatePendingInput(ctx, { type: 'discarded' });
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
  ctx.scope.listen(ctx.dom.el('changeOpponent'), 'click', () => {
    ctx.configOpen = !ctx.configOpen;
    ctxRender(ctx);
    if (ctx.configOpen) ctx.dom.selectEl('engineSelect').focus();
  });
  ctx.scope.listen(ctx.dom.query('.color-choice') ?? ctx.dom.root, 'click', (event: Event) => {
    const button = (event.target as HTMLElement | null)?.closest?.<HTMLButtonElement>('[data-color]');
    const select = ctx.dom.selectEl('colorSelect');
    if (!button?.dataset.color || button.dataset.color === select.value) return;
    select.value = button.dataset.color;
    select.dispatchEvent(new Event('change', { bubbles: true }));
  });
  ctx.scope.listen(ctx.dom.query('.rating-presets') ?? ctx.dom.root, 'click', (event: Event) => {
    const chip = (event.target as HTMLElement | null)?.closest?.<HTMLButtonElement>('[data-elo]');
    const input = ctx.dom.el('maia3Elo') as HTMLInputElement;
    if (!chip?.dataset.elo) return;
    input.value = chip.dataset.elo;
    input.dispatchEvent(new Event('input', { bubbles: true }));
  });
  ctx.scope.listen(ctx.dom.el('moveForm'), 'submit', (event: Event) => {
    event.preventDefault();
    ctxSubmitTypedMove(ctx);
  });
  ctx.scope.listen(ctx.dom.el('moveInput'), 'input', () => {
    const input = ctx.dom.el('moveInput') as HTMLInputElement;
    input.setCustomValidity('');
    input.classList.remove('invalid');
  });
  for (const link of ctx.dom.queryAll<HTMLAnchorElement>('[data-analyze]')) {
    ctx.scope.listen(link, 'click', (event: Event) => {
      if (!ctx.moves.length && !ctx.practicePosition) {
        event.preventDefault();
        return;
      }
      writeAnalysisHandoff(ctxExportPgn(ctx));
    });
  }
  ctx.scope.listen(ctx.dom.el('downloadPgn'), 'click', () => {
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
  ctx.scope.listen(ctx.dom.el('retryEngine'), 'click', () => {
    if (ctx.engineThinking || ctx.preparingEngine || ctx.gameOver) return;
    ctx.dom.buttonEl('retryEngine').hidden = true;
    ctxSetEngineNote(ctx, 'Retrying engine…');
    if (ctx.board.turn === ctx.humanColor) void ctxVerifySelectedEngineReady(ctx);
    else void ctxEngineTurn(ctx);
  });
  ctx.scope.listen(window, BROWSER_RUNTIME_AUDIT_EVENT, (event: Event) => {
    const detail = (event as CustomEvent<BrowserRuntimeAuditDetail>).detail;
    if (detail.surface && detail.surface !== 'play') return;
    const identity = browserRuntimeAuditIdentity(detail);
    if (detail.fallbackReason) ctx.runtimeFallbackAudits.set(identity, detail);
    else ctx.runtimeFallbackAudits.delete(identity);
    if (playOptionMatchesRuntimeAudit(ctxSelectedEngine(ctx), detail)) ctxRestoreSelectedRuntimeWarning(ctx);
  });
  ctx.scope.listen(ctx.dom.el('takeback'), 'click', () => ctxTakeback(ctx));
  ctx.scope.listen(ctx.dom.el('navFirst'), 'click', () => ctxNavigate(ctx, 'first'));
  ctx.scope.listen(ctx.dom.el('navPrev'), 'click', () => ctxNavigate(ctx, 'prev'));
  ctx.scope.listen(ctx.dom.el('navNext'), 'click', () => ctxNavigate(ctx, 'next'));
  ctx.scope.listen(ctx.dom.el('navLast'), 'click', () => ctxNavigate(ctx, 'last'));
  ctx.scope.listen(ctx.dom.el('moveList'), 'click', (event: Event) => {
    const span = (event.target as HTMLElement | null)?.closest?.('.san[data-ply]') as HTMLElement | null;
    if (!span) return;
    ctxSetViewPly(ctx, viewPlyFor(Number(span.dataset.ply), ctx.moves.length));
  });
  ctx.scope.listen(document, 'keydown', (event: Event) => {
    const keyEvent = event as KeyboardEvent;
    // The promotion overlay owns the arrow keys (and Escape) while it is open.
    if (keyEvent.defaultPrevented || ctx.pendingPromotion) return;
    const action = moveNavActionForKey(keyEvent);
    if (!action) return;
    keyEvent.preventDefault();
    ctxNavigate(ctx, action);
  });
  ctx.scope.listen(ctx.dom.el('resign'), 'click', () => ctxResign(ctx));
  ctx.scope.listen(ctx.dom.el('flip'), 'click', () => {
    ctx.orientation = ctx.orientation === 'white' ? 'black' : 'white';
    ctxRender(ctx);
  });
  ctx.scope.listen(ctx.dom.el('copyPgn'), 'click', () => {
    void writeClipboard(ctxExportPgn(ctx)).then((copied) => {
      if (ctx.disposed) return;
      const btn = ctx.dom.buttonEl('copyPgn');
      setButtonLabel(btn, copied ? 'Copied' : 'Copy failed');
      btn.dataset.copied = String(copied);
      if (!copied) ctx.dom.el('gameAnnouncement').textContent = 'Could not copy PGN to the clipboard. Use Download PGN to keep the game.';
      ctx.scope.timeout(() => {
        setButtonLabel(btn, 'Copy PGN');
        btn.dataset.copied = 'false';
      }, 1500);
    });
  });
  ctx.scope.listen(ctx.dom.el('confirmRestart'), 'click', () => ctxConfirmRestart(ctx));
  ctx.scope.listen(ctx.dom.el('dismissRestart'), 'click', () => ctxDismissRestart(ctx));
  ctx.scope.listen(ctx.dom.selectEl('colorSelect'), 'change', () => ctxMaybeQueueRestart(ctx));
  ctx.scope.listen(window, PHONE_LARGE_NET_CHANGE_EVENT, () => ctxRevokePhoneLargeNets(ctx));
  ctx.scope.listen(ctx.dom.selectEl('engineSelect'), 'change', async () => {
    const resumeFallback = !!ctx.pendingPreferredEngineId || !!ctx.pickerConsent?.resumeFallback;
    ctxCancelPickerConsent(ctx);
    const request = { controller: new AbortController(), resumeFallback };
    ctx.pickerConsent = request;
    const signal = AbortSignal.any([ctx.consentAbort.signal, ctx.abort.signal, request.controller.signal]);
    ctx.pendingPreferredEngineId = null;
    if (ctx.pendingPreferredEngineTimer) {
      ctx.scope.clearTimeout(ctx.pendingPreferredEngineTimer);
      ctx.pendingPreferredEngineTimer = null;
    }
    const previousEngineId = ctx.lastEngineId;
    const selectionGame = ctx.abort.signal;
    const resumeRevertedOpponent = () => {
      if (
        request.resumeFallback &&
        !ctx.disposed &&
        !selectionGame.aborted &&
        !signal.aborted &&
        ctx.pickerConsent === request &&
        !ctx.engineThinking &&
        !ctx.preparingEngine &&
        !ctx.gameOver &&
        ctx.lastEngineId === previousEngineId
      ) {
        ctxSetEngineNote(ctx, '');
        ctxPrepareOrStartEngine(ctx);
      }
    };
    if (ctx.engineThinking) {
      ctx.dom.selectEl('engineSelect').value = previousEngineId;
      ctxRender(ctx);
      return;
    }
    const option = ctxDraftEngine(ctx);
    if (option.family === 'lc0' && option.variant !== 'small') {
      ctx.dom.selectEl('engineSelect').value = previousEngineId;
      if (!(await requestLargeNetConsent(BIG_NETS[option.variant as BigNetKey], signal, false))) {
        resumeRevertedOpponent();
        return;
      }
      if (ctx.disposed || signal.aborted || ctx.pickerConsent !== request || ctx.engineThinking || ctx.lastEngineId !== previousEngineId) return;
      if (ctxEngineOptionState(option).disabled) {
        ctxRefreshEngineOptions(ctx);
        resumeRevertedOpponent();
        return;
      }
      ctxRefreshEngineOptions(ctx);
      ctx.dom.selectEl('engineSelect').value = option.id;
      if (ctx.restoredDownloadEngineId === option.id) ctx.restoredDownloadEngineId = null;
    }
    if (option.family === 'maia3' && previousEngineId.startsWith('maia-')) {
      const carried = Number(ctx.lastEngineId.slice('maia-'.length));
      if (Number.isFinite(carried)) {
        (ctx.dom.el('maia3Elo') as HTMLInputElement).value = String(Math.max(MAIA3_MIN_ELO, Math.min(MAIA3_MAX_ELO, carried)));
      }
    }
    ctx.lastEngineId = option.id;
    ctxRenderLevelOptions(ctx);
    ctxMaybeQueueRestart(ctx);
  });
  ctx.scope.listen(ctx.dom.selectEl('levelSelect'), 'change', () => ctxTuneOpponent(ctx));
  ctx.scope.listen(ctx.dom.selectEl('maia3Style'), 'change', () => ctxTuneOpponent(ctx));
  ctx.scope.listen(ctx.dom.el('maia3Elo'), 'input', () => ctxTuneOpponent(ctx));
  ctx.scope.listen(ctx.dom.el('maia3Temperature'), 'input', () => ctxTuneOpponent(ctx));
  ctx.scope.listen(ctx.dom.el('maia3TopP'), 'input', () => ctxTuneOpponent(ctx));
  const refreshIfMounted = () => {
    if (!ctxIsDisposed(ctx)) {
      ctxRefreshEngineOptions(ctx);
      ctxResolvePendingPreferredEngine(ctx);
    }
  };
  void probeBt4Support().then(refreshIfMounted);
  for (const option of ENGINE_OPTIONS) {
    if (option.family === 'lc0' && option.variant !== 'small') void checkBigNetAsset(BIG_NETS[option.variant as BigNetKey]).then(refreshIfMounted);
  }
  ctxRender(ctx);
  if (ctx.pendingPreferredEngineId) {
    ctxSetEngineNote(ctx, 'Checking your saved engine assets…');
    ctx.pendingPreferredEngineTimer = ctx.scope.timeout(() => {
      if (ctx.disposed || !ctx.pendingPreferredEngineId) return;
      ctx.pendingPreferredEngineId = null;
      ctx.pendingPreferredEngineTimer = null;
      ctxPersistPlayPreferences(ctx);
      ctx.engineReady = false;
      ctxSetRuntimeWarning(ctx, 'The saved-engine check timed out, so Play is using Maia. You can retry the engine from the selector.');
      ctxRender(ctx);
      ctxPrepareOrStartEngine(ctx);
    }, 10_000);
  } else if (!ctx.gameOver) ctxPrepareOrStartEngine(ctx);
}

function hasPlayBrowserDom(root: HTMLElement): boolean {
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
    'deviceLossWarning',
    'bigNetCpuWarning',
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
  const missing = required.filter((id) => root.id !== id && !root.querySelector(`[id=${JSON.stringify(id)}]`));
  if (!missing.length) return true;
  console.warn(`Play browser mount skipped; missing required DOM element(s): ${missing.map((id) => `#${id}`).join(', ')}`);
  return false;
}

/** Mounts Play on `scope.root`. Engines are not mount-scoped: they live in the module-level host and outlive this mount. */
export function mountPlayBrowser(scope: MountScope): void {
  if (!hasPlayBrowserDom(scope.root)) return;
  if (reloadForChangedOverrides(scope.url)) return;
  // Decided at mount: by teardown the URL may already show the next route.
  const retireCpuOnUnmount = isMemoryConstrainedDevice() && phoneReclaimSettings().retireCpu;
  const ctx = createPlayContext(scope);
  // Registered before init so a mount that throws midway still retires what it started.
  scope.defer(() => {
    ctx.autosave?.flush();
    ctx.autosave = null;
    ctx.disposed = true;
    ctx.consentAbort.abort();
    ctxCancelEngineTurn(ctx);
    ctx.abort.abort();
    engines.unmounted(retireCpuOnUnmount);
    ctx.keyboardBoard?.destroy();
    ctx.keyboardBoard = null;
    (ctx.ground as { destroy?: () => void } | null)?.destroy?.();
    ctx.ground = null;
  });
  ctxInit(ctx);

  // Test hook for automated browser checks: synthetic chessground drags are
  // unreliable, so smokes call this to route through the real user-move path.
  const hook = (from: string, to: string) => ctxOnUserMove(ctx, from as Key, to as Key);
  const g = globalThis as unknown as { __playUserMove?: (from: string, to: string) => void };
  g.__playUserMove = hook;
  // Only clear the hook if it is still ours (a newer mount may have installed its own).
  scope.defer(() => {
    if (g.__playUserMove === hook) delete g.__playUserMove;
  });
}
