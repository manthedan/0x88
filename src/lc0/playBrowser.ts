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
import { sampleHumanMove } from './humanSampling.ts';
import { loadLc0ModelForOrt } from './modelCache.ts';
import { CachedLc0Evaluator, Lc0OnnxEvaluator } from './onnxEvaluator.ts';
import { Lc0PolicyOnlyPlayer } from './policyOnlyPlayer.ts';
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

const params = new URLSearchParams(location.search);
const DEFAULT_MODEL_URL = '/models/lc0/t1-256x10-distilled-swa-2432500.batch1.f32.onnx';
const MODEL_URL = params.get('model') ?? DEFAULT_MODEL_URL;

type PlayFamily = 'maia' | 'maia3' | 'lc0' | 'sf' | 'reckless' | 'viridithas' | 'berserk' | 'plentychess';

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
const LEVELS: Record<Exclude<PlayFamily, 'maia' | 'maia3' | 'sf'>, number[]> = {
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

function maiaModelUrl(elo: string): string {
  return `/models/lc0/maia-${elo}.f32.onnx`;
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
let maia3Promise: Promise<Maia3BrowserEvaluator> | null = null;
/** One-line model/cache status shown in the caption once Maia3 has loaded. */
let maia3Status: string | null = null;

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

function ensureMaia3(): Promise<Maia3BrowserEvaluator> {
  if (maia3Promise) return maia3Promise;
  maia3Promise = (async () => {
    setEngineNote('Loading Maia3 human model…');
    const evaluator = await Maia3BrowserEvaluator.create({
      selfElo: selectedMaia3Elo(),
      oppoElo: selectedMaia3Elo(),
      onProgress: (loaded, total) => showDownloadProgress('Maia3', loaded, total),
    });
    hideDownloadProgress();
    const load = evaluator.modelLoad;
    const origin = load.cacheStatus === 'hit' ? 'from cache' : load.cacheStatus === 'miss' ? 'downloaded' : 'loaded';
    const integrity = load.sha256Valid === true ? ', sha256 ✓' : load.sha256Valid === false ? ', sha256 MISMATCH' : '';
    maia3Status = `Model ${origin} (${((load.bytes ?? 0) / 1e6).toFixed(0)}MB${integrity})`;
    setEngineNote('');
    return evaluator;
  })().catch((error: Error) => {
    maia3Promise = null;
    hideDownloadProgress();
    setEngineNote(`Maia3 load failed: ${error.message}`, true);
    throw error;
  });
  return maia3Promise;
}

function selectedEngine(): PlayEngineOption {
  const id = selectEl('engineSelect').value;
  return ENGINE_OPTIONS.find((option) => option.id === id) ?? ENGINE_OPTIONS[0];
}
function selectedLevel(): number {
  return Math.max(0, Math.min(LEVEL_COUNT - 1, Number(selectEl('levelSelect').value) || 0));
}
function selectedMaia3Elo(): number {
  const input = el('maia3Elo') as HTMLInputElement;
  return Math.max(MAIA3_MIN_ELO, Math.min(MAIA3_MAX_ELO, Number(input.value) || MAIA3_DEFAULT_ELO));
}
function selectedMaia3Style(): Maia3MoveStyle {
  return selectEl('maia3Style').value === 'argmax' ? 'argmax' : 'sample';
}
function selectedMaia3Temperature(): number {
  const input = el('maia3Temperature') as HTMLInputElement;
  return Math.max(0.01, Math.min(5, Number(input.value) || 1));
}
function selectedMaia3TopP(): number {
  const input = el('maia3TopP') as HTMLInputElement;
  return Math.max(0.01, Math.min(1, Number(input.value) || 1));
}
function strengthFor(option: PlayEngineOption, level: number): number {
  if (option.family === 'maia' || option.family === 'maia3') return 1;
  if (option.family === 'sf') return SF_LEVELS[level].depth;
  if (option.family === 'lc0' && option.variant !== 'small') {
    // Without WebGPU the big nets run on the wasm-CPU fallback: keep them
    // available but at per-net reduced visit ladders (seconds-per-move).
    if (!bt4SupportedSync()) return BIG_NETS[option.variant as BigNetKey].wasmLevels[level];
    return BIG_NET_LEVELS[level];
  }
  return LEVELS[option.family][level];
}
function strengthCaption(): string {
  const option = selectedEngine();
  const level = selectedLevel();
  if (option.family === 'maia') {
    return `Plays like a ~${option.variant}-rated human — moves are sampled from its human-move predictions, so games vary.`;
  }
  if (option.family === 'maia3') {
    const style = selectedMaia3Style();
    const suffix = style === 'argmax' ? 'deterministic top human move' : `sampled, temperature ${selectedMaia3Temperature().toFixed(2)}, top-p ${selectedMaia3TopP().toFixed(2)}`;
    const status = maia3Status ? ` ${maia3Status}.` : '';
    return `Maia3 predicts human moves at Elo ${selectedMaia3Elo()} — ${suffix}. No LC0/PUCT search; its WDL output predicts the human game outcome, not an engine eval.${status}`;
  }
  if (option.family === 'sf') {
    const sf = SF_LEVELS[level];
    return sf.skill >= 20 ? `full strength · depth ${sf.depth}` : `UCI skill ${sf.skill} · depth ${sf.depth}`;
  }
  const value = strengthFor(option, level);
  const cpuNote = option.family === 'lc0' && option.variant !== 'small' && !bt4SupportedSync()
    ? ' · CPU fallback (no WebGPU): expect several seconds per move'
    : '';
  if (option.variant === 'lqo') return `≈ ${value} visits per move — higher levels press harder for tricks${cpuNote}`;
  const base = option.family === 'lc0' ? `≈ ${value} visits per move` : `search depth ${value}`;
  return option.family === 'lc0' ? `${base} — strong even on Fastest; pick a Maia for a human-level opponent${cpuNote}` : base;
}

function renderLevelOptions(): void {
  const select = selectEl('levelSelect');
  const previous = select.value;
  const option = selectedEngine();
  const field = select.closest('.field') as HTMLElement;
  if (option.family === 'maia' || option.family === 'maia3') {
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

function renderMaia3Controls(): void {
  const isMaia3 = selectedEngine().family === 'maia3';
  el('maia3Controls').hidden = !isMaia3;
  const elo = selectedMaia3Elo();
  el('maia3EloValue').textContent = String(elo);
  // Disabled (not hidden) in argmax mode so the sampling knobs stay
  // discoverable; values are preserved for when sampling is re-selected.
  const sample = selectedMaia3Style() === 'sample';
  (el('maia3Temperature') as HTMLInputElement).disabled = !sample;
  (el('maia3TopP') as HTMLInputElement).disabled = !sample;
  (el('maia3TemperatureField') as HTMLElement).style.opacity = sample ? '' : '0.5';
  (el('maia3TopPField') as HTMLElement).style.opacity = sample ? '' : '0.5';
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
  if (option.family === 'maia3') {
    const player = await ensureMaia3();
    if (signal.aborted) return null;
    const choice = await player.chooseMove({ positions }, {
      selfElo: selectedMaia3Elo(),
      oppoElo: selectedMaia3Elo(),
      style: selectedMaia3Style(),
      temperature: selectedMaia3Temperature(),
      topP: selectedMaia3TopP(),
    });
    return choice.move;
  }
  if (option.family === 'lc0' && option.variant === 'small') {
    const searcher = await ensureLc0Small();
    const result = await searcher.search({ positions }, { visits: visitsOrDepth, signal, yieldEveryMs: 16, reuseTree: true, drawScore: PLAY_DRAW_SCORE, searchContemptLimit: PLAY_SEARCH_CONTEMPT_LIMIT });
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
        searchContemptLimit: option.variant === 'lqo' ? LQO_SEARCH_CONTEMPT_LIMIT : PLAY_SEARCH_CONTEMPT_LIMIT,
        ...(option.variant === 'lqo' ? { cpuct: LQO_CPUCT } : {}),
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

// Test hook for automated browser checks: synthetic chessground drags are
// unreliable, so smokes call this to route through the real user-move path.
(globalThis as unknown as { __playUserMove?: (from: string, to: string) => void }).__playUserMove
  = (from, to) => onUserMove(from as Key, to as Key);

function onUserMove(from: Key, to: Key): void {
  if (engineThinking || gameOver || board.turn !== humanColor) { render(); return; }
  const matching = matchUserMoves(board, from, to);
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
  el('promoPicker').hidden = true;
  if (!pendingPromotion) {
    hidePromotionOverlay(el('ground'));
    return;
  }
  showPromotionOverlay({
    boardContainer: el('ground'),
    orientation,
    color: humanColor,
    choices: pendingPromotion,
    onPick: (move) => applyHumanMove(move),
    onCancel: () => { pendingPromotion = null; render(); },
  });
}

function engineOptionState(option: PlayEngineOption): { disabled: boolean; suffix: string } {
  if (option.family !== 'lc0' || option.variant === 'small') return { disabled: false, suffix: '' };
  return bigNetOptionState(BIG_NETS[option.variant as BigNetKey]);
}

function engineOptionHtml(option: PlayEngineOption): string {
  const { disabled, suffix } = engineOptionState(option);
  return `<option value="${option.id}"${disabled ? ' disabled' : ''}>${option.label}${suffix}</option>`;
}

function refreshEngineOptions(): void {
  const select = selectEl('engineSelect');
  const selected = select.value;
  const group = (key: PlayEngineOption['group']) => ENGINE_OPTIONS.filter((option) => option.group === key).map(engineOptionHtml).join('');
  select.innerHTML = `<optgroup label="Human-like (Maia, plays like a rated human)">${group('human')}</optgroup>`
    + `<optgroup label="Odds bots (give you material, then hunt for tricks)">${group('odds')}</optgroup>`
    + `<optgroup label="Engines (strong at any level)">${group('engine')}</optgroup>`;
  const selectedOption = ENGINE_OPTIONS.find((option) => option.id === selected);
  if (selectedOption && !engineOptionState(selectedOption).disabled) select.value = selected;
  else select.value = 'maia3';
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
    const cpu = bt4SupportedSync() ? '' : ' No WebGPU here: runs on the CPU (wasm) fallback with reduced visit budgets — expect several seconds per move.';
    caution.textContent = `First move downloads the ~${config.approxMb}MB net.${odds}${memory ?? ''}${cpu}`.trimEnd();
    caution.hidden = false;
  } else {
    caution.hidden = true;
  }
}

function render(): void {
  el('status').textContent = statusText();
  el('status').classList.toggle('over', !!gameOver);
  renderMaia3Controls();
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
    check: boardCheck(board),
    highlight: { lastMove: true, check: true },
    animation: { enabled: true, duration: 160 },
    movable: {
      free: false,
      color: humanCanMove ? (humanColor === 'w' ? 'white' as const : 'black' as const) : undefined,
      dests: humanCanMove ? legalDests(board) : new Map<Key, Key[]>(),
      showDests: humanCanMove,
      events: { after: onUserMove },
    },
    lastMove: lastUci ? [lastUci.slice(0, 2) as Key, lastUci.slice(2, 4) as Key] : undefined,
  };
  if (!ground) ground = Chessground(el('ground'), config);
  else ground.set(config);
}

let lastEngineId = 'maia3';

function init(): void {
  refreshEngineOptions();
  selectEl('engineSelect').value = 'maia3';
  renderLevelOptions();
  el('newGame').addEventListener('click', newGame);
  el('takeback').addEventListener('click', takeback);
  el('resign').addEventListener('click', resign);
  el('flip').addEventListener('click', () => { orientation = orientation === 'white' ? 'black' : 'white'; render(); });
  el('exportPgn').addEventListener('click', () => { el('pgnOut').textContent = exportPgn(); });
  el('copyPgn').addEventListener('click', () => { void navigator.clipboard.writeText(exportPgn()); });
  selectEl('engineSelect').addEventListener('change', () => {
    const option = selectedEngine();
    // Switching from a fixed Maia level to Maia3 carries the rating over, so
    // "Maia 1700 → Maia3" plays at 1700 instead of snapping back to 1500.
    if (option.family === 'maia3' && lastEngineId.startsWith('maia-')) {
      const carried = Number(lastEngineId.slice('maia-'.length));
      if (Number.isFinite(carried)) {
        (el('maia3Elo') as HTMLInputElement).value = String(Math.max(MAIA3_MIN_ELO, Math.min(MAIA3_MAX_ELO, carried)));
      }
    }
    lastEngineId = option.id;
    renderLevelOptions();
    renderMaia3Controls();
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
  selectEl('maia3Style').addEventListener('change', render);
  el('maia3Elo').addEventListener('input', render);
  el('maia3Temperature').addEventListener('input', render);
  el('maia3TopP').addEventListener('input', render);
  void probeBt4Support().then(refreshEngineOptions);
  void checkBigNetAsset(BIG_NETS.bt4, refreshEngineOptions);
  void checkBigNetAsset(T3_NET, refreshEngineOptions);
  void checkBigNetAsset(LQO_NET, refreshEngineOptions);
  render();
}

init();
