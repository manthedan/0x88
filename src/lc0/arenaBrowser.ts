import { Chessground } from 'chessground';
import type { DrawShape } from 'chessground/draw';
import type { Key } from 'chessground/types';
import { boardToFen, parseFen, START_FEN, type BoardState } from '../chess/board.ts';
import { legalMoves, makeMove } from '../chess/movegen.ts';
import { moveToUci, type Move } from '../chess/moveCodec.ts';
import { gameTreeToPgn } from '../chess/pgn.ts';
import { BUILTIN_ARENA_OPENINGS, parseArenaOpenings, type ArenaOpening } from './arenaOpenings.ts';
import { gameOutcome, type GameResultCode } from './engineBattle.ts';
import { GameTree } from './gameTree.ts';
import { loadLc0ModelForOrt } from './modelCache.ts';
import { collectOrtRuntimeDiagnostics } from '../nn/ortRuntime.ts';
import { CachedLc0Evaluator, Lc0OnnxEvaluator, type Lc0Evaluation, type Lc0EvaluationCacheMetrics } from './onnxEvaluator.ts';
import { Lc0PolicyOnlyPlayer } from './policyOnlyPlayer.ts';
import { Lc0PuctSearcher, type Lc0SearchResult } from './search.ts';
import { Lc0WebHybridEvaluator } from './wgslMatmulAddProbe.ts';
import type { Node as PuctNode } from '../search/puct.ts';
import { StockfishEngine, stockfishFlavorUrl, type StockfishFlavor, type StockfishInfoLine } from './stockfishEngine.ts';
import { RecklessEngine, formatRecklessBrowserApiLoadStatus } from './recklessEngine.ts';
import { RECKLESS_VARIANTS, checkRecklessVariantAsset, hasExplicitRecklessVariant, recklessVariantAssetStatus, recklessVariantByKey, recklessVariantFromParams, normalizeRecklessVariant, resolveDefaultRecklessVariantAssetFallback, type RecklessVariant } from './recklessVariants.ts';
import { ViridithasEngine, canUsePersistentViridithasWasi } from './viridithasEngine.ts';
import { VIRIDITHAS_VARIANTS, checkViridithasVariantAsset, normalizeViridithasVariant, viridithasVariantAssetStatus, viridithasVariantByKey, viridithasVariantFromParams, type ViridithasVariant } from './viridithasVariants.ts';
import { Bt4WorkerSearcher, bt4LoadWarning, bt4SupportedSync, probeBt4Support, type Bt4SearchResult } from './bt4Engine.ts';

type Ground = ReturnType<typeof Chessground>;
type EngineFamily = 'lc0' | 'sf' | 'reckless' | 'viridithas';
type SeatId = 'A' | 'B';
interface EngineRow { family: EngineFamily; variant: string; strength: number; }
interface ArenaEngine {
  id: string;
  name: string;
  move(positions: BoardState[], signal: AbortSignal): Promise<string | null>;
  warmup?(signal: AbortSignal): Promise<void>;
}
interface GameRecord { pgn: string; }
interface MatchGame { whiteSeat: SeatId; opening: ArenaOpening; }
interface MatchScore { a: number; b: number; aWins: number; bWins: number; draws: number; games: number; }
interface Lc0TreeTelemetry {
  engineName: string;
  searches: number;
  rootReused: number;
  completedVisits: number;
  reusedRootVisits: number;
  evalCalls: number;
  cacheHits: number;
  neuralEvalMisses: number;
  transpositionHits: number;
  replyChecks: number;
  replyParentsExpanded: number;
  replyVisited: number;
  replyTopPolicy5: number;
  replyTopVisits5: number;
}
interface PendingLc0ReplyProbe {
  engineId: string;
  engineName: string;
  afterMoveFen: string;
  child: PuctNode | null;
}
interface EngineEvalBar {
  /** White expected score, with 0.5 as equal. Displayed board-oriented: black at top, white at bottom. */
  whiteScore: number;
  label: string;
}
interface EngineOutputSnapshot {
  engineId: string;
  engineName: string;
  kind: 'lc0' | 'uci';
  fen: string;
  move?: string;
  summary: string;
  shortEval: string;
  evalBar?: EngineEvalBar;
  detail?: string;
  pv?: string[];
}

const params = new URLSearchParams(location.search);
const DEFAULT_MODEL_URL = '/models/lc0/t1-256x10-distilled-swa-2432500.batch1.f32.onnx';
const MODEL_URL = params.get('model') ?? DEFAULT_MODEL_URL;
const DEFAULT_PACK_URL = '/models/lc0/t1-256x10-distilled-swa-2432500.batch8.f16.lc0web/model.lc0web.json';
const PACK_URL = params.get('pack') ?? params.get('modelPack') ?? DEFAULT_PACK_URL;
type Lc0ArenaRuntime = 'onnx' | 'hybrid-ort-heads' | 'hybrid-wgsl-heads';
const REQUESTED_RECKLESS_EXPLICIT = hasExplicitRecklessVariant(params);
let REQUESTED_RECKLESS_VARIANT = recklessVariantFromParams(params);
const REQUESTED_VIRIDITHAS_VARIANT = viridithasVariantFromParams(params);

let ground: Ground | null = null;
let board: BoardState = parseFen(START_FEN);
let historyBoards: BoardState[] = [board];
let lastUci: string | null = null;
let boardWhiteId: string | null = null;
let boardBlackId: string | null = null;
let boardWhiteName: string | null = null;
let boardBlackName: string | null = null;
let loadingLc0 = false;
let running = false;
let abort: AbortController | null = null;
let player: Lc0PolicyOnlyPlayer | null = null;
let searcher: Lc0PuctSearcher | null = null;
let lc0Cache: CachedLc0Evaluator | null = null;
let stockfishLite: StockfishEngine | null = null;
let stockfishFull: StockfishEngine | null = null;
const recklessByVariant = new Map<string, RecklessEngine>();
const viridithasByVariant = new Map<string, ViridithasEngine>();
// Lc0 BT4 runs in its own worker (lazy, WebGPU-gated, disposable). See bt4Engine.ts.
const bt4 = new Bt4WorkerSearcher();
let runtimeIsolation = typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated;
let runtimeSharedArrayBuffer = typeof SharedArrayBuffer !== 'undefined';
const engines = new Map<string, ArenaEngine>();
const lc0Searchers = new Map<string, Lc0PuctSearcher>();
const lastLc0SearchResults = new Map<string, Lc0SearchResult>();
const pendingLc0ReplyProbes = new Map<string, PendingLc0ReplyProbe>();
const lc0TreeTelemetry = new Map<string, Lc0TreeTelemetry>();
const engineOutputs = new Map<string, EngineOutputSnapshot>();
const thinkingEngineIds = new Set<string>();
let activeEngineIds: string[] = [];
const games: GameRecord[] = [];
const seatRows: Record<SeatId, EngineRow> = {
  A: { family: 'lc0', variant: 'small', strength: 100 },
  B: { family: 'sf', variant: 'lite', strength: 8 },
};

function el(id: string): HTMLElement {
  const node = document.getElementById(id);
  if (!node) throw new Error(`Missing #${id}`);
  return node;
}
function inputEl(id: string): HTMLInputElement { return el(id) as HTMLInputElement; }
function selectEl(id: string): HTMLSelectElement { return el(id) as HTMLSelectElement; }
function htmlEscape(value: unknown): string {
  return String(value).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]!));
}

function normalizeLc0Runtime(value: string | null): Lc0ArenaRuntime {
  const raw = (value ?? '').toLowerCase();
  if (raw === 'hybrid' || raw === 'lc0web' || raw === 'hybrid-ort-heads' || raw === 'wgsl-encoder') return 'hybrid-ort-heads';
  if (raw === 'hybrid-wgsl-heads' || raw === 'wgsl-heads' || raw === 'wgsl') return 'hybrid-wgsl-heads';
  return 'onnx';
}

function initialLc0Runtime(): Lc0ArenaRuntime {
  if (params.get('headBackend') === 'wgsl' || params.get('hybridHeads') === 'wgsl') return 'hybrid-wgsl-heads';
  return normalizeLc0Runtime(params.get('lc0Runtime') ?? params.get('runtime'));
}

function selectedLc0Runtime(): Lc0ArenaRuntime {
  return normalizeLc0Runtime(selectEl('lc0RuntimeSelect').value);
}

function lc0RuntimeLabel(runtime = selectedLc0Runtime()): string {
  if (runtime === 'hybrid-wgsl-heads') return 'WGSL encoder + WGSL heads';
  if (runtime === 'hybrid-ort-heads') return 'WGSL encoder + ORT heads';
  return 'ORT ONNX';
}

function lc0EncoderLayers(): number {
  return Math.min(32, Math.max(1, Math.floor(Number(params.get('encoderLayers') ?? params.get('layers') ?? '10') || 10)));
}

function percent(value: number): string {
  return `${(100 * value).toFixed(1)}%`;
}

function percent0(value: number): string {
  return `${Math.round(100 * value)}%`;
}

function signed(value: number, digits = 2): string {
  return `${value >= 0 ? '+' : ''}${value.toFixed(digits)}`;
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function fenTurn(fen: string): 'w' | 'b' {
  return fen.split(/\s+/)[1] === 'b' ? 'b' : 'w';
}

function stmScoreToWhiteScore(fen: string, stmScore: number): number {
  const score = clamp01(stmScore);
  return fenTurn(fen) === 'w' ? score : 1 - score;
}

function pvText(pv: string[] | undefined, maxMoves = 8): string {
  if (!pv?.length) return 'PV —';
  const shown = pv.slice(0, maxMoves).join(' ');
  return `PV ${shown}${pv.length > maxMoves ? ' …' : ''}`;
}

// All engine scores are displayed from White's perspective (positive = good for
// White), matching the eval bar and standard chess GUIs, so the sign no longer
// flips with the side to move. Engines report from the side-to-move POV, so flip
// when Black is to move (WDL swaps win/loss; Q and cp negate).
function toWhiteWdl(wdl: [number, number, number], fen: string): [number, number, number] {
  return fenTurn(fen) === 'w' ? wdl : [wdl[2], wdl[1], wdl[0]];
}
function toWhiteQ(q: number, fen: string): number {
  return fenTurn(fen) === 'w' ? q : -q;
}

function stockfishScoreText(info: StockfishInfoLine | undefined, fen: string): string {
  if (!info) return 'score unavailable';
  const w = fenTurn(fen) === 'w' ? 1 : -1;
  if (info.mateIn !== undefined) return `mate ${signed(w * info.mateIn, 0)} · d${info.depth}`;
  if (info.scoreCp !== undefined) return `${signed(w * info.scoreCp / 100, 2)} · d${info.depth}`;
  return `score unavailable · d${info.depth}`;
}

function stockfishScoreCompact(info: StockfishInfoLine | undefined, fen: string): string {
  if (!info) return 'eval —';
  const w = fenTurn(fen) === 'w' ? 1 : -1;
  if (info.mateIn !== undefined) return `M${signed(w * info.mateIn, 0)} · d${info.depth}`;
  if (info.scoreCp !== undefined) return `${signed(w * info.scoreCp / 100, 2)} · d${info.depth}`;
  return `d${info.depth}`;
}

function lc0WdlText(evaluation: Lc0Evaluation): string {
  const wdl = toWhiteWdl(evaluation.wdl, evaluation.fen);
  return `WDL ${percent(wdl[0])}/${percent(wdl[1])}/${percent(wdl[2])} · Q ${signed(toWhiteQ(evaluation.q, evaluation.fen), 3)} · MLH ${evaluation.mlh.toFixed(1)}`;
}

function lc0WdlCompact(wdl: [number, number, number], q: number, fen: string): string {
  const w = toWhiteWdl(wdl, fen);
  return `WDL ${percent0(w[0])}/${percent0(w[1])}/${percent0(w[2])} · Q ${signed(toWhiteQ(q, fen), 2)}`;
}

function lc0EvalBar(fen: string, wdl: [number, number, number]): EngineEvalBar {
  const stmExpectedScore = clamp01(wdl[0] + 0.5 * wdl[1]);
  const whiteScore = stmScoreToWhiteScore(fen, stmExpectedScore);
  return { whiteScore, label: percent0(whiteScore) };
}

function qEvalBar(fen: string, q: number): EngineEvalBar {
  const whiteScore = stmScoreToWhiteScore(fen, (clamp01((q + 1) / 2)));
  return { whiteScore, label: percent0(whiteScore) };
}

function stockfishEvalBar(fen: string, info: StockfishInfoLine | undefined): EngineEvalBar | undefined {
  if (!info) return undefined;
  const turn = fenTurn(fen);
  if (info.mateIn !== undefined) {
    const whiteMateSign = turn === 'w' ? info.mateIn : -info.mateIn;
    return { whiteScore: whiteMateSign > 0 ? 1 : 0, label: `M${Math.abs(whiteMateSign)}` };
  }
  if (info.scoreCp === undefined) return undefined;
  const whiteCp = turn === 'w' ? info.scoreCp : -info.scoreCp;
  const whiteScore = 1 / (1 + Math.exp(-whiteCp / 320));
  return { whiteScore: clamp01(whiteScore), label: signed(whiteCp / 100, 1) };
}

function searchWdlText(wdl: [number, number, number], q: number, fen: string): string {
  const w = toWhiteWdl(wdl, fen);
  return `WDL ${percent(w[0])}/${percent(w[1])}/${percent(w[2])} · search Q ${signed(toWhiteQ(q, fen), 3)}`;
}

function recordEngineOutput(snapshot: EngineOutputSnapshot): void {
  thinkingEngineIds.delete(snapshot.engineId);
  engineOutputs.set(snapshot.engineId, snapshot);
  renderSideLabels();
  renderEngineOutputs();
}

function shortEngineTag(name: string): string {
  const n = name.toLowerCase();
  if (n.includes('bt4')) return 'BT4';
  if (n.includes('lc0') || n.includes('leela')) return 'Lc0';
  if (n.includes('reckless')) return 'Reck';
  if (n.includes('viridithas')) return 'Viri';
  if (n.includes('stockfish') || /\bsf\b/.test(n)) return n.includes('lite') ? 'SF-L' : 'SF';
  return name.split(/[\s·|]+/)[0] || name;
}

// Optional favicon-sized engine logos (public/engine-logos/, not bundled by
// default). We probe once which files exist and only emit an <img> for those, so
// when a logo is absent the markup is unchanged — no per-render insert/remove jitter.
const availableEngineLogos = new Set<string>();

function engineLogoFamily(name: string): string {
  const n = name.toLowerCase();
  if (n.includes('bt4') || n.includes('lc0') || n.includes('leela')) return 'lc0';
  if (n.includes('reckless')) return 'reckless';
  if (n.includes('viridithas')) return 'viridithas';
  if (n.includes('stockfish') || /\bsf\b/.test(n)) return 'stockfish';
  return '';
}

function engineLogoHtml(name: string): string {
  const family = engineLogoFamily(name);
  return family && availableEngineLogos.has(family) ? `<img class="engine-logo" src="/engine-logos/${family}.png" alt="">` : '';
}

async function probeEngineLogos(): Promise<void> {
  await Promise.all(['lc0', 'stockfish', 'reckless', 'viridithas'].map(async (family) => {
    try { if ((await fetch(`/engine-logos/${family}.png`, { method: 'HEAD', cache: 'no-store' })).ok) availableEngineLogos.add(family); } catch { /* absent */ }
  }));
  if (availableEngineLogos.size) renderSideLabels();
}

function renderEvalBars(): void {
  const render = (id: string, color: 'White' | 'Black', engineId: string | null, engineName: string | null) => {
    const node = el(id);
    const output = engineId ? engineOutputs.get(engineId) : undefined;
    const bar = output?.evalBar;
    const thinking = engineId ? thinkingEngineIds.has(engineId) : false;
    const whiteScore = clamp01(bar?.whiteScore ?? 0.5);
    const label = bar?.label ?? (thinking ? 'thinking…' : 'eval —');
    node.classList.toggle('empty', !bar);
    node.classList.toggle('thinking', thinking);
    node.title = `${color} engine: ${engineName ?? '—'}${bar ? ` · ${bar.label}` : ''}`;
    node.innerHTML = `<div class="eval-fill" style="height:${(100 * whiteScore).toFixed(1)}%"></div><div class="eval-midline"></div><div class="eval-bar-caption">${color[0]}</div><div class="eval-bar-value">${htmlEscape(label)}</div>`;
    const chip = document.getElementById(id.replace('EvalBar', 'Chip'));
    if (chip) {
      chip.innerHTML = engineName ? `${engineLogoHtml(engineName)}<span>${htmlEscape(shortEngineTag(engineName))}</span>` : '';
      chip.title = engineName ? `${color} engine: ${engineName}` : '';
      chip.style.display = engineName ? '' : 'none';
    }
  };
  render('whiteEngineEvalBar', 'White', boardWhiteId, boardWhiteName);
  render('blackEngineEvalBar', 'Black', boardBlackId, boardBlackName);
}

function renderEngineOutputs(): void {
  const ids = activeEngineIds.length ? activeEngineIds : [...new Set([seatEngineId('A'), seatEngineId('B')])].filter((id) => engines.has(id));
  const cards = ids.map((id) => {
    const snapshot = engineOutputs.get(id);
    const name = snapshot?.engineName ?? engines.get(id)?.name ?? id;
    const thinking = thinkingEngineIds.has(id);
    // No per-ply "active" highlight here: at fast movetimes it strobes. Whose turn
    // it is is shown calmly by the last-move arrow on the board instead.
    if (!snapshot) return `<div class="eval-card"><strong>${htmlEscape(name)}</strong>${thinking ? 'thinking on current position…' : 'waiting for output…'}</div>`;
    const status = thinking ? '<span class="eval-status">thinking… keeping last eval</span>' : '';
    const detail = snapshot.detail ? `<br>${htmlEscape(snapshot.detail)}` : '';
    const pv = snapshot.pv?.length ? `<br>${htmlEscape(pvText(snapshot.pv))}` : '';
    const move = snapshot.move ? ` · move ${snapshot.move}` : '';
    return `<div class="eval-card"><strong>${htmlEscape(name)}${status}</strong>${htmlEscape(snapshot.summary)}${htmlEscape(move)}${detail}${pv}</div>`;
  });
  el('engineEvalInfo').innerHTML = cards.length ? cards.join('') : '<div class="eval-card">Engine outputs: waiting for a move…</div>';
}

function renderSideLabels() {
  // Side labels are static identity rows (chip + logo + engine + eval). Whose turn
  // it is is conveyed by the board's last-move arrow, not a per-ply highlight that
  // would strobe at fast movetimes.
  const update = (id: string, color: 'White' | 'Black', engineId: string | null, engineName: string | null) => {
    const node = el(id);
    const output = engineId ? engineOutputs.get(engineId) : undefined;
    const thinking = engineId ? thinkingEngineIds.has(engineId) : false;
    const evalText = output?.shortEval ?? (thinking ? 'thinking…' : 'eval —');
    node.classList.remove('active');
    node.innerHTML = `<span class="side-main"><span class="color">${color}</span> ${engineName ? engineLogoHtml(engineName) : ''}<span class="engine">${htmlEscape(engineName ?? '—')}</span> <span class="side-eval">${htmlEscape(evalText)}</span></span>`;
  };
  update('blackSideLabel', 'Black', boardBlackId, boardBlackName);
  update('whiteSideLabel', 'White', boardWhiteId, boardWhiteName);
  renderEvalBars();
}

function setBoardSideEngines(whiteId: string | null, whiteName: string | null, blackId: string | null, blackName: string | null): void {
  boardWhiteId = whiteId;
  boardBlackId = blackId;
  boardWhiteName = whiteName;
  boardBlackName = blackName;
  renderSideLabels();
}

function renderBoard() {
  const config = {
    orientation: 'white' as const,
    fen: boardToFen(board).split(' ')[0],
    coordinates: true,
    viewOnly: true,
    highlight: { lastMove: true, check: true },
    animation: { enabled: true, duration: 140 },
    lastMove: lastUci ? [lastUci.slice(0, 2) as Key, lastUci.slice(2, 4) as Key] : undefined,
    // Custom brushes in the two side identity colors so the last-move arrow also
    // shows which side just moved — a calm alternative to per-ply "to move" flashing.
    drawable: { enabled: false, brushes: {
      moveWhite: { key: 'moveWhite', color: '#2f6e7d', opacity: 0.9, lineWidth: 14 },
      moveBlack: { key: 'moveBlack', color: '#b15c2b', opacity: 0.9, lineWidth: 14 },
    } },
  };
  // Cast: chessground's DrawBrushes type has fixed keys, but custom brush keys
  // merge fine at runtime.
  const cfg = config as unknown as NonNullable<Parameters<typeof Chessground>[1]>;
  if (!ground) ground = Chessground(el('ground'), cfg);
  else ground.set(cfg);
  // The mover is the side NOT to move now; tint the arrow with their identity hue.
  const moverBrush = board.turn === 'w' ? 'moveBlack' : 'moveWhite';
  const shapes: DrawShape[] = lastUci && lastUci.length >= 4
    ? [{ orig: lastUci.slice(0, 2) as Key, dest: lastUci.slice(2, 4) as Key, brush: moverBrush }] : [];
  ground.setAutoShapes(shapes);
  renderSideLabels();
  renderEngineOutputs();
}

// The arena is a two-seat head-to-head. Each seat uses the same staged selector
// pattern as the analysis page: family → variant → strength.
function strengthMeta(family: EngineFamily): { unit: string; min: number; max: number; def: number } {
  if (family === 'lc0') return { unit: 'visits', min: 1, max: 100000, def: 100 };
  if (family === 'sf') return { unit: 'depth', min: 1, max: 40, def: 8 };
  if (family === 'viridithas') return { unit: 'depth', min: 1, max: 20, def: 6 };
  return { unit: 'depth', min: 1, max: 30, def: 4 };
}

function defaultVariant(family: EngineFamily): string {
  if (family === 'reckless') return REQUESTED_RECKLESS_VARIANT.key;
  if (family === 'viridithas') return REQUESTED_VIRIDITHAS_VARIANT.key;
  return family === 'sf' ? 'lite' : 'small';
}

function variantOptions(family: EngineFamily): { value: string; label: string; disabled?: boolean }[] {
  if (family === 'lc0') return [{ value: 'small', label: 'Small' }, { value: 'bt4', label: 'BT4', disabled: !bt4SupportedSync() }];
  if (family === 'sf') return [{ value: 'lite', label: 'Lite' }, { value: 'full', label: 'Full' }];
  if (family === 'viridithas') return availableViridithasVariants().map((v) => ({ value: v.key, label: v.label }));
  return availableRecklessVariants().map((v) => ({ value: v.key, label: v.label }));
}

function clampStrength(row: EngineRow): void {
  const meta = strengthMeta(row.family);
  row.strength = Math.max(meta.min, Math.min(meta.max, Math.floor(Number(row.strength) || meta.def)));
}

function rowLabel(row: EngineRow): string {
  if (row.family === 'lc0') return row.variant === 'bt4' ? 'Lc0 BT4' : 'Lc0';
  if (row.family === 'sf') return row.variant === 'lite' ? 'Stockfish Lite' : 'Stockfish';
  if (row.family === 'viridithas') return viridithasVariantForKey(row.variant).label;
  return recklessVariantForKey(row.variant).label;
}

function engineIdForRow(row: EngineRow): string {
  return `${row.family}:${row.variant}:${row.strength}`;
}

function activeSeatRows(): EngineRow[] {
  return [seatRows.A, seatRows.B];
}

function seatEngineId(seat: SeatId): string {
  return engineIdForRow(seatRows[seat]);
}

function renderSeatSelectors(): void {
  const families: [EngineFamily, string][] = [['lc0', 'Lc0'], ['sf', 'Stockfish'], ['reckless', 'Reckless'], ['viridithas', 'Viridithas']];
  el('arenaSeatList').innerHTML = (['A', 'B'] as const).map((seat) => {
    const row = seatRows[seat];
    const meta = strengthMeta(row.family);
    const famSel = families.map(([value, label]) => `<option value="${value}"${row.family === value ? ' selected' : ''}>${label}</option>`).join('');
    const varSel = variantOptions(row.family).map((option) => `<option value="${option.value}"${row.variant === option.value ? ' selected' : ''}${option.disabled ? ' disabled' : ''}>${htmlEscape(option.label)}</option>`).join('');
    const label = `Engine ${seat === 'A' ? '1' : '2'}`;
    return `<div class="engine-row seat-row" data-seat="${seat}"><span class="seat-name">${label}</span><select class="seat-fam" data-seat="${seat}" aria-label="${label} family">${famSel}</select><span class="arrow">→</span><select class="seat-var" data-seat="${seat}" aria-label="${label} variant">${varSel}</select><span class="arrow">→</span><input class="seat-strength row-strength" data-seat="${seat}" aria-label="${label} strength" type="number" min="${meta.min}" max="${meta.max}" step="1" value="${row.strength}" title="${meta.unit}"><span class="row-unit">${meta.unit}</span></div>`;
  }).join('');
}

function syncSeatRowsFromDom(): void {
  for (const seat of ['A', 'B'] as const) {
    const host = el('arenaSeatList');
    const family = host.querySelector<HTMLSelectElement>(`.seat-fam[data-seat="${seat}"]`)?.value as EngineFamily | undefined;
    if (family === 'lc0' || family === 'sf' || family === 'reckless' || family === 'viridithas') seatRows[seat].family = family;
    const variant = host.querySelector<HTMLSelectElement>(`.seat-var[data-seat="${seat}"]`)?.value;
    if (variant) seatRows[seat].variant = variant;
    const strength = host.querySelector<HTMLInputElement>(`.seat-strength[data-seat="${seat}"]`)?.value;
    if (strength != null) seatRows[seat].strength = Number(strength);
    clampStrength(seatRows[seat]);
  }
}

function populateSeats(): void {
  const options = [...engines.values()].map((engine) => `<option value="${htmlEscape(engine.id)}">${htmlEscape(engine.name)}</option>`).join('');
  selectEl('seatA').innerHTML = options;
  selectEl('seatB').innerHTML = options;
  selectEl('seatA').value = seatEngineId('A');
  selectEl('seatB').value = seatEngineId('B');
  renderSeatSelectors();
}

function refreshSeatControls(): void {
  selectEl('seatA').disabled = running;
  selectEl('seatB').disabled = running;
  selectEl('lc0RuntimeSelect').disabled = running || loadingLc0;
  for (const selector of ['.seat-fam', '.seat-var', '.seat-strength']) {
    for (const node of el('arenaSeatList').querySelectorAll<HTMLInputElement | HTMLSelectElement>(selector)) node.disabled = running;
  }
}

function arenaBudgetMode(): 'fixed' | 'movetime' {
  return selectEl('budgetModeSelect').value === 'movetime' ? 'movetime' : 'fixed';
}

function arenaMovetimeMs(): number {
  return Math.max(10, Math.min(60000, Math.floor(Number(inputEl('movetimeInput').value) || 500)));
}

function arenaCacheEntries(): number {
  return Math.max(0, Math.min(100000, Math.floor(Number(inputEl('cacheEntriesInput').value) || 0)));
}

function stockfishThreads(): number {
  const requested = Math.max(1, Math.min(32, Math.floor(Number(inputEl('stockfishThreadsInput').value) || 1)));
  return threadedStockfishAvailable() ? requested : 1;
}

function cacheMetricsText(metrics: Lc0EvaluationCacheMetrics | undefined): string {
  if (!metrics) return 'NN cache: unavailable';
  return `NN cache: ${metrics.entries}/${metrics.maxEntries} entries · ${metrics.hits} hit${metrics.hits === 1 ? '' : 's'} · ${metrics.misses} miss${metrics.misses === 1 ? '' : 'es'}`;
}

function renderCacheInfo(): void {
  lc0Cache?.setMaxEntries(arenaCacheEntries());
  el('cacheInfo').textContent = cacheMetricsText(lc0Cache?.metrics());
  renderSearchTelemetryInfo();
}

function emptyTreeTelemetry(engineName: string): Lc0TreeTelemetry {
  return {
    engineName,
    searches: 0,
    rootReused: 0,
    completedVisits: 0,
    reusedRootVisits: 0,
    evalCalls: 0,
    cacheHits: 0,
    neuralEvalMisses: 0,
    transpositionHits: 0,
    replyChecks: 0,
    replyParentsExpanded: 0,
    replyVisited: 0,
    replyTopPolicy5: 0,
    replyTopVisits5: 0,
  };
}

function telemetryFor(engineId: string, engineName: string): Lc0TreeTelemetry {
  const existing = lc0TreeTelemetry.get(engineId);
  if (existing) {
    existing.engineName = engineName;
    return existing;
  }
  const created = emptyTreeTelemetry(engineName);
  lc0TreeTelemetry.set(engineId, created);
  return created;
}

function ratioText(numerator: number, denominator: number): string {
  return denominator > 0 ? `${numerator}/${denominator}` : '0/0';
}

function searchTelemetryText(): string {
  const entries = [...lc0TreeTelemetry.values()].filter((t) => t.searches || t.replyChecks);
  if (!entries.length) return 'LC0 tree: waiting for searches…';
  return `LC0 tree: ${entries.map((t) => {
    const fresh = Math.max(0, t.searches - t.rootReused);
    return `${t.engineName}: reuse ${ratioText(t.rootReused, t.searches)} (${fresh} fresh) · reused visits ${t.reusedRootVisits} · reply parent ${ratioText(t.replyParentsExpanded, t.replyChecks)} · reply visited ${ratioText(t.replyVisited, t.replyChecks)} · opp reply top-policy≤5 ${ratioText(t.replyTopPolicy5, t.replyChecks)} · top-visits≤5 ${ratioText(t.replyTopVisits5, t.replyChecks)} · evals ${t.evalCalls} · cache hits ${t.cacheHits} · trans ${t.transpositionHits}`;
  }).join(' | ')}`;
}

function renderSearchTelemetryInfo(): void {
  el('searchTelemetryInfo').textContent = searchTelemetryText();
}

function isLc0SearchEngine(engineId: string): boolean {
  return engineId.startsWith('lc0:small:');
}

function childForRootMove(result: Lc0SearchResult | undefined, uci: string): PuctNode | null {
  const root = result?.search.root;
  if (!root?.expanded) return null;
  return root.edges.find((edge) => moveToUci(edge.move) === uci)?.child ?? null;
}

function recordLc0SearchTelemetry(engineId: string, engineName: string, result: Lc0SearchResult): void {
  const stats = result.search.stats;
  const t = telemetryFor(engineId, engineName);
  const completed = stats?.completedVisits ?? 0;
  t.searches += 1;
  if (stats?.rootReused) t.rootReused += 1;
  t.completedVisits += completed;
  t.reusedRootVisits += Math.max(0, result.visits - completed);
  t.evalCalls += stats?.evalCalls ?? 0;
  t.cacheHits += stats?.cacheHits ?? 0;
  t.neuralEvalMisses += stats?.neuralEvalMisses ?? 0;
  t.transpositionHits += stats?.transpositionHits ?? 0;
  lastLc0SearchResults.set(engineId, result);
  renderSearchTelemetryInfo();
}

function recordLc0PolicyOutput(engineId: string, engineName: string, evaluation: Lc0Evaluation, move?: string): void {
  recordEngineOutput({
    engineId,
    engineName,
    kind: 'lc0',
    fen: evaluation.fen,
    move,
    summary: lc0WdlText(evaluation),
    shortEval: lc0WdlCompact(evaluation.wdl, evaluation.q, evaluation.fen),
    evalBar: lc0EvalBar(evaluation.fen, evaluation.wdl),
    detail: `top policy ${evaluation.legalPriors.slice(0, 5).map((p) => `${p.uci} ${(100 * p.prior).toFixed(1)}%`).join(', ') || '—'}`,
  });
}

function recordLc0SearchOutput(engineId: string, engineName: string, result: Lc0SearchResult): void {
  const stats = result.search.stats;
  const rootWdl = result.search.root?.evaluation?.wdl;
  recordEngineOutput({
    engineId,
    engineName,
    kind: 'lc0',
    fen: result.fen,
    move: result.move,
    summary: rootWdl ? searchWdlText(rootWdl, result.value, result.fen) : `search Q ${signed(toWhiteQ(result.value, result.fen), 3)}`,
    shortEval: rootWdl ? lc0WdlCompact(rootWdl, result.value, result.fen) : `Q ${signed(toWhiteQ(result.value, result.fen), 2)}`,
    evalBar: rootWdl ? lc0EvalBar(result.fen, rootWdl) : qEvalBar(result.fen, result.value),
    detail: `visits ${result.visits} · evals ${stats?.evalCalls ?? 0} · cache hits ${stats?.cacheHits ?? 0}`,
    pv: result.pv,
  });
}

function recordBt4SearchOutput(engineId: string, engineName: string, result: Bt4SearchResult): void {
  recordEngineOutput({
    engineId,
    engineName,
    kind: 'lc0',
    fen: result.fen,
    move: result.move ?? undefined,
    summary: `search Q ${signed(toWhiteQ(result.value, result.fen), 3)}`,
    shortEval: `Q ${signed(toWhiteQ(result.value, result.fen), 2)}`,
    evalBar: qEvalBar(result.fen, result.value),
    detail: `visits ${result.visits} · evals ${result.stats?.evalCalls ?? 0} · cache hits ${result.stats?.cacheHits ?? 0}`,
    pv: result.pv,
  });
}

function recordUciOutput(engineId: string, engineName: string, label: string, fen: string, move: string | null, lines: StockfishInfoLine[]): void {
  const best = lines[0];
  recordEngineOutput({
    engineId,
    engineName,
    kind: 'uci',
    fen,
    move: move ?? undefined,
    summary: `${label} ${stockfishScoreText(best, fen)}`,
    shortEval: stockfishScoreCompact(best, fen),
    evalBar: stockfishEvalBar(fen, best),
    detail: lines.length > 1 ? `MultiPV ${lines.slice(0, 3).map((line) => `#${line.multipv} ${stockfishScoreText(line, fen)}`).join(' · ')}` : undefined,
    pv: best?.pvUci,
  });
}

function recordStockfishOutput(engineId: string, engineName: string, fen: string, move: string | null, lines: StockfishInfoLine[]): void {
  recordUciOutput(engineId, engineName, 'SF', fen, move, lines);
}

function recordRecklessOutput(engineId: string, engineName: string, fen: string, move: string | null, lines: StockfishInfoLine[]): void {
  recordUciOutput(engineId, engineName, 'Reckless', fen, move, lines);
}

function recordViridithasOutput(engineId: string, engineName: string, fen: string, move: string | null, lines: StockfishInfoLine[]): void {
  recordUciOutput(engineId, engineName, 'Viridithas', fen, move, lines);
}

function recordEngineThinking(engine: ArenaEngine): void {
  thinkingEngineIds.add(engine.id);
  renderSideLabels();
  renderEngineOutputs();
}

function noteLc0MoveForReplyProbe(engine: ArenaEngine, move: Move, nextBoard: BoardState): void {
  if (!isLc0SearchEngine(engine.id)) return;
  const result = lastLc0SearchResults.get(engine.id);
  const child = childForRootMove(result, moveToUci(move));
  pendingLc0ReplyProbes.set(engine.id, { engineId: engine.id, engineName: engine.name, afterMoveFen: boardToFen(nextBoard), child });
}

function recordPendingLc0ReplyProbes(replyEngine: ArenaEngine, replyBoard: BoardState, replyMove: Move): void {
  const replyFen = boardToFen(replyBoard);
  const replyUci = moveToUci(replyMove);
  for (const [engineId, pending] of [...pendingLc0ReplyProbes]) {
    if (engineId === replyEngine.id || pending.afterMoveFen !== replyFen) continue;
    const t = telemetryFor(pending.engineId, pending.engineName);
    t.replyChecks += 1;
    const child = pending.child;
    if (child?.expanded) {
      t.replyParentsExpanded += 1;
      const policyRank = child.edges.slice().sort((a, b) => b.prior - a.prior).findIndex((edge) => moveToUci(edge.move) === replyUci);
      const visitsRank = child.edges.slice().sort((a, b) => b.visits - a.visits || b.prior - a.prior).findIndex((edge) => moveToUci(edge.move) === replyUci);
      const replyEdge = child.edges.find((edge) => moveToUci(edge.move) === replyUci);
      if (replyEdge && replyEdge.visits > 0) t.replyVisited += 1;
      if (policyRank >= 0 && policyRank < 5) t.replyTopPolicy5 += 1;
      if (visitsRank >= 0 && visitsRank < 5) t.replyTopVisits5 += 1;
    }
    pendingLc0ReplyProbes.delete(engineId);
  }
  renderSearchTelemetryInfo();
}

function threadedStockfishAvailable(): boolean {
  return runtimeIsolation && runtimeSharedArrayBuffer;
}

function stockfishFlavorFor(kind: 'lite' | 'full'): StockfishFlavor {
  const threaded = threadedStockfishAvailable() && stockfishThreads() > 1;
  if (kind === 'lite') return threaded ? 'lite-threaded' : 'lite-single';
  return threaded ? 'threaded' : 'single';
}

// Stockfish Lite and full are independent engines (they may face each other), so
// each has its own lazily-created instance.
function stockfishEngineFor(kind: 'lite' | 'full'): StockfishEngine {
  if (kind === 'lite') {
    if (!stockfishLite) stockfishLite = new StockfishEngine({ depth: 4, threads: stockfishThreads() }, stockfishFlavorUrl(stockfishFlavorFor('lite')));
    return stockfishLite;
  }
  if (!stockfishFull) stockfishFull = new StockfishEngine({ depth: 4, threads: stockfishThreads() }, stockfishFlavorUrl(stockfishFlavorFor('full')));
  return stockfishFull;
}

function disposeStockfish(): void {
  stockfishLite?.dispose();
  stockfishLite = null;
  stockfishFull?.dispose();
  stockfishFull = null;
}

function availableRecklessVariants(): RecklessVariant[] {
  return REQUESTED_RECKLESS_VARIANT.key === 'custom' ? [...RECKLESS_VARIANTS, REQUESTED_RECKLESS_VARIANT] : [...RECKLESS_VARIANTS];
}

function recklessVariantForKey(variantKey: string): RecklessVariant {
  const key = normalizeRecklessVariant(variantKey);
  if (key === 'custom' && REQUESTED_RECKLESS_VARIANT.key === 'custom') return REQUESTED_RECKLESS_VARIANT;
  return recklessVariantByKey(key);
}

function recklessCacheKey(variant: RecklessVariant): string {
  return `${variant.key}:${variant.wasmUrl}:${variant.nnueUrl ?? ''}:${variant.backend ?? 'wasi'}`;
}

function getRecklessFor(variantKey: string): RecklessEngine {
  const variant = recklessVariantForKey(variantKey);
  const key = recklessCacheKey(variant);
  let engine = recklessByVariant.get(key);
  if (!engine) {
    engine = new RecklessEngine({ depth: 4, hashMb: 16 }, variant.wasmUrl, { backend: variant.backend ?? 'wasi', nnueUrl: variant.nnueUrl, onStatus: renderRecklessRuntimeInfo });
    recklessByVariant.set(key, engine);
  }
  return engine;
}

function prewarmReckless(engine: RecklessEngine): void {
  void engine.prewarm()
    .then(renderRecklessRuntimeInfo)
    .catch((error) => {
      if ((error as Error).name !== 'AbortError') console.warn('Reckless prewarm failed', error);
      renderRecklessRuntimeInfo();
    });
}

function renderRecklessRuntimeInfo(): void {
  const rows = activeSeatRows().filter((row) => row.family === 'reckless');
  if (!rows.length) { el('recklessRuntimeInfo').textContent = 'Reckless: not selected'; return; }
  const sab = typeof SharedArrayBuffer !== 'undefined' ? 'SAB yes' : 'SAB no';
  const parts = rows.map((row) => {
    const variant = recklessVariantForKey(row.variant);
    const engine = recklessByVariant.get(recklessCacheKey(variant));
    const status = engine?.runtimeStatus();
    const mode = engine?.runtimeLabel() ?? (typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated ? 'persistent available' : 'one-shot fallback');
    const asset = recklessVariantAssetStatus(variant);
    if (asset === 'unknown') void checkRecklessVariantAsset(variant, renderRecklessRuntimeInfo);
    const assetText = asset === 'present' ? 'asset ok' : asset === 'missing' ? 'asset missing' : 'checking asset';
    const loadText = formatRecklessBrowserApiLoadStatus(status?.browserApiLoad);
    return `${variant.label} d${row.strength} · ${mode} · ${sab} · ${assetText}${loadText ? ` · ${loadText}` : ''}${status?.persistentDisabled ? ' · persistent disabled after fallback' : ''}`;
  });
  el('recklessRuntimeInfo').textContent = `Reckless: ${[...new Set(parts)].join(' | ')}`;
}

function refreshRecklessVariantUi(): void {
  const select = selectEl('recklessVariantSelect');
  if (!select.options.length) {
    select.innerHTML = availableRecklessVariants().map((variant) => `<option value="${variant.key}">${htmlEscape(variant.label)}</option>`).join('');
  }
  select.disabled = running;
  renderRecklessRuntimeInfo();
}

function availableViridithasVariants(): ViridithasVariant[] {
  return REQUESTED_VIRIDITHAS_VARIANT.key === 'custom' ? [...VIRIDITHAS_VARIANTS, REQUESTED_VIRIDITHAS_VARIANT] : [...VIRIDITHAS_VARIANTS];
}

function viridithasVariantForKey(variantKey: string): ViridithasVariant {
  const key = normalizeViridithasVariant(variantKey);
  if (key === 'custom' && REQUESTED_VIRIDITHAS_VARIANT.key === 'custom') return REQUESTED_VIRIDITHAS_VARIANT;
  return viridithasVariantByKey(key);
}

function viridithasCacheKey(variant: ViridithasVariant): string {
  return `${variant.key}:${variant.wasmUrl}`;
}

function getViridithasFor(variantKey: string): ViridithasEngine {
  const variant = viridithasVariantForKey(variantKey);
  const key = viridithasCacheKey(variant);
  let engine = viridithasByVariant.get(key);
  if (!engine) {
    engine = new ViridithasEngine({ depth: 4, hashMb: 16 }, variant.wasmUrl);
    viridithasByVariant.set(key, engine);
  }
  return engine;
}

function renderViridithasRuntimeInfo(): void {
  const rows = activeSeatRows().filter((row) => row.family === 'viridithas');
  if (!rows.length) { el('viridithasRuntimeInfo').textContent = 'Viridithas: not selected'; return; }
  const sab = typeof SharedArrayBuffer !== 'undefined' ? 'SAB yes' : 'SAB no';
  const parts = rows.map((row) => {
    const variant = viridithasVariantForKey(row.variant);
    const engine = viridithasByVariant.get(viridithasCacheKey(variant));
    const status = engine?.runtimeStatus();
    const mode = engine?.runtimeLabel() ?? (canUsePersistentViridithasWasi() ? 'persistent available' : 'one-shot fallback');
    const asset = viridithasVariantAssetStatus(variant);
    if (asset === 'unknown') void checkViridithasVariantAsset(variant, renderViridithasRuntimeInfo);
    const assetText = asset === 'ok' ? 'asset ok' : asset === 'missing' ? 'asset missing' : 'checking asset';
    return `${variant.label} d${row.strength} · ${mode} · ${sab} · ${assetText}${status?.persistentDisabled ? ' · persistent disabled after fallback' : ''}`;
  });
  el('viridithasRuntimeInfo').textContent = `Viridithas: ${[...new Set(parts)].join(' | ')}`;
}

function refreshViridithasVariantUi(): void {
  const select = selectEl('viridithasVariantSelect');
  if (!select.options.length) {
    select.innerHTML = availableViridithasVariants().map((variant) => `<option value="${variant.key}">${htmlEscape(variant.label)}</option>`).join('');
  }
  select.disabled = running;
  renderViridithasRuntimeInfo();
}

function refreshStockfishControls(): void {
  inputEl('stockfishThreadsInput').disabled = running || !threadedStockfishAvailable();
  inputEl('stockfishThreadsInput').value = String(stockfishThreads());
}

// Lc0 BT4 is WebGPU-only; disable/downgrade its staged option when WebGPU is unusable.
async function refreshBt4Availability(): Promise<void> {
  const ok = await probeBt4Support();
  if (!ok) {
    for (const row of activeSeatRows()) if (row.family === 'lc0' && row.variant === 'bt4') row.variant = 'small';
    buildEngines();
    populateSeats();
  } else {
    renderSeatSelectors();
  }
  refreshSeatControls();
}

async function renderRuntimeBadge(): Promise<void> {
  const badge = el('runtimeBadge');
  try {
    const diag = await collectOrtRuntimeDiagnostics({ probeAdapter: true });
    runtimeIsolation = diag.crossOriginIsolated === true;
    runtimeSharedArrayBuffer = diag.wasm.sharedArrayBuffer === true;
    refreshStockfishControls();
    const webgpu = diag.webgpuAvailable ? (diag.adapter?.ok === false ? 'WebGPU unavailable/blocked' : 'WebGPU available') : 'WebGPU unavailable';
    const isolated = runtimeIsolation ? 'isolated' : 'not isolated';
    const sab = runtimeSharedArrayBuffer ? 'SAB yes' : 'SAB no';
    const sfThreads = threadedStockfishAvailable() ? 'SF threaded yes' : 'SF threaded no';
    const ortThreads = `ORT wasm threads ${diag.wasm.numThreads ?? '?'}`;
    const ortSessions = `ORT sessions ${diag.sessions.active} active`;
    badge.textContent = `Runtime: ${isolated} · ${sab} · ${webgpu} · ${diag.describe} · ${ortThreads} · ${ortSessions} · ${sfThreads}`;
    badge.classList.toggle('ready', runtimeIsolation && (diag.webgpuAvailable || runtimeSharedArrayBuffer));
    badge.classList.toggle('warn', !runtimeIsolation || !diag.webgpuAvailable);
  } catch (error) {
    badge.textContent = `Runtime detection failed: ${(error as Error).message}`;
    badge.classList.add('warn');
  }
}

function lc0SearcherFor(engineId: string): Lc0PuctSearcher {
  const existing = lc0Searchers.get(engineId);
  if (existing) return existing;
  if (!lc0Cache) throw new Error('LC0 evaluator is not initialized');
  const created = new Lc0PuctSearcher(lc0Cache);
  lc0Searchers.set(engineId, created);
  return created;
}

function resetLc0SearchTrees(ids?: string[]): void {
  const targets = ids ? ids.map((id) => lc0Searchers.get(id)).filter((entry): entry is Lc0PuctSearcher => !!entry) : [...lc0Searchers.values()];
  for (const search of targets) search.resetTree();
}

function pruneUnusedLc0Searchers(activeIds: Set<string>): void {
  for (const id of lc0Searchers.keys()) {
    if (activeIds.has(id)) continue;
    lc0Searchers.delete(id);
    lastLc0SearchResults.delete(id);
    lc0TreeTelemetry.delete(id);
  }
}

function disposeUnusedUciEngines(): void {
  const activeReckless = new Set(activeSeatRows().filter((row) => row.family === 'reckless').map((row) => recklessCacheKey(recklessVariantForKey(row.variant))));
  for (const [key, engine] of recklessByVariant) {
    if (!activeReckless.has(key)) { engine.dispose(); recklessByVariant.delete(key); }
  }
  const activeViridithas = new Set(activeSeatRows().filter((row) => row.family === 'viridithas').map((row) => viridithasCacheKey(viridithasVariantForKey(row.variant))));
  for (const [key, engine] of viridithasByVariant) {
    if (!activeViridithas.has(key)) { engine.dispose(); viridithasByVariant.delete(key); }
  }
}

function buildEngines() {
  for (const row of activeSeatRows()) clampStrength(row);
  const activeIds = new Set(activeSeatRows().map(engineIdForRow));
  pruneUnusedLc0Searchers(activeIds);
  engines.clear();
  disposeUnusedUciEngines();
  const warmupPositions = [parseFen(START_FEN)];
  const lc0Search = (engineId: string, row: EngineRow): ArenaEngine['move'] => async (positions, signal) => {
    const timed = arenaBudgetMode() === 'movetime';
    const result = await lc0SearcherFor(engineId).search({ positions }, {
      visits: timed ? undefined : row.strength,
      movetimeMs: timed ? arenaMovetimeMs() : undefined,
      signal,
      yieldEveryMs: 16,
      reuseTree: true,
    });
    const engineName = engines.get(engineId)?.name ?? engineId;
    recordLc0SearchTelemetry(engineId, engineName, result);
    recordLc0SearchOutput(engineId, engineName, result);
    return result.move ?? null;
  };
  const lc0Bt4Move = (engineId: string, row: EngineRow): ArenaEngine['move'] => async (positions, signal) => {
    const onAbort = () => bt4.cancel();
    signal.addEventListener('abort', onAbort, { once: true });
    try {
      const timed = arenaBudgetMode() === 'movetime';
      const result = await bt4.search({ positions }, {
        visits: timed ? undefined : row.strength,
        movetimeMs: timed ? arenaMovetimeMs() : undefined,
        reuseTree: true,
      });
      if (result.cancelled) return null;
      recordBt4SearchOutput(engineId, engines.get(engineId)?.name ?? 'Lc0 BT4', result);
      return result.move ?? null;
    } finally {
      signal.removeEventListener('abort', onAbort);
    }
  };
  const sf = (engineId: string, row: EngineRow, kind: 'lite' | 'full'): ArenaEngine['move'] => async (positions, signal) => {
    const engine = stockfishEngineFor(kind);
    if (arenaBudgetMode() === 'movetime') engine.setOptions({ depth: undefined, movetimeMs: arenaMovetimeMs(), threads: stockfishThreads() });
    else engine.setOptions({ depth: row.strength, movetimeMs: undefined, threads: stockfishThreads() });
    const fen = boardToFen(positions[positions.length - 1]);
    const move = await engine.bestMove(fen, signal);
    recordStockfishOutput(engineId, engines.get(engineId)?.name ?? (kind === 'lite' ? 'Stockfish Lite' : 'Stockfish'), fen, move, engine.lastInfo());
    return move;
  };
  const recklessMove = (engineId: string, row: EngineRow, engine: RecklessEngine): ArenaEngine['move'] => async (positions, signal) => {
    if (arenaBudgetMode() === 'movetime') engine.setOptions({ depth: undefined, movetimeMs: arenaMovetimeMs() });
    else engine.setOptions({ depth: row.strength, movetimeMs: undefined });
    const fen = boardToFen(positions[positions.length - 1]);
    const move = await engine.bestMove(fen, signal);
    recordRecklessOutput(engineId, engines.get(engineId)?.name ?? 'Reckless', fen, move, engine.lastInfo());
    renderRecklessRuntimeInfo();
    return move;
  };
  const viridithasMove = (engineId: string, row: EngineRow, engine: ViridithasEngine): ArenaEngine['move'] => async (positions, signal) => {
    if (arenaBudgetMode() === 'movetime') engine.setOptions({ depth: undefined, movetimeMs: arenaMovetimeMs() });
    else engine.setOptions({ depth: row.strength, movetimeMs: undefined });
    const fen = boardToFen(positions[positions.length - 1]);
    const move = await engine.bestMove(fen, signal);
    recordViridithasOutput(engineId, engines.get(engineId)?.name ?? 'Viridithas', fen, move, engine.lastInfo());
    renderViridithasRuntimeInfo();
    return move;
  };
  const lc0SearchWarmup = (engineId: string) => async (signal: AbortSignal) => {
    const search = lc0SearcherFor(engineId);
    await search.search({ positions: warmupPositions }, { visits: 1, signal, yieldEveryMs: 16 });
    search.resetTree();
    renderCacheInfo();
  };
  const lc0Bt4Warmup = async (signal: AbortSignal) => {
    const onAbort = () => bt4.cancel();
    signal.addEventListener('abort', onAbort, { once: true });
    try {
      await bt4.search({ positions: warmupPositions }, { visits: 1 });
      await bt4.resetTree();
    } finally {
      signal.removeEventListener('abort', onAbort);
    }
  };
  const stockfishWarmup = (kind: 'lite' | 'full') => async (signal: AbortSignal) => {
    const engine = stockfishEngineFor(kind);
    engine.setOptions({ depth: 1, movetimeMs: undefined, threads: stockfishThreads() });
    await engine.bestMove(START_FEN, signal);
  };
  const recklessWarmup = (engine: RecklessEngine) => async (signal: AbortSignal) => {
    engine.setOptions({ depth: 1, movetimeMs: undefined });
    await engine.bestMove(START_FEN, signal);
    renderRecklessRuntimeInfo();
  };
  const viridithasWarmup = (engine: ViridithasEngine) => async (signal: AbortSignal) => {
    engine.setOptions({ depth: 1, movetimeMs: undefined });
    await engine.bestMove(START_FEN, signal);
    await engine.newGame(signal);
    renderViridithasRuntimeInfo();
  };
  for (const row of activeSeatRows()) {
    const id = engineIdForRow(row);
    if (engines.has(id)) continue;
    if (row.family === 'lc0') {
      if (row.variant === 'bt4') engines.set(id, { id, name: `${rowLabel(row)} v${row.strength}`, move: lc0Bt4Move(id, row), warmup: lc0Bt4Warmup });
      else engines.set(id, { id, name: `${rowLabel(row)} v${row.strength}`, move: lc0Search(id, row), warmup: lc0SearchWarmup(id) });
    } else if (row.family === 'sf') {
      const kind = row.variant === 'full' ? 'full' : 'lite';
      engines.set(id, { id, name: `${rowLabel(row)} d${row.strength}`, move: sf(id, row, kind), warmup: stockfishWarmup(kind) });
    } else if (row.family === 'reckless') {
      const engine = getRecklessFor(row.variant);
      prewarmReckless(engine);
      engines.set(id, { id, name: `${rowLabel(row)} d${row.strength}`, move: recklessMove(id, row, engine), warmup: recklessWarmup(engine) });
    } else {
      const engine = getViridithasFor(row.variant);
      engines.set(id, { id, name: `${rowLabel(row)} d${row.strength}`, move: viridithasMove(id, row, engine), warmup: viridithasWarmup(engine) });
    }
  }
  renderRecklessRuntimeInfo();
  renderViridithasRuntimeInfo();
}

function selectedOpenings(): ArenaOpening[] {
  const mode = selectEl('startingPositionSelect').value;
  if (mode === 'built-in') return BUILTIN_ARENA_OPENINGS;
  if (mode === 'custom') {
    const parsed = parseArenaOpenings((el('openingText') as HTMLTextAreaElement).value);
    if (!parsed.length) throw new Error('Add at least one custom FEN, or choose Start position.');
    return parsed;
  }
  return [{ name: 'Start position', fen: START_FEN }];
}

function openingHistoryBoards(opening: ArenaOpening): BoardState[] {
  if (opening.positions?.length) return [...opening.positions];
  return [parseFen(opening.fen)];
}

function gameTreeFromOpening(opening: ArenaOpening): GameTree {
  const tree = new GameTree(opening.startFen ?? opening.fen);
  for (const uci of opening.moves ?? []) {
    if (!tree.addUci(uci)) throw new Error(`Opening ${opening.name} contains illegal move ${uci}`);
  }
  return tree;
}

function openingPgnSetupTags(opening: ArenaOpening): Record<string, string> {
  if (opening.moves?.length) {
    const startFen = opening.startFen ?? START_FEN;
    return startFen !== START_FEN ? { SetUp: '1', FEN: startFen } : {};
  }
  return opening.fen !== START_FEN ? { SetUp: '1', FEN: opening.fen } : {};
}

function setOpeningPreview(opening: ArenaOpening): void {
  historyBoards = openingHistoryBoards(opening);
  board = historyBoards[historyBoards.length - 1];
  lastUci = null;
  setBoardSideEngines(null, null, null, null);
  renderBoard();
}

function refreshOpeningPreview(): void {
  const select = selectEl('startingPositionSelect');
  const textarea = el('openingText') as HTMLTextAreaElement;
  const custom = select.value === 'custom';
  select.disabled = running;
  textarea.disabled = running || !custom;
  if (running) return;
  try {
    const openings = selectedOpenings();
    el('openingInfo').textContent = `${openings.length} starting position${openings.length === 1 ? '' : 's'} · each pair plays every selected position with the configured color schedule.`;
    setOpeningPreview(openings[0]);
  } catch (error) {
    el('openingInfo').textContent = `Opening setup error: ${(error as Error).message}`;
  }
}

function formatScoreHalf(value: number): string {
  return value % 1 ? `${Math.floor(value) || ''}½` : String(value);
}

function renderMatchScore(nameA: string, nameB: string, sameEngine: boolean, score: MatchScore): void {
  if (score.games === 0) { el('matchScore').textContent = 'No games played yet.'; return; }
  const games = `${score.games} game${score.games === 1 ? '' : 's'}`;
  el('matchScore').textContent = sameEngine
    ? `${nameA} mirror · ${score.aWins}–${score.bWins}–${score.draws} (W–L–D, Engine 1 seat) over ${games}`
    : `${nameA} ${formatScoreHalf(score.a)} – ${formatScoreHalf(score.b)} ${nameB} · ${score.aWins}W ${score.draws}D ${score.bWins}L over ${games}`;
}

function appendLog(text: string) {
  const div = document.createElement('div');
  div.textContent = text;
  el('log').prepend(div);
}

function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    if (ms <= 0 || signal.aborted) return resolve();
    const timer = setTimeout(resolve, ms);
    signal.addEventListener('abort', () => { clearTimeout(timer); resolve(); }, { once: true });
  });
}

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === 'AbortError';
}

async function warmUpSelectedEngines(ids: string[], signal: AbortSignal): Promise<void> {
  const warmed = new Set<string>();
  for (const id of ids) {
    if (signal.aborted) return;
    const engine = engines.get(id);
    if (!engine?.warmup || warmed.has(id)) continue;
    el('message').textContent = `Warming up ${engine.name}…`;
    await engine.warmup(signal);
    warmed.add(id);
    renderCacheInfo();
  }
}

function legalFromUci(current: BoardState, uci: string | null): Move | undefined {
  return uci ? legalMoves(current).find((m) => moveToUci(m) === uci) : undefined;
}

async function playArenaGame(white: ArenaEngine, black: ArenaEngine, opening: ArenaOpening, signal: AbortSignal): Promise<{ result: GameResultCode; reason: string; tree: GameTree }> {
  pendingLc0ReplyProbes.clear();
  const tree = gameTreeFromOpening(opening);
  historyBoards = tree.historyBoards();
  board = historyBoards[historyBoards.length - 1];
  lastUci = null;
  activeEngineIds = [white.id, black.id];
  engineOutputs.delete(white.id);
  engineOutputs.delete(black.id);
  thinkingEngineIds.delete(white.id);
  thinkingEngineIds.delete(black.id);
  setBoardSideEngines(white.id, white.name, black.id, black.name);
  renderBoard();
  renderEngineOutputs();
  const priorFens: string[] = historyBoards.slice(0, -1).map(boardToFen);
  const delay = Math.max(0, Math.floor(Number(inputEl('delayInput').value) || 0));
  for (let ply = 0; ply < 300; ply++) {
    if (signal.aborted) return { result: '1/2-1/2', reason: 'cancelled', tree };
    const outcome = gameOutcome(board, priorFens);
    if (outcome) return { ...outcome, tree };
    const engine = board.turn === 'w' ? white : black;
    recordEngineThinking(engine);
    let uci: string | null;
    try {
      uci = await engine.move(historyBoards, signal);
    } catch (error) {
      if (isAbortError(error)) return { result: '1/2-1/2', reason: 'cancelled', tree };
      throw error;
    }
    if (signal.aborted) return { result: '1/2-1/2', reason: 'cancelled', tree };
    const move = legalFromUci(board, uci);
    if (!move) return { result: board.turn === 'w' ? '0-1' : '1-0', reason: uci ? `illegal ${uci}` : 'resigned', tree };
    recordPendingLc0ReplyProbes(engine, board, move);
    priorFens.push(boardToFen(board));
    board = makeMove(board, move);
    noteLc0MoveForReplyProbe(engine, move, board);
    historyBoards.push(board);
    lastUci = moveToUci(move);
    tree.addUci(lastUci);
    renderBoard();
    await sleep(delay, signal);
  }
  return { result: '1/2-1/2', reason: 'max plies', tree };
}

async function startMatch() {
  if (running) return;
  syncSeatRowsFromDom();
  buildEngines();
  populateSeats();
  const idA = seatEngineId('A');
  const idB = seatEngineId('B');
  const engineA = engines.get(idA);
  const engineB = engines.get(idB);
  if (!engineA || !engineB) { el('message').textContent = 'Pick two engines.'; return; }
  const usesBt4 = activeSeatRows().some((row) => row.family === 'lc0' && row.variant === 'bt4');
  if (usesBt4 && !(await probeBt4Support())) {
    el('message').textContent = 'Lc0 BT4 needs WebGPU, which is unavailable in this browser.';
    return;
  }
  const sameEngine = idA === idB;
  const seatIds = sameEngine ? [idA] : [idA, idB];
  const gamesPerOpening = Math.max(1, Math.floor(Number(inputEl('gamesInput').value) || 2));
  let schedule: MatchGame[];
  try {
    const openings = selectedOpenings();
    schedule = [];
    // Alternate colors each game so a full set is color-balanced.
    for (let g = 0; g < gamesPerOpening; g++) {
      for (const opening of openings) schedule.push({ whiteSeat: g % 2 === 0 ? 'A' : 'B', opening });
    }
  } catch (error) {
    el('message').textContent = `Opening setup error: ${(error as Error).message}`;
    return;
  }
  if (!schedule.length) { el('message').textContent = 'No games to play.'; return; }

  running = true;
  abort = new AbortController();
  refreshOpeningPreview();
  refreshStockfishControls();
  refreshRecklessVariantUi();
  refreshViridithasVariantUi();
  refreshSeatControls();
  games.length = 0;
  activeEngineIds = [];
  engineOutputs.clear();
  thinkingEngineIds.clear();
  lastLc0SearchResults.clear();
  pendingLc0ReplyProbes.clear();
  lc0TreeTelemetry.clear();
  el('log').innerHTML = '';
  renderEngineOutputs();
  renderSearchTelemetryInfo();
  el('start').toggleAttribute('disabled', true);
  el('stop').toggleAttribute('disabled', false);
  const score: MatchScore = { a: 0, b: 0, aWins: 0, bWins: 0, draws: 0, games: 0 };
  renderMatchScore(engineA.name, engineB.name, sameEngine, score);
  try {
    resetLc0SearchTrees(seatIds);
    await warmUpSelectedEngines(seatIds, abort.signal);
    if (abort.signal.aborted) return;
    for (let i = 0; i < schedule.length; i++) {
      if (abort.signal.aborted) break;
      const { whiteSeat, opening } = schedule[i];
      const whiteEngine = whiteSeat === 'A' ? engineA : engineB;
      const blackEngine = whiteSeat === 'A' ? engineB : engineA;
      // Reset trees per game (fresh game tree); within a game the shared tree is
      // reused across both sides' plies — i.e. self-play when both seats match.
      resetLc0SearchTrees(seatIds);
      if (usesBt4 && bt4.loaded) await bt4.resetTree();
      for (const engine of viridithasByVariant.values()) await engine.newGame(abort.signal);
      setBoardSideEngines(whiteEngine.id, whiteEngine.name, blackEngine.id, blackEngine.name);
      el('pairing').textContent = `Game ${i + 1}/${schedule.length}: ${whiteEngine.name} (W) vs ${blackEngine.name} (B) · ${opening.name}`;
      el('message').textContent = 'Playing…';
      const { result, reason, tree } = await playArenaGame(whiteEngine, blackEngine, opening, abort.signal);
      if (reason === 'cancelled') break;
      score.games += 1;
      if (result === '1/2-1/2') { score.draws += 1; score.a += 0.5; score.b += 0.5; }
      else {
        const winnerIsA = (whiteSeat === 'A') === (result === '1-0');
        if (winnerIsA) { score.a += 1; score.aWins += 1; } else { score.b += 1; score.bWins += 1; }
      }
      const tags: Record<string, string> = { Event: 'LC0 arena', White: whiteEngine.name, Black: blackEngine.name, Opening: opening.name, ...openingPgnSetupTags(opening) };
      games.push({ pgn: gameTreeToPgn(tree, tags, result) });
      renderCacheInfo();
      appendLog(`${i + 1}. ${whiteEngine.name} vs ${blackEngine.name} [${opening.name}]: ${result} (${reason}) · ${cacheMetricsText(lc0Cache?.metrics())} · ${searchTelemetryText()}`);
      renderMatchScore(engineA.name, engineB.name, sameEngine, score);
    }
    el('message').textContent = abort.signal.aborted ? `Stopped after ${score.games} game(s).` : `Match done (${score.games} game${score.games === 1 ? '' : 's'}).`;
    el('pairing').textContent = abort.signal.aborted ? 'Match stopped.' : 'Match finished.';
  } catch (error) {
    if (isAbortError(error) || abort?.signal.aborted) {
      el('message').textContent = `Stopped after ${score.games} game(s).`;
      el('pairing').textContent = 'Match stopped.';
    } else {
      el('message').textContent = `Match failed: ${(error as Error).message}`;
    }
  } finally {
    running = false;
    abort = null;
    activeEngineIds = [];
    engineOutputs.clear();
    thinkingEngineIds.clear();
    el('start').toggleAttribute('disabled', false);
    el('stop').toggleAttribute('disabled', true);
    refreshOpeningPreview();
    refreshStockfishControls();
    refreshRecklessVariantUi();
    refreshViridithasVariantUi();
    refreshSeatControls();
  }
}

function exportPgn() {
  inputEl('pgnOut').value = games.map((g) => g.pgn).join('\n\n');
  el('message').textContent = games.length ? `Exported ${games.length} game(s) as PGN.` : 'No games to export yet.';
}

function disposeLc0Resources(): void {
  void lc0Cache?.dispose();
  lc0Cache = null;
  player = null;
  searcher = null;
  lc0Searchers.clear();
  lastLc0SearchResults.clear();
  pendingLc0ReplyProbes.clear();
  engineOutputs.clear();
  thinkingEngineIds.clear();
  activeEngineIds = [];
}

function disposeRuntimeResources(): void {
  abort?.abort();
  abort = null;
  disposeLc0Resources();
  disposeStockfish();
  for (const engine of recklessByVariant.values()) engine.dispose();
  recklessByVariant.clear();
  for (const engine of viridithasByVariant.values()) engine.dispose();
  viridithasByVariant.clear();
  bt4.dispose();
}

async function createSelectedLc0Evaluator(): Promise<Lc0OnnxEvaluator | Lc0WebHybridEvaluator> {
  const runtime = selectedLc0Runtime();
  if (runtime === 'onnx') {
    const modelLoad = await loadLc0ModelForOrt(MODEL_URL, { cache: false });
    return Lc0OnnxEvaluator.create(modelLoad.model);
  }
  return new Lc0WebHybridEvaluator({
    packUrl: PACK_URL,
    layers: lc0EncoderLayers(),
    verifyShards: params.get('packVerify') !== '0',
    headBackend: runtime === 'hybrid-wgsl-heads' ? 'wgsl' : 'ort',
    wgslBatchMode: 'physical',
    inputBackend: 'js',
    legalPriorsBackend: 'js',
  });
}

async function loadLc0Evaluator(): Promise<void> {
  const runtime = selectedLc0Runtime();
  loadingLc0 = true;
  el('start').toggleAttribute('disabled', true);
  refreshSeatControls();
  el('message').textContent = `Loading LC0 ${lc0RuntimeLabel(runtime)}…`;
  try {
    const evaluator = await createSelectedLc0Evaluator();
    lc0Cache = new CachedLc0Evaluator(evaluator, { maxEntries: arenaCacheEntries() });
    lc0Searchers.clear();
    player = new Lc0PolicyOnlyPlayer(lc0Cache);
    searcher = new Lc0PuctSearcher(lc0Cache);
    renderCacheInfo();
    el('start').toggleAttribute('disabled', false);
    el('message').textContent = `Ready (${lc0RuntimeLabel(runtime)}). Pick engines and start a tournament.`;
  } catch (error) {
    el('message').textContent = `LC0 ${lc0RuntimeLabel(runtime)} load failed: ${(error as Error).message}`;
  } finally {
    loadingLc0 = false;
    refreshSeatControls();
  }
}

async function reloadLc0Evaluator(): Promise<void> {
  abort?.abort();
  disposeLc0Resources();
  renderCacheInfo();
  await loadLc0Evaluator();
}

function wireEvents() {
  el('start').addEventListener('click', () => { void startMatch(); });
  el('stop').addEventListener('click', () => { abort?.abort(); el('message').textContent = 'Stopping…'; });
  el('exportPgn').addEventListener('click', exportPgn);
  el('arenaSeatList').addEventListener('change', (event) => {
    if (running) return;
    const target = event.target as HTMLInputElement | HTMLSelectElement;
    const seat = target.dataset.seat as SeatId | undefined;
    if (seat !== 'A' && seat !== 'B') return;
    const row = seatRows[seat];
    if (target.classList.contains('seat-fam')) {
      row.family = target.value as EngineFamily;
      row.variant = defaultVariant(row.family);
      row.strength = strengthMeta(row.family).def;
    } else if (target.classList.contains('seat-var')) {
      if (row.family === 'lc0' && target.value === 'bt4' && !window.confirm(`${bt4LoadWarning()}\n\nUse Lc0 BT4?`)) { target.value = row.variant; return; }
      row.variant = target.value;
    } else if (target.classList.contains('seat-strength')) {
      row.strength = Number(target.value);
      clampStrength(row);
    }
    if (!activeSeatRows().some((r) => r.family === 'lc0' && r.variant === 'bt4')) bt4.dispose();
    buildEngines();
    populateSeats();
  });
  el('arenaSeatList').addEventListener('input', (event) => {
    if (running) return;
    const target = event.target as HTMLInputElement;
    const seat = target.dataset.seat as SeatId | undefined;
    if ((seat === 'A' || seat === 'B') && target.classList.contains('seat-strength')) {
      seatRows[seat].strength = Number(target.value);
      clampStrength(seatRows[seat]);
      renderRecklessRuntimeInfo();
      renderViridithasRuntimeInfo();
    }
  });
  el('startingPositionSelect').addEventListener('change', refreshOpeningPreview);
  el('openingText').addEventListener('input', refreshOpeningPreview);
  el('cacheEntriesInput').addEventListener('input', () => { renderCacheInfo(); resetLc0SearchTrees(); });
  el('lc0RuntimeSelect').addEventListener('change', () => { if (!running) void reloadLc0Evaluator(); });
  el('budgetModeSelect').addEventListener('change', () => resetLc0SearchTrees());
  el('movetimeInput').addEventListener('input', () => resetLc0SearchTrees());
  el('stockfishThreadsInput').addEventListener('input', () => {
    if (running) return;
    inputEl('stockfishThreadsInput').value = String(stockfishThreads());
    // Threads flips single<->threaded flavor (different wasm); rebuild on next use.
    disposeStockfish();
  });
  window.addEventListener('pagehide', (event) => {
    if (!(event as PageTransitionEvent).persisted) disposeRuntimeResources();
  });
}

async function init() {
  REQUESTED_RECKLESS_VARIANT = await resolveDefaultRecklessVariantAssetFallback(REQUESTED_RECKLESS_VARIANT, REQUESTED_RECKLESS_EXPLICIT, renderRecklessRuntimeInfo);
  renderBoard();
  selectEl('lc0RuntimeSelect').value = initialLc0Runtime();
  refreshRecklessVariantUi();
  refreshViridithasVariantUi();
  selectEl('recklessVariantSelect').value = REQUESTED_RECKLESS_VARIANT.key;
  selectEl('viridithasVariantSelect').value = REQUESTED_VIRIDITHAS_VARIANT.key;
  renderRecklessRuntimeInfo();
  renderViridithasRuntimeInfo();
  inputEl('stockfishThreadsInput').value = String(Math.max(1, Math.min(32, Math.floor(Number(params.get('sfThreads') ?? '1') || 1))));
  refreshStockfishControls();
  void renderRuntimeBadge();
  buildEngines();
  populateSeats();
  void refreshBt4Availability();
  void probeEngineLogos();
  wireEvents();
  refreshOpeningPreview();
  await loadLc0Evaluator();
  void renderRuntimeBadge();
}

void init();
