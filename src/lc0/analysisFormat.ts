import { parseFen, type BoardState } from '../chess/board.ts';
import { uciLineToSan } from '../chess/san.ts';

/**
 * Pure formatting helpers shared by the analysis UI: LC0 value <-> centipawn,
 * eval strings, eval-bar mapping, and analysis-line construction. Kept free of
 * DOM/engine state so they are unit-testable.
 */

export interface AnalysisLine {
  engine: string;
  /** 1-based MultiPV rank. */
  multipv: number;
  /** Score from White's perspective, in centipawns (mate encoded separately). */
  scoreCp?: number;
  mateIn?: number;
  scoreText: string;
  /** Depth (Stockfish) or visit count (LC0) summary. */
  detail: string;
  nodes?: number;
  nps?: number;
  pvUci: string[];
  pvSan: string;
}

// LC0's standard value->centipawn mapping (used by lc0 `WDL`/`Q` reporting).
const Q_TO_CP = 111.714640912;
const Q_TO_CP_K = 1.5620688421;

/** Convert an LC0 Q value in [-1, 1] (side-to-move perspective) to centipawns. */
export function qToCentipawns(q: number): number {
  const clamped = Math.max(-0.9999, Math.min(0.9999, q));
  return Math.round(Q_TO_CP * Math.tan(Q_TO_CP_K * clamped));
}

/** Format a centipawn score (side-to-move) as a signed pawn string, e.g. "+1.23". */
export function formatCentipawns(cp: number): string {
  const pawns = cp / 100;
  const sign = pawns > 0 ? '+' : pawns < 0 ? '' : '+';
  return `${sign}${pawns.toFixed(2)}`;
}

/** Format a score for display: mate takes priority over centipawns. */
export function formatScore(scoreCp: number | undefined, mateIn: number | undefined): string {
  if (mateIn !== undefined) return mateIn === 0 ? '#' : `#${mateIn > 0 ? '' : '-'}${Math.abs(mateIn)}`;
  if (scoreCp === undefined) return '—';
  return formatCentipawns(scoreCp);
}

/**
 * White-advantage percentage [0, 100] for a vertical eval bar, from a
 * White-perspective score. Mate maps to the extreme for the mating side.
 */
export function evalBarWhitePercent(scoreCp: number | undefined, mateIn: number | undefined): number {
  let whiteAdvantage: number;
  if (mateIn !== undefined) whiteAdvantage = mateIn >= 0 ? 1 : -1;
  else if (scoreCp === undefined) whiteAdvantage = 0;
  else whiteAdvantage = (2 / (1 + Math.exp(-0.004 * scoreCp))) - 1; // logistic squash to [-1, 1]
  return Math.max(0, Math.min(100, 50 + 50 * whiteAdvantage));
}

export type EngineColorKey = 'green' | 'blue' | 'red' | 'purple' | 'orange' | 'cyan' | 'pink' | 'yellow';

export interface EngineBrushes {
  /** Chessground brush for the engine's best move. */
  primary: string;
  /** Chessground brush for the engine's alternative (lower MultiPV) moves. */
  alt: string;
  /** Hex matching the chessground brush, for legend/panel swatches. */
  swatch: string;
}

export const ANALYSIS_DRAWABLE_BRUSHES = {
  green: { key: 'g', color: '#15781B', opacity: 1, lineWidth: 10 },
  paleGreen: { key: 'pg', color: '#15781B', opacity: 0.4, lineWidth: 15 },
  blue: { key: 'b', color: '#003088', opacity: 1, lineWidth: 10 },
  paleBlue: { key: 'pb', color: '#003088', opacity: 0.4, lineWidth: 15 },
  red: { key: 'r', color: '#882020', opacity: 1, lineWidth: 10 },
  paleRed: { key: 'pr', color: '#882020', opacity: 0.4, lineWidth: 15 },
  purple: { key: 'purple', color: '#68217a', opacity: 0.85, lineWidth: 10 },
  palePurple: { key: 'palePurple', color: '#68217a', opacity: 0.35, lineWidth: 15 },
  orange: { key: 'orange', color: '#c45a11', opacity: 0.85, lineWidth: 10 },
  paleOrange: { key: 'paleOrange', color: '#c45a11', opacity: 0.35, lineWidth: 15 },
  cyan: { key: 'cyan', color: '#007a87', opacity: 0.85, lineWidth: 10 },
  paleCyan: { key: 'paleCyan', color: '#007a87', opacity: 0.35, lineWidth: 15 },
  pink: { key: 'pink', color: '#b02080', opacity: 0.75, lineWidth: 10 },
  palePink: { key: 'palePink', color: '#b02080', opacity: 0.3, lineWidth: 15 },
  yellow: { key: 'y', color: '#e68f00', opacity: 1, lineWidth: 10 },
  paleGrey: { key: 'pgr', color: '#4a4a4a', opacity: 0.35, lineWidth: 15 },
} as const;

const ENGINE_BRUSHES: Record<EngineColorKey, EngineBrushes> = {
  green: { primary: 'green', alt: 'paleGreen', swatch: '#15781B' },
  blue: { primary: 'blue', alt: 'paleBlue', swatch: '#003088' },
  red: { primary: 'red', alt: 'paleRed', swatch: '#882020' },
  purple: { primary: 'purple', alt: 'palePurple', swatch: '#68217a' },
  orange: { primary: 'orange', alt: 'paleOrange', swatch: '#c45a11' },
  cyan: { primary: 'cyan', alt: 'paleCyan', swatch: '#007a87' },
  pink: { primary: 'pink', alt: 'palePink', swatch: '#b02080' },
  yellow: { primary: 'yellow', alt: 'paleGrey', swatch: '#e68f00' },
};

/** Stable color family per engine/family for multi-engine analysis overlays. */
export function engineColorKey(engine: string): EngineColorKey {
  const lower = engine.toLowerCase();
  if (lower.startsWith('lc0')) return 'green';
  if (lower.startsWith('sf') || lower.startsWith('stockfish')) return 'blue';
  if (lower.startsWith('reckless')) return 'red';
  if (lower.startsWith('viridithas')) return 'purple';
  if (lower.startsWith('berserk')) return 'orange';
  if (lower.startsWith('plentychess')) return 'cyan';
  return 'pink';
}

export function engineBrushes(engine: string): EngineBrushes {
  return ENGINE_BRUSHES[engineColorKey(engine)];
}

export interface StockfishInfoLineLike {
  multipv: number;
  depth: number;
  scoreCp?: number;
  mateIn?: number;
  nodes?: number;
  nps?: number;
  pvUci: string[];
}

function formatCompactNumber(value: number): string {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(value >= 10_000_000 ? 0 : 1)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(value >= 10_000 ? 0 : 1)}k`;
  return String(Math.round(value));
}

function stockfishDetail(info: StockfishInfoLineLike): string {
  const parts = [`depth ${info.depth}`];
  if (info.nodes !== undefined) parts.push(`${formatCompactNumber(info.nodes)} nodes`);
  if (info.nps !== undefined) parts.push(`${formatCompactNumber(info.nps)} nps`);
  return parts.join(' · ');
}

/** Build analysis lines from Stockfish MultiPV info for the given FEN. */
export function stockfishAnalysisLines(infos: StockfishInfoLineLike[], fen: string, engine = 'Stockfish'): AnalysisLine[] {
  const board = parseFen(fen);
  // UCI reports from the side-to-move POV; display from White's perspective.
  const w = board.turn === 'w' ? 1 : -1;
  return infos
    .filter((info) => info.pvUci.length)
    .map((info) => {
      const scoreCp = info.scoreCp === undefined ? undefined : w * info.scoreCp;
      const mateIn = info.mateIn === undefined ? undefined : w * info.mateIn;
      return {
        engine,
        multipv: info.multipv,
        scoreCp,
        mateIn,
        scoreText: formatScore(scoreCp, mateIn),
        detail: stockfishDetail(info),
        ...(info.nodes !== undefined ? { nodes: info.nodes } : {}),
        ...(info.nps !== undefined ? { nps: info.nps } : {}),
        pvUci: info.pvUci,
        pvSan: uciLineToSan(board, info.pvUci, 12),
      };
    });
}

export interface Lc0SearchLike {
  value: number;
  multiPv?: string[][];
  pv: string[];
  children: { uci: string; visits: number; q: number }[];
  visits: number;
}

/**
 * Build analysis lines from an LC0 search result for the given root FEN. Each
 * MultiPV line is scored from its root child's Q, displayed from White's view.
 */
export function lc0AnalysisLines(result: Lc0SearchLike, fen: string, engine = 'LC0'): AnalysisLine[] {
  const board = parseFen(fen);
  const w = board.turn === 'w' ? 1 : -1;
  const lines = (result.multiPv && result.multiPv.length ? result.multiPv : [result.pv]).filter((line) => line.length);
  return lines.map((pvUci, index) => {
    const rootUci = pvUci[0];
    const child = result.children.find((entry) => entry.uci === rootUci);
    // children[].q is the move's value from the root mover's perspective
    // (edgeQForParentInNode), matching result.value; positive favors the mover.
    // Negate for Black so the displayed cp is from White's perspective.
    const q = (child ? child.q : result.value) * w;
    const visits = child?.visits ?? result.visits;
    const scoreCp = qToCentipawns(q);
    return {
      engine,
      multipv: index + 1,
      scoreCp,
      scoreText: formatCentipawns(scoreCp),
      detail: `${visits} visits`,
      nodes: visits,
      pvUci,
      pvSan: uciLineToSan(board, pvUci, 12),
    };
  });
}
