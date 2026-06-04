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
  /** Score from the side-to-move perspective, in centipawns (mate encoded separately). */
  scoreCp?: number;
  mateIn?: number;
  scoreText: string;
  /** Depth (Stockfish) or visit count (LC0) summary. */
  detail: string;
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
 * side-to-move score. Mate maps to the extreme for the mating side.
 */
export function evalBarWhitePercent(scoreCp: number | undefined, mateIn: number | undefined, sideToMove: BoardState['turn']): number {
  let stmAdvantage: number;
  if (mateIn !== undefined) stmAdvantage = mateIn >= 0 ? 1 : -1;
  else if (scoreCp === undefined) stmAdvantage = 0;
  else stmAdvantage = (2 / (1 + Math.exp(-0.004 * scoreCp))) - 1; // logistic squash to [-1, 1]
  const whiteAdvantage = sideToMove === 'w' ? stmAdvantage : -stmAdvantage;
  return Math.max(0, Math.min(100, 50 + 50 * whiteAdvantage));
}

export type EngineColorKey = 'green' | 'blue' | 'red' | 'yellow';

export interface EngineBrushes {
  /** Chessground brush for the engine's best move. */
  primary: string;
  /** Chessground brush for the engine's alternative (lower MultiPV) moves. */
  alt: string;
  /** Hex matching the chessground brush, for legend/panel swatches. */
  swatch: string;
}

const ENGINE_BRUSHES: Record<EngineColorKey, EngineBrushes> = {
  green: { primary: 'green', alt: 'paleGreen', swatch: '#15781B' },
  blue: { primary: 'blue', alt: 'paleBlue', swatch: '#003088' },
  red: { primary: 'red', alt: 'paleRed', swatch: '#882020' },
  yellow: { primary: 'yellow', alt: 'paleGrey', swatch: '#e68f00' },
};

/** Stable color family per engine: LC0 green, Stockfish blue, Reckless red, others fall back. */
export function engineColorKey(engine: string): EngineColorKey {
  const lower = engine.toLowerCase();
  if (lower.startsWith('lc0')) return 'green';
  if (lower.startsWith('sf') || lower.startsWith('stockfish')) return 'blue';
  if (lower.startsWith('reckless')) return 'red';
  return 'yellow';
}

export function engineBrushes(engine: string): EngineBrushes {
  return ENGINE_BRUSHES[engineColorKey(engine)];
}

export interface StockfishInfoLineLike {
  multipv: number;
  depth: number;
  scoreCp?: number;
  mateIn?: number;
  pvUci: string[];
}

/** Build analysis lines from Stockfish MultiPV info for the given FEN. */
export function stockfishAnalysisLines(infos: StockfishInfoLineLike[], fen: string, engine = 'Stockfish'): AnalysisLine[] {
  const board = parseFen(fen);
  return infos
    .filter((info) => info.pvUci.length)
    .map((info) => ({
      engine,
      multipv: info.multipv,
      scoreCp: info.scoreCp,
      mateIn: info.mateIn,
      scoreText: formatScore(info.scoreCp, info.mateIn),
      detail: `depth ${info.depth}`,
      pvUci: info.pvUci,
      pvSan: uciLineToSan(board, info.pvUci, 12),
    }));
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
 * MultiPV line is scored from its root child's Q (side-to-move), with SAN PV.
 */
export function lc0AnalysisLines(result: Lc0SearchLike, fen: string, engine = 'LC0'): AnalysisLine[] {
  const board = parseFen(fen);
  const lines = (result.multiPv && result.multiPv.length ? result.multiPv : [result.pv]).filter((line) => line.length);
  return lines.map((pvUci, index) => {
    const rootUci = pvUci[0];
    const child = result.children.find((entry) => entry.uci === rootUci);
    // children[].q is already the move's value from the root mover's perspective
    // (edgeQForParentInNode), matching result.value; positive favors the mover.
    const q = child ? child.q : result.value;
    const visits = child?.visits ?? result.visits;
    const scoreCp = qToCentipawns(q);
    return {
      engine,
      multipv: index + 1,
      scoreCp,
      scoreText: formatCentipawns(scoreCp),
      detail: `${visits} visits`,
      pvUci,
      pvSan: uciLineToSan(board, pvUci, 12),
    };
  });
}
