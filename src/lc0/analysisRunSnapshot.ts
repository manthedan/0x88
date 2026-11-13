import { type BoardState, boardToFen } from '../chess/board.ts';
import { analysisResultCacheKey } from './analysisCache.ts';
import { type EngineRow, engineResourceProfile, isLc0BigNetVariant } from './engineCatalog.ts';
import type { PerformanceDial } from './resourceBroker.ts';

export type RunSnapshot = Readonly<{
  positionKey: string;
  fen: string;
  positions: readonly BoardState[];
  cpuPosition: Readonly<{ fen: string; moves: readonly string[] }>;
  rows: readonly Readonly<EngineRow>[];
  multiPv: number;
  runtime: string;
  backendGeneration: number;
  servedModels: Readonly<Record<string, string>>;
  broker: Readonly<{ dial: PerformanceDial; cpuBudget: number; participantsKey: string }>;
  selectionKey: string;
  rowKeys: readonly string[];
}>;

export function analysisRunBroker(rows: readonly EngineRow[], budget: { dial: PerformanceDial; cpuBudget: number }): RunSnapshot['broker'] {
  const participants = [...new Set(rows
    .filter((row) => engineResourceProfile(row.family).resourceClass === 'cpu')
    .map((row) => `${row.family}:${row.variant}`))].sort();
  return Object.freeze({ dial: budget.dial, cpuBudget: budget.cpuBudget, participantsKey: JSON.stringify(participants) });
}

type KeyInputs = Pick<RunSnapshot, 'fen' | 'positions' | 'multiPv' | 'runtime' | 'servedModels' | 'broker'>;

/** Shared by captured runs and the live view's cache lookup. */
export function analysisRunRowKey(input: KeyInputs, row: EngineRow): string {
  const runtime = row.family !== 'lc0'
    ? ''
    : isLc0BigNetVariant(row.variant)
      ? (input.servedModels[row.variant === 't3' ? 't3' : 'bt4'] ?? '')
      : `${input.runtime}|${input.servedModels.small ?? ''}`;
  const { dial, cpuBudget, participantsKey } = input.broker;
  const cpu = engineResourceProfile(row.family).resourceClass === 'cpu' ? JSON.stringify([dial, cpuBudget, JSON.parse(participantsKey)]) : '';
  return analysisResultCacheKey(input.fen, input.positions.map(boardToFen), row, { multiPv: input.multiPv, runtime, cpu });
}

/** Capture synchronously after preparation validation; no controls or workers are retained. */
export function captureAnalysisRun(input: Omit<RunSnapshot, 'positionKey' | 'selectionKey' | 'rowKeys'>): RunSnapshot {
  const positions = Object.freeze(input.positions.map((board) => {
    const squares = [...board.squares];
    Object.freeze(squares);
    return Object.freeze({ ...board, squares });
  }));
  const snapshot = {
    ...input,
    positions,
    positionKey: JSON.stringify([input.fen, positions.map(boardToFen)]),
    cpuPosition: Object.freeze({ fen: input.cpuPosition.fen, moves: Object.freeze([...input.cpuPosition.moves]) }),
    rows: Object.freeze(input.rows.map((row) => Object.freeze({ ...row }))),
    servedModels: Object.freeze({ ...input.servedModels }),
    broker: Object.freeze({ ...input.broker }),
  };
  const rowKeys = Object.freeze(snapshot.rows.map((row) => analysisRunRowKey(snapshot, row)));
  return Object.freeze({ ...snapshot, rowKeys, selectionKey: rowKeys.join('\u0001') });
}
