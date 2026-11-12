import { analysisResultCacheKey } from './analysisCache.ts';
import type { AnalysisLine } from './analysisFormat.ts';
import type { EngineRow } from './engineCatalog.ts';
import { criticalReviewPlies, type GameReview, type ReviewedMove } from './gameReview.ts';

/** A deferred interactive search belongs to one exact history/settings selection, not just a FEN. */
export class CriticalAnalysisRefresh {
  private pending?: { key: string; force: boolean };

  defer(key: string, force = false): void {
    this.pending = { key, force: force || (this.pending?.key === key && this.pending.force) };
  }

  navigate(key: string, hasSnapshot: boolean): void {
    if (hasSnapshot || this.pending?.key !== key) this.clear();
  }

  take(key: string): { force: boolean } | undefined {
    const pending = this.pending;
    this.clear();
    return pending?.key === key ? { force: pending.force } : undefined;
  }

  clear(): void {
    this.pending = undefined;
  }
}

export interface CriticalMoment {
  move: ReviewedMove;
  /** The position BEFORE the played error, with complete known history including it. */
  fen: string;
  history: string[];
}

export interface CriticalProvenance {
  engine: string;
  model: string;
  build: string;
  runtime: string;
  budget: string;
  options: string;
}

export interface CriticalResult {
  provenance: CriticalProvenance;
  /** Includes FEN, history, engine/model, budget and options; retained with every result. */
  cacheKey: string;
  line?: AnalysisLine;
  /** Legal-move probabilities only. Human expected points are deliberately not stored. */
  policy?: Record<string, number>;
  rating?: number;
}

export interface CriticalLane {
  id: string;
  provenance: CriticalProvenance;
  row: EngineRow;
  evaluate(moment: CriticalMoment, signal: AbortSignal, progress: (text: string) => void): Promise<CriticalResult>;
}

export interface CriticalCell {
  status: 'pending' | 'running' | 'complete' | 'failed' | 'cancelled';
  result?: CriticalResult;
  error?: string;
  cached?: boolean;
}

export interface CriticalReport {
  signature: string;
  baseline: string;
  moments: CriticalMoment[];
  lanes: { id: string; provenance: CriticalProvenance; completed: number; progress: string }[];
  cells: Record<string, CriticalCell[]>;
  state: 'running' | 'complete' | 'cancelled';
  stale: boolean;
}

export function criticalMoments(review: GameReview, fens: readonly string[]): CriticalMoment[] {
  return criticalReviewPlies(review, 5).map((ply) => {
    if (!fens[ply - 1]) throw new Error('Critical moment is missing its pre-move position.');
    return { move: { ...review.moves[ply - 1] }, fen: fens[ply - 1], history: fens.slice(0, ply) };
  });
}

export function criticalReportSignature(mainline: string, settings: unknown): string {
  return JSON.stringify([mainline, settings]);
}

export function criticalReportIsStale(report: Pick<CriticalReport, 'signature' | 'stale'>, signature: string): boolean {
  return report.stale || report.signature !== signature;
}

export function criticalMomentCacheKey(moment: CriticalMoment, lane: Pick<CriticalLane, 'row' | 'provenance'>): string {
  const p = lane.provenance;
  return analysisResultCacheKey(moment.fen, moment.history, lane.row, {
    multiPv: 1,
    runtime: JSON.stringify([p.engine, p.model, p.build, p.runtime, p.options, p.budget]),
    cpu: '', // Resource/thread options are explicit in provenance.options, not a mutable UI snapshot.
  });
}

export function createCriticalReport(signature: string, baseline: string, moments: CriticalMoment[], lanes: CriticalLane[]): CriticalReport {
  return {
    signature,
    baseline,
    moments,
    stale: false,
    state: 'running',
    lanes: lanes.map(({ id, provenance }) => ({ id, provenance: { ...provenance }, completed: 0, progress: 'Waiting' })),
    cells: Object.fromEntries(lanes.map(({ id }) => [id, moments.map(() => ({ status: 'pending' as const }))])),
  };
}

/** Serial within each engine; independent failures never discard another engine's results. */
export async function runCriticalReport(
  report: CriticalReport,
  lanes: CriticalLane[],
  cache: Map<string, CriticalResult>,
  signal: AbortSignal,
  changed: () => void,
): Promise<void> {
  await Promise.all(
    lanes.map(async (lane, laneIndex) => {
      const state = report.lanes[laneIndex];
      for (let i = 0; i < report.moments.length; i++) {
        if (signal.aborted) break;
        const cell = report.cells[lane.id][i];
        const moment = report.moments[i];
        const key = criticalMomentCacheKey(moment, lane);
        cell.status = 'running';
        state.progress = `Moment ${i + 1} of ${report.moments.length}`;
        changed();
        try {
          const cached = cache.get(key);
          const result =
            cached ??
            (await lane.evaluate(moment, signal, (text) => {
              if (!signal.aborted) {
                state.progress = `Moment ${i + 1} of ${report.moments.length} · ${text}`;
                changed();
              }
            }));
          if (signal.aborted) break;
          if (!result.line?.pvUci[0] && !result.policy) throw new Error('No scored best move returned.');
          if (result.line && (result.line.error || (!Number.isFinite(result.line.scoreCp) && !Number.isFinite(result.line.mateIn))))
            throw new Error('Engine returned no finite evaluation.');
          // Re-key after a fresh evaluation: a lane may refine its provenance while
          // loading (Lc0 Small records which model file it actually ran).
          const storedKey = cached ? key : criticalMomentCacheKey(moment, lane);
          result.cacheKey = storedKey;
          cell.result = result;
          cell.cached = !!cached;
          cell.status = 'complete';
          cache.set(storedKey, result);
          if (cache.size > 128) cache.delete(cache.keys().next().value!);
        } catch (error) {
          if (signal.aborted) break;
          cell.status = 'failed';
          cell.error = error instanceof Error ? error.message : String(error);
        }
        state.completed++;
        changed();
        // Cached work also yields so the cancel control remains usable.
        await new Promise<void>((resolve) => setTimeout(resolve, 0));
      }
      const cells = report.cells[lane.id];
      for (const cell of cells) if (cell.status === 'pending' || cell.status === 'running') cell.status = 'cancelled';
      const failures = cells.filter((cell) => cell.status === 'failed').length;
      state.progress = signal.aborted ? 'Cancelled' : failures ? `Finished with ${failures} ${failures === 1 ? 'failure' : 'failures'}` : 'Finished';
      changed();
    }),
  );
  report.state = signal.aborted ? 'cancelled' : 'complete';
  changed();
}
