import { buildBoardHistoryFromMoves } from './history.ts';
import type { Lc0EvaluatorInput } from './onnxEvaluator.ts';
import type { Lc0PuctSearcher } from './search.ts';

/**
 * Position-set drift/perf sweep utilities shared by the Node CLI/tests and,
 * later, the browser. Kept evaluator-agnostic (any object with evaluate()) so
 * the same comparison can run across f32/WASM, f16/WASM, and f16/WebGPU, and on
 * the main thread or a worker, by passing different backends in.
 */

export interface SweepFixture {
  id: string;
  fen?: string;
  startFen?: string;
  moves?: string[];
}

export interface SweepEvaluator {
  evaluate(input: Lc0EvaluatorInput): Promise<{
    wdl: [number, number, number];
    q: number;
    mlh: number;
    bestMove?: string;
    legalPriors: { uci: string; prior: number }[];
  }>;
}

export interface SweepEvalRecord {
  id: string;
  bestMove?: string;
  wdl: [number, number, number];
  q: number;
  mlh: number;
  topPriors: { uci: string; prior: number }[];
  elapsedMs: number;
}

export interface EvalSweepResult {
  label: string;
  records: SweepEvalRecord[];
  evalsPerSecond: number;
  totalMs: number;
}

export interface SearchSweepRecord {
  id: string;
  bestMove?: string;
  visits: number;
  elapsedMs: number;
}

export interface SearchSweepResult {
  label: string;
  visits: number;
  records: SearchSweepRecord[];
  visitsPerSecond: number;
  totalMs: number;
}

export interface DriftMetrics {
  id: string;
  bestMoveMatch: boolean;
  baselineBestMove?: string;
  candidateBestMove?: string;
  wdlDrift: number;
  qDrift: number;
  mlhDrift: number;
  topPriorDrift: number;
}

export interface SweepComparison {
  baselineLabel: string;
  candidateLabel: string;
  perFixture: DriftMetrics[];
  bestMoveMismatches: number;
  maxWdlDrift: number;
  maxQDrift: number;
  maxMlhDrift: number;
  maxTopPriorDrift: number;
}

function nowMs(): number {
  return typeof performance === 'undefined' ? Date.now() : performance.now();
}

/** Convert a fixture into an evaluator input, reconstructing explicit history. */
export function sweepFixtureInput(fixture: SweepFixture): Lc0EvaluatorInput {
  if (fixture.moves) return { positions: buildBoardHistoryFromMoves(fixture.moves, fixture.startFen) };
  if (fixture.fen) return fixture.fen;
  throw new Error(`Sweep fixture ${fixture.id} has neither moves nor fen`);
}

export async function runEvalSweep(
  label: string,
  evaluator: SweepEvaluator,
  fixtures: SweepFixture[],
  topK = 5,
): Promise<EvalSweepResult> {
  const records: SweepEvalRecord[] = [];
  const startedAll = nowMs();
  for (const fixture of fixtures) {
    const input = sweepFixtureInput(fixture);
    const started = nowMs();
    const ev = await evaluator.evaluate(input);
    records.push({
      id: fixture.id,
      bestMove: ev.bestMove,
      wdl: ev.wdl,
      q: ev.q,
      mlh: ev.mlh,
      topPriors: ev.legalPriors.slice(0, topK).map((p) => ({ uci: p.uci, prior: p.prior })),
      elapsedMs: nowMs() - started,
    });
  }
  const totalMs = nowMs() - startedAll;
  return { label, records, evalsPerSecond: records.length / Math.max(1e-9, totalMs / 1000), totalMs };
}

export async function runSearchSweep(
  label: string,
  searcher: Pick<Lc0PuctSearcher, 'search'>,
  fixtures: SweepFixture[],
  visits: number,
): Promise<SearchSweepResult> {
  const records: SearchSweepRecord[] = [];
  let totalVisits = 0;
  const startedAll = nowMs();
  for (const fixture of fixtures) {
    const input = sweepFixtureInput(fixture);
    const started = nowMs();
    const result = await searcher.search(input, { visits });
    totalVisits += result.visits;
    records.push({ id: fixture.id, bestMove: result.move, visits: result.visits, elapsedMs: nowMs() - started });
  }
  const totalMs = nowMs() - startedAll;
  return { label, visits, records, visitsPerSecond: totalVisits / Math.max(1e-9, totalMs / 1000), totalMs };
}

/** Compare two eval sweeps over the same fixture ids and summarize drift. */
export function compareEvalSweeps(baseline: EvalSweepResult, candidate: EvalSweepResult, topK = 5): SweepComparison {
  const byId = new Map(candidate.records.map((r) => [r.id, r]));
  const perFixture: DriftMetrics[] = [];
  for (const base of baseline.records) {
    const cand = byId.get(base.id);
    if (!cand) continue;
    const wdlDrift = Math.max(...base.wdl.map((v, i) => Math.abs(v - cand.wdl[i])));
    let topPriorDrift = 0;
    for (const bp of base.topPriors.slice(0, topK)) {
      const cp = cand.topPriors.find((p) => p.uci === bp.uci);
      // A missing shared move counts its full baseline mass as drift.
      topPriorDrift = Math.max(topPriorDrift, cp ? Math.abs(bp.prior - cp.prior) : bp.prior);
    }
    perFixture.push({
      id: base.id,
      bestMoveMatch: base.bestMove === cand.bestMove,
      baselineBestMove: base.bestMove,
      candidateBestMove: cand.bestMove,
      wdlDrift,
      qDrift: Math.abs(base.q - cand.q),
      mlhDrift: Math.abs(base.mlh - cand.mlh),
      topPriorDrift,
    });
  }
  return {
    baselineLabel: baseline.label,
    candidateLabel: candidate.label,
    perFixture,
    bestMoveMismatches: perFixture.filter((m) => !m.bestMoveMatch).length,
    maxWdlDrift: Math.max(0, ...perFixture.map((m) => m.wdlDrift)),
    maxQDrift: Math.max(0, ...perFixture.map((m) => m.qDrift)),
    maxMlhDrift: Math.max(0, ...perFixture.map((m) => m.mlhDrift)),
    maxTopPriorDrift: Math.max(0, ...perFixture.map((m) => m.topPriorDrift)),
  };
}
