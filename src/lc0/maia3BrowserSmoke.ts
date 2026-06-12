import { START_FEN, parseFen } from '../chess/board.ts';
import { legalMoves } from '../chess/movegen.ts';
import { moveToUci } from '../chess/moveCodec.ts';
import { Maia3BrowserEvaluator, type Maia3MovePolicyEntry, type Maia3MoveStyle } from './maia3.ts';

interface Fixture {
  name: string;
  fen: string;
  expectLegal?: string[];
}

const FIXTURES: Fixture[] = [
  { name: 'startpos-white', fen: START_FEN },
  { name: 'startpos-after-e4-black', fen: 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1' },
  { name: 'white-castling-both-sides', fen: 'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1', expectLegal: ['e1g1', 'e1c1'] },
  { name: 'white-en-passant', fen: 'rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3', expectLegal: ['e5f6'] },
  { name: 'white-promotions', fen: '8/1P5k/8/8/8/8/8/K7 w - - 0 1', expectLegal: ['b7b8q', 'b7b8r', 'b7b8b', 'b7b8n'] },
  { name: 'black-promotions', fen: '7k/8/8/8/8/8/1p6/K7 b - - 0 1', expectLegal: ['b2b1q', 'b2b1r', 'b2b1b', 'b2b1n'] },
  { name: 'checkmate-terminal', fen: '7k/6Q1/6K1/8/8/8/8/8 b - - 0 1' },
  { name: 'stalemate-terminal', fen: '7k/5K2/6Q1/8/8/8/8/8 b - - 0 1' },
];

function el(id: string): HTMLElement {
  const node = document.getElementById(id);
  if (!node) throw new Error(`Missing #${id}`);
  return node;
}

function parsePositiveInt(value: string | null, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? Math.floor(parsed) : fallback;
}

function parseFloatParam(value: string | null, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function legalUcis(fen: string): string[] {
  const board = parseFen(fen);
  return legalMoves(board).map(moveToUci).sort();
}

function sum(values: number[]): number {
  return values.reduce((acc, value) => acc + value, 0);
}

function top(policy: Maia3MovePolicyEntry[], n = 5): Maia3MovePolicyEntry[] {
  return policy.slice(0, n).map((entry) => ({ uci: entry.uci, prior: entry.prior, logit: entry.logit, index: entry.index }));
}

function validateFixture(name: string, expectedLegal: string[], policy: Maia3MovePolicyEntry[], move: string | null, expectedMoves: string[] | undefined): string[] {
  const errors: string[] = [];
  const legalSet = new Set(expectedLegal);
  const policyUcis = policy.map((entry) => entry.uci).sort();
  const duplicatePolicyUcis = policyUcis.filter((uci, i) => i > 0 && policyUcis[i - 1] === uci);
  const duplicateIndices = policy.map((entry) => entry.index).sort((a, b) => a - b).filter((index, i, sorted) => i > 0 && sorted[i - 1] === index);
  if (duplicatePolicyUcis.length) errors.push(`${name}: duplicate legal policy UCIs ${duplicatePolicyUcis.join(',')}`);
  if (duplicateIndices.length) errors.push(`${name}: duplicate legal policy indices ${duplicateIndices.join(',')}`);
  if (policy.length !== expectedLegal.length) errors.push(`${name}: policy has ${policy.length} legal moves, expected ${expectedLegal.length}`);
  for (const entry of policy) {
    if (!legalSet.has(entry.uci)) errors.push(`${name}: illegal policy move ${entry.uci}`);
    if (!(entry.index >= 0 && entry.index < 4352)) errors.push(`${name}: out-of-range Maia3 index ${entry.index} for ${entry.uci}`);
    if (!(entry.prior >= 0 && entry.prior <= 1)) errors.push(`${name}: invalid prior ${entry.prior} for ${entry.uci}`);
  }
  if (expectedLegal.length) {
    if (!move || !legalSet.has(move)) errors.push(`${name}: chose illegal/null move ${move ?? 'null'}`);
    const priorSum = sum(policy.map((entry) => entry.prior));
    if (Math.abs(priorSum - 1) > 1e-4) errors.push(`${name}: legal priors sum to ${priorSum}`);
  } else if (move !== null || policy.length !== 0) {
    errors.push(`${name}: terminal/no-legal position returned move=${move ?? 'null'} policy=${policy.length}`);
  }
  if (expectedMoves) {
    for (const expected of expectedMoves) {
      if (!legalSet.has(expected)) errors.push(`${name}: move generator missing expected move ${expected}`);
      if (!policy.some((entry) => entry.uci === expected)) errors.push(`${name}: Maia3 mask missing expected move ${expected}`);
    }
  }
  return errors;
}

async function main(): Promise<void> {
  const params = new URLSearchParams(location.search);
  const cycles = parsePositiveInt(params.get('cycles'), 2);
  const selfElo = parsePositiveInt(params.get('selfElo'), 1500);
  const oppoElo = parsePositiveInt(params.get('oppoElo'), selfElo);
  const style: Maia3MoveStyle = params.get('style') === 'sample' ? 'sample' : 'argmax';
  const temperature = parseFloatParam(params.get('temperature'), 1);
  const topP = parseFloatParam(params.get('topP'), 1);
  const epParam = params.get('ortEp');
  const ep = epParam === 'wasm' || epParam === 'webgpu' || epParam === 'webgpu,wasm' || epParam === 'auto' ? epParam : undefined;
  const modelUrl = params.get('model') ?? undefined;
  const started = performance.now();
  const result = {
    ok: false,
    cycles,
    selfElo,
    oppoElo,
    style,
    temperature,
    topP,
    requestedEp: ep ?? 'auto',
    backend: 'unknown',
    evalCount: 0,
    evalElapsedMs: 0,
    gridSize: 0,
    gridMsPerBatch: 0,
    model: null as null | { url: string; bytes?: number; sha256?: string; sha256Valid?: boolean; cacheStatus: string; mode: string },
    loads: [] as Array<{ cycle: number; bytes?: number; sha256?: string; sha256Valid?: boolean; cacheStatus: string; mode: string; elapsedMs: number }>,
    inputNames: [] as string[],
    outputNames: [] as string[],
    rows: [] as Array<Record<string, unknown>>,
    errors: [] as string[],
    elapsedMs: 0,
  };
  for (let cycle = 0; cycle < cycles; cycle++) {
    el('status').textContent = `Loading Maia3 cycle ${cycle + 1}/${cycles}…`;
    const evaluator = await Maia3BrowserEvaluator.create({ selfElo, oppoElo, ep, modelUrl });
    result.backend = evaluator.backend;
    result.model ??= {
      url: evaluator.modelLoad.url,
      bytes: evaluator.modelLoad.bytes,
      sha256: evaluator.modelLoad.sha256,
      sha256Valid: evaluator.modelLoad.sha256Valid,
      cacheStatus: evaluator.modelLoad.cacheStatus,
      mode: evaluator.modelLoad.mode,
    };
    result.loads.push({
      cycle,
      bytes: evaluator.modelLoad.bytes,
      sha256: evaluator.modelLoad.sha256,
      sha256Valid: evaluator.modelLoad.sha256Valid,
      cacheStatus: evaluator.modelLoad.cacheStatus,
      mode: evaluator.modelLoad.mode,
      elapsedMs: evaluator.modelLoad.elapsedMs,
    });
    result.inputNames = evaluator.inputNames;
    result.outputNames = evaluator.outputNames;
    try {
      for (const fixture of FIXTURES) {
        el('status').textContent = `Evaluating ${fixture.name} (${cycle + 1}/${cycles})…`;
        const expectedLegal = legalUcis(fixture.fen);
        const evalStarted = performance.now();
        const choice = await evaluator.chooseMove(fixture.fen, { selfElo, oppoElo, style, temperature, topP });
        result.evalElapsedMs += performance.now() - evalStarted;
        result.evalCount += 1;
        const valueSum = sum(choice.evaluation.valueProbabilities);
        if (choice.evaluation.valueProbabilities.length && Math.abs(valueSum - 1) > 1e-4) {
          result.errors.push(`${fixture.name}: value probabilities sum to ${valueSum}`);
        }
        result.errors.push(...validateFixture(fixture.name, expectedLegal, choice.evaluation.legalPriors, choice.move, fixture.expectLegal));
        result.rows.push({
          cycle,
          name: fixture.name,
          evalMs: Math.round((performance.now() - evalStarted) * 10) / 10,
          fen: fixture.fen,
          legalCount: expectedLegal.length,
          move: choice.move,
          top5: top(choice.evaluation.legalPriors),
          valueLogits: choice.evaluation.valueLogits,
          valueProbabilities: choice.evaluation.valueProbabilities,
        });
      }
    } finally {
      if (cycle < cycles - 1) await evaluator.dispose();
      else {
        // Optional batched-grid benchmark (rating-inference workload) on the
        // last live session: one position under gridSize (selfElo, oppoElo)
        // conditions per run; first run dropped as warmup.
        const gridSize = parsePositiveInt(params.get('gridSize'), 0);
        if (gridSize > 0) {
          const conditions = Array.from({ length: gridSize }, (_, i) => ({ selfElo: 600 + ((2000 / Math.max(1, gridSize - 1)) * i | 0), oppoElo: 1500 }));
          const runs = 6;
          let total = 0;
          for (let run = 0; run < runs; run += 1) {
            const t0 = performance.now();
            await evaluator.evaluateConditions(FIXTURES[0].fen, conditions);
            if (run > 0) total += performance.now() - t0;
          }
          result.gridSize = gridSize;
          result.gridMsPerBatch = Math.round((total / (runs - 1)) * 10) / 10;
        }
        await evaluator.dispose();
      }
    }
  }
  result.elapsedMs = performance.now() - started;
  result.ok = result.errors.length === 0;
  el('benchResult').textContent = JSON.stringify(result, null, 2);
  el('status').textContent = result.ok ? 'MAIA3_BROWSER_SMOKE_DONE' : 'MAIA3_BROWSER_SMOKE_FAILED';
  if (!result.ok) throw new Error(result.errors.join('\n'));
}

main().catch((error) => {
  const result = { ok: false, errors: [error instanceof Error ? error.message : String(error)] };
  el('benchResult').textContent = JSON.stringify(result, null, 2);
  el('status').textContent = 'MAIA3_BROWSER_SMOKE_FAILED';
  throw error;
});
