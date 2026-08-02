#!/usr/bin/env node
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { performance } from 'node:perf_hooks';
import { buildBoardHistoryFromMoves } from '../src/lc0/history.ts';
import { Lc0OnnxEvaluator } from '../src/lc0/onnxEvaluator.ts';
import { collectOrtRuntimeDiagnostics } from '../src/nn/ortRuntime.ts';

function parseArgs(argv) {
  const options = { warmup: 1, iterations: 3, threads: 1 };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--baseline') options.baseline = argv[++i];
    else if (arg === '--candidate') options.candidate = argv[++i];
    else if (arg === '--fixtures') options.fixtures = argv[++i];
    else if (arg === '--warmup') options.warmup = Number(argv[++i]);
    else if (arg === '--iterations') options.iterations = Number(argv[++i]);
    else if (arg === '--threads') options.threads = Number(argv[++i]);
    else if (arg === '--out') options.out = argv[++i];
    else throw new Error(`unknown argument: ${arg}`);
  }
  if (!options.baseline || !options.candidate || !options.fixtures) {
    throw new Error(
      'usage: lc0_compare_onnx_models.mjs --baseline MODEL --candidate MODEL --fixtures JSON [--warmup N] [--iterations N] [--threads N] [--out JSON]',
    );
  }
  for (const path of [options.baseline, options.candidate, options.fixtures]) {
    if (!existsSync(path)) throw new Error(`file not found: ${path}`);
  }
  for (const [name, value, minimum] of [
    ['warmup', options.warmup, 0],
    ['iterations', options.iterations, 1],
    ['threads', options.threads, 1],
  ]) {
    if (!Number.isFinite(value) || !Number.isInteger(value) || value < minimum) {
      throw new Error(`--${name} must be an integer >= ${minimum}`);
    }
  }
  return options;
}

function fixtureInput(fixture) {
  if (!fixture.moves) return fixture.fen;
  return { positions: buildBoardHistoryFromMoves(fixture.moves, fixture.startFen) };
}

function percentile(values, fraction) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.max(0, Math.floor((sorted.length - 1) * fraction)))];
}

function timingSummary(values) {
  return {
    samples: values.length,
    medianMs: percentile(values, 0.5),
    p90Ms: percentile(values, 0.9),
    minMs: values.length ? Math.min(...values) : null,
    maxMs: values.length ? Math.max(...values) : null,
  };
}

async function timedEvaluate(evaluator, input) {
  const started = performance.now();
  const evaluation = await evaluator.evaluate(input);
  return { evaluation, elapsedMs: performance.now() - started };
}

function assertFiniteEvaluation(label, evaluation) {
  const values = [...evaluation.wdl, evaluation.q, evaluation.mlh, ...evaluation.legalPriors.map((entry) => entry.prior)];
  if (values.some((value) => !Number.isFinite(value))) {
    throw new Error(`${label} produced a non-finite WDL, Q, MLH, or legal-prior value`);
  }
}

function compareEvaluations(id, baseline, candidate) {
  assertFiniteEvaluation(`baseline fixture ${id}`, baseline);
  assertFiniteEvaluation(`candidate fixture ${id}`, candidate);
  const baselinePriors = new Map(baseline.legalPriors.map((entry) => [entry.index, entry.prior]));
  const candidatePriors = new Map(candidate.legalPriors.map((entry) => [entry.index, entry.prior]));
  const baselineTop = [...baseline.legalPriors].sort((a, b) => b.prior - a.prior);
  const candidateTop = [...candidate.legalPriors].sort((a, b) => b.prior - a.prior);
  const indices = new Set([...baselinePriors.keys(), ...candidatePriors.keys()]);
  const priorErrors = [...indices].map((index) => Math.abs((baselinePriors.get(index) ?? 0) - (candidatePriors.get(index) ?? 0)));
  let jensenShannonDivergence = 0;
  for (const index of indices) {
    const baselinePrior = baselinePriors.get(index) ?? 0;
    const candidatePrior = candidatePriors.get(index) ?? 0;
    const midpoint = (baselinePrior + candidatePrior) / 2;
    if (baselinePrior > 0) jensenShannonDivergence += 0.5 * baselinePrior * Math.log(baselinePrior / midpoint);
    if (candidatePrior > 0) jensenShannonDivergence += 0.5 * candidatePrior * Math.log(candidatePrior / midpoint);
  }
  const baselineTopTwo = new Set(baselineTop.slice(0, 2).map((entry) => entry.index));
  const candidateTopTwo = new Set(candidateTop.slice(0, 2).map((entry) => entry.index));
  return {
    id,
    bestMoveAgreement: baseline.bestMove === candidate.bestMove,
    mutualTopTwoAgreement:
      baselineTop[0] !== undefined && candidateTop[0] !== undefined && candidateTopTwo.has(baselineTop[0].index) && baselineTopTwo.has(candidateTop[0].index),
    baselineBestMove: baseline.bestMove,
    candidateBestMove: candidate.bestMove,
    maxWdlAbsError: Math.max(...baseline.wdl.map((value, index) => Math.abs(value - candidate.wdl[index]))),
    qAbsError: Math.abs(baseline.q - candidate.q),
    mlhAbsError: Math.abs(baseline.mlh - candidate.mlh),
    maxLegalPriorAbsError: priorErrors.length ? Math.max(...priorErrors) : 0,
    meanLegalPriorAbsError: priorErrors.length ? priorErrors.reduce((sum, value) => sum + value, 0) / priorErrors.length : 0,
    legalPolicyTotalVariation: priorErrors.reduce((sum, value) => sum + value, 0) / 2,
    legalPolicyJensenShannonDivergence: jensenShannonDivergence,
    baselineTopMoves: baselineTop.slice(0, 3).map(({ uci, prior }) => ({ uci, prior })),
    candidateTopMoves: candidateTop.slice(0, 3).map(({ uci, prior }) => ({ uci, prior })),
    baselineTopMoveMargin: baselineTop.length > 1 ? baselineTop[0].prior - baselineTop[1].prior : null,
    candidateTopMoveMargin: candidateTop.length > 1 ? candidateTop[0].prior - candidateTop[1].prior : null,
  };
}

const options = parseArgs(process.argv.slice(2));
process.env.ORT_NUM_THREADS = String(options.threads);
const fixtures = JSON.parse(readFileSync(options.fixtures, 'utf8'));
if (!Array.isArray(fixtures) || fixtures.length === 0) throw new Error('fixtures must be a non-empty JSON array');
const baselineEvaluator = await Lc0OnnxEvaluator.create(readFileSync(options.baseline));
const candidateEvaluator = await Lc0OnnxEvaluator.create(readFileSync(options.candidate));
const baselineTimes = [];
const candidateTimes = [];
const comparisons = [];
const timingSamples = [];

try {
  for (const [fixtureIndex, fixture] of fixtures.entries()) {
    const input = fixtureInput(fixture);
    for (let i = 0; i < options.warmup; i += 1) {
      if ((fixtureIndex + i) % 2 === 0) {
        await baselineEvaluator.evaluate(input);
        await candidateEvaluator.evaluate(input);
      } else {
        await candidateEvaluator.evaluate(input);
        await baselineEvaluator.evaluate(input);
      }
    }
    let baseline;
    let candidate;
    for (let i = 0; i < options.iterations; i += 1) {
      const baselineFirst = (fixtureIndex + i) % 2 === 0;
      const first = await timedEvaluate(baselineFirst ? baselineEvaluator : candidateEvaluator, input);
      const second = await timedEvaluate(baselineFirst ? candidateEvaluator : baselineEvaluator, input);
      const baselineResult = baselineFirst ? first : second;
      const candidateResult = baselineFirst ? second : first;
      baseline = baselineResult.evaluation;
      candidate = candidateResult.evaluation;
      assertFiniteEvaluation(`baseline fixture ${fixture.id} iteration ${i}`, baseline);
      assertFiniteEvaluation(`candidate fixture ${fixture.id} iteration ${i}`, candidate);
      baselineTimes.push(baselineResult.elapsedMs);
      candidateTimes.push(candidateResult.elapsedMs);
      timingSamples.push({
        fixtureId: fixture.id,
        iteration: i,
        first: baselineFirst ? 'baseline' : 'candidate',
        baselineMs: baselineResult.elapsedMs,
        candidateMs: candidateResult.elapsedMs,
      });
    }
    comparisons.push(compareEvaluations(fixture.id, baseline, candidate));
  }
} finally {
  await baselineEvaluator.dispose();
  await candidateEvaluator.dispose();
}

const summary = {
  schema: 'lc0_browser.onnx_model_comparison.v1',
  baselineModel: options.baseline,
  candidateModel: options.candidate,
  fixtureCount: comparisons.length,
  warmup: options.warmup,
  iterations: options.iterations,
  threads: options.threads,
  measurementOrder: 'alternating by fixture and iteration',
  bestMoveAgreement: comparisons.filter((row) => row.bestMoveAgreement).length / comparisons.length,
  mutualTopTwoAgreement: comparisons.filter((row) => row.mutualTopTwoAgreement).length / comparisons.length,
  maxWdlAbsError: Math.max(...comparisons.map((row) => row.maxWdlAbsError)),
  maxQAbsError: Math.max(...comparisons.map((row) => row.qAbsError)),
  maxMlhAbsError: Math.max(...comparisons.map((row) => row.mlhAbsError)),
  maxLegalPriorAbsError: Math.max(...comparisons.map((row) => row.maxLegalPriorAbsError)),
  maxLegalPolicyTotalVariation: Math.max(...comparisons.map((row) => row.legalPolicyTotalVariation)),
  maxLegalPolicyJensenShannonDivergence: Math.max(...comparisons.map((row) => row.legalPolicyJensenShannonDivergence)),
  baselineTiming: timingSummary(baselineTimes),
  candidateTiming: timingSummary(candidateTimes),
  ortDiagnostics: await collectOrtRuntimeDiagnostics(),
};
const report = { summary, fixtures: comparisons, timingSamples };
if (options.out) {
  mkdirSync(dirname(options.out), { recursive: true });
  writeFileSync(options.out, `${JSON.stringify(report, null, 2)}\n`);
}
console.log(JSON.stringify(report, null, 2));
