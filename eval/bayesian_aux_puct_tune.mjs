#!/usr/bin/env node
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';

function arg(name, fallback = undefined) {
  const prefix = `${name}=`;
  const inline = process.argv.find((value) => value.startsWith(prefix));
  if (inline) return inline.slice(prefix.length);
  const i = process.argv.indexOf(name);
  return i >= 0 ? process.argv[i + 1] : fallback;
}
function num(name, fallback) { return Number(arg(name, String(fallback))); }
function boolArg(name, fallback = false) { return ['1', 'true', 'yes', 'on'].includes(String(arg(name, fallback ? '1' : '0')).toLowerCase()); }
function listNums(name, fallback) {
  return String(arg(name, fallback)).split(',').map((s) => Number(s.trim())).filter((x) => Number.isFinite(x));
}
function mulberry32(seed) {
  let t = seed >>> 0;
  return () => {
    t += 0x6D2B79F5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}
function keyOf(w) { return `${w.av.toFixed(5)}_${w.rank.toFixed(5)}_${w.regret.toFixed(5)}_${w.risk.toFixed(5)}_${w.uncertainty.toFixed(5)}_${w.cpuct.toFixed(4)}_${w.fpu.toFixed(4)}`; }
function nameOf(w, idx) {
  const enc = (x) => String(Math.round(x * 10000)).padStart(3, '0');
  const encP = (x) => String(Math.round(x * 1000)).padStart(4, '0');
  const encS = (x) => `${x < 0 ? 'm' : 'p'}${String(Math.round(Math.abs(x) * 1000)).padStart(3, '0')}`;
  return `cand${idx}_a${enc(w.av)}_r${enc(w.rank)}_g${enc(w.regret)}_k${enc(w.risk)}_u${enc(w.uncertainty)}_c${encP(w.cpuct)}_f${encS(w.fpu)}`;
}
function scoreRateFromArena(path, candidateName) {
  const data = JSON.parse(readFileSync(path, 'utf8'));
  const pair = data.pairs.find((p) => (p.a === 'classic' && p.b === candidateName) || (p.b === 'classic' && p.a === candidateName));
  if (!pair) throw new Error(`No classic/${candidateName} pair in ${path}`);
  const candScore = pair.a === candidateName ? pair.aScore : pair.games - pair.aScore;
  return { scoreRate: candScore / Math.max(1, pair.games), score: candScore, games: pair.games };
}
function readObs(path) {
  if (!existsSync(path)) return [];
  return readFileSync(path, 'utf8').split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line));
}
function appendObs(path, obs) { appendFileSync(path, JSON.stringify(obs) + '\n'); }
function appendLedger(path, row) { appendFileSync(path, JSON.stringify(row) + '\n'); }
function dist2(a, b, maxWeight, cpuctScale, fpuScale) {
  const s = (x) => x / maxWeight;
  return (s(a.av)-s(b.av))**2 + (s(a.rank)-s(b.rank))**2 + (s(a.regret)-s(b.regret))**2 + (s(a.risk)-s(b.risk))**2 + (s(a.uncertainty)-s(b.uncertainty))**2 + ((a.cpuct-b.cpuct)/cpuctScale)**2 + ((a.fpu-b.fpu)/fpuScale)**2;
}
function surrogate(x, obs, maxWeight, lengthScale, beta, cpuctScale, fpuScale, costAware) {
  if (!obs.length) return { mean: 0.5, std: 0.25, acqRaw: 0.5 + beta * 0.25, acq: 0.5 + beta * 0.25, pseudoGames: 0, predictedWallSeconds: 1 };
  let sw = 0, sy = 0, sg = 0, scost = 0, scostw = 0;
  for (const o of obs) {
    const k = Math.exp(-dist2(x, o.weights, maxWeight, cpuctScale, fpuScale) / (2 * lengthScale * lengthScale));
    sw += k * o.games;
    sy += k * o.score;
    sg += k * o.games;
    if (Number.isFinite(o.wallSeconds) && o.wallSeconds > 0) { scost += k * o.wallSeconds; scostw += k; }
  }
  const mean = sg > 1e-9 ? sy / sg : 0.5;
  let rv = 0, rw = 0;
  for (const o of obs) {
    const k = Math.exp(-dist2(x, o.weights, maxWeight, cpuctScale, fpuScale) / (2 * lengthScale * lengthScale));
    const y = o.scoreRate;
    rv += k * o.games * (y - mean) ** 2;
    rw += k * o.games;
  }
  const residual = rw > 1e-9 ? rv / rw : 0.03;
  const binom = Math.max(0.0025, mean * (1 - mean)) / Math.max(1, sw + 2);
  const prior = 0.02 / Math.sqrt(1 + sw);
  const std = Math.sqrt(binom + residual + prior * prior);
  const acqRaw = mean + beta * std;
  const predictedWallSeconds = scostw > 1e-9 ? scost / scostw : 1;
  const acq = costAware ? acqRaw / Math.sqrt(Math.max(1, predictedWallSeconds)) : acqRaw;
  return { mean, std, acqRaw, acq, pseudoGames: sw, predictedWallSeconds };
}
function normalizePoint(w, baseCpuct, baseFpu) {
  return { av:0, rank:0, regret:0, risk:0, uncertainty:0, cpuct: baseCpuct, fpu: baseFpu, ...w };
}
function parsePriorBest(path, visits, baseCpuct, baseFpu) {
  if (!path || !existsSync(path)) return [];
  const lines = readFileSync(path, 'utf8').split(/\r?\n/).filter(Boolean);
  const header = lines.shift()?.split('\t') ?? [];
  const out = [];
  for (const line of lines) {
    const cols = line.split('\t');
    const row = Object.fromEntries(header.map((h, i) => [h, cols[i]]));
    if (String(row.visits) !== String(visits)) continue;
    const av = Number(row.av ?? 0), rank = Number(row.rank ?? 0), regret = Number(row.regret ?? 0);
    const cpuct = Number(row.cpuct ?? baseCpuct), fpu = Number(row.fpu ?? baseFpu);
    out.push(normalizePoint({ av, rank, regret, cpuct: Number.isFinite(cpuct) ? cpuct : baseCpuct, fpu: Number.isFinite(fpu) ? fpu : baseFpu }, baseCpuct, baseFpu));
  }
  return out;
}
function makePool(rng, maxWeight, poolSize, dims, baseCpuct, baseFpu, cpuctValues, fpuValues, priorPoints) {
  const points = [];
  const add = (w) => points.push(normalizePoint(w, baseCpuct, baseFpu));
  const vals = [0.001, 0.0025, 0.005, 0.01, 0.02].filter((v) => v <= maxWeight + 1e-12);

  for (const p of priorPoints) {
    add(p);
    for (const scale of [0.5, 0.75, 1.25, 1.5]) {
      const w = {};
      for (const d of dims) w[d] = Math.min(maxWeight, Math.max(0, (p[d] ?? 0) * scale));
      add({ ...w, cpuct: p.cpuct, fpu: p.fpu });
    }
    for (const c of cpuctValues) add({ av: p.av, rank: p.rank, regret: p.regret, risk: p.risk, uncertainty: p.uncertainty, cpuct: c, fpu: p.fpu });
    for (const f of fpuValues) add({ av: p.av, rank: p.rank, regret: p.regret, risk: p.risk, uncertainty: p.uncertainty, cpuct: p.cpuct, fpu: f });
  }

  for (const c of cpuctValues) add({ cpuct: c, fpu: baseFpu });
  for (const f of fpuValues) add({ cpuct: baseCpuct, fpu: f });
  for (const c of cpuctValues) for (const f of fpuValues) add({ cpuct: c, fpu: f });

  for (const d of dims) for (const v of vals) add({ [d]: v });
  for (const v of vals) {
    const w = {};
    for (const d of dims) w[d] = v;
    add(w);
  }
  if (dims.includes('av') && dims.includes('rank')) for (const v of vals) add({ av: v, rank: v });
  if (dims.includes('av') && dims.includes('regret')) for (const v of vals) add({ av: v, regret: v });
  if (dims.includes('rank') && dims.includes('regret')) for (const v of vals) add({ rank: v, regret: v });

  while (points.length < poolSize) {
    const w = { cpuct: cpuctValues[Math.floor(rng() * cpuctValues.length)] ?? baseCpuct, fpu: fpuValues[Math.floor(rng() * fpuValues.length)] ?? baseFpu };
    for (const d of dims) {
      if (rng() < 0.58) {
        const raw = Math.exp(Math.log(0.0005) + rng() * (Math.log(maxWeight) - Math.log(0.0005)));
        w[d] = Math.round(raw / 0.0005) * 0.0005;
      }
    }
    add(w);
  }
  const seen = new Set();
  return points.filter((w) => {
    const k = keyOf(w);
    if (seen.has(k)) return false;
    seen.add(k);
    return w.av !== 0 || w.rank !== 0 || w.regret !== 0 || w.risk !== 0 || w.uncertainty !== 0 || w.cpuct !== baseCpuct || w.fpu !== baseFpu;
  });
}
function runArena({ model, meta, outPath, name, weights, games, visits, batchSize, baseCpuct, baseFpu, maxPlies, openingsFile, maxOpenings, judgeModel, judgeMeta, adjudicateThreshold, ortThreads, openingOffset }) {
  const players = `classic:${model}:${meta}:puct:0:0:0:0:0:${baseCpuct}:${baseFpu},${name}:${model}:${meta}:aux:${weights.av}:${weights.rank}:${weights.regret}:${weights.risk}:${weights.uncertainty}:${weights.cpuct}:${weights.fpu}`;
  const args = [
    ...process.execArgv,
    'eval/search_mode_arena.mjs',
    `--players=${players}`,
    '--games-per-pair', String(games),
    '--visits', String(visits),
    '--batch-size', String(batchSize),
    '--cpuct', String(baseCpuct),
    '--fpu', String(baseFpu),
    '--max-plies', String(maxPlies),
    '--openings-file', openingsFile,
    '--opening-offset', String(openingOffset),
    '--out', outPath,
  ];
  if (maxOpenings > 0) args.push('--max-openings', String(maxOpenings));
  if (judgeModel && judgeMeta) args.push('--judge-model', judgeModel, '--judge-meta', judgeMeta, '--adjudicate-threshold', adjudicateThreshold);
  return spawnSync(process.execPath, args, { stdio: 'inherit', env: { ...process.env, ORT_INTRA_OP_NUM_THREADS: ortThreads } });
}

const model = arg('--model');
const meta = arg('--meta');
const outDir = arg('--out-dir', 'artifacts/head_ablation_1m/bayesian_aux_puct_tune');
if (!model || !meta) throw new Error('usage: node --experimental-strip-types eval/bayesian_aux_puct_tune.mjs --model m.onnx --meta m.meta.json --out-dir out');
const visits = num('--visits', 32);
const batchSize = num('--batch-size', 16);
const gamesPerCandidate = num('--games-per-candidate', 4);
const iterations = num('--iterations', 20);
const baseCpuct = num('--cpuct', 1.5);
const baseFpu = num('--fpu', 0);
const maxPlies = num('--max-plies', 100);
const maxWeight = num('--max-weight', 0.02);
const poolSize = num('--pool-size', 384);
const seed = num('--seed', 23);
const ortThreads = arg('--ort-threads', process.env.ORT_INTRA_OP_NUM_THREADS ?? '2');
const openingsFile = arg('--openings-file', 'eval/opening_suite_uho_lite_v1.fen');
const maxOpenings = num('--max-openings', 0);
const judgeModel = arg('--judge-model', '');
const judgeMeta = arg('--judge-meta', '');
const adjudicateThreshold = arg('--adjudicate-threshold', '0.05');
const beta = num('--beta', 0.9);
const lengthScale = num('--length-scale', 0.35);
const initialBudget = num('--initial-budget', 8);
const dims = arg('--dims', 'av,rank,regret').split(',').map(s => s.trim()).filter(Boolean);
const cpuctValues = listNums('--cpuct-values', `${baseCpuct},1.2,1.8`).filter((v, i, a) => a.indexOf(v) === i);
const fpuValues = listNums('--fpu-values', `${baseFpu},-0.1,0.1`).filter((v, i, a) => a.indexOf(v) === i);
const cpuctScale = num('--cpuct-scale', 0.35);
const fpuScale = num('--fpu-scale', 0.15);
const costAware = boolArg('--cost-aware', true);
const priorBestTsv = arg('--prior-best-tsv', '');
const confirmTopK = num('--confirm-top-k', 0);
const confirmGames = num('--confirm-games', 24);
const confirmOpeningOffset = num('--confirm-opening-offset', 97);
mkdirSync(outDir, { recursive: true });
const obsPath = join(outDir, 'observations.jsonl');
const ledgerPath = join(outDir, 'ledger.jsonl');
const summaryPath = join(outDir, 'summary.tsv');
const confirmPath = join(outDir, 'confirmation.tsv');
const statePath = join(outDir, 'state.json');
if (!existsSync(summaryPath)) writeFileSync(summaryPath, 'iter\tname\tav\trank\tregret\trisk\tuncertainty\tcpuct\tfpu\tscore_rate\tscore\tgames\twall_seconds\tposterior_mean\tposterior_std\tposterior_acq\tposterior_acq_raw\tpredicted_wall_seconds\tarena_json\n');
if (confirmTopK > 0 && !existsSync(confirmPath)) writeFileSync(confirmPath, 'rank\tname\tav\trank_weight\tregret\trisk\tuncertainty\tcpuct\tfpu\tscore_rate\tscore\tgames\twall_seconds\tarena_json\n');
const rng = mulberry32(seed);
const priorPoints = parsePriorBest(priorBestTsv, visits, baseCpuct, baseFpu);
const pool = makePool(rng, maxWeight, poolSize, dims, baseCpuct, baseFpu, cpuctValues, fpuValues, priorPoints);
let obs = readObs(obsPath);
const evaluated = new Set(obs.map((o) => keyOf(o.weights)));
console.error(`[bayes-aux] out=${outDir} visits=${visits} games_per_candidate=${gamesPerCandidate} dims=${dims.join(',')} existing=${obs.length} pool=${pool.length} initial_budget=${initialBudget} cost_aware=${costAware} cpuct_values=${cpuctValues.join(',')} fpu_values=${fpuValues.join(',')} prior_points=${priorPoints.length}`);
for (let iter = obs.length; iter < iterations; iter++) {
  let chosen = null, pred = null;
  const initial = pool.slice(0, Math.min(initialBudget, pool.length)).filter((w) => !evaluated.has(keyOf(w)));
  if (obs.length < initialBudget && initial.length) chosen = initial[0];
  else {
    let best = null;
    for (const w of pool) {
      if (evaluated.has(keyOf(w))) continue;
      const s = surrogate(w, obs, maxWeight, lengthScale, beta, cpuctScale, fpuScale, costAware);
      if (!best || s.acq > best.s.acq) best = { w, s };
    }
    if (!best) break;
    chosen = best.w;
    pred = best.s;
  }
  if (!pred) pred = surrogate(chosen, obs, maxWeight, lengthScale, beta, cpuctScale, fpuScale, costAware);
  const name = nameOf(chosen, iter + 1);
  const arenaPath = join(outDir, `${String(iter + 1).padStart(3, '0')}_${name}.json`);
  console.error(`[bayes-aux] iter=${iter + 1}/${iterations} eval ${name} av=${chosen.av} rank=${chosen.rank} regret=${chosen.regret} cpuct=${chosen.cpuct} fpu=${chosen.fpu} pred_mean=${pred.mean.toFixed(3)} pred_std=${pred.std.toFixed(3)} acq=${pred.acq.toFixed(4)} raw=${pred.acqRaw.toFixed(3)}`);
  const started = Date.now();
  try {
    if (!existsSync(arenaPath)) {
      const child = runArena({ model, meta, outPath: arenaPath, name, weights: chosen, games: gamesPerCandidate, visits, batchSize, baseCpuct, baseFpu, maxPlies, openingsFile, maxOpenings, judgeModel, judgeMeta, adjudicateThreshold, ortThreads, openingOffset: 0 });
      if (child.status !== 0) throw new Error(`arena failed status=${child.status}`);
    }
    const result = scoreRateFromArena(arenaPath, name);
    const wallSeconds = (Date.now() - started) / 1000;
    const nextObs = { iter: iter + 1, name, weights: chosen, ...result, wallSeconds, posteriorBefore: pred, arenaPath, createdUtc: new Date().toISOString() };
    appendObs(obsPath, nextObs);
    appendLedger(ledgerPath, { trial_id: name, status: 'succeeded', params: { visits, cpuct: chosen.cpuct, fpu: chosen.fpu, av_weight: chosen.av, rank_weight: chosen.rank, regret_weight: chosen.regret, risk_weight: chosen.risk, uncertainty_weight: chosen.uncertainty }, score: result.scoreRate, raw_metrics: result, cost: { wall_seconds: wallSeconds, gpu_hours: 0, games: result.games, visits }, artifacts: [arenaPath], createdUtc: nextObs.createdUtc });
    obs.push(nextObs);
    evaluated.add(keyOf(chosen));
    writeFileSync(summaryPath, readFileSync(summaryPath, 'utf8') + `${iter + 1}\t${name}\t${chosen.av}\t${chosen.rank}\t${chosen.regret}\t${chosen.risk}\t${chosen.uncertainty}\t${chosen.cpuct}\t${chosen.fpu}\t${result.scoreRate.toFixed(4)}\t${result.score}\t${result.games}\t${wallSeconds.toFixed(3)}\t${pred.mean.toFixed(4)}\t${pred.std.toFixed(4)}\t${pred.acq.toFixed(4)}\t${pred.acqRaw.toFixed(4)}\t${pred.predictedWallSeconds.toFixed(3)}\t${arenaPath}\n`);
    const ranked = obs.slice().sort((a, b) => b.scoreRate - a.scoreRate || b.games - a.games);
    writeFileSync(statePath, JSON.stringify({ protocol: { model, meta, visits, batchSize, gamesPerCandidate, iterations, baseCpuct, baseFpu, maxPlies, maxWeight, dims, beta, lengthScale, initialBudget, seed, ortThreads, openingsFile, maxOpenings, judgeModel, judgeMeta, adjudicateThreshold, cpuctValues, fpuValues, costAware, priorBestTsv, confirmTopK, confirmGames }, best: ranked[0] ?? null, observations: obs, updatedUtc: new Date().toISOString() }, null, 2));
  } catch (err) {
    const wallSeconds = (Date.now() - started) / 1000;
    appendLedger(ledgerPath, { trial_id: name, status: 'failed', params: { visits, cpuct: chosen.cpuct, fpu: chosen.fpu, av_weight: chosen.av, rank_weight: chosen.rank, regret_weight: chosen.regret, risk_weight: chosen.risk, uncertainty_weight: chosen.uncertainty }, score: null, error: String(err?.message ?? err), cost: { wall_seconds: wallSeconds, gpu_hours: 0, games: 0, visits }, artifacts: [arenaPath], createdUtc: new Date().toISOString() });
    throw err;
  }
}
const ranked = obs.slice().sort((a, b) => b.scoreRate - a.scoreRate || b.games - a.games);
if (confirmTopK > 0) {
  const confirmed = new Set(existsSync(confirmPath) ? readFileSync(confirmPath, 'utf8').split(/\r?\n/).slice(1).filter(Boolean).map((line) => line.split('\t')[1]) : []);
  for (const [idx, o] of ranked.slice(0, confirmTopK).entries()) {
    if (confirmed.has(o.name)) continue;
    const arenaPath = join(outDir, `confirm_${String(idx + 1).padStart(2, '0')}_${o.name}.json`);
    const started = Date.now();
    console.error(`[bayes-aux] confirm rank=${idx + 1}/${confirmTopK} ${o.name} games=${confirmGames}`);
    const child = runArena({ model, meta, outPath: arenaPath, name: o.name, weights: o.weights, games: confirmGames, visits, batchSize, baseCpuct, baseFpu, maxPlies, openingsFile, maxOpenings, judgeModel, judgeMeta, adjudicateThreshold, ortThreads, openingOffset: confirmOpeningOffset + idx * confirmGames });
    if (child.status !== 0) throw new Error(`confirmation arena failed status=${child.status}`);
    const result = scoreRateFromArena(arenaPath, o.name);
    const wallSeconds = (Date.now() - started) / 1000;
    writeFileSync(confirmPath, readFileSync(confirmPath, 'utf8') + `${idx + 1}\t${o.name}\t${o.weights.av}\t${o.weights.rank}\t${o.weights.regret}\t${o.weights.risk}\t${o.weights.uncertainty}\t${o.weights.cpuct}\t${o.weights.fpu}\t${result.scoreRate.toFixed(4)}\t${result.score}\t${result.games}\t${wallSeconds.toFixed(3)}\t${arenaPath}\n`);
    appendLedger(ledgerPath, { trial_id: `${o.name}-confirm`, status: 'succeeded', params: { visits, cpuct: o.weights.cpuct, fpu: o.weights.fpu, av_weight: o.weights.av, rank_weight: o.weights.rank, regret_weight: o.weights.regret, risk_weight: o.weights.risk, uncertainty_weight: o.weights.uncertainty, confirmation: true }, score: result.scoreRate, raw_metrics: result, cost: { wall_seconds: wallSeconds, gpu_hours: 0, games: result.games, visits }, artifacts: [arenaPath], createdUtc: new Date().toISOString() });
  }
}
const finalRanked = obs.slice().sort((a, b) => b.scoreRate - a.scoreRate || b.games - a.games);
console.log('BAYES_AUX_TOP');
for (const o of finalRanked.slice(0, 10)) console.log(`rank iter=${o.iter} name=${o.name} scoreRate=${o.scoreRate.toFixed(4)} games=${o.games} av=${o.weights.av} rankW=${o.weights.rank} regret=${o.weights.regret} cpuct=${o.weights.cpuct} fpu=${o.weights.fpu}`);
