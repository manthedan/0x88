#!/usr/bin/env node
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
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
function mulberry32(seed) {
  let t = seed >>> 0;
  return () => {
    t += 0x6D2B79F5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}
function keyOf(w) { return `${w.av.toFixed(5)}_${w.rank.toFixed(5)}_${w.regret.toFixed(5)}_${w.risk.toFixed(5)}_${w.uncertainty.toFixed(5)}`; }
function nameOf(w, idx) {
  const enc = (x) => String(Math.round(x * 10000)).padStart(3, '0');
  return `cand${idx}_a${enc(w.av)}_r${enc(w.rank)}_g${enc(w.regret)}_k${enc(w.risk)}_u${enc(w.uncertainty)}`;
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
function appendObs(path, obs) {
  const prev = existsSync(path) ? readFileSync(path, 'utf8') : '';
  writeFileSync(path, prev + JSON.stringify(obs) + '\n');
}
function dist2(a, b, maxWeight) {
  const s = (x) => x / maxWeight;
  return (s(a.av)-s(b.av))**2 + (s(a.rank)-s(b.rank))**2 + (s(a.regret)-s(b.regret))**2 + (s(a.risk)-s(b.risk))**2 + (s(a.uncertainty)-s(b.uncertainty))**2;
}
function surrogate(x, obs, maxWeight, lengthScale, beta) {
  if (!obs.length) return { mean: 0.5, std: 0.25, acq: 0.5 + beta * 0.25, pseudoGames: 0 };
  let sw = 0, sy = 0, sg = 0;
  for (const o of obs) {
    const k = Math.exp(-dist2(x, o.weights, maxWeight) / (2 * lengthScale * lengthScale));
    sw += k * o.games;
    sy += k * o.score;
    sg += k * o.games;
  }
  const mean = sg > 1e-9 ? sy / sg : 0.5;
  let rv = 0, rw = 0;
  for (const o of obs) {
    const k = Math.exp(-dist2(x, o.weights, maxWeight) / (2 * lengthScale * lengthScale));
    const y = o.scoreRate;
    rv += k * o.games * (y - mean) ** 2;
    rw += k * o.games;
  }
  const residual = rw > 1e-9 ? rv / rw : 0.03;
  const binom = Math.max(0.0025, mean * (1 - mean)) / Math.max(1, sw + 2);
  const prior = 0.02 / Math.sqrt(1 + sw);
  const std = Math.sqrt(binom + residual + prior * prior);
  return { mean, std, acq: mean + beta * std, pseudoGames: sw };
}
function makePool(rng, maxWeight, poolSize, dims) {
  const zero = { av:0, rank:0, regret:0, risk:0, uncertainty:0 };
  const points = [];
  const add = (w) => points.push({ ...zero, ...w });
  const vals = [0.0025, 0.005, 0.01, 0.02].filter((v) => v <= maxWeight + 1e-12);
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
    const w = { ...zero };
    for (const d of dims) {
      if (rng() < 0.72) {
        // Log-biased toward tiny weights, rounded to 0.0005 for resumable deduping.
        const raw = Math.exp(Math.log(0.001) + rng() * (Math.log(maxWeight) - Math.log(0.001)));
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
    return Object.values(w).some((v) => v !== 0);
  });
}

const model = arg('--model');
const meta = arg('--meta');
const outDir = arg('--out-dir', 'artifacts/head_ablation_1m/bayesian_aux_puct_tune');
if (!model || !meta) throw new Error('usage: node --experimental-strip-types eval/bayesian_aux_puct_tune.mjs --model m.onnx --meta m.meta.json --out-dir out');
const visits = num('--visits', 32);
const batchSize = num('--batch-size', 16);
const gamesPerCandidate = num('--games-per-candidate', 4);
const iterations = num('--iterations', 12);
const cpuct = num('--cpuct', 1.5);
const maxPlies = num('--max-plies', 100);
const maxWeight = num('--max-weight', 0.02);
const poolSize = num('--pool-size', 256);
const seed = num('--seed', 23);
const ortThreads = arg('--ort-threads', process.env.ORT_INTRA_OP_NUM_THREADS ?? '2');
const openingsFile = arg('--openings-file', 'eval/opening_suite_uho_lite_v1.fen');
const maxOpenings = num('--max-openings', 0);
const judgeModel = arg('--judge-model', '');
const judgeMeta = arg('--judge-meta', '');
const adjudicateThreshold = arg('--adjudicate-threshold', '0.05');
const beta = num('--beta', 0.9);
const lengthScale = num('--length-scale', 0.35);
const dims = arg('--dims', 'av,rank,regret').split(',').map(s => s.trim()).filter(Boolean);
mkdirSync(outDir, { recursive: true });
const obsPath = join(outDir, 'observations.jsonl');
const summaryPath = join(outDir, 'summary.tsv');
const statePath = join(outDir, 'state.json');
if (!existsSync(summaryPath)) writeFileSync(summaryPath, 'iter\tname\tav\trank\tregret\trisk\tuncertainty\tscore_rate\tscore\tgames\tposterior_mean\tposterior_std\tposterior_acq\tarena_json\n');
const rng = mulberry32(seed);
const pool = makePool(rng, maxWeight, poolSize, dims);
let obs = readObs(obsPath);
const evaluated = new Set(obs.map((o) => keyOf(o.weights)));
console.error(`[bayes-aux] out=${outDir} visits=${visits} games_per_candidate=${gamesPerCandidate} dims=${dims.join(',')} existing=${obs.length} pool=${pool.length}`);
for (let iter = obs.length; iter < iterations; iter++) {
  let chosen = null, pred = null;
  const initialBudget = Math.min(8, pool.length);
  const initial = pool.slice(0, initialBudget).filter((w) => !evaluated.has(keyOf(w)));
  if (obs.length < initialBudget && initial.length) chosen = initial[0];
  else {
    let best = null;
    for (const w of pool) {
      if (evaluated.has(keyOf(w))) continue;
      const s = surrogate(w, obs, maxWeight, lengthScale, beta);
      if (!best || s.acq > best.s.acq) best = { w, s };
    }
    if (!best) break;
    chosen = best.w;
    pred = best.s;
  }
  if (!pred) pred = surrogate(chosen, obs, maxWeight, lengthScale, beta);
  const name = nameOf(chosen, iter + 1);
  const arenaPath = join(outDir, `${String(iter + 1).padStart(3, '0')}_${name}.json`);
  console.error(`[bayes-aux] iter=${iter + 1}/${iterations} eval ${name} av=${chosen.av} rank=${chosen.rank} regret=${chosen.regret} pred_mean=${pred.mean.toFixed(3)} pred_std=${pred.std.toFixed(3)} acq=${pred.acq.toFixed(3)}`);
  if (!existsSync(arenaPath)) {
    const players = `classic:${model}:${meta}:puct,${name}:${model}:${meta}:aux:${chosen.av}:${chosen.rank}:${chosen.regret}:${chosen.risk}:${chosen.uncertainty}`;
    const args = [
      ...process.execArgv,
      'eval/search_mode_arena.mjs',
      `--players=${players}`,
      '--games-per-pair', String(gamesPerCandidate),
      '--visits', String(visits),
      '--batch-size', String(batchSize),
      '--cpuct', String(cpuct),
      '--max-plies', String(maxPlies),
      '--openings-file', openingsFile,
      '--out', arenaPath,
    ];
    if (maxOpenings > 0) args.push('--max-openings', String(maxOpenings));
    if (judgeModel && judgeMeta) args.push('--judge-model', judgeModel, '--judge-meta', judgeMeta, '--adjudicate-threshold', adjudicateThreshold);
    const child = spawnSync(process.execPath, args, { stdio: 'inherit', env: { ...process.env, ORT_INTRA_OP_NUM_THREADS: ortThreads } });
    if (child.status !== 0) throw new Error(`arena failed status=${child.status}`);
  }
  const result = scoreRateFromArena(arenaPath, name);
  const nextObs = { iter: iter + 1, name, weights: chosen, ...result, posteriorBefore: pred, arenaPath, createdUtc: new Date().toISOString() };
  appendObs(obsPath, nextObs);
  obs.push(nextObs);
  evaluated.add(keyOf(chosen));
  writeFileSync(summaryPath, readFileSync(summaryPath, 'utf8') + `${iter + 1}\t${name}\t${chosen.av}\t${chosen.rank}\t${chosen.regret}\t${chosen.risk}\t${chosen.uncertainty}\t${result.scoreRate.toFixed(4)}\t${result.score}\t${result.games}\t${pred.mean.toFixed(4)}\t${pred.std.toFixed(4)}\t${pred.acq.toFixed(4)}\t${arenaPath}\n`);
  const ranked = obs.slice().sort((a, b) => b.scoreRate - a.scoreRate || b.games - a.games);
  writeFileSync(statePath, JSON.stringify({ protocol: { model, meta, visits, batchSize, gamesPerCandidate, iterations, cpuct, maxPlies, maxWeight, dims, beta, lengthScale, seed, ortThreads, openingsFile, maxOpenings, judgeModel, judgeMeta, adjudicateThreshold }, best: ranked[0] ?? null, observations: obs, updatedUtc: new Date().toISOString() }, null, 2));
}
const ranked = obs.slice().sort((a, b) => b.scoreRate - a.scoreRate || b.games - a.games);
console.log('BAYES_AUX_TOP');
for (const o of ranked.slice(0, 10)) console.log(`rank iter=${o.iter} name=${o.name} scoreRate=${o.scoreRate.toFixed(4)} games=${o.games} av=${o.weights.av} rankW=${o.weights.rank} regret=${o.weights.regret}`);
