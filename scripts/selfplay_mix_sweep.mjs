#!/usr/bin/env node
import { execFileSync, spawn } from 'node:child_process';
import { existsSync } from 'node:fs';

function arg(name, fallback = undefined) {
  const prefix = `${name}=`;
  const inline = process.argv.find((value) => value.startsWith(prefix));
  if (inline) return inline.slice(prefix.length);
  const i = process.argv.indexOf(name);
  return i >= 0 ? process.argv[i + 1] : fallback;
}

function run(cmd, args) {
  return execFileSync(cmd, args, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'inherit'] });
}

function runAsync(cmd, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(cmd, args, { stdio: ['ignore', 'pipe', 'pipe'] });
    let stdout = '';
    child.stdout.on('data', (chunk) => { stdout += chunk; });
    child.stderr.on('data', (chunk) => { process.stderr.write(chunk); });
    child.on('error', reject);
    child.on('close', (code) => code === 0 ? resolve(stdout) : reject(new Error(`${cmd} exited ${code}`)));
  });
}

function metrics(output) {
  return Object.fromEntries([...output.matchAll(/^METRIC ([^=]+)=(-?[0-9.]+)/gm)].map((m) => [m[1], Number(m[2])]));
}

const backend = arg('--backend', process.env.TINY_LEELA_BACKEND ?? 'rust');
const weights = arg('--weights', '0,0.1,0.25').split(',').map(Number);
const selfplayPath = arg('--selfplay', 'artifacts/selfplay_mix_sweep.jsonl');
const candidatePrefix = arg('--candidate-prefix', 'artifacts/selfplay_mix_candidate');
const games = arg('--games', '2');
const selfplayVisits = arg('--selfplay-visits', '1');
const maxPlies = arg('--max-plies', '8');
const epochs = arg('--epochs', '20');
const lr = arg('--lr', '0.05');
const arenaGames = arg('--arena-games', '2');
const arenaVisits = arg('--arena-visits', '1');
const adjudicate = arg('--adjudicate', 'terminal');
const adjudicateThreshold = arg('--adjudicate-threshold', '0.02');
const primaryConvArch = arg('--primary-conv-arch', '');
const parallel = process.argv.includes('--parallel-candidates') || process.env.TINY_LEELA_PARALLEL_CANDIDATES === '1';
const featureCache = arg('--feature-cache', primaryConvArch ? `artifacts/cache/conv_features_${primaryConvArch}.json` : '');
const trainPaths = arg('--train', 'data/teacher_labels.jsonl,data/stockfish_teacher_labels.jsonl').split(',').filter(Boolean);

const startedAt = Date.now();
const logProgress = (message) => process.stderr.write(`[mix-sweep backend=${backend}] ${message}\n`);
logProgress(`start weights=${weights.join(',')} selfplay=${selfplayPath}`);

if (!existsSync(selfplayPath) || process.argv.includes('--regenerate')) {
  logProgress(`generating self-play rows at ${selfplayPath}`);
  const selfplayOutput = run('npm', ['run', 'selfplay:generate', '--silent', '--', `--backend=${backend}`, `--games=${games}`, `--visits=${selfplayVisits}`, `--max-plies=${maxPlies}`, `--out=${selfplayPath}`, `--adjudicate=${adjudicate}`, `--adjudicate-threshold=${adjudicateThreshold}`]);
  process.stderr.write(selfplayOutput);
  const selfplayMetrics = metrics(selfplayOutput);
  for (const [key, value] of Object.entries(selfplayMetrics)) console.log(`METRIC mix_${key}=${value.toFixed(6)}`);
}

if (featureCache && primaryConvArch) {
  logProgress(`rust feature cache prewarm start cache=${featureCache}`);
  run('cargo', ['build', '--release', '--quiet', '--manifest-path', 'rust/tiny_leela_core/Cargo.toml', '--bin', 'tiny-leela-rust-feature-cache']);
  const cacheOutput = run('rust/tiny_leela_core/target/release/tiny-leela-rust-feature-cache', [`--arch=${primaryConvArch}`, `--out=${featureCache}`, `--inputs=${[...trainPaths, selfplayPath].join(',')}`]);
  process.stderr.write(cacheOutput);
  for (const [key, value] of Object.entries(metrics(cacheOutput))) console.log(`METRIC mix_${key}=${value.toFixed(6)}`);
}

async function evaluateWeight(weight) {
  const out = `${candidatePrefix}_w${String(weight).replace(/\./g, 'p')}.json`;
  logProgress(`weight=${weight} train start out=${out} elapsed_s=${((Date.now() - startedAt) / 1000).toFixed(1)}`);
  const trainArgs = ['training/train_student.py', '--train', ...trainPaths, '--merge-fen', '--epochs', epochs, '--lr', lr, '--out', out, '--selfplay-weight', String(weight)];
  if (weight > 0) trainArgs.push('--selfplay-train', selfplayPath);
  if (primaryConvArch) trainArgs.push('--average-weights', '--average-policy-only', '--primary-conv-arch', primaryConvArch);
  if (featureCache) trainArgs.push('--feature-cache', featureCache);
  const trainOutput = parallel ? await runAsync('python3', trainArgs) : run('python3', trainArgs);
  const trainMetrics = metrics(trainOutput);
  logProgress(`weight=${weight} train done score=${(trainMetrics.distill_student_score ?? 0).toFixed(6)} arena start elapsed_s=${((Date.now() - startedAt) / 1000).toFixed(1)}`);
  const arenaArgs = ['run', 'eval:arena', '--silent', '--', `--backend=${backend}`, `--candidate=${out}`, '--baseline=artifacts/student_distill_benchmark.json', `--games=${arenaGames}`, `--visits=${arenaVisits}`, `--max-plies=${maxPlies}`, `--adjudicate=${adjudicate}`, `--adjudicate-threshold=${adjudicateThreshold}`];
  const arenaOutput = parallel ? await runAsync('npm', arenaArgs) : run('npm', arenaArgs);
  const arenaMetrics = metrics(arenaOutput);
  logProgress(`weight=${weight} arena done score=${(arenaMetrics.arena_score_rate ?? 0).toFixed(6)} illegal=${arenaMetrics.arena_illegal_losses ?? 0} elapsed_s=${((Date.now() - startedAt) / 1000).toFixed(1)}`);
  return { weight, out, trainMetrics, arenaMetrics };
}

const results = parallel ? await Promise.all(weights.map(evaluateWeight)) : [];
if (!parallel) for (const weight of weights) results.push(await evaluateWeight(weight));
let best = null;
for (const candidate of results) {
  const { weight, trainMetrics, arenaMetrics } = candidate;
  const prefix = `mix_${String(weight).replace(/\./g, 'p')}`;
  console.log(`METRIC ${prefix}_selfplay_weight=${weight.toFixed(6)}`);
  console.log(`METRIC ${prefix}_distill_student_score=${(trainMetrics.distill_student_score ?? 0).toFixed(6)}`);
  console.log(`METRIC ${prefix}_arena_score_rate=${(arenaMetrics.arena_score_rate ?? 0).toFixed(6)}`);
  console.log(`METRIC ${prefix}_arena_candidate_elo_estimate=${(arenaMetrics.arena_candidate_elo_estimate ?? 0).toFixed(6)}`);
  console.log(`METRIC ${prefix}_arena_illegal_losses=${arenaMetrics.arena_illegal_losses ?? 0}`);
  console.log(`METRIC ${prefix}_arena_adjudicated_rate=${(arenaMetrics.arena_adjudicated_rate ?? 0).toFixed(6)}`);
  if (!best || (arenaMetrics.arena_score_rate ?? -1) > (best.arenaMetrics.arena_score_rate ?? -1)) best = candidate;
}

console.log(`METRIC mix_backend_${backend}=1`);
console.log(`METRIC mix_weights_tested=${weights.length}`);
console.log(`METRIC mix_best_selfplay_weight=${(best?.weight ?? 0).toFixed(6)}`);
console.log(`METRIC mix_best_arena_score_rate=${(best?.arenaMetrics.arena_score_rate ?? 0).toFixed(6)}`);
console.log(`METRIC mix_best_distill_student_score=${(best?.trainMetrics.distill_student_score ?? 0).toFixed(6)}`);
