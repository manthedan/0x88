#!/usr/bin/env node
import { readFileSync } from 'node:fs';

const spec = JSON.parse(readFileSync(new URL('./benchmark_spec.json', import.meta.url), 'utf8'));
const variant = process.env.TINY_LEELA_SEARCH_VARIANT ?? 'argmax';
const eps = 1e-12;

function normalize(entries) {
  const total = Object.values(entries).reduce((a, b) => a + b, 0);
  return Object.fromEntries(Object.entries(entries).map(([k, v]) => [k, v / total]));
}

function rootDistribution(policy) {
  const moves = Object.keys(policy);
  if (variant === 'argmax') {
    const best = moves.reduce((a, b) => policy[a] >= policy[b] ? a : b);
    return Object.fromEntries(moves.map((m) => [m, m === best ? 1 : 0]));
  }
  if (variant === 'prior_proportional') return { ...policy };
  if (variant === 'sqrt_prior') return normalize(Object.fromEntries(moves.map((m) => [m, Math.sqrt(policy[m])])));
  throw new Error(`Unknown TINY_LEELA_SEARCH_VARIANT=${variant}`);
}

let top1 = 0;
let top3 = 0;
let crossEntropy = 0;
let suiteScore = 0;

for (const pos of spec.fixed_policy_suite) {
  const teacher = pos.teacher_policy;
  const dist = rootDistribution(teacher);
  const teacherSorted = Object.entries(teacher).sort((a, b) => b[1] - a[1]).map(([m]) => m);
  const predictedSorted = Object.entries(dist).sort((a, b) => b[1] - a[1]).map(([m]) => m);
  if (predictedSorted[0] === teacherSorted[0]) top1 += 1;
  if (predictedSorted.slice(0, 3).some((m) => teacherSorted.slice(0, 3).includes(m))) top3 += 1;
  const ce = Object.entries(teacher).reduce((acc, [move, prob]) => acc - prob * Math.log(Math.max(eps, dist[move] ?? 0)), 0);
  crossEntropy += ce;
  // Bounded score rewards matching visit distribution and top move under frozen fixtures.
  suiteScore += 100 / (1 + ce) + (predictedSorted[0] === teacherSorted[0] ? 25 : 0);
}

const n = spec.fixed_policy_suite.length;
console.log(`METRIC fixed_playout_suite_score=${(suiteScore / n).toFixed(6)}`);
console.log(`METRIC policy_top1_acc=${(top1 / n).toFixed(6)}`);
console.log(`METRIC policy_top3_acc=${(top3 / n).toFixed(6)}`);
console.log(`METRIC policy_cross_entropy=${(crossEntropy / n).toFixed(6)}`);
console.log(`METRIC search_suite_positions=${n}`);
console.log(`METRIC phase_b_benchmark_ready=1`);
