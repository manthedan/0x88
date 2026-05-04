#!/usr/bin/env node
import { readFileSync } from 'node:fs';

const spec = JSON.parse(readFileSync(new URL('./benchmark_spec.json', import.meta.url), 'utf8'));

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

assert(spec.benchmark_id === 'tiny_leela_phase_b_fixed_metrics_v1', 'benchmark id drifted');
assert(Array.isArray(spec.primary_research_metrics) && spec.primary_research_metrics.length >= 8, 'not enough metrics');
assert(Array.isArray(spec.fixed_policy_suite) && spec.fixed_policy_suite.length >= 3, 'not enough fixed positions');

const metricNames = new Set(spec.primary_research_metrics.map((m) => m.name));
for (const name of ['policy_top1_acc', 'wdl_cross_entropy', 'fixed_playout_suite_score', 'median_move_latency_ms', 'model_size_mb']) {
  assert(metricNames.has(name), `missing metric ${name}`);
}

for (const pos of spec.fixed_policy_suite) {
  assert(typeof pos.id === 'string' && pos.id, 'position missing id');
  assert(typeof pos.fen === 'string' && pos.fen.split(' ').length >= 4, `${pos.id} missing FEN`);
  const sum = Object.values(pos.teacher_policy).reduce((a, b) => a + b, 0);
  assert(Math.abs(sum - 1) < 1e-9, `${pos.id} policy does not sum to 1`);
  assert(Array.isArray(pos.wdl) && pos.wdl.length === 3, `${pos.id} missing WDL`);
}

console.log(`METRIC phase_b_benchmark_ready=1`);
console.log(`METRIC fixed_positions=${spec.fixed_policy_suite.length}`);
console.log(`METRIC research_metrics_defined=${spec.primary_research_metrics.length}`);
console.log(`METRIC runtime_profiles_defined=${spec.fixed_runtime_profiles.length}`);
