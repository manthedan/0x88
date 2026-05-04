import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const spec = JSON.parse(readFileSync(new URL('../eval/benchmark_spec.json', import.meta.url), 'utf8'));

test('Phase B benchmark freezes real research metrics', () => {
  assert.equal(spec.benchmark_id, 'tiny_leela_phase_b_fixed_metrics_v1');
  const names = spec.primary_research_metrics.map((m) => m.name);
  assert.ok(names.includes('policy_top1_acc'));
  assert.ok(names.includes('wdl_cross_entropy'));
  assert.ok(names.includes('fixed_playout_suite_score'));
  assert.ok(names.includes('median_move_latency_ms'));
  assert.ok(!names.includes('tiny_leela_score'));
});

test('fixed policy suite is deterministic and normalized', () => {
  assert.ok(spec.fixed_policy_suite.length >= 3);
  for (const position of spec.fixed_policy_suite) {
    const sum = Object.values(position.teacher_policy).reduce((a, b) => a + b, 0);
    assert.ok(Math.abs(sum - 1) < 1e-9, `${position.id} sums to ${sum}`);
  }
});
