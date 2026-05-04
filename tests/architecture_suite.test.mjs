import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';

function run(spec) {
  const output = execFileSync('node', ['--experimental-strip-types', 'eval/architecture_suite.mjs'], {
    encoding: 'utf8',
    env: { ...process.env, TINY_LEELA_MODEL_SPEC: spec },
  });
  return Object.fromEntries([...output.matchAll(/^METRIC ([^=]+)=([0-9.]+)/gm)].map((m) => [m[1], Number(m[2])]));
}

test('architecture suite distinguishes micro and balanced student specs', () => {
  const micro = run('micro_16x2_int8');
  const balanced = run('balanced_48x5_int8');
  assert.ok(balanced.policy_top1_acc > micro.policy_top1_acc);
  assert.ok(balanced.model_size_mb > micro.model_size_mb);
  assert.ok(balanced.architecture_score > micro.architecture_score);
});
