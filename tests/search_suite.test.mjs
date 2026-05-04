import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';

function run(variant) {
  const output = execFileSync('node', ['eval/search_suite.mjs'], {
    encoding: 'utf8',
    env: { ...process.env, TINY_LEELA_SEARCH_VARIANT: variant },
  });
  return Object.fromEntries([...output.matchAll(/^METRIC ([^=]+)=([0-9.]+)/gm)].map((m) => [m[1], Number(m[2])]));
}

test('search suite rewards visit distributions, not just argmax moves', () => {
  const argmax = run('argmax');
  const proportional = run('prior_proportional');
  assert.equal(argmax.policy_top1_acc, 1);
  assert.equal(proportional.policy_top1_acc, 1);
  assert.ok(proportional.policy_cross_entropy < argmax.policy_cross_entropy);
  assert.ok(proportional.fixed_playout_suite_score > argmax.fixed_playout_suite_score);
});
