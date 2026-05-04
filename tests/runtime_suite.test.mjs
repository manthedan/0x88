import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';

function run(strategy) {
  const output = execFileSync('node', ['--experimental-strip-types', 'eval/runtime_suite.mjs'], {
    encoding: 'utf8',
    env: { ...process.env, TINY_LEELA_RUNTIME_STRATEGY: strategy },
  });
  return Object.fromEntries([...output.matchAll(/^METRIC ([^=]+)=([0-9.]+)/gm)].map((m) => [m[1], Number(m[2])]));
}

test('runtime suite distinguishes backend fallback strategies', () => {
  const webgpuOnly = run('webgpu_only');
  const fallback = run('progressive_fallback');
  assert.ok(fallback.compatible_profiles > webgpuOnly.compatible_profiles);
  assert.ok(fallback.browser_compatibility_score > webgpuOnly.browser_compatibility_score);
});
