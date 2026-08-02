import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import test from 'node:test';

test('final validation runs tests and SvelteKit build serially', () => {
  const stdout = execFileSync(process.execPath, ['scripts/validate_change.mjs', '--mode', 'final', '--with-build', '--dry-run'], { encoding: 'utf8' });
  const plan = JSON.parse(stdout);
  assert.equal(plan.parallel, false);
  assert.deepEqual(
    plan.steps.map((step) => step.name),
    ['full-test', 'build-client'],
  );
});
