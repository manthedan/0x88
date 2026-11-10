import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import test from 'node:test';

function dryRunPlan(...args) {
  return JSON.parse(execFileSync(process.execPath, ['scripts/validate_change.mjs', ...args, '--dry-run'], { encoding: 'utf8' }));
}

test('final validation runs tests and SvelteKit build serially', () => {
  const plan = dryRunPlan('--mode', 'final', '--with-build');
  assert.equal(plan.parallel, false);
  assert.deepEqual(
    plan.steps.map((step) => step.name),
    ['biome', 'full-test', 'build-client'],
  );
});

test('both validation modes run Biome', () => {
  for (const args of [
    ['--mode', 'quick'],
    ['--mode', 'quick', '--skip-typecheck'],
    ['--mode', 'final'],
  ]) {
    const biome = dryRunPlan(...args).steps.find((step) => step.name === 'biome');
    assert.deepEqual(biome && [biome.command, ...biome.args], ['npm', 'run', 'check:biome'], args.join(' '));
  }
});
