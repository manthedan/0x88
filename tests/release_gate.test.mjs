import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { test } from 'node:test';
import { checkReleaseGate, checkReleaseTree } from '../scripts/lib/release_gate.mjs';

const HEAD = 'a'.repeat(40);
const green = { status: 'completed', conclusion: 'success' };
const red = { status: 'completed', conclusion: 'failure' };

function harness({ dirty = '', lintStatus = 0, ciRuns = [green], smokeRuns = [{ ...green, event: 'schedule' }], ghFailure = false, gitFailure = false } = {}) {
  const calls = [];
  const logs = [];
  let elapsed = 0;
  let ciIndex = 0;
  const deps = {
    exec(bin, args, options) {
      calls.push([bin, args]);
      assert.ok(options.timeout > 0, 'all subprocesses have bounded timeouts');
      const ok = (stdout) => ({ status: 0, stdout, stderr: '' });
      if (bin === 'git') {
        if (gitFailure) return { status: 128, stdout: '', stderr: 'not a repository' };
        if (args[0] === 'rev-parse') return ok(HEAD);
        assert.deepEqual(args, ['status', '--porcelain', '--untracked-files=no']);
        return ok(dirty);
      }
      if (bin === 'npm' || bin === '/custom/npm') {
        assert.deepEqual(args, ['run', 'check:biome:tracked']);
        return { status: lintStatus, stdout: '', stderr: lintStatus ? 'Biome formatting error' : '' };
      }
      assert.equal(bin, 'gh');
      if (ghFailure) return { status: 1, stdout: '', stderr: 'authentication required' };
      if (args.includes('ci')) {
        assert.deepEqual(args, ['run', 'list', '--commit', HEAD, '--event', 'push', '--workflow', 'ci', '--limit', '1', '--json', 'conclusion,status']);
        const runs = ciRuns[Math.min(ciIndex++, ciRuns.length - 1)];
        return ok(JSON.stringify(runs ? [runs] : []));
      }
      assert.deepEqual(args, ['run', 'list', '--workflow', 'production-smoke', '--limit', '1', '--json', 'conclusion,status,event,displayTitle']);
      return ok(JSON.stringify(smokeRuns));
    },
    now: () => elapsed,
    sleep: async (ms) => {
      elapsed += ms;
    },
    log: (line) => logs.push(line),
    ciTimeoutMs: 30,
    pollIntervalMs: 10,
  };
  return { deps, calls, logs };
}

test('release gate passes a clean tracked tree, lint, exact green HEAD ci and latest green smoke', async () => {
  const h = harness();
  const result = await checkReleaseGate({ prod: true, npmBin: '/custom/npm' }, h.deps);
  assert.deepEqual(result, { commit: HEAD, ci: 'success', smoke: 'success', overrides: [] });
  assert.equal(h.calls.length, 5);
});

for (const dirty of [' M tracked.ts', 'M  staged.ts']) {
  test(`release gate rejects tracked changes: ${dirty}`, async () => {
    const h = harness({ dirty });
    await assert.rejects(
      checkReleaseGate({ prod: true, skipCiCheck: 'emergency', acceptRedSmoke: 'known smoke failure' }, h.deps),
      /Tracked working tree is dirty.*[\s\S]*Commit or discard/,
    );
    assert.equal(h.calls.length, 2);
  });
}

test('release gate does not treat unavailable git as a clean tree', async () => {
  const h = harness({ gitFailure: true });
  await assert.rejects(checkReleaseGate({ prod: true }, h.deps), /not a repository.*fix git/);
});

test('release gate rejects lint failure, even with both overrides', async () => {
  const h = harness({ lintStatus: 1 });
  await assert.rejects(
    checkReleaseGate({ prod: true, skipCiCheck: 'emergency', acceptRedSmoke: 'known failure' }, h.deps),
    /Biome formatting error.*npx biome check --write/,
  );
  assert.equal(h.calls.length, 3);
});

for (const conclusion of ['failure', 'cancelled', 'timed_out', 'skipped']) {
  test(`release gate rejects completed ci that concluded ${conclusion}`, async () => {
    const h = harness({ ciRuns: [{ status: 'completed', conclusion }] });
    await assert.rejects(checkReleaseGate({ prod: true }, h.deps), new RegExp(`ci for HEAD ${HEAD} concluded ${conclusion}.*--skip-ci-check`));
  });
}

test('release gate rejects missing exact-HEAD ci', async () => {
  const h = harness({ ciRuns: [] });
  await assert.rejects(checkReleaseGate({ prod: true }, h.deps), /No ci workflow run found.*Push this commit/);
});

test('release gate polls running ci with progress until success', async () => {
  const h = harness({ ciRuns: [{ status: 'queued', conclusion: '' }, { status: 'in_progress', conclusion: '' }, green] });
  assert.equal((await checkReleaseGate({ prod: true }, h.deps)).ci, 'success');
  assert.match(h.logs.join('\n'), /queued; waiting.*\n.*in_progress; waiting/);
  assert.equal(h.calls.filter(([bin, args]) => bin === 'gh' && args.includes('ci')).length, 3);
});

test('release gate times out instead of waiting forever for running ci', async () => {
  const h = harness({ ciRuns: [{ status: 'in_progress', conclusion: '' }] });
  await assert.rejects(checkReleaseGate({ prod: true }, h.deps), /Timed out after .*waiting for ci.*--skip-ci-check/);
  assert.equal(h.calls.filter(([bin]) => bin === 'gh').length, 3);
});

test('release gate rejects red smoke without override', async () => {
  const h = harness({ smokeRuns: [red] });
  await assert.rejects(checkReleaseGate({ prod: true }, h.deps), /production-smoke run is completed\/failure.*--accept-red-smoke/);
});

test('smoke override is trimmed, printed and returned for the deploy log', async () => {
  const h = harness({ smokeRuns: [red] });
  const result = await checkReleaseGate({ prod: true, acceptRedSmoke: '  known upstream outage  ' }, h.deps);
  assert.equal(result.smoke, 'overridden');
  assert.deepEqual(result.overrides, ['--accept-red-smoke: known upstream outage']);
  assert.match(h.logs.join('\n'), /WARNING override --accept-red-smoke: known upstream outage/);
  assert.equal(h.calls.filter(([bin]) => bin === 'gh').length, 1);
});

test('explicit ci override does not skip smoke', async () => {
  const h = harness({ ciRuns: [red] });
  const result = await checkReleaseGate({ prod: true, skipCiCheck: 'CI service outage' }, h.deps);
  assert.equal(result.ci, 'skipped');
  assert.equal(result.smoke, 'success');
  assert.deepEqual(result.overrides, ['--skip-ci-check: CI service outage']);
  assert.match(h.logs.join('\n'), /WARNING override --skip-ci-check: CI service outage/);
  assert.ok(h.calls.some(([bin, args]) => bin === 'gh' && args.includes('production-smoke')));
});

test('draft deployment skips ci with a warning, but still checks lint and smoke', async () => {
  const h = harness({ ciRuns: [red] });
  const result = await checkReleaseGate({ prod: false }, h.deps);
  assert.equal(result.ci, 'skipped');
  assert.match(h.logs.join('\n'), /WARNING: draft deploy.*skipping GitHub ci/);
  assert.ok(h.calls.some(([bin]) => bin === 'npm'));
  assert.equal(h.calls.filter(([bin]) => bin === 'gh').length, 1);
  await assert.rejects(checkReleaseGate({ prod: false }, harness({ smokeRuns: [red] }).deps), /production-smoke/);
});

for (const key of ['skipCiCheck', 'acceptRedSmoke']) {
  for (const reason of ['', ' \n\t ']) {
    test(`release gate rejects empty ${key} reason ${JSON.stringify(reason)} before commands`, async () => {
      const h = harness();
      await assert.rejects(checkReleaseGate({ prod: true, [key]: reason }, h.deps), /requires a non-empty reason/);
      assert.equal(h.calls.length, 0);
    });
  }
}

for (const smokeRuns of [[], [{ status: 'in_progress', conclusion: '' }]]) {
  test(`release gate fails closed on unknown smoke health: ${JSON.stringify(smokeRuns)}`, async () => {
    await assert.rejects(checkReleaseGate({ prod: true }, harness({ smokeRuns }).deps), /production health is not green/);
  });
}

test('green manual production smoke is accepted when its target is verified', async () => {
  const smokeRuns = [{ ...green, event: 'workflow_dispatch', displayTitle: 'production-smoke (https://0x88.app)' }];
  assert.equal((await checkReleaseGate({ prod: true }, harness({ smokeRuns }).deps)).smoke, 'success');
});

for (const displayTitle of ['production-smoke (https://preview.example)', 'production-smoke']) {
  test(`green manual smoke cannot clear production health for ${displayTitle}`, async () => {
    const smokeRuns = [{ ...green, event: 'workflow_dispatch', displayTitle }];
    await assert.rejects(checkReleaseGate({ prod: true }, harness({ smokeRuns }).deps), /did not verify https:\/\/0x88.app.*--accept-red-smoke/);
  });
}

test('manual smoke run-name and actual target use the same expression', async () => {
  const workflow = await readFile('.github/workflows/production-smoke.yml', 'utf8');
  const target = `\${{ inputs.base_url || 'https://0x88.app' }}`;
  assert.ok(workflow.includes(`run-name: production-smoke (${target})`));
  assert.ok(workflow.includes(`PRODUCTION_BASE_URL: ${target}`));
  assert.ok(workflow.includes('--base-url "$PRODUCTION_BASE_URL"'));
});

test('release gate explains how to repair a failed GitHub lookup', async () => {
  await assert.rejects(checkReleaseGate({ prod: true }, harness({ ghFailure: true }).deps), /authentication required.*gh auth login/);
});

for (const stdout of ['not json', '{}', '[null]']) {
  test(`release gate rejects malformed gh output: ${stdout}`, async () => {
    const h = harness();
    const exec = (bin, args, options) => (bin === 'gh' ? { status: 0, stdout } : h.deps.exec(bin, args, options));
    await assert.rejects(checkReleaseGate({ prod: true }, { ...h.deps, exec }), /gh returned .*invalid.*ci/);
  });
}

test('pre-deploy recheck rejects changed HEAD or build-generated tracked changes', () => {
  const h = harness();
  assert.equal(checkReleaseTree(HEAD, h.deps.exec), HEAD);
  assert.throws(() => checkReleaseTree('b'.repeat(40), h.deps.exec), /HEAD changed.*rerun/);
  assert.throws(() => checkReleaseTree(HEAD, harness({ dirty: ' M generated.ts' }).deps.exec), /Tracked working tree is dirty/);
});
