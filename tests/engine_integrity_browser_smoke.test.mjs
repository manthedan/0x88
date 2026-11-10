import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import test from 'node:test';
import { parseIntegritySmokeArgs } from '../scripts/lib/engine_integrity_smoke.mjs';

const script = 'scripts/engine_integrity_browser_smoke.mjs';

test('integrity smoke parses flag-only invocations before resolving the dist directory', () => {
  const result = spawnSync(process.execPath, [script, '--ui-only', '--help'], { encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /Usage:.*\[dist-directory\].*\[--ui-only\]/);
});

test('integrity smoke accepts flags independently of its optional positional path', () => {
  assert.deepEqual(parseIntegritySmokeArgs([]), { dist: resolve('dist-client'), uiOnly: false, requireAll: false });
  for (const argv of [['--ui-only'], ['dist-client', '--ui-only'], ['--ui-only', 'dist-client']]) {
    assert.deepEqual(parseIntegritySmokeArgs(argv), { dist: resolve('dist-client'), uiOnly: true, requireAll: false });
  }
  assert.deepEqual(parseIntegritySmokeArgs(['other-dist']), { dist: resolve('other-dist'), uiOnly: false, requireAll: false });
  assert.throws(() => parseIntegritySmokeArgs(['a', 'b']), /at most one dist directory/);
  assert.throws(() => parseIntegritySmokeArgs(['--unknown']), /Unknown option/);
});

test('UI journeys require their own engines and full checks can report missing artifacts', async () => {
  const { integrityEngineSpecs, integrityUiAssets, missingIntegrityAssets, integritySmokeExitCode } = await import('../scripts/lib/engine_integrity_smoke.mjs');
  const play = integrityUiAssets('/app/play/');
  const arena = integrityUiAssets('/app/arena/');
  assert.ok(play.some((path) => path.startsWith('/stockfish/')));
  assert.ok(arena.some((path) => path.startsWith('/berserk/')));
  assert.ok([...play, ...arena].every((path) => !/reckless|viridithas|plentychess|stormphrax|stockfish-19\.wasm/.test(path)));
  const reckless = integrityEngineSpecs.find((spec) => spec.family === 'reckless');
  assert.deepEqual(
    missingIntegrityAssets(
      reckless.assets,
      (path) => path,
      () => false,
    ),
    reckless.assets,
  );
  assert.deepEqual(
    missingIntegrityAssets(
      play,
      (path) => path,
      () => true,
    ),
    [],
  );
  assert.equal(integritySmokeExitCode({ failures: 0, skipped: 2 }, false), 0);
  assert.equal(integritySmokeExitCode({ failures: 0, skipped: 2 }, true), 1);
  assert.equal(integritySmokeExitCode({ failures: 1, skipped: 0 }, false), 1);
});

test('integrity smoke can make skipped checks fatal with require-all', () => {
  assert.equal(parseIntegritySmokeArgs(['--require-all']).requireAll, true);
  assert.equal(parseIntegritySmokeArgs(['--ui-only', 'dist-client', '--require-all']).uiOnly, true);
});

test('integrity smoke reports missing artifacts as skips and require-all makes them fatal', () => {
  const directory = mkdtempSync(join(tmpdir(), 'integrity-smoke-test-'));
  try {
    for (const [flags, expectedSkipped, expectedExit] of [
      [[], 18, 0],
      [['--require-all'], 18, 1],
      [['--ui-only'], 2, 0],
    ]) {
      const result = spawnSync(process.execPath, [script, directory, ...flags], {
        encoding: 'utf8',
        env: { ...process.env, ENGINE_INTEGRITY_BROWSER_SCRATCH: directory },
        timeout: 30_000,
      });
      assert.equal(result.status, expectedExit, result.stdout + result.stderr);
      assert.equal((result.stdout.match(/^SKIPPED /gm) ?? []).length, expectedSkipped, result.stdout);
      assert.match(result.stdout, /SKIPPED .*missing artifacts:/);
      assert.match(result.stdout, new RegExp(`RESULT 0 passed, 0 failed, ${expectedSkipped} skipped`));
      if (flags.includes('--ui-only')) assert.doesNotMatch(result.stdout, /Reckless|Viridithas|PlentyChess|Stormphrax/);
    }
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

test('only confirmed optional HEAD resource errors are excluded from UI console failures', async () => {
  const { actionableIntegrityErrors } = await import('../scripts/lib/engine_integrity_smoke.mjs');
  const text = 'Failed to load resource: the server responded with a status of 404 (Not Found)';
  const optional = 'http://localhost/reckless/reckless.wasm';
  const entries = [
    { kind: 'console', text, url: optional },
    { kind: 'console', text, url: 'http://localhost/stockfish/required.wasm' },
    { kind: 'console', text: 'Uncaught engine error', url: optional },
    { kind: 'pageerror', text, url: optional },
  ];
  assert.deepEqual(actionableIntegrityErrors(entries, new Set([optional])), [text, 'Uncaught engine error', text]);
  assert.deepEqual(
    actionableIntegrityErrors(entries, new Set([optional]), new Set([optional])),
    entries.map((entry) => entry.text),
    'a failed GET remains actionable even if a HEAD probe also failed for the same URL',
  );
  assert.deepEqual(
    actionableIntegrityErrors(entries, new Set()),
    entries.map((entry) => entry.text),
  );
});
