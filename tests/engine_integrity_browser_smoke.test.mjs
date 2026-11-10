import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { resolve } from 'node:path';
import test from 'node:test';
import { parseIntegritySmokeArgs } from '../scripts/lib/engine_integrity_smoke.mjs';

const script = 'scripts/engine_integrity_browser_smoke.mjs';

test('integrity smoke parses flag-only invocations before resolving the dist directory', () => {
  const result = spawnSync(process.execPath, [script, '--ui-only', '--help'], { encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /Usage:.*\[dist-directory\].*\[--ui-only\]/);
});

test('integrity smoke accepts flags independently of its optional positional path', () => {
  assert.deepEqual(parseIntegritySmokeArgs([]), { dist: resolve('dist-client'), uiOnly: false });
  for (const argv of [['--ui-only'], ['dist-client', '--ui-only'], ['--ui-only', 'dist-client']]) {
    assert.deepEqual(parseIntegritySmokeArgs(argv), { dist: resolve('dist-client'), uiOnly: true });
  }
  assert.deepEqual(parseIntegritySmokeArgs(['other-dist']), { dist: resolve('other-dist'), uiOnly: false });
  assert.throws(() => parseIntegritySmokeArgs(['a', 'b']), /at most one dist directory/);
  assert.throws(() => parseIntegritySmokeArgs(['--unknown']), /Unknown option/);
});
