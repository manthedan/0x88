import assert from 'node:assert/strict';
import { existsSync, statSync } from 'node:fs';
import { resolve } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

// Assets CI genuinely cannot have. Anything else a test reads must exist in CI:
// models come from `npm run test:stage-lc0-assets` (scripts/stage_lc0_test_assets.mjs
// REQUIRED_ASSETS), fixtures are committed.
const CI_UNAVAILABLE = [
  // The lc0 C++ source checkout (../../repos/lc0); CI clones only this repo.
  /(^|\/)repos\/lc0\//,
  // The project's Python ONNX environment; CI does not provision Python.
  /(^|\/)\.venv-onnx\//,
];

const inCi = !!process.env.CI && process.env.CI !== 'false' && process.env.CI !== '0';

function present(path) {
  if (!existsSync(path)) return false;
  const stat = statSync(path);
  return stat.isDirectory() || stat.size > 0;
}

/**
 * Value for a test's `skip` option: false when every path is present, else a
 * reason. Locally a missing asset skips. Under CI it also registers a failing
 * test, unless every missing path is allowlisted above, so an unstaged asset can
 * no longer pass as coverage.
 */
export function skipUnlessAssets(...paths) {
  const missing = paths.map((path) => (path instanceof URL ? fileURLToPath(path) : resolve(String(path)))).filter((path) => !present(path));
  if (!missing.length) return false;
  const reason = `missing ${missing.join(', ')}`;
  const unexpected = missing.filter((path) => !CI_UNAVAILABLE.some((pattern) => pattern.test(path.replace(/\\/g, '/'))));
  if (inCi && unexpected.length) {
    test(`CI has the test assets: ${unexpected.join(', ')}`, () => {
      assert.fail(`${reason}. Stage it in CI (scripts/stage_lc0_test_assets.mjs) or allowlist it in tests/lib/required_assets.mjs.`);
    });
  }
  return reason;
}
