import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { buildImportGraph, selectTargetedTests } from '../scripts/lib/targeted_tests.mjs';

const repoRoot = fileURLToPath(new URL('..', import.meta.url));
const graph = buildImportGraph(repoRoot);

function runCli(args) {
  return spawnSync(process.execPath, ['scripts/run_targeted_tests.mjs', ...args], { cwd: repoRoot, encoding: 'utf8' });
}

test('import graph build stays cheap enough for the quick path', () => {
  assert.ok(graph.buildMs < 5000, `graph build took ${graph.buildMs}ms`);
  assert.ok(graph.tests.length > 50);
});

test('a puct.ts-only change selects the search suites and the tests that import it', () => {
  const decision = selectTargetedTests(['src/search/puct.ts'], graph);
  assert.equal(decision.status, 'run');
  assert.equal(decision.uncovered.length, 0);
  for (const required of [
    'tests/puct_core.test.mjs',
    'tests/puct_contempt.test.mjs',
    'tests/puct_elo_contempt.test.mjs',
    'tests/puct_search_contempt.test.mjs',
    'tests/lc0_search.test.mjs',
    'tests/search_metamorphic.test.mjs',
    'tests/lc0_moves_left_utility.test.mjs',
    'tests/chess_ocean.test.mjs',
  ]) {
    assert.ok(decision.tests.includes(required), `missing ${required}`);
  }
});

test('search.ts is covered by the import walk, not only the PUCT family list', () => {
  const decision = selectTargetedTests(['src/lc0/search.ts'], graph);
  assert.equal(decision.status, 'run');
  assert.ok(decision.tests.includes('tests/lc0_search.test.mjs'));
  assert.ok(decision.tests.includes('tests/lc0_engine_battle.test.mjs'));
});

test('an engine adapter change selects that engine family', () => {
  const decision = selectTargetedTests(['src/lc0/stockfishEngine.ts'], graph);
  assert.equal(decision.status, 'run');
  assert.ok(decision.tests.some((file) => file.includes('stockfish')));
});

test('executable source with no matched tests fails closed', () => {
  const decision = selectTargetedTests(['src/not_a_module.ts'], graph);
  assert.equal(decision.status, 'fail');
  assert.deepEqual(decision.uncovered, ['src/not_a_module.ts']);
  assert.equal(decision.tests.length, 0);
});

test('a matched file does not hide an unmatched executable file', () => {
  const decision = selectTargetedTests(['src/search/puct.ts', 'src/not_a_module.ts'], graph);
  assert.equal(decision.status, 'fail');
  assert.deepEqual(decision.uncovered, ['src/not_a_module.ts']);
  assert.ok(decision.tests.includes('tests/puct_core.test.mjs'));
});

test('docs-only changes do not pretend to have run tests', () => {
  const decision = selectTargetedTests(['docs/perf_canary.md', 'README.md'], graph);
  assert.equal(decision.status, 'skip');
  assert.equal(decision.tests.length, 0);
  assert.deepEqual(decision.uncovered, []);
});

test('a known non-source mapping still selects its test', () => {
  const decision = selectTargetedTests(['.github/workflows/production-smoke.yml'], graph);
  assert.equal(decision.status, 'run');
  assert.ok(decision.tests.includes('tests/production_browser_smoke.test.mjs'));
});

test('CLI refuses a zero-test pass for puct no longer, and refuses an unknown source file', () => {
  const puct = runCli(['--files', 'src/search/puct.ts', '--dry-run']);
  assert.equal(puct.status, 0, puct.stderr);
  assert.match(puct.stdout, /tests\/puct_core\.test\.mjs/);
  assert.match(puct.stdout, /tests\/lc0_search\.test\.mjs/);
  assert.match(puct.stdout, /tests\/search_metamorphic\.test\.mjs/);
  assert.doesNotMatch(`${puct.stdout}\n${puct.stderr}`, /No targeted tests inferred/);

  const unknown = runCli(['--files', 'src/not_a_module.ts', '--dry-run']);
  assert.equal(unknown.status, 1);
  assert.match(unknown.stderr, /Refusing a zero-test pass/);
  assert.equal(unknown.stdout.trim(), '');

  const fallback = runCli(['--files', 'src/not_a_module.ts', '--on-unmatched', 'full', '--dry-run']);
  assert.equal(fallback.status, 0, fallback.stderr);
  assert.match(fallback.stdout, /^npm test$/m);
  assert.match(fallback.stderr, /--on-unmatched full/);

  const docs = runCli(['--files', 'docs/perf_canary.md', '--dry-run']);
  assert.equal(docs.status, 0, docs.stderr);
  assert.match(docs.stdout, /No executable source changes matched tests/);
});
