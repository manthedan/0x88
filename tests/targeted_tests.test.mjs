import assert from 'node:assert/strict';
import { execFileSync, spawnSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { buildImportGraph, selectTargetedTests } from '../scripts/lib/targeted_tests.mjs';

const repoRoot = fileURLToPath(new URL('..', import.meta.url));
const graph = buildImportGraph(repoRoot);

function runCli(args, cwd = repoRoot) {
  return spawnSync(process.execPath, [join(repoRoot, 'scripts/run_targeted_tests.mjs'), ...args], { cwd, encoding: 'utf8' });
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

function gitFixture(t, files) {
  const cwd = mkdtempSync(join(tmpdir(), 'targeted-test-deletions-'));
  t.after(() => rmSync(cwd, { recursive: true, force: true }));
  const git = (...args) => execFileSync('git', ['-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', ...args], { cwd });
  git('init', '-q');
  for (const [file, contents] of Object.entries(files)) {
    const target = join(cwd, file);
    mkdirSync(dirname(target), { recursive: true });
    writeFileSync(target, contents);
  }
  git('add', ...Object.keys(files));
  git('commit', '-qm', 'fixture');
  return { cwd, git };
}

test('Git deletions still select mapped tests without failing on an unmapped deletion', (t) => {
  const { cwd } = gitFixture(t, {
    'src/obsolete.ts': '',
    'tests/obsolete.test.mjs': '',
    'src/covered.ts': '',
    'tests/covered.test.mjs': '',
  });
  for (const file of ['src/obsolete.ts', 'tests/obsolete.test.mjs', 'src/covered.ts']) rmSync(join(cwd, file));

  const result = runCli(['--base', 'HEAD', '--dry-run'], cwd);
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /tests\/covered\.test\.mjs/);
  assert.doesNotMatch(result.stdout, /tests\/obsolete\.test\.mjs/);
});

test('a source deletion in the base diff selects surviving tests that import it', (t) => {
  const { cwd, git } = gitFixture(t, {
    'src/lc0/searchWorkerCore.ts': '',
    'tests/search_worker_init_abort.test.mjs': "import '../src/lc0/searchWorkerCore.ts';",
  });
  git('rm', 'src/lc0/searchWorkerCore.ts');
  git('commit', '-qm', 'delete worker core');

  const result = runCli(['--base', 'HEAD^', '--on-unmatched', 'full', '--dry-run'], cwd);
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /tests\/search_worker_init_abort\.test\.mjs/);
});

test('a deleted source with a surviving importer but no matching test triggers the full fallback', (t) => {
  const { cwd } = gitFixture(t, {
    'src/obsolete.ts': '',
    'src/importer.ts': "import './obsolete';",
  });
  rmSync(join(cwd, 'src/obsolete.ts'));

  const result = runCli(['--base', 'HEAD', '--on-unmatched', 'full', '--dry-run'], cwd);
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /^npm test$/m);
});

test('a staged addition missing from disk still fails closed when unmatched', (t) => {
  const { cwd, git } = gitFixture(t, { 'src/retained.ts': '' });
  const added = 'src/new_module.ts';
  writeFileSync(join(cwd, added), '');
  git('add', added);
  rmSync(join(cwd, added));

  const result = runCli(['--staged', '--dry-run'], cwd);
  assert.equal(result.status, 1);
  assert.match(result.stderr, /Refusing a zero-test pass/);
  assert.ok(result.stderr.includes(added));
});

test('an unmatched staged deletion is ignored even if the file is still on disk', (t) => {
  const { cwd, git } = gitFixture(t, { 'src/obsolete.ts': '' });
  git('rm', '--cached', 'src/obsolete.ts');

  const result = runCli(['--staged', '--dry-run'], cwd);
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /Ignored: src\/obsolete\.ts/);
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
