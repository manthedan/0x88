import assert from 'node:assert/strict';
import test from 'node:test';
import { ALLOWLIST_PATH, compareWithAllowlist, inventory, loadAllowlist, scanTestSource, whoReads } from '../scripts/check_source_text_reads.mjs';

// Paths under src/fake/ do not exist, so this file never shows up in the real inventory.
const tree = {
  files: ['src/fake/analysisBrowser.ts', 'src/fake/playBrowser.ts', 'src/fake/arenaBrowser.ts', 'src/fake/util.ts'],
  dirs: new Set(['src/', 'src/fake/']),
};
const scan = (text) => scanTestSource(text, 'tests/x.test.mjs', tree);
// Template holes are written #{x} so these fixture strings are not mistaken for templates by the linter.
const holes = (text) => text.replaceAll('#{', '${');

test('every test that reads src/ as text is in the allowlist with a reason, and reads exactly what it lists', () => {
  const problems = compareWithAllowlist(inventory(), loadAllowlist());
  assert.deepEqual(
    problems,
    [],
    `Prefer testing behavior through exports. If a test must read source, add it to ${ALLOWLIST_PATH} via\n` +
      '  node scripts/check_source_text_reads.mjs --update --reason "<why>"',
  );
});

test('template paths expand to the files named by literals in the test', () => {
  const text = holes(
    "import { readFileSync } from 'node:fs';\nfor (const surface of ['analysis', 'play']) readFileSync(new URL(`../src/fake/#{surface}Browser.ts`, import.meta.url));",
  );
  assert.deepEqual(scan(text), ['src/fake/analysisBrowser.ts', 'src/fake/playBrowser.ts']);
});

test('computed template holes fall back to every matching file', () => {
  const text = holes("import { readFileSync } from 'node:fs';\nconst s = pick(); readFileSync(new URL(`../src/fake/#{s}Browser.ts`, import.meta.url));");
  assert.deepEqual(scan(text), ['src/fake/analysisBrowser.ts', 'src/fake/arenaBrowser.ts', 'src/fake/playBrowser.ts']);
});

test('imports, scratch-tree joins and files without a read API are not source reads', () => {
  const imports = "import { readFileSync } from 'node:fs';\nimport { x } from '../src/fake/util.ts';\nawait import('../src/fake/playBrowser.ts');";
  assert.deepEqual(scan(imports), []);
  assert.deepEqual(scan("import { readFileSync } from 'node:fs';\nreadFileSync(join(tmpDir, 'src', 'unit.cpp'));"), []);
  assert.deepEqual(scan("run(['--files', 'src/fake/util.ts']);"), []);
});

test('cwd-relative, root-joined and helper-argument paths count', () => {
  const text =
    "import { readFile } from 'node:fs/promises';\nreadFile('src/fake/util.ts');\nextract('../src/fake/arenaBrowser.ts', 'a', 'b');\nscan(join(ROOT, 'src'));";
  assert.deepEqual(scan(text), ['src/', 'src/fake/arenaBrowser.ts', 'src/fake/util.ts']);
});

test('who-reads matches exact files and directory scans that cover them', () => {
  const found = { 'tests/a.test.mjs': ['src/fake/util.ts'], 'tests/b.test.mjs': ['src/'], 'tests/c.test.mjs': ['src/fake/playBrowser.ts'] };
  assert.deepEqual(
    whoReads(found, 'src/fake/util.ts').map((hit) => hit.test),
    ['tests/a.test.mjs', 'tests/b.test.mjs'],
  );
});
