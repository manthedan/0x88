import assert from 'node:assert/strict';
import test from 'node:test';
import { checkSyncPhraseComments, syncPhraseCommentLines, syncPhraseComments } from '../scripts/check_comment_sync_phrases.mjs';

test('no comment in src/ asks to keep code in sync, outside the shrinking allowlist', () => {
  assert.deepEqual(checkSyncPhraseComments(), [], 'Share a constant, type or helper (or pin the relation in a test) instead of a "keep in sync" comment.');
});

test('sync phrases are found in line, block and JSDoc comments, not in code or strings', () => {
  const text = [
    'const a = 1; // keep in step with b',
    "/* Mirror lineChartSvg's layout */",
    '/**',
    ' * Copied from playBrowser.ts.',
    ' */',
    "const label = 'copied from clipboard';",
    'const ok = 2;',
  ].join('\n');
  assert.deepEqual(syncPhraseCommentLines(text), [1, 2, 3]);
});

test('each allowlist entry permits one matching comment, and an entry with no comment left fails', () => {
  const allowlist = [{ path: 'src/a.ts', comment: 'keep in step with foo', reason: 'wave 2' }];
  const permitted = { path: 'src/a.ts', line: 3, text: 'keep in step with foo' };
  assert.deepEqual(checkSyncPhraseComments([permitted], allowlist), []);
  assert.deepEqual(checkSyncPhraseComments([{ path: 'src/b.ts', line: 3, text: 'keep in step with foo' }], allowlist), [
    'src/a.ts: allowlisted comment "keep in step with foo" not found; remove it from ALLOWLIST',
    'src/b.ts:3: comment asks to keep code in sync',
  ]);
  assert.deepEqual(checkSyncPhraseComments([], allowlist), ['src/a.ts: allowlisted comment "keep in step with foo" not found; remove it from ALLOWLIST']);
});

test('a new sync comment in an allowlisted file still fails', () => {
  const allowlist = [{ path: 'src/a.ts', comment: 'keep in step with foo', reason: 'wave 2' }];
  const hits = [
    { path: 'src/a.ts', line: 3, text: 'keep in step with foo' },
    { path: 'src/a.ts', line: 9, text: 'Copied from bar.ts' },
    { path: 'src/a.ts', line: 12, text: 'also keep in step with foo' },
  ];
  assert.deepEqual(checkSyncPhraseComments(hits, allowlist), [
    'src/a.ts:9: comment asks to keep code in sync',
    'src/a.ts:12: comment asks to keep code in sync',
  ]);
});

test('an adjacent sync directive cannot hide in an allowlisted comment', () => {
  const path = 'src/a.ts';
  const allowlist = [{ path, comment: 'keep in step with foo', reason: 'wave 2' }];
  const text = 'const key = foo(); // keep in step with foo\n// keep in sync with another helper\nconst value = bar();';
  const hits = syncPhraseComments(text, path).map((hit) => ({ path, ...hit }));
  assert.match(checkSyncPhraseComments(hits, allowlist).join('\n'), /src\/a\.ts:\d+: comment asks to keep code in sync/);
});

test('comment markers inside strings, templates and regexes are not comments', () => {
  const text = [
    "const a = 'https://example.test was copied from the clipboard';",
    'const b = `/* keep in sync with b */` + a;',
    'const c = /\\/\\/ copied from/;',
    'const d = "<!-- copied from -->";',
  ].join('\n');
  assert.deepEqual(syncPhraseCommentLines(text), []);
});

test('a sync phrase split across comment lines is still found, at the comment start', () => {
  const text = [
    'const a = 1;',
    '/**',
    ' * Keep in sync',
    ' * with the chart layout.',
    ' */',
    'const b = 2;',
    '// These values were copied',
    '// from the old table.',
    'const c = 3;',
  ].join('\n');
  assert.deepEqual(syncPhraseCommentLines(text), [2, 7]);
});

test('svelte files: script comments, markup HTML comments and style comments, not markup text', () => {
  const text = [
    '<script lang="ts">',
    "const url = 'https://example.test copied from';",
    '// keep in sync',
    '// with Board.svelte',
    '</script>',
    '<p>Text copied from elsewhere</p>',
    '<!-- keep in step with',
    '     the header -->',
    '<style>',
    ".a::after { content: '/* copied from */'; }",
    "/* Mirror chart's layout */",
    '</style>',
  ].join('\n');
  assert.deepEqual(syncPhraseCommentLines(text, 'src/Fixture.svelte'), [3, 7, 11]);
});
