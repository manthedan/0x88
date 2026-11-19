import assert from 'node:assert/strict';
import test from 'node:test';
import { checkSyncPhraseComments, syncPhraseCommentLines } from '../scripts/check_comment_sync_phrases.mjs';

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

test('the allowlist must shrink: a fixed file left on it is a failure', () => {
  assert.deepEqual(checkSyncPhraseComments([{ path: 'src/a.ts', line: 3 }], {}), ['src/a.ts:3: comment asks to keep code in sync']);
  assert.deepEqual(checkSyncPhraseComments([], { 'src/b.ts': 'wave 2' }), [
    'src/b.ts: allowlisted but has no sync-phrase comment left; remove it from ALLOWLIST',
  ]);
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
