import assert from 'node:assert/strict';
import test from 'node:test';
import { checkSyncPhraseComments, syncPhraseCommentLines } from '../scripts/check_comment_sync_phrases.mjs';

test('no comment in src/ asks to keep code in sync, outside the shrinking allowlist', () => {
  assert.deepEqual(checkSyncPhraseComments(), [], 'Share a constant, type or helper (or pin the relation in a test) instead of a "keep in sync" comment.');
});

test('sync phrases are found in line, block, JSDoc, CSS and HTML comments, not in code or strings', () => {
  const text = [
    'const a = 1; // keep in step with b',
    "/* Mirror lineChartSvg's layout */",
    '/**',
    ' * Copied from playBrowser.ts.',
    ' */',
    '<!-- keep in sync with the header -->',
    "const label = 'copied from clipboard';",
    'const ok = 2;',
  ].join('\n');
  assert.deepEqual(syncPhraseCommentLines(text), [1, 2, 4, 6]);
});

test('the allowlist must shrink: a fixed file left on it is a failure', () => {
  assert.deepEqual(checkSyncPhraseComments([{ path: 'src/a.ts', line: 3 }], {}), ['src/a.ts:3: comment asks to keep code in sync']);
  assert.deepEqual(checkSyncPhraseComments([], { 'src/b.ts': 'wave 2' }), [
    'src/b.ts: allowlisted but has no sync-phrase comment left; remove it from ALLOWLIST',
  ]);
});
