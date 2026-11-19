import assert from 'node:assert/strict';
import test from 'node:test';
import { diagBlockHtml } from '../src/lc0/arenaDiagnostics.ts';

test('diagBlockHtml escapes label and values so markup renders as text', () => {
  assert.equal(
    diagBlockHtml('A<b>', ['x < y & "z"', 'plain']),
    '<span class="diag-label">A&lt;b&gt;</span><span class="diag-value">x &lt; y &amp; &quot;z&quot;</span>' +
      '<span class="diag-label"></span><span class="diag-value">plain</span>',
  );
});

test('diagBlockHtml shows a dash when there are no values', () => {
  assert.equal(diagBlockHtml('Reckless', []), '<span class="diag-label">Reckless</span><span class="diag-value">—</span>');
});
