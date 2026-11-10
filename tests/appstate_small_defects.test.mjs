import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import { test } from 'node:test';
import { gameTreeToPgn } from '../src/chess/pgn.ts';
import { arenaAdjudicationLabel, arenaAdjudicationTags } from '../src/lc0/arenaAdjudication.ts';
import { writeClipboard } from '../src/lc0/clipboard.ts';
import { GameTree } from '../src/lc0/gameTree.ts';

const analysis = readFileSync(new URL('../src/lc0/analysisBrowser.ts', import.meta.url), 'utf8');

test('Analysis FEN and PGN copy show failure and retain text when the clipboard rejects or is missing', async () => {
  for (const clipboard of [
    undefined,
    {
      async writeText() {
        throw Error('denied');
      },
    },
  ]) {
    const elements = new Map(['message', 'pgnInput', 'fenInput'].map((key) => [key, { value: '', textContent: '' }]));
    const tree = new GameTree();
    tree.addUci('e2e4');
    const code = analysis.slice(analysis.indexOf('async function copyFen()'), analysis.indexOf('function closePvPreview()'));
    const dependencies = {
      tree,
      workspaceHeaders: {},
      workspaceResult: '*',
      reviewMetadata: new Map(),
      reviewSignature: () => '',
      pgnResultForExport: (headers, result) => ({ headers, result }),
      gameTreeToPgn,
      writeClipboard: (text) => writeClipboard(text, clipboard),
      mountAbort: new AbortController(),
      isStaleMount: () => false,
      el: (key) => elements.get(key),
      inputEl: (key) => elements.get(key),
    };
    const api = new Function(...Object.keys(dependencies), `${stripTypeScriptTypes(code)}; return {copyFen, copyPgn};`)(...Object.values(dependencies));
    assert.equal(await api.copyFen(), false);
    assert.match(elements.get('message').textContent, /Could not copy FEN/);
    assert.equal(elements.get('fenInput').value, tree.current.fen);
    assert.equal(await api.copyPgn(), false);
    assert.match(elements.get('message').textContent, /Could not copy PGN/);
    assert.match(elements.get('pgnInput').value, /1\. e4/);
  }
  let copied;
  assert.equal(
    await writeClipboard('PGN', {
      async writeText(text) {
        copied = text;
      },
    }),
    true,
  );
  assert.equal(copied, 'PGN');
});

test('pending Analysis clipboard writes cannot update another mount or access removed status nodes', async () => {
  const copies = analysis.slice(analysis.indexOf('async function copyFen()'), analysis.indexOf('function closePvPreview()'));
  const review = analysis.slice(analysis.indexOf('async function copyReviewPgn()'), analysis.indexOf('function hasActiveAnalysisRun()'));
  for (const action of ['copyFen', 'copyPgn', 'copyReviewPgn']) {
    for (const rejected of [false, true]) {
      let finish;
      const clipboard = {
        writeText: () =>
          new Promise((resolve, reject) => {
            finish = () => (rejected ? reject(Error('denied')) : resolve());
          }),
      };
      const controller = new AbortController();
      const dependencies = {
        tree: new GameTree(),
        workspaceHeaders: {},
        workspaceResult: '*',
        reviewMetadata: new Map(),
        reviewSignature: () => '',
        pgnResultForExport: (headers, result) => ({ headers, result }),
        gameTreeToPgn,
        reviewedPgn: () => '1. e4 *',
        navigator: { clipboard },
        writeClipboard: (text) => writeClipboard(text, clipboard),
        mountAbort: controller,
        isStaleMount: (signal) => signal.aborted,
      };
      let unmounted = false;
      const status = { value: '', textContent: 'New page status' };
      const el = () => {
        assert.equal(unmounted, false, 'stale clipboard handler touched the DOM');
        return status;
      };
      const api = new Function(
        ...Object.keys(dependencies),
        'el',
        'inputEl',
        `${stripTypeScriptTypes(copies + review)}; return {copyFen, copyPgn, copyReviewPgn};`,
      )(...Object.values(dependencies), el, el);
      const pending = api[action]();
      controller.abort();
      unmounted = true;
      finish();
      await assert.doesNotReject(pending, `${action} (${rejected ? 'rejected' : 'resolved'})`);
      assert.equal(status.textContent, 'New page status');
    }
  }
});

test('move-limit draws export explicit adjudication and have a visible, descriptive label', () => {
  const tags = arenaAdjudicationTags('max plies');
  const pgn = gameTreeToPgn(new GameTree(), tags, '1/2-1/2');
  assert.match(pgn, /\[Termination "adjudication"\]/);
  assert.match(pgn, /\[Adjudication "Draw at the 300-ply limit"\]/);
  assert.match(arenaAdjudicationLabel('max plies'), /Adjudicated draw.*300-ply limit/);
  assert.deepEqual(arenaAdjudicationTags('checkmate'), {});
  assert.equal(arenaAdjudicationLabel('checkmate'), null);
});
