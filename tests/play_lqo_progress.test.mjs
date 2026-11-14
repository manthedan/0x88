import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import test from 'node:test';

// Run the real request function with a deferred worker evaluation: no DOM,
// browser, network or model assets. UI writes record what the next mount sees.
const source = readFileSync(new URL('../src/lc0/playBrowser.ts', import.meta.url), 'utf8');
const requestSource = source.slice(source.indexOf('async function ctxRequestEngineMove('), source.indexOf('async function ctxVerifySelectedEngineReady('));
for (const lifecycle of ['current', 'cancelled', 'disposed']) {
  for (const rejected of [false, true]) {
    test(`LQO policy progress belongs to its live turn (${lifecycle}, rejected=${rejected})`, async () => {
      const evaluation = Promise.withResolvers();
      const entered = Promise.withResolvers();
      const writes = [];
      const searcher = {
        loaded: false,
        config: { name: 'LQO', approxMb: 100 },
        evaluate: () => {
          entered.resolve();
          return evaluation.promise;
        },
      };
      const deps = {
        ctxSelectedEngine: () => ({ family: 'lc0', variant: 'lqo' }),
        ctxStrengthFor: () => 1,
        acquireBigNetSearcher: () => searcher,
        ctxSetEngineNote: (...args) => writes.push(['note', ...args]),
        ctxShowDownloadProgress: (...args) => writes.push(['show', ...args]),
        ctxHideDownloadProgress: () => writes.push(['hide']),
        ctxIsDisposed: (ctx) => ctx.disposed,
        moveToUci: (move) => move,
        lqoWhiteFirstPolicyBookMove: () => 'e2e4',
      };
      const request = new Function(...Object.keys(deps), `${stripTypeScriptTypes(requestSource)}; return ctxRequestEngineMove;`)(...Object.values(deps));
      const abort = new AbortController();
      const ctx = { opponent: { active: { level: 0 } }, humanColor: 'b', moves: [], positions: [], disposed: false };
      const running = request(ctx, abort.signal);
      await entered.promise;
      searcher.onDownloadProgress(10, 100);
      assert.ok(
        writes.some(([kind]) => kind === 'show'),
        'current load still reports progress',
      );
      if (lifecycle !== 'current') abort.abort();
      ctx.disposed = lifecycle === 'disposed';
      writes.length = 0; // After cancellation, the replacement owns the shared DOM.
      searcher.onDownloadProgress(20, 100);
      if (rejected) evaluation.reject(new Error('retired worker disposed'));
      else evaluation.resolve({ legalPriors: [] });
      if (rejected) await assert.rejects(running, /retired worker/);
      else assert.equal(await running, lifecycle === 'current' ? 'e2e4' : null);
      if (lifecycle === 'current') {
        assert.deepEqual(
          writes.map(([kind]) => kind),
          rejected ? ['show', 'hide'] : ['show', 'hide', 'note'],
        );
      } else {
        assert.deepEqual(writes, [], "late progress and finally must not show/hide the new game's progress");
      }
    });
  }
}
