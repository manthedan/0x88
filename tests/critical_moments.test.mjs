import assert from 'node:assert/strict';
import test from 'node:test';
import { START_FEN } from '../src/chess/board.ts';
import { stockfishAnalysisLines, lc0AnalysisLines } from '../src/lc0/analysisFormat.ts';
import { criticalAgreement, criticalPositionLines, criticalReportMarkup } from '../src/lc0/criticalMomentsPresentation.ts';
import { CriticalAnalysisRefresh, createCriticalReport, criticalMomentCacheKey, criticalMoments, criticalReportIsStale, criticalReportSignature, runCriticalReport } from '../src/lc0/criticalMoments.ts';
import { reviewGame } from '../src/lc0/gameReview.ts';
import { GameTree } from '../src/lc0/gameTree.ts';

const line = (uci = 'e2e4', scoreCp = 20, engine = 'Stockfish Lite') => ({ engine, multipv: 1, pvUci: [uci], pvSan: uci === 'e2e4' ? 'e4' : 'd4', scoreCp, scoreText: '+0.20', detail: 'depth 12' });
const provenance = (engine = 'Stockfish Lite') => ({ engine, model: 'test-model', build: 'pinned-build', budget: 'depth 12', runtime: 'test-runtime', options: 'MultiPV 1; threads 1' });
const moment = () => ({ move: { ply: 1, san: 'd4', uci: 'd2d4', side: 'w', class: 'mistake', moverLoss: 0.15 }, fen: START_FEN, history: [START_FEN] });
const lane = (id = 'sf', evaluate = async () => ({ cacheKey: '', provenance: provenance(), line: line() })) => ({ id, provenance: provenance(id === 'sf' ? 'Stockfish Lite' : id), row: { family: id === 'sf' ? 'sf' : 'lc0', variant: id === 'sf' ? 'lite' : id, strength: 12 }, evaluate });
const run = async (lanes, moments = [moment()], cache = new Map(), signal = new AbortController().signal) => {
  const report = createCriticalReport('snapshot', 'Stockfish Lite · depth 8', moments, lanes);
  await runCriticalReport(report, lanes, cache, signal, () => {});
  return report;
};

test('deferred interactive refreshes merge force only for the same history/settings selection and drain once', () => {
  const refresh = new CriticalAnalysisRefresh();
  refresh.defer('history-A/settings-A', true);
  refresh.defer('history-A/settings-A');
  assert.deepEqual(refresh.take('history-A/settings-A'), { force: true });
  assert.equal(refresh.take('history-A/settings-A'), undefined);
  refresh.defer('history-A/settings-A', true);
  refresh.defer('history-A/settings-B');
  assert.deepEqual(refresh.take('history-A/settings-B'), { force: false });
});

test('opening a cached or report snapshot discards an older interactive refresh instead of overwriting it', () => {
  const refresh = new CriticalAnalysisRefresh();
  for (const selected of ['position-A', 'position-B']) {
    refresh.defer('position-A', true);
    refresh.navigate(selected, true);
    assert.equal(refresh.take(selected), undefined);
  }
  refresh.defer('position-A');
  assert.equal(refresh.take('position-B'), undefined, 'never resume on another history or setup');
});

test('normal uncached navigation can queue its own refresh but cannot inherit a previous position request', () => {
  const refresh = new CriticalAnalysisRefresh();
  refresh.defer('position-A', true);
  refresh.navigate('position-B', false);
  assert.equal(refresh.take('position-B'), undefined);
  refresh.defer('position-B');
  refresh.navigate('position-B', false);
  assert.deepEqual(refresh.take('position-B'), { force: false });
});

test('same best move agrees even when engine-specific evaluations differ greatly', () => {
  assert.equal(criticalAgreement([line('e2e4', 10), line('e2e4', 900, 'Lc0 Small')]), 'Both engines pick e4');
});

test('different best moves with similar evals are not called a tactical error', () => {
  assert.equal(criticalAgreement([line('e2e4', 10), line('d2d4', 11, 'Lc0 Small')]), 'Engines split: e4 / d4');
  assert.equal(criticalAgreement([line()]), 'Not enough engine results to compare moves.');
});

test('critical positions are ranked, capped at five, and BEFORE each error with full history', () => {
  const game = new GameTree();
  for (const uci of ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1c4', 'g8f6', 'd2d3']) game.addUci(uci);
  const nodes = game.lineTo();
  const review = reviewGame([0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1].map((winWhite) => ({ winWhite, bestUci: null })), nodes.slice(1).map((n) => ({ san: n.san, uci: '' })));
  const moments = criticalMoments(review, nodes.map((node) => node.fen));
  assert.equal(moments.length, 5);
  for (const m of moments) {
    assert.equal(m.fen, nodes[m.move.ply - 1].fen);
    assert.deepEqual(m.history, nodes.slice(0, m.move.ply).map((node) => node.fen));
  }
});

test('partial engine failures preserve other lanes and earlier completed results', async () => {
  let calls = 0;
  const lanes = [lane(), lane('lc0', async () => { if (++calls === 2) throw new Error('GPU lost'); return { line: line('d2d4', 22, 'Lc0 Small'), provenance: provenance('Lc0 Small') }; })];
  const report = await run(lanes, [moment(), { ...moment(), history: [START_FEN, START_FEN] }]);
  assert.equal(report.state, 'complete');
  assert.equal(report.cells.sf[1].status, 'complete');
  assert.equal(report.cells.lc0[0].status, 'complete');
  assert.equal(report.cells.lc0[1].status, 'failed');
  assert.equal(report.cells.lc0[1].error, 'GPU lost');
  assert.match(criticalReportMarkup(report), /Failed: GPU lost/);
});

test('history-specific cache keys separate identical-FEN transpositions and retain provenance', async () => {
  const histories = [
    ['g1f3', 'g8f6', 'f3g1', 'f6g8', 'g1f3', 'g8f6', 'f3g1', 'f6g8'],
    ['g1f3', 'g8f6', 'f3h4', 'f6h5', 'h4f3', 'h5f6', 'f3g1', 'f6g8'],
  ].map((moves) => { const game = new GameTree(); for (const move of moves) game.addUci(move); return game.lineTo().map((node) => node.fen); });
  assert.equal(histories[0].at(-1), histories[1].at(-1));
  const moments = histories.map((history) => ({ ...moment(), fen: history.at(-1), history }));
  assert.notEqual(criticalMomentCacheKey(moments[0], lane()), criticalMomentCacheKey(moments[1], lane()));
  let calls = 0;
  const engine = lane('sf', async () => { calls++; return { line: line(), provenance: provenance() }; });
  const cache = new Map();
  await run([engine], moments, cache);
  const report = await run([engine], moments, cache);
  assert.equal(calls, 2);
  assert.equal(report.cells.sf[0].cached, true);
  assert.equal(report.cells.sf[0].result.cacheKey, criticalMomentCacheKey(moments[0], engine));
  assert.deepEqual(report.cells.sf[0].result.provenance, engine.provenance);
  for (const field of ['model', 'build', 'runtime', 'budget', 'options']) {
    const different = { ...engine, provenance: { ...engine.provenance, [field]: 'different' } };
    assert.notEqual(criticalMomentCacheKey(moments[0], engine), criticalMomentCacheKey(moments[0], different));
  }
});

test('stale reports detect mainline, rating and engine settings changes, not board navigation', () => {
  const settings = { rating: 1500, depth: 12, engines: ['sf', 'lc0'] };
  const signature = criticalReportSignature('mainline', settings);
  const report = { signature, stale: false };
  assert.equal(criticalReportIsStale(report, criticalReportSignature('mainline', { ...settings })), false);
  for (const signature of [criticalReportSignature('edited', settings), criticalReportSignature('mainline', { ...settings, rating: 1900 }), criticalReportSignature('mainline', { ...settings, depth: 8 })]) assert.equal(criticalReportIsStale(report, signature), true);
  assert.equal(criticalReportIsStale({ ...report, stale: true }, signature), true, 'invalidated reports stay stale even after reverting settings');
});

test('cancellation retains completed cells, discards late work and starts no further moments', async () => {
  const controller = new AbortController();
  let calls = 0;
  const engine = lane('sf', async () => {
    calls++;
    if (calls === 2) controller.abort();
    return { line: line(), provenance: provenance() };
  });
  const report = await run([engine], [moment(), { ...moment(), history: [START_FEN, START_FEN] }, { ...moment(), history: [START_FEN, START_FEN, START_FEN] }], new Map(), controller.signal);
  assert.equal(report.state, 'cancelled');
  assert.equal(calls, 2);
  assert.equal(report.cells.sf[0].status, 'complete');
  assert.equal(report.cells.sf[1].status, 'cancelled');
  assert.equal(report.cells.sf[2].status, 'cancelled');
  assert.equal(report.cells.sf[1].result, undefined);
});

test('human plausibility reports played and unique engine-move probabilities without expected points', async () => {
  const lanes = [lane(), lane('lc0'), lane('maia', async () => ({ provenance: provenance('Maia3'), rating: 1500, policy: { d2d4: 0.12, e2e4: 0.35 } }))];
  const report = await run(lanes);
  const markup = criticalReportMarkup(report);
  assert.match(markup, /Played d4: 12.0% · e4: 35.0%/);
  assert.match(markup, /Original review: <strong>Mistake<\/strong>/);
  assert.match(markup, /Stockfish score in pawns/);
  assert.match(markup, /Lc0 score converted to pawns/);
  assert.doesNotMatch(markup, /Q-mapped|Best child|Baseline review/);
  assert.match(markup, /Recommended move searched:/);
  assert.doesNotMatch(markup, /expected points|accuracy/i);
  report.stale = true;
  assert.match(criticalReportMarkup(report), /data-critical-target="board"[^>]* disabled/);
});

test('White perspective is normalized exactly once while engine scales stay distinct', () => {
  const fen = START_FEN.replace(' w ', ' b ');
  const sf = stockfishAnalysisLines([{ multipv: 1, depth: 12, scoreCp: 120, pvUci: ['e7e5'] }], fen)[0];
  const lc0 = lc0AnalysisLines({ visits: 400, value: 0.2, children: [{ uci: 'e7e5', visits: 400, q: 0.2 }], pv: ['e7e5'] }, fen)[0];
  assert.equal(sf.scoreCp, -120);
  assert.ok(lc0.scoreCp < 0);
  assert.equal(criticalAgreement([sf, lc0]), 'Both engines pick e5');
});

test('pending and partial report snapshots preserve cached position lines and accept late engine results', () => {
  const sf = line();
  const lc0 = line('d2d4', 30, 'Lc0 Small');
  assert.equal(criticalPositionLines(undefined, []), undefined);
  assert.deepEqual(criticalPositionLines([sf, lc0], []), [sf, lc0], 'a pending report must not erase valid position results');
  const freshSf = line('d2d4', 35);
  const partial = criticalPositionLines([sf, lc0], [freshSf]);
  assert.deepEqual(partial, [lc0, freshSf]);
  const finished = criticalPositionLines(partial, [freshSf, { ...lc0, scoreCp: 40 }]);
  assert.deepEqual(finished, [freshSf, { ...lc0, scoreCp: 40 }]);
  assert.equal(finished.length, 2, 'finishing report engines replace their own snapshot, not duplicate it');
});

test('missing or unscored engine results are per-engine failures, never fabricated neutral scores', async () => {
  const report = await run([lane('sf', async () => ({ provenance: provenance(), line: { ...line(), scoreCp: undefined } })), lane('lc0')]);
  assert.equal(report.cells.sf[0].status, 'failed');
  assert.equal(report.cells.lc0[0].status, 'complete');
});
