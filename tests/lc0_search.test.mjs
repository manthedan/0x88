import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { test } from 'node:test';
import { boardToFen, parseFen, START_FEN } from '../src/chess/board.ts';
import { legalMoves } from '../src/chess/movegen.ts';
import { moveToUci } from '../src/chess/moveCodec.ts';
import { buildBoardHistoryFromMoves } from '../src/lc0/history.ts';
import { Lc0PuctSearcher, Lc0SearchEvaluator } from '../src/lc0/search.ts';

const MODEL = '../models/lc0-bestnets/onnx/t1-256x10-distilled-swa-2432500.batch1.f32.onnx';
const NATIVE_FEN_SEARCH = 'fixtures/lc0/native_search_fen_only_blas_nodes32.jsonl';
const NATIVE_HISTORY_SEARCH = 'fixtures/lc0/native_search_history_blas_nodes32.jsonl';
// Higher-visit native BLAS fixtures for stronger best-move sanity. Exact visit
// distributions are intentionally not asserted (see docs/lc0_search_parity_strictness.md).
const HIGHER_VISIT_SUITES = [
  { visits: 64, files: ['fixtures/lc0/native_search_fen_only_blas_nodes64.jsonl', 'fixtures/lc0/native_search_history_blas_nodes64.jsonl'] },
  { visits: 128, files: ['fixtures/lc0/native_search_fen_only_blas_nodes128.jsonl', 'fixtures/lc0/native_search_history_blas_nodes128.jsonl'] },
];

function readJsonl(path) {
  return readFileSync(path, 'utf8').trim().split('\n').filter(Boolean).map((line) => JSON.parse(line));
}

function nativeCastlingToStandard(uci) {
  switch (uci) {
    case 'e1h1': return 'e1g1';
    case 'e1a1': return 'e1c1';
    case 'e8h8': return 'e8g8';
    case 'e8a8': return 'e8c8';
    default: return uci;
  }
}

test('LC0 search evaluator reconstructs explicit history from search context', async () => {
  const calls = [];
  const root = parseFen(START_FEN);
  const current = parseFen('rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2');
  const evaluator = new Lc0SearchEvaluator({
    evaluate(input) {
      calls.push(input);
      return { fen: boardToFen(current), wdl: [0.4, 0.3, 0.3], q: 0.1, mlh: 80, legalPriors: [] };
    },
  });
  await evaluator.evaluate(current, { historyFens: [boardToFen(root)] });
  assert.equal(calls.length, 1);
  assert.equal(calls[0].positions.length, 2);
  assert.equal(boardToFen(calls[0].positions[0]), boardToFen(root));
  assert.equal(boardToFen(calls[0].positions[1]), boardToFen(current));
});

test('LC0 batched PUCT groups leaf evaluations through the LC0 adapter', async () => {
  const batchSizes = [];
  const fakeLc0 = {
    async evaluateBatch(inputs) {
      batchSizes.push(inputs.length);
      return inputs.map((input) => {
        const board = typeof input === 'object' && input !== null && 'positions' in input ? input.positions[input.positions.length - 1] : input;
        const parsed = typeof board === 'string' ? parseFen(board) : board;
        const priors = legalMoves(parsed).map((move) => ({ uci: moveToUci(move), index: 0, logit: 0, prior: 1 }));
        return { fen: boardToFen(parsed), wdl: [0.34, 0.32, 0.34], q: 0, mlh: 80, legalPriors: priors };
      });
    },
    async evaluate(input) {
      return (await this.evaluateBatch([input]))[0];
    },
  };
  const result = await new Lc0PuctSearcher(fakeLc0).search(START_FEN, { visits: 8, batchSize: 4 });
  assert.equal(result.search.stats?.completedVisits, 8);
  assert.equal(result.search.stats?.batchEvalCalls, 2);
  assert.equal(result.search.stats?.maxEvalBatch, 4);
  assert.ok(batchSizes.includes(4), `expected a batch of 4, got ${batchSizes.join(',')}`);
});

function uniformEvaluator() {
  return {
    async evaluateBatch(inputs) {
      return inputs.map((input) => {
        const board = typeof input === 'object' && input !== null && 'positions' in input ? input.positions[input.positions.length - 1] : input;
        const parsed = typeof board === 'string' ? parseFen(board) : board;
        const priors = legalMoves(parsed).map((move) => ({ uci: moveToUci(move), index: 0, logit: 0, prior: 1 }));
        return { fen: boardToFen(parsed), wdl: [0.34, 0.32, 0.34], q: 0, mlh: 80, legalPriors: priors };
      });
    },
    async evaluate(input) {
      return (await this.evaluateBatch([input]))[0];
    },
  };
}

test('LC0 search can reuse the previous subtree after a played move', async () => {
  const searcher = new Lc0PuctSearcher(uniformEvaluator());
  const first = await searcher.search(START_FEN, { visits: 12, reuseTree: true });
  assert.equal(first.search.stats?.rootReused, false, 'first search starts from a fresh root');
  assert.ok(first.move, 'first search chooses a move');

  const positions = buildBoardHistoryFromMoves([first.move]);
  const second = await searcher.search({ positions }, { visits: 16, reuseTree: true });
  assert.equal(second.search.stats?.rootReused, true, 'second search reuses the subtree for the played move');
  assert.equal(second.visits, 16, 'reused subtree is topped up to the requested visit budget');
  assert.ok((second.search.stats?.completedVisits ?? 16) < 16, 'reuse avoids rerunning the full target budget');

  searcher.resetTree();
  const fresh = await searcher.search({ positions }, { visits: 16, reuseTree: true });
  assert.equal(fresh.search.stats?.rootReused, false, 'resetTree clears cached search state');
});

test('LC0 search can reuse a deeper subtree after an opponent reply', async () => {
  const searcher = new Lc0PuctSearcher(uniformEvaluator());
  const first = await searcher.search(START_FEN, { visits: 80, reuseTree: true });
  const root = first.search.root;
  const playedEdge = root?.edges.find((edge) => moveToUci(edge.move) === first.move);
  const replyEdge = playedEdge?.child?.edges.find((edge) => edge.child);
  assert.ok(first.move && replyEdge, 'initial search explored at least one reply subtree');

  const reply = moveToUci(replyEdge.move);
  const positions = buildBoardHistoryFromMoves([first.move, reply]);
  const second = await searcher.search({ positions }, { visits: 88, reuseTree: true });
  assert.equal(second.search.stats?.rootReused, true, `reused subtree after ${first.move} ${reply}`);
  assert.equal(second.visits, 88, 'reused reply subtree is topped up to requested visits');
});

test('LC0 search exposes a principal variation of legal UCI moves', async () => {
  const result = await new Lc0PuctSearcher(uniformEvaluator()).search(START_FEN, { visits: 24 });
  assert.ok(Array.isArray(result.pv), 'pv is an array');
  assert.ok(result.pv.length >= 1, `pv has at least one move, got ${result.pv.length}`);
  assert.equal(result.pv[0], result.move, 'pv starts with the chosen root move');
  const startLegal = new Set(legalMoves(parseFen(START_FEN)).map(moveToUci));
  assert.ok(startLegal.has(result.pv[0]), `pv[0] ${result.pv[0]} is legal at the root`);
});

test('LC0 search returns MultiPV lines, one per top root move', async () => {
  const result = await new Lc0PuctSearcher(uniformEvaluator()).search(START_FEN, { visits: 40, multiPv: 3 });
  assert.ok(Array.isArray(result.multiPv), 'multiPv is an array');
  assert.ok(result.multiPv.length >= 2 && result.multiPv.length <= 3, `2-3 lines, got ${result.multiPv.length}`);
  // The first MultiPV line is the principal variation / chosen move.
  assert.equal(result.multiPv[0][0], result.move, 'first MultiPV line starts with the best move');
  assert.deepEqual(result.multiPv[0], result.pv, 'first MultiPV line equals the single PV');
  const startLegal = new Set(legalMoves(parseFen(START_FEN)).map(moveToUci));
  const rootMoves = new Set();
  for (const line of result.multiPv) {
    assert.ok(line.length >= 1, 'each line has at least the root move');
    assert.ok(startLegal.has(line[0]), `line root ${line[0]} is legal`);
    rootMoves.add(line[0]);
  }
  assert.equal(rootMoves.size, result.multiPv.length, 'each MultiPV line has a distinct root move');
});

test('LC0 search omits multiPv when multiPv <= 1', async () => {
  const result = await new Lc0PuctSearcher(uniformEvaluator()).search(START_FEN, { visits: 16, multiPv: 1 });
  assert.equal(result.multiPv, undefined, 'no multiPv lines for multiPv=1');
  assert.ok(result.pv.length >= 1, 'single pv is still present');
});

test('LC0 movetime search returns best-so-far instead of aborting on soft timeout', async () => {
  const slowEvaluator = {
    async evaluate(input) {
      await new Promise((resolve) => setTimeout(resolve, 8));
      const board = typeof input === 'object' && input !== null && 'positions' in input ? input.positions[input.positions.length - 1] : input;
      const parsed = typeof board === 'string' ? parseFen(board) : board;
      const priors = legalMoves(parsed).map((move) => ({ uci: moveToUci(move), index: 0, logit: 0, prior: 1 }));
      return { fen: boardToFen(parsed), wdl: [0.34, 0.32, 0.34], q: 0, mlh: 80, legalPriors: priors };
    },
  };
  const result = await new Lc0PuctSearcher(slowEvaluator).search(START_FEN, { movetimeMs: 25 });
  assert.ok(result.move, 'soft timeout still returns a move');
  assert.equal(result.search.stats?.stopReason, 'movetime');
  assert.equal(result.search.stats?.requestedVisits, Number.MAX_SAFE_INTEGER, 'movetime-only search uses deadline rather than a reachable visit cap');
  assert.ok((result.search.stats?.completedVisits ?? Number.MAX_SAFE_INTEGER) < 200, `completed only best-so-far visits, got ${result.search.stats?.completedVisits}`);
});

test('LC0 search throws AbortError when given an already-aborted signal', async () => {
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(
    () => new Lc0PuctSearcher(uniformEvaluator()).search(START_FEN, { visits: 64, signal: controller.signal }),
    (error) => error.name === 'AbortError',
  );
});

test('LC0 search honors a signal aborted mid-flight and stops early', async () => {
  const controller = new AbortController();
  let evalCount = 0;
  const slowEvaluator = {
    async evaluate(input) {
      evalCount += 1;
      if (evalCount === 3) controller.abort();
      const board = typeof input === 'object' && input !== null && 'positions' in input ? input.positions[input.positions.length - 1] : input;
      const parsed = typeof board === 'string' ? parseFen(board) : board;
      const priors = legalMoves(parsed).map((move) => ({ uci: moveToUci(move), index: 0, logit: 0, prior: 1 }));
      return { fen: boardToFen(parsed), wdl: [0.34, 0.32, 0.34], q: 0, mlh: 80, legalPriors: priors };
    },
  };
  await assert.rejects(
    () => new Lc0PuctSearcher(slowEvaluator).search(START_FEN, { visits: 200, signal: controller.signal }),
    (error) => error.name === 'AbortError',
  );
  assert.ok(evalCount < 200, `aborted search stopped early after ${evalCount} evals`);
});

test('LC0 fixed-visit PUCT matches native BLAS nodes32 fixture best moves', { skip: (!existsSync(MODEL) || !existsSync(NATIVE_FEN_SEARCH) || !existsSync(NATIVE_HISTORY_SEARCH)) && 'missing model or native search artifacts' }, async () => {
  const searcher = await Lc0PuctSearcher.create(readFileSync(MODEL));
  const records = [...readJsonl(NATIVE_FEN_SEARCH), ...readJsonl(NATIVE_HISTORY_SEARCH)];
  for (const native of records) {
    const input = native.moves ? { positions: buildBoardHistoryFromMoves(native.moves, native.startFen) } : native.fen;
    const result = await searcher.search(input, { visits: 32 });
    const expected = nativeCastlingToStandard(native.bestmove);
    assert.equal(result.move, expected, `${native.id} best move`);
    assert.equal(result.children[0]?.uci, expected, `${native.id} top child by visits`);
    assert.equal(result.search.stats?.completedVisits, 32, `${native.id} completed visits`);
    for (const prior of native.topPriors.slice(0, 5)) {
      const uci = nativeCastlingToStandard(prior.uci);
      const actual = result.children.find((entry) => entry.uci === uci);
      assert.ok(actual, `${native.id} has ${uci}`);
      assert.ok(Math.abs(actual.prior - prior.prior) < 0.0035, `${native.id} ${uci} prior native=${prior.prior} search=${actual.prior}`);
    }
  }
});

for (const { visits, files } of HIGHER_VISIT_SUITES) {
  const present = existsSync(MODEL) && files.every(existsSync);
  test(`LC0 fixed-visit PUCT matches native BLAS nodes${visits} fixture best moves`, { skip: !present && `missing model or native nodes${visits} artifacts` }, async () => {
    const searcher = await Lc0PuctSearcher.create(readFileSync(MODEL));
    const records = files.flatMap(readJsonl);
    for (const native of records) {
      const input = native.moves ? { positions: buildBoardHistoryFromMoves(native.moves, native.startFen) } : native.fen;
      const result = await searcher.search(input, { visits });
      const expected = nativeCastlingToStandard(native.bestmove);
      // Best-move parity is the strict criterion; root prior consistency is a
      // soft check. Exact visit distributions are not asserted at higher visits.
      assert.equal(result.move, expected, `nodes${visits} ${native.id} best move`);
      assert.equal(result.children[0]?.uci, expected, `nodes${visits} ${native.id} top child by visits`);
      assert.equal(result.search.stats?.completedVisits, visits, `nodes${visits} ${native.id} completed visits`);
      for (const prior of native.topPriors.slice(0, 3)) {
        const uci = nativeCastlingToStandard(prior.uci);
        const actual = result.children.find((entry) => entry.uci === uci);
        assert.ok(actual, `nodes${visits} ${native.id} has ${uci}`);
        assert.ok(Math.abs(actual.prior - prior.prior) < 0.0035, `nodes${visits} ${native.id} ${uci} prior native=${prior.prior} search=${actual.prior}`);
      }
    }
  });
}
