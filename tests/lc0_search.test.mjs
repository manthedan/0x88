import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { test } from 'node:test';
import { boardToFen, parseFen, START_FEN } from '../src/chess/board.ts';
import { buildBoardHistoryFromMoves } from '../src/lc0/history.ts';
import { Lc0PuctSearcher, Lc0SearchEvaluator } from '../src/lc0/search.ts';

const MODEL = '../models/lc0-bestnets/onnx/t1-256x10-distilled-swa-2432500.batch1.f32.onnx';
const NATIVE_FEN_SEARCH = 'fixtures/lc0/native_search_fen_only_blas_nodes32.jsonl';
const NATIVE_HISTORY_SEARCH = 'fixtures/lc0/native_search_history_blas_nodes32.jsonl';

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
