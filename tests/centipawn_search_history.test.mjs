import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { boardToFen, parseFen } from '../src/chess/board.ts';
import { automaticDrawReason, repetitionCount, searchHistoryFens } from '../src/chess/drawRules.ts';
import { moveToActionId, moveToUci } from '../src/chess/moveCodec.ts';
import { legalMoves, makeMove } from '../src/chess/movegen.ts';
import { CachedEvaluator } from '../src/nn/evaluator.ts';
import { chooseMove } from '../src/search/puct.ts';

function play(ucis, fen) {
  let board = parseFen(fen);
  const positions = [board];
  for (const uci of ucis) {
    const move = legalMoves(board).find((candidate) => moveToUci(candidate) === uci);
    assert.ok(move, uci);
    board = makeMove(board, move);
    positions.push(board);
  }
  return positions;
}

// A 10-ply cycle: the start position recurs at plies 0, 10 and 20. At ply 19,
// Black's d7e8 makes the third occurrence, but the first is 20 plies before it,
// outside the 16-position window the pages used to pass.
const START = '4k3/8/8/8/8/8/8/R3K3 w - - 0 1';
const CYCLE = ['a1a2', 'e8f8', 'a2a3', 'f8f7', 'a3a4', 'f7e7', 'a4a5', 'e7d7', 'a5a1', 'd7e8'];
const positions = play([...CYCLE, ...CYCLE.slice(0, 9)], START);

test('searchHistoryFens keeps the game back to the last irreversible move, and at least the net window', () => {
  assert.equal(searchHistoryFens(positions, 16).length, 19, 'no capture or pawn move: the whole game');
  assert.deepEqual(searchHistoryFens(positions, 16), positions.slice(0, -1).map(boardToFen).reverse());
  const afterPawn = play(['a1a2', 'e8d8', 'e2e4', 'd8e8', 'a2a1', 'e8d8'], '4k3/8/8/8/8/8/4P3/R3K3 w - - 0 1');
  // Stops one position before e4: the position after e4 must not be the
  // oldest entry, which is keyed as a FEN root (en-passant square kept).
  assert.deepEqual(searchHistoryFens(afterPawn), afterPawn.slice(2, -1).map(boardToFen).reverse(), 'stops before e4');
  assert.equal(searchHistoryFens(afterPawn, 16).length, 6, 'net window still reaches back past e4');
  assert.deepEqual(searchHistoryFens(positions.slice(0, 1), 16), []);
});

test('a position reached by a double push repeats through a history cut at that push', () => {
  // After 1.e4 (uncapturable e3) a 10-ply cycle returns to it at plies 11 and 21.
  const cycle = ['e8f8', 'a1a2', 'f8f7', 'a2a3', 'f7e7', 'a3a4', 'e7d7', 'a4a5', 'd7e8', 'a5a1'];
  const game = play(['e2e4', ...cycle, ...cycle], '4k3/8/8/8/8/8/4P3/R3K3 w - - 0 1');
  assert.match(boardToFen(game[1]), / e3 /);
  assert.equal(boardToFen(game[21]).split(' ', 4).join(' '), boardToFen(game[11]).split(' ', 4).join(' '));
  const history = searchHistoryFens(game);
  assert.equal(history.length, 21);
  assert.equal(repetitionCount(game[21], history), 3);
  assert.equal(automaticDrawReason(game[21], history), 'threefold');
});

function uniformEvaluator(seen) {
  return {
    async evaluate(board, context = {}) {
      seen.push(context.historyFens?.length ?? 0);
      const moves = context.legalMoves ?? legalMoves(board);
      return { policy: new Map(moves.map((move) => [moveToActionId(move), 1 / moves.length])), wdl: [0.6, 0.3, 0.1] };
    },
  };
}

async function repetitionChild(historyFens) {
  const seen = [];
  const evaluator = new CachedEvaluator(uniformEvaluator(seen), { maxHistoryPositions: 16 });
  const result = await chooseMove(positions.at(-1), evaluator, { visits: 64, batchSize: 1, historyFens });
  const edge = result.root.edges.find((candidate) => moveToUci(candidate.move) === 'd7e8');
  assert.ok(edge?.child?.expanded, 'the repeating move was searched');
  return { terminalValue: edge.child.terminalValue, seen };
}

test('Centipawn search scores a repetition that spans more than 16 plies as a draw', async () => {
  const child = makeMove(
    positions.at(-1),
    legalMoves(positions.at(-1)).find((move) => moveToUci(move) === 'd7e8'),
  );
  assert.equal(automaticDrawReason(child, [boardToFen(positions.at(-1)), ...searchHistoryFens(positions, 16)]), 'threefold');

  const full = await repetitionChild(searchHistoryFens(positions, 16));
  assert.equal(full.terminalValue, 0, 'third occurrence is a draw in search');
  assert.ok(Math.max(...full.seen) <= 16, `net saw at most 16 history positions (max ${Math.max(...full.seen)})`);

  const truncated = await repetitionChild(positions.slice(0, -1).map(boardToFen).reverse().slice(0, 16));
  assert.equal(truncated.terminalValue, null, 'the old 16-position history missed it');
});

test('CachedEvaluator bounds history for keys and the net, so a longer search history hits the same entries', async () => {
  const seen = [];
  const evaluator = new CachedEvaluator(uniformEvaluator(seen), { maxHistoryPositions: 16 });
  const history = searchHistoryFens(positions, 16);
  await evaluator.evaluate(positions.at(-1), { historyFens: history.slice(0, 16) });
  await evaluator.evaluate(positions.at(-1), { historyFens: history });
  assert.deepEqual(seen, [16]);
  assert.deepEqual(evaluator.metrics().hits, 1);
});

test('Play, Analysis and Arena retain the long history for native repetition features', () => {
  const source = (file) => readFileSync(new URL(`../src/lc0/${file}`, import.meta.url), 'utf8');
  for (const page of ['playBrowser.ts', 'analysisBrowser.ts', 'arenaBrowser.ts']) {
    assert.doesNotMatch(source(page), /reverse\(\)\.slice\(0, 16\)/, `${page} still slices search history`);
    assert.match(source(page), /searchHistoryFens\([^)]*, (?:ctx\.)?positions.length\)/, page);
  }
  // The new model scans older positions for repetition; legacy nets retain the bound.
  assert.doesNotMatch(source('playBrowser.ts'), /maxHistoryPositions:/);
  assert.match(source('centipawnEvaluatorCache.ts'), /nativeHistory \? undefined : CENTIPAWN_NET_HISTORY_POSITIONS/);
  for (const page of ['analysisBrowser.ts', 'arenaBrowser.ts']) assert.match(source(page), /createCentipawnEvaluatorCache\(/, page);
});
