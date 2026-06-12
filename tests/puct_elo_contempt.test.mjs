import assert from 'node:assert/strict';
import { test } from 'node:test';
import { parseFen, boardToFen, START_FEN } from '../src/chess/board.ts';
import { makeMove } from '../src/chess/movegen.ts';
import { moveFromUci, moveToActionId, moveToUci } from '../src/chess/moveCodec.ts';
import { applyEloContempt, searchRoot } from '../src/search/puct.ts';

// ---------------------------------------------------------------------------
// Oracle: native Monty (release 0950aff1, /tmp/monty-macos-aarch64) `eval`
// output. Each sample is the printed "wdl material" row (transform input) and
// "wdl contempt" row (transform output) at the given Contempt setting, in
// percent at 2dp. Regenerate by piping
//   setoption name Contempt value <C> / position fen <fen> / eval
// into the native binary. Tolerance covers the 2dp print rounding (and f32 vs
// f64): inputs are only known to +-0.005%, so outputs are checked at 0.02%.
// ---------------------------------------------------------------------------
const MONTY_ORACLE = [
  // [contempt, [material w,d,l %], [expected contempt w,d,l %]]
  [200, [28.44, 56.32, 15.24], [29.56, 55.88, 14.55]],
  [200, [34.12, 50.79, 15.10], [35.49, 50.17, 14.34]],
  [200, [92.17, 6.53, 1.30], [92.71, 6.09, 1.21]],
  [200, [0.28, 5.53, 94.19], [0.30, 5.78, 93.93]],
  [200, [0.70, 98.85, 0.45], [0.71, 98.85, 0.44]],
  [600, [28.44, 56.32, 15.24], [31.88, 54.87, 13.25]],
  [600, [34.12, 50.79, 15.10], [38.29, 48.79, 12.92]],
  [600, [92.17, 6.53, 1.30], [93.68, 5.28, 1.03]],
  [600, [0.28, 5.53, 94.19], [0.33, 6.30, 93.37]],
  [600, [0.70, 98.85, 0.45], [0.73, 98.84, 0.43]],
  [-400, [28.44, 56.32, 15.24], [26.27, 57.02, 16.71]],
  [-400, [34.12, 50.79, 15.10], [31.46, 51.83, 16.71]],
  [-400, [92.17, 6.53, 1.30], [90.98, 7.51, 1.51]],
  [-400, [0.28, 5.53, 94.19], [0.26, 5.06, 94.68]],
  [-400, [0.70, 98.85, 0.45], [0.68, 98.86, 0.46]],
];

const pct = (wdl) => wdl.map((x) => x / 100);

test('applyEloContempt matches the native Monty oracle', () => {
  for (const [contempt, material, expected] of MONTY_ORACLE) {
    const got = applyEloContempt(pct(material), contempt);
    for (let i = 0; i < 3; i += 1) {
      const diff = Math.abs(got[i] - expected[i] / 100);
      assert.ok(
        diff < 2e-4,
        `C=${contempt} material=[${material}] component ${i}: got ${(got[i] * 100).toFixed(3)}%, oracle ${expected[i]}% (diff ${(diff * 100).toFixed(3)}%)`,
      );
    }
  }
});

test('applyEloContempt identity and guards', () => {
  const wdl = [0.3, 0.5, 0.2];
  assert.equal(applyEloContempt(wdl, 0), wdl, 'zero contempt is exact identity');
  // Near-certain WDLs are untouched (Monty EPS guard), e.g. the decided-rook
  // endgame oracle row 99.80/0.19/0.00 at every contempt setting.
  const decided = pct([99.8, 0.19, 0.0]);
  assert.deepEqual(applyEloContempt(decided, 600), decided);
  assert.deepEqual(applyEloContempt(pct([0.0, 0.19, 99.8]), 600), pct([0.0, 0.19, 99.8]));
});

test('contempt sign moves the expected score the right way and fades when decided', () => {
  const score = (wdl) => wdl[0] + 0.5 * wdl[1];
  const open = [0.3, 0.45, 0.25];
  assert.ok(score(applyEloContempt(open, 400)) > score(open), 'positive contempt presses');
  assert.ok(score(applyEloContempt(open, -400)) < score(open), 'negative contempt concedes');
  // s^2 scaling: an already-decided position moves far less than an open one.
  const decided = [0.9, 0.07, 0.03];
  const openShift = score(applyEloContempt(open, 400)) - score(open);
  const decidedShift = score(applyEloContempt(decided, 400)) - score(decided);
  assert.ok(decidedShift > 0 && decidedShift < openShift / 2, `decided-position shift ${decidedShift} should be far below open-position shift ${openShift}`);
});

// ---------------------------------------------------------------------------
// Search-level behavior (same threefold harness as the drawScore tests).
// ---------------------------------------------------------------------------
function shuffleHistory() {
  const ucis = ['g1f3', 'g8f6', 'f3g1', 'f6g8', 'g1f3', 'g8f6', 'f3g1'];
  let board = parseFen(START_FEN);
  const fens = [boardToFen(board)];
  for (const uci of ucis) {
    board = makeMove(board, moveFromUci(uci));
    fens.push(boardToFen(board));
  }
  return { board, historyFens: fens.slice(0, -1).reverse() };
}

const flatEvaluator = {
  async evaluate(board, context) {
    const moves = context?.legalMoves ?? [];
    const uniform = moves.length ? 1 / moves.length : 0;
    return { policy: new Map(moves.map((move) => [moveToActionId(move), uniform])), wdl: [0.36, 0.30, 0.34] };
  },
};

function policySnapshot(result) {
  return result.policy.map((entry) => `${moveToUci(entry.move)}:${entry.visits}:${entry.q.toFixed(12)}`).join('|');
}

test('contemptElo 0 is bit-identical to no contempt option', async () => {
  const { board, historyFens } = shuffleHistory();
  const plain = await searchRoot(board, flatEvaluator, { visits: 200, historyFens });
  const zeroed = await searchRoot(board, flatEvaluator, { visits: 200, historyFens, contemptElo: 0 });
  assert.equal(policySnapshot(zeroed), policySnapshot(plain));
});

test('contemptElo sign decides the threefold like a calibrated drawScore', async () => {
  const { board, historyFens } = shuffleHistory();
  // Stronger side (+600): evaluated continuations rescale above the terminal
  // draw, so the repetition is refused.
  const press = await searchRoot(board, flatEvaluator, { visits: 400, historyFens, contemptElo: 600 });
  assert.notEqual(moveToUci(press.move), 'f6g8', 'positive contemptElo must avoid the repetition');
  // Weaker side (-600): continuations rescale below the draw, so repeat.
  const concede = await searchRoot(board, flatEvaluator, { visits: 400, historyFens, contemptElo: -600 });
  assert.equal(moveToUci(concede.move), 'f6g8', 'negative contemptElo must take the repetition');
});

test('changed contemptElo does not reuse a stale search tree', async () => {
  const { board, historyFens } = shuffleHistory();
  const press = await searchRoot(board, flatEvaluator, { visits: 200, historyFens, contemptElo: 600 });
  const neutral = await searchRoot(board, flatEvaluator, { visits: 200, historyFens, root: press.root });
  assert.equal(neutral.stats?.rootReused, false, 'contemptElo changes must refuse tree reuse');
});
