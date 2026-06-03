import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { test } from 'node:test';
import { START_FEN, parseFen } from '../src/chess/board.ts';
import { moveToUci } from '../src/chess/moveCodec.ts';
import { legalMoves } from '../src/chess/movegen.ts';
import { Lc0PolicyOnlyPlayer } from '../src/lc0/policyOnlyPlayer.ts';

const MODEL = '../models/lc0-bestnets/onnx/t1-256x10-distilled-swa-2432500.batch1.f32.onnx';
const NATIVE_PRIORS = 'fixtures/lc0/native_fen_only_blas.jsonl';

function nativeCastlingToStandard(uci) {
  switch (uci) {
    case 'e1h1': return 'e1g1';
    case 'e1a1': return 'e1c1';
    case 'e8h8': return 'e8g8';
    case 'e8a8': return 'e8c8';
    default: return uci;
  }
}

test('LC0 policy-only player returns evaluator argmax without search', async () => {
  const player = new Lc0PolicyOnlyPlayer({
    async evaluate(fen) {
      return {
        fen,
        wdl: [0.2, 0.6, 0.2],
        q: 0,
        mlh: 42,
        legalPriors: [
          { uci: 'g1f3', index: 159, logit: 2, prior: 0.7 },
          { uci: 'd2d4', index: 293, logit: 1, prior: 0.3 },
        ],
        bestMove: 'g1f3',
      };
    },
  });
  const choice = await player.chooseMove(START_FEN);
  assert.equal(choice.move, 'g1f3');
  assert.equal(choice.evaluation.legalPriors[0].uci, 'g1f3');
  assert.deepEqual(choice.evaluation.wdl, [0.2, 0.6, 0.2]);
});

test('LC0 policy-only player matches native BLAS fixture best moves', { skip: (!existsSync(MODEL) || !existsSync(NATIVE_PRIORS)) && 'missing model or native prior artifact' }, async () => {
  const nativeRecords = readFileSync(NATIVE_PRIORS, 'utf8').trim().split('\n').map((line) => JSON.parse(line));
  const player = await Lc0PolicyOnlyPlayer.create(readFileSync(MODEL));
  for (const native of nativeRecords) {
    const choice = await player.chooseMove(native.fen);
    const expected = nativeCastlingToStandard(native.bestmove);
    const legal = new Set(legalMoves(parseFen(native.fen)).map(moveToUci));
    assert.equal(choice.move, expected, `${native.id} bestmove`);
    assert.equal(legal.has(choice.move), true, `${native.id} move is legal`);
    assert.equal(choice.evaluation.legalPriors[0].uci, expected, `${native.id} top prior`);
  }
});
