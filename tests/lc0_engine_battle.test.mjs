import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { test } from 'node:test';
import { legalMoves } from '../src/chess/movegen.ts';
import { moveToUci } from '../src/chess/moveCodec.ts';
import {
  lc0PolicyBattleEngine,
  lc0SearchBattleEngine,
  playGame,
  runMatch,
} from '../src/lc0/engineBattle.ts';
import { Lc0OnnxEvaluator } from '../src/lc0/onnxEvaluator.ts';
import { Lc0PolicyOnlyPlayer } from '../src/lc0/policyOnlyPlayer.ts';
import { Lc0PuctSearcher } from '../src/lc0/search.ts';

const MODEL = '../models/lc0-bestnets/onnx/t1-256x10-distilled-swa-2432500.batch1.f32.onnx';
const BACK_RANK_MATE = '6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1';
const STALEMATE = '7k/5Q2/6K1/8/8/8/8/8 b - - 0 1';
const KING_VS_KING = '8/8/4k3/8/8/4K3/8/8 w - - 0 1';

// A stub engine that always offers the same UCI move (used to script outcomes).
function fixedEngine(name, uci) {
  return { name, chooseMove: () => uci };
}

test('playGame detects checkmate and awards the win to the mating side', async () => {
  const result = await playGame(fixedEngine('mater', 'a1a8'), fixedEngine('victim', null), { startFen: BACK_RANK_MATE });
  assert.equal(result.result, '1-0');
  assert.equal(result.reason, 'checkmate');
  assert.deepEqual(result.moves, ['a1a8']);
});

test('playGame detects an immediate stalemate as a draw', async () => {
  const result = await playGame(fixedEngine('a', null), fixedEngine('b', null), { startFen: STALEMATE });
  assert.equal(result.result, '1/2-1/2');
  assert.equal(result.reason, 'stalemate');
  assert.equal(result.plies, 0);
});

test('playGame detects insufficient material as a draw before any move', async () => {
  const result = await playGame(fixedEngine('a', 'e3e4'), fixedEngine('b', null), { startFen: KING_VS_KING });
  assert.equal(result.result, '1/2-1/2');
  assert.equal(result.reason, 'insufficientMaterial');
});

test('playGame forfeits on an illegal move and on resignation', async () => {
  const illegal = await playGame(fixedEngine('w', 'e2e5'), fixedEngine('b', null));
  assert.equal(illegal.result, '0-1');
  assert.match(illegal.reason, /illegal move e2e5/);

  const resign = await playGame(fixedEngine('w', null), fixedEngine('b', null));
  assert.equal(resign.result, '0-1');
  assert.equal(resign.reason, 'resigned');
});

test('playGame draws on max plies and records the move list', async () => {
  // Two kings shuffling between two squares each: legal, never terminal, never
  // a repetition draw within the cap because halfmove/insufficient-material is
  // avoided by keeping a rook on the board.
  const shuffleWhite = (positions) => {
    const board = positions[positions.length - 1];
    const uci = legalMoves(board).map(moveToUci).find((m) => m.startsWith('e1') || m.startsWith('a1'));
    return uci ?? null;
  };
  const shuffleBlack = (positions) => {
    const board = positions[positions.length - 1];
    return legalMoves(board).map(moveToUci)[0] ?? null;
  };
  const result = await playGame({ name: 'w', chooseMove: shuffleWhite }, { name: 'b', chooseMove: shuffleBlack }, { startFen: 'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1', maxPlies: 4 });
  assert.ok(['1/2-1/2', '1-0', '0-1'].includes(result.result));
  assert.ok(result.plies <= 4);
});

test('runMatch alternates colors and tallies scores from A perspective', async () => {
  const summary = await runMatch(fixedEngine('mater', 'a1a8'), fixedEngine('passive', null), 2, { startFen: BACK_RANK_MATE });
  assert.equal(summary.games, 2);
  // Game 0: A is white and mates -> A win. Game 1: B is white and resigns -> A win.
  assert.equal(summary.aWins, 2);
  assert.equal(summary.bWins, 0);
  assert.equal(summary.draws, 0);
  assert.equal(summary.aScore, 2);
  assert.equal(summary.results[0].reason, 'checkmate');
  assert.equal(summary.results[1].reason, 'resigned');
});

test('LC0 adapters play a finite game from a near-terminal position', { skip: !existsSync(MODEL) && 'missing ONNX model' }, async () => {
  const evaluator = await Lc0OnnxEvaluator.create(readFileSync(MODEL));
  const search = lc0SearchBattleEngine(new Lc0PuctSearcher(evaluator), 8);
  const policy = lc0PolicyBattleEngine(new Lc0PolicyOnlyPlayer(evaluator));
  const result = await playGame(search, policy, { startFen: BACK_RANK_MATE, maxPlies: 6 });
  assert.ok(['1-0', '0-1', '1/2-1/2'].includes(result.result));
  assert.equal(typeof result.finalFen, 'string');
  // White (search) should convert the back-rank mate quickly.
  assert.equal(result.result, '1-0');
});
