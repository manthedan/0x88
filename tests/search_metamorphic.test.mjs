import test from 'node:test';
import assert from 'node:assert/strict';
import { parseFen } from '../src/chess/board.ts';
import { legalMoves } from '../src/chess/movegen.ts';
import { moveToActionId, moveToUci } from '../src/chess/moveCodec.ts';
import { searchRoot } from '../src/search/puct.ts';

class PoisonIllegalEvaluator {
  evaluate(board, context = {}) {
    const legal = context.legalMoves ?? legalMoves(board);
    const policy = new Map();
    // Deliberately put almost all mass on an illegal action id. Search expansion
    // must ignore it and renormalize over legal moves only.
    policy.set(0, 1_000_000);
    if (legal[0]) policy.set(moveToActionId(legal[0]), 1e-9);
    return { policy, wdl: [0.34, 0.33, 0.33] };
  }
  evaluateBatch(boards, contexts = []) {
    return boards.map((board, i) => this.evaluate(board, contexts[i] ?? {}));
  }
}

class UniformContextEvaluator {
  evaluate(board, context = {}) {
    const legal = context.legalMoves ?? legalMoves(board);
    const policy = new Map();
    const p = legal.length ? 1 / legal.length : 0;
    for (const move of legal) policy.set(moveToActionId(move), p);
    return { policy, wdl: [0.33, 0.34, 0.33] };
  }
  evaluateBatch(boards, contexts = []) {
    return boards.map((board, i) => this.evaluate(board, contexts[i] ?? {}));
  }
}

function legalUcis(board) {
  return new Set(legalMoves(board).map(moveToUci));
}

function roundedPolicy(result) {
  return Object.fromEntries(result.policy.map((entry) => [moveToUci(entry.move), Number(entry.probability.toFixed(10))]).sort());
}

test('PUCT masks illegal evaluator policy mass instead of selecting illegal moves', async () => {
  const board = parseFen('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');
  const result = await searchRoot(board, new PoisonIllegalEvaluator(), { visits: 16, temperature: 0, batchSize: 4 });
  assert.ok(result.move, 'expected a legal move');
  assert.ok(legalUcis(board).has(moveToUci(result.move)), `selected illegal move ${moveToUci(result.move)}`);
  assert.equal(result.policy.length, legalMoves(board).length, 'root policy should only contain legal moves');
  assert.ok(result.policy.every((entry) => legalUcis(board).has(moveToUci(entry.move))), 'root policy contains illegal moves');
});

test('batched and unbatched PUCT are deterministic-equivalent for a stateless evaluator', async () => {
  const board = parseFen('r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1');
  const evaluator = new UniformContextEvaluator();
  const a = await searchRoot(board, evaluator, { visits: 24, temperature: 1, batchSize: 1 });
  const b = await searchRoot(board, evaluator, { visits: 24, temperature: 1, batchSize: 4 });
  assert.deepEqual(roundedPolicy(b), roundedPolicy(a));
  assert.equal(moveToUci(b.move), moveToUci(a.move));
  assert.equal(b.visits, a.visits);
});
