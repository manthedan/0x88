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

class MildlyPeakedEvaluator {
  batchSizes = [];

  evaluate(board, context = {}) {
    const legal = context.legalMoves ?? legalMoves(board);
    const policy = new Map();
    if (legal.length) {
      policy.set(moveToActionId(legal[0]), 0.1);
      const rest = legal.length > 1 ? 0.9 / (legal.length - 1) : 0;
      for (const move of legal.slice(1)) policy.set(moveToActionId(move), rest);
    }
    return { policy, wdl: [0.33, 0.34, 0.33] };
  }

  evaluateBatch(boards, contexts = []) {
    this.batchSizes.push(boards.length);
    return boards.map((board, i) => this.evaluate(board, contexts[i] ?? {}));
  }
}

class DominantFirstMoveEvaluator {
  evaluate(board, context = {}) {
    const legal = context.legalMoves ?? legalMoves(board);
    const policy = new Map();
    legal.forEach((move, i) => policy.set(moveToActionId(move), i === 0 ? 1 : 0));
    return { policy, wdl: [0.34, 0.32, 0.34] };
  }

  evaluateBatch(boards, contexts = []) {
    return boards.map((board, i) => this.evaluate(board, contexts[i] ?? {}));
  }
}

class SlowDominantFirstMoveEvaluator extends DominantFirstMoveEvaluator {
  async evaluate(board, context = {}) {
    await new Promise((resolve) => setTimeout(resolve, 3));
    return super.evaluate(board, context);
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

test('batched PUCT retries in-flight leaf collisions to keep evaluator batches full', async () => {
  const board = parseFen('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');
  const evaluator = new MildlyPeakedEvaluator();
  const result = await searchRoot(board, evaluator, { visits: 4, temperature: 1, batchSize: 4 });
  assert.equal(result.stats?.completedVisits, 4);
  assert.equal(result.stats?.maxEvalBatch, 4, `expected a full physical leaf batch, got ${evaluator.batchSizes.join(',')}`);
  assert.ok((result.stats?.batchLeafCollisions ?? 0) > 0, 'collision retry path was exercised');
  assert.equal(result.root?.edges.reduce((sum, edge) => sum + edge.virtualVisits, 0), 0, 'temporary virtual visits were unwound');
});

test('pipelined PUCT backup mode skips cross-batch in-flight leaf collisions', async () => {
  const board = parseFen('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');
  const result = await searchRoot(board, new DominantFirstMoveEvaluator(), {
    visits: 8,
    temperature: 1,
    batchSize: 2,
    batchPipelineDepth: 2,
    batchCollisionMode: 'backup',
  });
  assert.equal(result.stats?.completedVisits, 8);
  assert.ok((result.stats?.batchLeafCollisions ?? 0) > 0, 'cross-batch in-flight collision path was exercised');
  assert.equal(result.root?.edges.reduce((sum, edge) => sum + edge.virtualVisits, 0), 0, 'temporary virtual visits were unwound');
});

test('PUCT can stop early when the best root move is stable', async () => {
  const board = parseFen('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');
  const result = await searchRoot(board, new DominantFirstMoveEvaluator(), {
    visits: 200,
    batchSize: 4,
    earlyStop: 'best-stable',
    bestStableMinVisits: 8,
    bestStableCheckInterval: 4,
    bestStableChecks: 2,
    bestStableMinVisitLead: 4,
    bestStableMaxQDelta: 1,
  });
  assert.equal(result.stats?.stopReason, 'best-stable');
  assert.ok((result.stats?.completedVisits ?? 200) < 200, `stopped before full budget, got ${result.stats?.completedVisits}`);
});

test('PUCT root-dominance early stop works for fixed visit budgets', async () => {
  const board = parseFen('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');
  const result = await searchRoot(board, new DominantFirstMoveEvaluator(), { visits: 40, earlyStop: 'root-dominance' });
  assert.equal(result.stats?.stopReason, 'root-dominance');
  assert.ok((result.stats?.completedVisits ?? 40) < 40, `dominant root move stopped early at ${result.stats?.completedVisits}`);
});

test('PUCT best-stable does not relabel full-budget completion', async () => {
  const board = parseFen('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');
  const result = await searchRoot(board, new DominantFirstMoveEvaluator(), {
    visits: 16,
    earlyStop: 'best-stable',
    bestStableMinVisits: 8,
    bestStableCheckInterval: 4,
    bestStableChecks: 2,
    bestStableMinVisitLead: 4,
    bestStableMaxQDelta: 1,
  });
  assert.equal(result.stats?.completedVisits, 16);
  assert.equal(result.stats?.stopReason, 'visit-budget');
});

test('PUCT best-stable default min visits is fixed for movetime searches', async () => {
  const board = parseFen('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');
  const result = await searchRoot(board, new SlowDominantFirstMoveEvaluator(), {
    movetimeMs: 25,
    earlyStop: 'best-stable',
    bestStableCheckInterval: 4,
    bestStableChecks: 1,
    bestStableMinVisitLead: 1,
    bestStableMaxQDelta: 1,
  });
  assert.notEqual(result.stats?.stopReason, 'best-stable');
  assert.ok((result.stats?.completedVisits ?? 0) < 32, `test should stay below default min visits, got ${result.stats?.completedVisits}`);
});

test('PUCT root-dominance does not relabel full-budget completion', async () => {
  const board = parseFen('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');
  const result = await searchRoot(board, new UniformContextEvaluator(), { visits: 4, batchSize: 4, earlyStop: 'root-dominance' });
  assert.equal(result.stats?.completedVisits, 4);
  assert.equal(result.stats?.stopReason, 'visit-budget');
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
