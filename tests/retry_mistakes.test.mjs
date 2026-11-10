import assert from 'node:assert/strict';
import test from 'node:test';
import { legalMoves } from '../src/chess/movegen.ts';
import { parseFen } from '../src/chess/board.ts';
import { parsePgnGame } from '../src/chess/pgn.ts';
import { GameTree } from '../src/lc0/gameTree.ts';
import { MistakeRetry, acceptableRetryScore, retryProblems, retrySideFromTags, retryTreeAt } from '../src/lc0/retryMistakes.ts';
import { quarantineRetryPanels } from '../src/lc0/retrySpoilers.ts';
import { serializeTree } from '../src/lc0/workspaceAutosave.ts';

function fixture() {
  const source = parsePgnGame('1. f3 e5 2. g4 Qh4# (2... Nc6)').tree;
  const nodes = [source.root, ...source.mainlineFrom()];
  const moves = nodes.slice(1).map((node, i) => ({ ply: i + 1, side: i % 2 ? 'b' : 'w', class: 'blunder', moverLoss: .4 - i / 20, san: node.san, uci: ['f2f3', 'e7e5', 'g2g4', 'd8h4'][i], bestUci: ['e2e4', 'd7d5', 'e2e4', 'b8c6'][i] }));
  return { source, nodes, review: { moves } };
}

test('selected side is filtered before the five-position limit and inaccuracies are excluded', () => {
  const { source, nodes, review } = fixture();
  assert.deepEqual(retryProblems(review, nodes, 'w').map((p) => p.move.ply), [1, 3]);
  assert.deepEqual(retryProblems(review, nodes, 'b').map((p) => p.move.ply), [2, 4]);
  assert.equal(retryProblems({ moves: review.moves.map((m) => ({ ...m, class: 'inaccuracy' })) }, nodes, 'w').length, 0);
  const long = new GameTree();
  for (const uci of Array(4).fill(['g1f3', 'g8f6', 'f3g1', 'f6g8']).flat()) long.addUci(uci);
  const longNodes = [long.root, ...long.mainlineFrom()];
  const moves = longNodes.slice(1).map((node, i) => ({ ...review.moves[0], ply: i + 1, side: i % 2 ? 'b' : 'w', moverLoss: i % 2 ? .9 : .2, bestUci: i % 4 === 0 ? 'e2e4' : i % 4 === 2 ? 'e2e4' : 'e7e5' }));
  assert.equal(retryProblems({ moves }, longNodes, 'w').length, 5);
  assert.equal(source.root.fen, nodes[0].fen);
});

test('single-legal-move positions are skipped even when a stale review calls them a blunder', () => {
  const source = new GameTree('7k/8/5K2/8/8/8/8/7R b - - 0 1');
  assert.equal(legalMoves(parseFen(source.root.fen)).length, 1);
  const review = { moves: [{ ply: 1, side: 'b', class: 'blunder', moverLoss: .5, bestUci: 'h8g8' }] };
  assert.deepEqual(retryProblems(review, [source.root], 'b'), []);
});

test('best match avoids a search; two equally good alternatives are both accepted', async () => {
  const { source, nodes, review } = fixture();
  const problems = retryProblems(review, nodes, 'w');
  const exact = new MistakeRetry(source, problems);
  await exact.attempt('e2e4', async () => { throw new Error('Should not search'); });
  assert.equal(exact.phase, 'solved');
  for (const alternative of ['d2d4', 'g1f3']) {
    const session = new MistakeRetry(source, problems);
    const calls = [];
    await session.attempt(alternative, async (tree, signal) => { calls.push(tree); assert.equal(signal.aborted, false); return .61; });
    assert.equal(session.phase, 'solved');
    assert.equal(session.solved, 1);
    assert.equal(calls.length, 2);
    assert.notEqual(calls[0], calls[1]);
  }
});

test('expected-score tolerance is mover-relative, inclusive and rejects missing scores', () => {
  assert.equal(acceptableRetryScore('w', .6, .58), true);
  assert.equal(acceptableRetryScore('w', .6, .579), false);
  assert.equal(acceptableRetryScore('b', .4, .42), true);
  assert.equal(acceptableRetryScore('b', .4, .421), false);
  assert.equal(acceptableRetryScore('b', .4, .1), true);
  assert.throws(() => acceptableRetryScore('w', NaN, .5), /scored result/);
});

test('bad moves say try again, remain on the pre-error board and do not count as solved', async () => {
  const { source, nodes, review } = fixture();
  const session = new MistakeRetry(source, retryProblems(review, nodes, 'w'));
  const before = session.tree.current.fen;
  let call = 0;
  await session.attempt('f2f3', async () => ++call === 1 ? .6 : .3);
  assert.equal(session.phase, 'wrong');
  assert.equal(session.tree.current.fen, before);
  assert.equal(session.solved, 0);
  assert.equal(session.remaining, 2);
  assert.equal(session.isLast, false);
  session.reveal();
  assert.equal(session.remaining, 1);
  session.next();
  assert.equal(session.isLast, true, 'the last item offers finish, not another position');
  assert.equal(session.tree.current.fen, nodes[2].fen, 'before the next error, not after it');
  session.reveal();
  session.next();
  assert.equal(session.phase, 'complete');
  assert.equal(session.remaining, 0);
});

test('cancel on skip/reveal/exit prevents stale verification from solving another problem', async () => {
  for (const action of ['next', 'reveal', 'cancel']) {
    const { source, nodes, review } = fixture();
    const session = new MistakeRetry(source, retryProblems(review, nodes, 'w'));
    let resolve;
    let signal;
    const pending = session.attempt('d2d4', (_, s) => { signal = s; return new Promise((r) => { resolve = r; }); });
    assert.equal(session.phase, 'checking');
    session[action]();
    assert.equal(signal.aborted, true);
    resolve(.5);
    await pending;
    assert.equal(session.solved, 0);
    assert.notEqual(session.phase, 'solved');
  }
});

test('temporary replay and all attempts preserve the exact tree, branches, comments and current object', async () => {
  const { source, nodes, review } = fixture();
  nodes[1].comment = 'Original note';
  source.goTo(nodes[1]);
  source.addUci('c7c5');
  const current = source.current;
  const snapshot = serializeTree(source);
  const session = new MistakeRetry(source, retryProblems(review, nodes, 'w'));
  assert.notEqual(retryTreeAt(source, nodes[2]).root, source.root);
  await session.attempt('d2d4', async () => .5);
  session.next();
  session.reveal();
  session.next();
  session.cancel();
  assert.equal(session.phase, 'complete');
  assert.equal(source.current, current);
  assert.deepEqual(serializeTree(source), snapshot);
});

test('verification failures stay unsolved and an illegal move cannot trigger a search', async () => {
  const { source, nodes, review } = fixture();
  const session = new MistakeRetry(source, retryProblems(review, nodes, 'w'));
  await session.attempt('e7e5', async () => { throw new Error('Should not search an illegal move'); });
  assert.equal(session.phase, 'attempt');
  await assert.rejects(session.attempt('d2d4', async () => { throw new Error('No scored result'); }), /No scored/);
  assert.equal(session.phase, 'attempt');
  assert.equal(session.solved, 0);
  assert.throws(() => new MistakeRetry(source, []), /No retry positions/);
});

test('black alternatives are checked from Black’s perspective with the preceding history intact', async () => {
  const { source, nodes, review } = fixture();
  const session = new MistakeRetry(source, retryProblems(review, nodes, 'b'));
  let calls = 0;
  await session.attempt('c7c5', async (candidate) => {
    assert.equal(candidate.lineTo()[1].san, 'f3');
    return ++calls === 1 ? .3 : .31;
  });
  assert.equal(session.phase, 'solved');
  assert.equal(calls, 2);
});

test('side defaults use explicit user/human PGN tags, never guess from names or ambiguous humans', () => {
  assert.equal(retrySideFromTags({ White: 'Alice', Black: 'Bob' }), undefined);
  assert.equal(retrySideFromTags({ White: 'You', Black: 'Stockfish' }), 'w');
  assert.equal(retrySideFromTags({ WhiteType: 'program', BlackType: 'human' }), 'b');
  assert.equal(retrySideFromTags({ WhiteType: 'human', BlackType: 'human' }), undefined);
  assert.equal(retrySideFromTags({ UserColor: 'black' }), 'b');
});

// Small DOM contract double: inspect all children/attributes, including hidden/aria content.
class Element {
  constructor(tagName, id = '', text = '', attrs = {}, children = []) { Object.assign(this, { tagName, id, text, attrs, childNodes: children, hidden: false, ownerDocument: { createElement: (tag) => new Element(tag) } }); }
  querySelectorAll() { return this.childNodes.flatMap((node) => [node, ...node.querySelectorAll()]).filter((node) => node.id); }
  replaceChildren(...nodes) { this.childNodes = nodes; }
}
test('quarantine leaves zero spoiler text or attributes in hidden and aria panels, and restores exact nodes', () => {
  const spoilers = new Element('DIV', 'line', 'Qh4# +9.8 blunder 88%', { 'aria-label': 'Best Qh4#', title: 'blunder', 'data-pv': 'd8h4' });
  const host = new Element('DIV', 'humanPanel', '', {}, [spoilers]);
  host.hidden = true;
  const restore = quarantineRetryPanels([host]);
  assert.doesNotMatch(JSON.stringify(host.childNodes), /Qh4|blunder|88%|data-pv|aria-label|9\.8/);
  assert.equal(host.childNodes[0].id, 'line', 'empty lifecycle hosts still exist');
  restore();
  assert.equal(host.childNodes[0], spoilers);
  assert.equal(host.hidden, true);
});
