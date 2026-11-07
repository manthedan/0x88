import assert from 'node:assert/strict';
import test from 'node:test';
import { analysisAvailability, analysisConsensus, analysisEngineName, analysisMoveMarkup, analysisPvText } from '../src/lc0/analysisPresentation.ts';
import { GameTree } from '../src/lc0/gameTree.ts';

const line = (engine, move, san, scoreCp, mateIn) => ({ engine, pvUci: [move], pvSan: san, scoreCp, mateIn, multipv: 1, scoreText: '', detail: '' });

test('PVs number White and initial Black moves, including the full continuation', () => {
  const tree = new GameTree();
  const pvUci = ['e2e4', 'e7e5', 'g1f3', 'b8c6'];
  assert.equal(analysisPvText(tree.current.fen, { pvUci }), '1. e4 e5 2. Nf3 Nc6');
  tree.addUci('e2e4');
  assert.equal(analysisPvText(tree.current.fen, { pvUci: pvUci.slice(1) }), '1… e5 2. Nf3 Nc6');
  const late = new GameTree('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 12');
  assert.equal(analysisPvText(late.current.fen, { pvUci: ['e7e5', 'e2e4'] }), '12… e5 13. e4');
});

test('comparison describes ties honestly and uses an unsigned spread', () => {
  assert.equal(analysisConsensus([line('Lc0', 'c3e2', 'Ne2 Nh5', 25), line('SF Lite', 'g5f6', 'gxf6 Qxc3', 36)]), 'Engines disagree: Ne2 / gxf6 · 11 cp spread');
  assert.equal(analysisConsensus([line('A', 'e2e4', 'e4', -20), line('B', 'e2e4', 'e4', 10)]), 'Engines agree: e4 · 30 cp spread');
});

test('only a strict majority is described as a preference', () => {
  const a = line('A', 'e2e4', 'e4', 20);
  const b = line('B', 'd2d4', 'd4', 30);
  assert.match(analysisConsensus([a, a, b]), /^2\/3 engines prefer e4/);
  assert.match(analysisConsensus([a, a, b, b]), /^Engines disagree: e4 \/ d4/);
  assert.doesNotMatch(analysisConsensus([a, line('B', 'e2e4', 'e4', undefined, 3)]), /cp spread/);
});

test('completed failures and empty results are not labeled pending', () => {
  assert.deepEqual(analysisAvailability(2, 1, 0, true), { pending: 1, unavailable: 0 });
  assert.deepEqual(analysisAvailability(2, 1, 1, true), { pending: 0, unavailable: 1 });
  assert.deepEqual(analysisAvailability(2, 1, 0, false), { pending: 0, unavailable: 1 });
  assert.deepEqual(analysisAvailability(2, 2, 0, false), { pending: 0, unavailable: 0 });
});

test('friendly names do not require changing engine/cache identities', () => {
  assert.equal(analysisEngineName('SF Lite'), 'Stockfish Lite');
  assert.equal(analysisEngineName('Lc0'), 'Lc0 Small');
  assert.equal(analysisEngineName('Lc0 BT4'), 'Lc0 BT4');
  assert.equal(analysisEngineName('Lc0 t3-512'), analysisEngineName('Lc0 t3-512 distill'));
});

test('history pairs numbered plies with named buttons and exactly one current move', () => {
  const tree = new GameTree();
  tree.addUci('e2e4'); tree.addUci('e7e5'); tree.addUci('g1f3');
  const html = analysisMoveMarkup(tree.root, tree.current);
  assert.equal((html.match(/class="move-pair"/g) ?? []).length, 2);
  assert.equal((html.match(/data-node=/g) ?? []).length, 3);
  assert.equal((html.match(/aria-current="step"/g) ?? []).length, 1);
  assert.match(html, /aria-label="1\.\.\. e5"/);
  assert.match(html, /<button type="button" class="mv current"[^>]*>Nf3<\/button>/);
});

test('history retains nested variations without duplicating sibling branches', () => {
  const tree = new GameTree();
  tree.addUci('e2e4'); tree.addUci('e7e5');
  tree.toStart(); tree.addUci('d2d4'); tree.addUci('d7d5');
  tree.back(); tree.addUci('g8f6');
  const html = analysisMoveMarkup(tree.root, tree.current);
  assert.equal((html.match(/data-node=/g) ?? []).length, 5);
  assert.equal((html.match(/class="variation"/g) ?? []).length, 2);
  assert.match(html, /aria-current="step"[^>]*>Nf6/);
});

test('Black-to-move FEN history leaves the White ply empty and escapes imported notation', () => {
  const tree = new GameTree('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 12');
  tree.addUci('e7e5');
  tree.current.san = '<img onerror="bad">';
  const html = analysisMoveMarkup(tree.root, tree.current);
  assert.match(html, /move-number">12\./);
  assert.match(html, /missing-ply/);
  assert.match(html, /&lt;img/);
  assert.doesNotMatch(html, /<img/);
  assert.match(analysisMoveMarkup(new GameTree().root, tree.current), /No moves yet/);
});
