import assert from 'node:assert/strict';
import test from 'node:test';
import { analysisAgreementHeadline, analysisArrowCandidates, analysisSpreadText, looksLikeFen, sharedMarkerCorner, analysisAvailability, analysisConsensus, analysisEngineName, analysisMoveGroups, analysisMoveMarkup, analysisPvText, analysisShortEngineNames, defaultAnalysisEngineRows, engineMonogram, nextAnalysisEvalSource, sharedArrowMarkerSvg } from '../src/lc0/analysisPresentation.ts';
import { ANALYSIS_DRAWABLE_BRUSHES } from '../src/lc0/analysisFormat.ts';
import { GameTree } from '../src/lc0/gameTree.ts';

const line = (engine, move, san, scoreCp, mateIn) => ({ engine, pvUci: [move], pvSan: san, scoreCp, mateIn, multipv: 1, scoreText: '', detail: '' });

test('first visit and reset compare two small engines without sharing mutable rows', () => {
  const rows = defaultAnalysisEngineRows();
  assert.deepEqual(rows, [{ family: 'lc0', variant: 'small', strength: 400 }, { family: 'sf', variant: 'lite', strength: 12 }]);
  rows[0].variant = 'bt4';
  assert.equal(defaultAnalysisEngineRows()[0].variant, 'small');
});

test('default arrows show only engine best moves; opt-in alternatives never cover a best move', () => {
  const lc0 = line('Lc0', 'd2d4', 'd4', 20);
  const sf = line('SF Lite', 'e2e4', 'e4', 30);
  const alternative = { ...line('Lc0', 'e2e4', 'e4', 10), multipv: 2 };
  const third = { ...line('Lc0', 'g1f3', 'Nf3', 0), multipv: 3 };
  const failed = { ...line('Berserk', 'a2a4', 'a4', 0), error: 'Unavailable' };
  const lines = [alternative, lc0, third, sf, failed];
  assert.deepEqual(analysisArrowCandidates(lines, false), [{ uci: 'e2e4', brush: 'blue' }, { uci: 'd2d4', brush: 'violet' }]);
  assert.deepEqual(analysisArrowCandidates(lines, true), [{ uci: 'g1f3', brush: 'paleViolet' }, { uci: 'e2e4', brush: 'blue' }, { uci: 'd2d4', brush: 'violet' }]);
  assert.equal(lines[0], alternative, 'rendering does not reorder cached lines');
  assert.ok(Object.values(ANALYSIS_DRAWABLE_BRUSHES).every((brush) => brush.lineWidth === 15));
  assert.equal(analysisArrowCandidates([lc0, { ...sf, pvUci: ['d2d4'] }], true).length, 1);
});

test('engine focus never borrows another engine’s arrow while results are pending or failed', () => {
  const lc0 = line('Lc0', 'd2d4', 'd4', 20);
  const sf = line('SF Lite', 'e2e4', 'e4', 30);
  assert.deepEqual(analysisArrowCandidates([lc0], false, 'Stockfish Lite'), []);
  assert.deepEqual(analysisArrowCandidates([lc0, { ...sf, error: 'Unavailable' }], true, 'Stockfish Lite'), []);
  assert.deepEqual(analysisArrowCandidates([lc0, sf], false, 'Stockfish Lite'), [{ uci: 'e2e4', brush: 'blue' }]);
});

test('eval source cycling skips unavailable engines and wraps through ready sources', () => {
  const engines = ['Lc0 Small', 'Stockfish Lite', 'Reckless'];
  const lc0 = line('Lc0', 'd2d4', 'd4', 20);
  const sf = { ...line('SF Lite', 'e2e4', 'e4', 30), error: 'Unavailable' };
  const reckless = line('Reckless', 'g1f3', 'Nf3', 10);
  const results = [lc0, sf, reckless];
  assert.equal(nextAnalysisEvalSource(engines, results, 'Lc0 Small'), 'Reckless');
  assert.equal(nextAnalysisEvalSource(engines, results, 'Reckless'), 'Lc0 Small');
  assert.equal(nextAnalysisEvalSource(engines, results, 'Stockfish Lite'), 'Lc0 Small');
  assert.equal(nextAnalysisEvalSource(engines, [lc0], 'Lc0 Small'), 'Lc0 Small');
  assert.equal(nextAnalysisEvalSource(engines, [], null), null);
});

test('a shared best move is one arrow that names every backing engine colour', () => {
  const lc0 = line('Lc0', 'c3e2', 'Ne2', 25);
  const sf = line('SF Lite', 'c3e2', 'Ne2', 39);
  const berserk = line('Berserk', 'c3a4', 'Na4', 48);
  const candidates = analysisArrowCandidates([lc0, sf, berserk], false);
  assert.equal(candidates.length, 2);
  assert.deepEqual(candidates.find((c) => c.uci === 'c3e2'), { uci: 'c3e2', brush: 'violet', shared: ['Lc0', 'SF Lite'] });
  assert.equal(candidates.find((c) => c.uci === 'c3a4').shared, undefined);
  const marker = sharedArrowMarkerSvg(['Lc0', 'SF Lite']);
  assert.match(marker, /#7a4fd6/);
  assert.match(marker, /#1f5fcf/);
  assert.equal((marker.match(/<rect /g) ?? []).length, 2);
  assert.match(marker, /Shared move: Lc0 Small \+ Stockfish Lite/);
  assert.match(marker, />L0<\/text>/);
  assert.match(marker, />SF<\/text>/);
  assert.doesNotMatch(marker, /translate\(/, 'Chessground custom viewBox already aligns the SVG with square bounds');
  assert.equal(analysisArrowCandidates([lc0, lc0, sf], false)[0].shared.length, 2, 'duplicate results never double-count a backer');
});

test('shared-move identity badges stay inside the destination square in either orientation', () => {
  const engines = ['Lc0', 'SF Lite', 'Berserk', 'PlentyChess', 'Reckless', 'Viridithas', 'Stormphrax', '<unsafe&>'];
  for (const corner of [[1, -1], [-1, -1]]) {
    const marker = sharedArrowMarkerSvg(engines.slice(0, 4), corner);
    const badges = [...marker.matchAll(/<rect x="([\d.-]+)" y="([\d.-]+)" width="([\d.]+)" height="([\d.]+)"/g)].map(([, ...values]) => values.map(Number));
    assert.equal(badges.length, 4);
    // Include the 2px outline in the bounds; the custom viewBox makes these square-local.
    assert.ok(badges.every(([x, y, w, h]) => x >= 1 && x + w <= 99 && y >= 1 && y + h <= 99));
    assert.ok(badges.every(([x, , w]) => corner[0] > 0 ? x + w / 2 > 50 : x + w / 2 < 50));
    assert.ok(badges.every(([, y, , h]) => y + h < 50));
  }
  const all = sharedArrowMarkerSvg(engines);
  assert.equal((all.match(/<rect /g) ?? []).length, 4);
  assert.match(all, />\+5<\/text>/);
  const many = sharedArrowMarkerSvg(Array.from({ length: 20 }, (_, i) => `Engine ${i}`));
  assert.equal((many.match(/<rect /g) ?? []).length, 4);
  assert.match(many, />\+17<\/text>/);
  assert.match(many, /Engine 19/, 'all contributors remain in the full title');
  assert.match(all, /&lt;unsafe&amp;&gt;/);
  assert.doesNotMatch(all, /<unsafe/);
  assert.equal(sharedArrowMarkerSvg([]), '');
  assert.deepEqual(sharedMarkerCorner('c3e2', 'white'), [1, -1]);
  assert.deepEqual(sharedMarkerCorner('c3e2', 'black'), [-1, -1]);
  assert.deepEqual(sharedMarkerCorner('g1e2', 'white'), [-1, -1]);
  assert.deepEqual(sharedMarkerCorner('e2e4', 'white'), [1, -1]);
});

test('callout groups engines by move and states agreement explicitly', () => {
  const lc0 = line('Lc0', 'c3e2', 'Ne2', 25);
  const sf = line('SF Lite', 'c3e2', 'Ne2', 39);
  const berserk = line('Berserk', 'c3a4', 'Na4', 48);
  const plenty = line('PlentyChess', 'g5f6', 'gxf6', 9);
  const groups = analysisMoveGroups([berserk, lc0, sf, plenty]);
  assert.deepEqual(groups.map((g) => [g.san, g.engines.length]), [['Ne2', 2], ['Na4', 1], ['gxf6', 1]]);
  assert.equal(analysisAgreementHeadline(groups, 4), '2 of 4 agree on Ne2');
  assert.equal(analysisAgreementHeadline(analysisMoveGroups([lc0, berserk]), 2), 'Different top moves: Ne2 / Na4');
  assert.equal(analysisAgreementHeadline(analysisMoveGroups([lc0, berserk]), 2, 23), 'Different top moves: Ne2 / Na4');
  assert.equal(analysisAgreementHeadline(analysisMoveGroups([lc0, berserk]), 2, 140), 'Engines disagree: Ne2 / Na4');
  assert.equal(analysisAgreementHeadline(analysisMoveGroups([lc0, sf]), 2), 'Engines agree on Ne2');
  assert.equal(analysisAgreementHeadline(analysisMoveGroups([lc0, sf, berserk, { ...plenty, pvUci: ['c3a4'] }]), 4), 'Split: Ne2 / Na4');
  assert.deepEqual(analysisSpreadText(26), { text: 'Eval spread: 0.26 pawns', level: 'close' });
  assert.deepEqual(analysisSpreadText(74), { text: 'Eval spread: 0.74 pawns', level: 'apart' });
  assert.equal(analysisSpreadText(120).level, 'wide');
  assert.deepEqual([...analysisShortEngineNames(['Lc0', 'SF Lite']).values()], ['Lc0', 'Stockfish']);
  assert.deepEqual([...analysisShortEngineNames(['Lc0', 'Lc0 BT4']).values()], ['Lc0 Small', 'Lc0 BT4']);
});

test('engine avatars use one two-character uppercase monogram rule', () => {
  assert.equal(engineMonogram('Lc0'), 'L0');
  assert.equal(engineMonogram('SF Lite'), 'SF');
  assert.equal(engineMonogram('PlentyChess'), 'PC');
  assert.equal(engineMonogram('Berserk'), 'BE');
});

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
  const empty = analysisMoveMarkup(new GameTree().root, tree.current);
  assert.match(empty, /No moves yet/);
  assert.match(empty, /<textarea[^>]+data-quick-pgn/);
  assert.match(empty, /<button[^>]+data-quick-pgn-load>Load<\/button>/);
  assert.match(empty, /PGN or FEN/);
  assert.ok(looksLikeFen('rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1'));
  assert.ok(looksLikeFen('8/8/8/8/8/8/8/K6k'));
  assert.ok(!looksLikeFen('1. e4 c5 2. Nf3 d6'));
  assert.ok(!looksLikeFen('[Event "x"]\n1. e4 e5'));
});
