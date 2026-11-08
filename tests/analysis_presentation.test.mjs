import assert from 'node:assert/strict';
import test from 'node:test';
import { analysisAgreementHeadline, analysisArrowCandidates, analysisChipColor, analysisSpreadText, isEmptyAnalysisGame, looksLikeFen, sharedMarkerPosition, sharedArrowMarkerText, analysisAvailability, analysisConsensus, analysisEngineName, analysisMoveGroups, analysisMoveMarkup, analysisPvText, analysisShortEngineNames, defaultAnalysisEngineRows, engineMonogram, nextAnalysisEvalSource, sharedArrowMarkerSvg } from '../src/lc0/analysisPresentation.ts';
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
  assert.match(marker, /#6941b8/);
  assert.equal((marker.match(/<rect /g) ?? []).length, 1);
  assert.match(marker, /aria-label="2 engines share this move: Lc0 Small, Stockfish Lite"/);
  assert.match(marker, /<title>2 engines share this move: Lc0 Small, Stockfish Lite<\/title>/);
  assert.match(marker, />L0 SF<\/text>/);
  assert.match(marker, /font-size="22"/);
  assert.match(marker, /pointer-events:none/, 'markers must not intercept chess-square taps');
  assert.match(marker, /tabindex="0"/, 'engine attribution is still available to keyboard users');
  assert.doesNotMatch(marker, /translate\(/, 'Chessground custom viewBox already aligns the SVG with square bounds');
  assert.equal(analysisArrowCandidates([lc0, lc0, sf], false)[0].shared.length, 2, 'duplicate results never double-count a backer');
});

test('shared pills show two or three monograms and use a count only for a wide 4+ group', () => {
  assert.equal(sharedArrowMarkerText(['Lc0', 'PlentyChess']), 'L0 PC');
  assert.equal(sharedArrowMarkerText(['Lc0', 'SF Lite', 'PlentyChess']), 'L0 SF PC');
  const engines = ['Lc0', 'SF Lite', 'Berserk', 'PlentyChess', 'Reckless', 'Viridithas', 'Stormphrax', '<unsafe&>'];
  assert.equal(sharedArrowMarkerText(engines.slice(0, 4)), '4');
  const all = sharedArrowMarkerSvg(engines);
  assert.equal((all.match(/<rect /g) ?? []).length, 1);
  assert.match(all, />8<\/text>/);
  assert.match(all, /&lt;unsafe&amp;&gt;/);
  assert.doesNotMatch(all, /<unsafe/);
  assert.equal(sharedArrowMarkerSvg([]), '');
});

test('pills stay on board and clear the destination square in both orientations, including captures', () => {
  const fen = 'r2q1rk1/3nbppp/p2pbn2/4p1P1/1p2P3/1NN1BP2/PPPQ3P/2KR1B1R w - - 0 13';
  for (const uci of ['c3e2', 'g5f6', 'e4e5', 'a2a3', 'h2h3']) {
    for (const orientation of ['white', 'black']) {
      for (const engines of [['Lc0', 'PlentyChess'], ['Lc0', 'SF Lite', 'PlentyChess'], ['Lc0', 'SF Lite', 'Berserk', 'PlentyChess']]) {
        const [x, y] = sharedMarkerPosition(uci, orientation, fen, engines);
        const sign = orientation === 'white' ? 1 : -1;
        const fromX = orientation === 'white' ? uci.charCodeAt(0) - 97 : 104 - uci.charCodeAt(0);
        const fromY = orientation === 'white' ? 8 - Number(uci[1]) : Number(uci[1]) - 1;
        const width = sharedArrowMarkerText(engines).length * 13.2 + 16;
        assert.ok(fromX * 100 + x - width / 2 >= 0);
        assert.ok(fromX * 100 + x + width / 2 <= 800);
        assert.ok(fromY * 100 + y - 16 >= 0 && fromY * 100 + y + 16 <= 800);
        const destX = 50 + (uci.charCodeAt(2) - uci.charCodeAt(0)) * sign * 100;
        const destY = 50 - (Number(uci[3]) - Number(uci[1])) * sign * 100;
        assert.ok(Math.abs(x - destX) >= width / 2 + 50 || Math.abs(y - destY) >= 66, `${uci}/${orientation}: pill must clear destination`);
      }
    }
  }
});

test('Analysis monogram fills have at least 4.5:1 contrast with white without changing arrows', () => {
  const luminance = (hex) => {
    const channels = hex.slice(1).match(/../g).map((x) => parseInt(x, 16) / 255).map((x) => x <= .04045 ? x / 12.92 : ((x + .055) / 1.055) ** 2.4);
    return channels[0] * .2126 + channels[1] * .7152 + channels[2] * .0722;
  };
  for (const engine of ['Lc0', 'SF Lite', 'Berserk', 'PlentyChess', 'Viridithas', 'Reckless', 'Stormphrax']) {
    assert.ok(1.05 / (luminance(analysisChipColor(engine)) + .05) >= 4.5, engine);
  }
  assert.equal(ANALYSIS_DRAWABLE_BRUSHES.orange.color, '#c45a11');
  assert.equal(ANALYSIS_DRAWABLE_BRUSHES.cyan.color, '#007a87');
});

test('mobile Load shortcut belongs only to a standard empty game, including after Reset', () => {
  const tree = new GameTree();
  assert.ok(isEmptyAnalysisGame(tree.root));
  tree.addUci('e2e4');
  assert.ok(!isEmptyAnalysisGame(tree.root));
  tree.toStart();
  assert.ok(!isEmptyAnalysisGame(tree.root), 'navigating to start does not make a loaded game empty');
  assert.ok(!isEmptyAnalysisGame(new GameTree('8/8/8/8/8/8/8/K6k w - - 0 1').root));
  assert.ok(isEmptyAnalysisGame(new GameTree().root));
});

test('callout groups engines by move and states agreement explicitly', () => {
  const lc0 = line('Lc0', 'c3e2', 'Ne2', 25);
  const sf = line('SF Lite', 'c3e2', 'Ne2', 39);
  const berserk = line('Berserk', 'c3a4', 'Na4', 48);
  const plenty = line('PlentyChess', 'g5f6', 'gxf6', 9);
  const groups = analysisMoveGroups([berserk, lc0, sf, plenty]);
  assert.deepEqual(groups.map((g) => [g.san, g.engines.length]), [['Ne2', 2], ['Na4', 1], ['gxf6', 1]]);
  assert.equal(analysisAgreementHeadline(groups, 4), '2 of 4 engines pick Ne2');
  assert.equal(analysisAgreementHeadline(analysisMoveGroups([lc0, berserk]), 2), 'Engines split: Ne2 / Na4');
  assert.equal(analysisAgreementHeadline(analysisMoveGroups([lc0, sf]), 2), 'Both engines pick Ne2');
  assert.equal(analysisAgreementHeadline(analysisMoveGroups([lc0, sf, berserk, { ...plenty, pvUci: ['c3a4'] }]), 4), 'Engines split: Ne2 / Na4');
  assert.equal(analysisAgreementHeadline([], 0), 'No best move yet');
  assert.equal(analysisAgreementHeadline(analysisMoveGroups([lc0]), 1), 'Best move: Ne2');
  assert.equal(analysisAgreementHeadline(analysisMoveGroups([lc0, sf, { ...berserk, pvUci: ['c3e2'] }]), 3), 'All 3 engines pick Ne2');
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
  assert.match(empty, /<label for="quickPgn">PGN game or FEN position<\/label>/);
  assert.match(empty, /<textarea[^>]+id="quickPgn"[^>]+data-quick-pgn/);
  assert.match(empty, /<button[^>]+data-quick-pgn-load>Load<\/button>/);
  assert.match(empty, /PGN or FEN/);
  assert.ok(looksLikeFen('rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1'));
  assert.ok(looksLikeFen('8/8/8/8/8/8/8/K6k'));
  assert.ok(!looksLikeFen('1. e4 c5 2. Nf3 d6'));
  assert.ok(!looksLikeFen('[Event "x"]\n1. e4 e5'));
});
