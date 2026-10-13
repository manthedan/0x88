import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { boardToFen, parseFen, START_FEN } from '../src/chess/board.ts';
import { rankFlipColorSwapBoard } from '../src/chess/boardNormalization.ts';
import { moveToActionId, moveToUci } from '../src/chess/moveCodec.ts';
import { legalMoves } from '../src/chess/movegen.ts';
import { BATCHING_STRENGTH_MATCHES } from '../src/lc0/batchingStrengthMatch.ts';
import {
  LQO_BLACK_MISSING_QUEEN_FEN,
  LQO_WHITE_MISSING_QUEEN_FEN,
  lqoQueenOddsOpenings,
  scheduleGame,
  smallNetOpenings,
} from '../src/lc0/batchingStrengthOpenings.ts';
import { strengthRandom, strengthSeed } from '../src/lc0/batchingStrengthRandom.ts';
import { candidatePoints, equalVisitGate, pentanomialElo, queenSidePoints, uciLineToPgn } from '../src/lc0/batchingStrengthScore.ts';
import { swapColorsUci } from '../src/lc0/bt4Engine.ts';
import { resolveOrtPhysicalBatchSize } from '../src/lc0/ortPhysicalBatch.ts';
import { searchRoot } from '../src/search/puct.ts';

const WAVE1_NET_VIEW = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR b KQkq - 0 1';

function queens(fen) {
  const board = parseFen(fen);
  let w = 0;
  let b = 0;
  for (const piece of board.squares) {
    if (piece === 'wq') w += 1;
    else if (piece === 'bq') b += 1;
  }
  return { w, b };
}

function pairGames(pairs, evenResult, oddResult, evenWhite = true) {
  const games = [];
  for (let pair = 0; pair < pairs; pair++) {
    const even = pair * 2;
    games.push({ index: even, result: evenResult, candidateWhite: evenWhite });
    games.push({ index: even + 1, result: oddResult, candidateWhite: !evenWhite });
  }
  return games;
}

test('physical batch resolver keeps fixed graphs and defaults dynamic dims to serial batch-1', () => {
  assert.equal(resolveOrtPhysicalBatchSize('batch', 8, 'declared'), 1);
  assert.equal(resolveOrtPhysicalBatchSize('batch', 8, 'logical'), 8);
  assert.equal(resolveOrtPhysicalBatchSize(undefined, 4), 1);
  assert.equal(resolveOrtPhysicalBatchSize(-1, 4, 'logical'), 4);
  assert.equal(resolveOrtPhysicalBatchSize(0, 4, 'logical'), 4);
  assert.equal(resolveOrtPhysicalBatchSize(Number.NaN, 5, 'declared'), 1);
  assert.equal(resolveOrtPhysicalBatchSize(8.9, 3, 'logical'), 8);
  assert.equal(resolveOrtPhysicalBatchSize(8, 3, 'declared'), 8);
  assert.equal(resolveOrtPhysicalBatchSize(8, 3, 'logical'), 8);
  assert.equal(resolveOrtPhysicalBatchSize('batch', 0, 'logical'), 1);
});

test('lab ScLimit RNG is reproducible and sensitive to the full input history', () => {
  const seed = strengthSeed(START_FEN);
  const a = strengthRandom(seed);
  const b = strengthRandom(seed);
  const values = Array.from({ length: 50 }, () => a());
  assert.deepEqual(
    values,
    Array.from({ length: 50 }, () => b()),
  );
  assert.ok(values.every((value) => value >= 0 && value < 1));
  assert.notEqual(strengthSeed(`${START_FEN}\ne2e4`), seed);
  assert.notEqual(strengthRandom(seed + 1)(), values[0]);
});

test('product evaluator and worker default to declared physical batching', () => {
  const evaluator = readFileSync(new URL('../src/lc0/onnxEvaluator.ts', import.meta.url), 'utf8');
  const worker = readFileSync(new URL('../src/lc0/searchWorkerCore.ts', import.meta.url), 'utf8');
  assert.match(evaluator, /physicalBatchMode === 'logical' \? 'logical' : 'declared'/);
  assert.match(worker, /physicalBatchMode === 'logical' \? 'logical' : 'declared'/);
  assert.match(worker, /Lc0OnnxEvaluator\.create\(modelLoad\.model, \{ physicalBatchMode \}\)/);
});

test('Play and LQO search constants stay locked to the product source', () => {
  const play = readFileSync(new URL('../src/lc0/playBrowser.ts', import.meta.url), 'utf8');
  const match = readFileSync(new URL('../src/lc0/batchingStrengthMatch.ts', import.meta.url), 'utf8');
  const catalog = readFileSync(new URL('../src/lc0/engineCatalog.ts', import.meta.url), 'utf8');
  for (const literal of [
    'const PLAY_DRAW_SCORE = -0.25;',
    'const PLAY_SEARCH_CONTEMPT_LIMIT = 16;',
    'const LQO_CPUCT = 1.5;',
    'const LQO_FPU_VALUE = 0.4;',
    'const LQO_WHITE_DRAW_SCORE = -0.4;',
    'const LQO_BLACK_DRAW_SCORE = -0.6;',
    'const LQO_WHITE_SEARCH_CONTEMPT_LIMIT = 40;',
    'const LQO_BLACK_SEARCH_CONTEMPT_LIMIT = 32;',
    'const BIG_NET_LEVELS = [4, 16, 64, 256, 800];',
  ]) {
    assert.match(play, new RegExp(literal.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')));
    if (!literal.startsWith('const BIG_NET'))
      assert.match(match, new RegExp(literal.replace('const ', 'export const ').replace(/[.*+?^${}()|[\]\\]/g, '\\$&')));
  }
  assert.match(catalog, /levels: \[8, 32, 100, 400, 1600\]/);
});

test('match catalog covers the ladders where moves changed and skips 1600', () => {
  const big = new Set([4, 16, 64, 256, 800]);
  const small = new Set([8, 32, 100, 400, 1600]);
  assert.equal(BATCHING_STRENGTH_MATCHES['lqo-v32'].visits, 32);
  for (const spec of Object.values(BATCHING_STRENGTH_MATCHES)) {
    assert.notEqual(spec.visits, 1600);
    assert.notEqual(spec.visits, 800);
    if (spec.movetimeMs) continue;
    if (spec.kind === 'lqo' && spec.visits === 32) continue;
    const ladder = spec.kind === 'lqo' ? big : small;
    assert.equal(ladder.has(spec.visits), true, spec.id);
  }
  assert.equal(BATCHING_STRENGTH_MATCHES['b1-v8'].candidate.batchSize, 8);
  assert.equal(BATCHING_STRENGTH_MATCHES['b1-v8'].baseline.batchSize, 8);
  assert.equal(BATCHING_STRENGTH_MATCHES['b2-v8'].baseline.batchSize, 1);
  assert.equal(BATCHING_STRENGTH_MATCHES['lqo-v32'].candidate.physicalBatchMode, 'logical');
  assert.equal(BATCHING_STRENGTH_MATCHES['lqo-v32'].baseline.physicalBatchMode, 'declared');
});

test('physical serial vs grouped evaluation cannot change fixed-visit logical batching with identical outputs', async () => {
  function evaluator(serial) {
    const evaluate = (board, context = {}) => {
      const moves = context.legalMoves ?? legalMoves(board);
      return { policy: new Map(moves.map((move, i) => [moveToActionId(move), i === 0 ? 0.7 : 0.3 / Math.max(1, moves.length - 1)])), wdl: [0.36, 0.34, 0.3] };
    };
    return {
      evaluate,
      async evaluateBatch(boards, contexts = []) {
        if (!serial) return boards.map((board, i) => evaluate(board, contexts[i]));
        const out = [];
        for (let i = 0; i < boards.length; i++) {
          // Separate promise turns mimic serial physical runs, without timing a benchmark.
          out.push(await Promise.resolve(evaluate(boards[i], contexts[i])));
        }
        return out;
      },
    };
  }
  for (const visits of [8, 32, 100, 400]) {
    const options = { visits, batchSize: 8, batchPipelineDepth: 1, temperature: 0, earlyStop: 'none', traceSearchVisits: true };
    const serial = await searchRoot(parseFen(START_FEN), evaluator(true), options);
    const grouped = await searchRoot(parseFen(START_FEN), evaluator(false), options);
    assert.equal(moveToUci(serial.move), moveToUci(grouped.move));
    assert.equal(serial.visits, visits);
    assert.equal(grouped.visits, visits);
    assert.deepEqual(serial.policy, grouped.policy);
    assert.deepEqual(serial.stats.searchTrace, grouped.stats.searchTrace);
    assert.ok(serial.stats.batchLeafCollisions > 0);
  }
});

test('queen-odds openings include both product starts and keep the odds', () => {
  const openings = lqoQueenOddsOpenings(100);
  assert.equal(openings.length, 100);
  assert.equal(new Set(openings.map((opening) => opening.fen)).size, 100);
  assert.equal(openings[0].name, 'lqo-product-black-missing');
  assert.equal(openings[0].fen, LQO_BLACK_MISSING_QUEEN_FEN);
  assert.equal(openings[1].fen, LQO_WHITE_MISSING_QUEEN_FEN);
  const swapped = boardToFen(rankFlipColorSwapBoard(parseFen(LQO_BLACK_MISSING_QUEEN_FEN)));
  assert.equal(swapped, WAVE1_NET_VIEW);
  assert.equal(swapColorsUci('g8f6'), 'g1f3');
  let blackMissing = 0;
  let whiteMissing = 0;
  for (const opening of openings) {
    const counts = queens(opening.fen);
    assert.ok(opening.positions.length >= 1);
    if (opening.queenLessColor === 'b') {
      blackMissing += 1;
      assert.deepEqual(counts, { w: 1, b: 0 });
    } else {
      whiteMissing += 1;
      assert.equal(opening.queenLessColor, 'w');
      assert.deepEqual(counts, { w: 0, b: 1 });
    }
  }
  assert.ok(blackMissing >= 70, `black-missing openings ${blackMissing}`);
  assert.ok(whiteMissing >= 8, `white-missing openings ${whiteMissing}`);
});

test('small-net openings keep both queens and start with the arena book', () => {
  const epd = `${START_FEN}\n${LQO_BLACK_MISSING_QUEEN_FEN}\nrnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1\n`;
  const openings = smallNetOpenings(epd, 20);
  assert.equal(openings[0].name, 'Start position');
  assert.equal(openings.length, 17);
  assert.ok(openings.every((opening) => opening.queenLessColor === null));
  for (const opening of openings) assert.deepEqual(queens(opening.fen), { w: 1, b: 1 });
});

test('scheduleGame pairs an opening with colors swapped', () => {
  assert.deepEqual(scheduleGame(100, 0), { pair: 0, openingIndex: 0, candidateWhite: true });
  assert.deepEqual(scheduleGame(100, 1), { pair: 0, openingIndex: 0, candidateWhite: false });
  assert.deepEqual(scheduleGame(100, 2), { pair: 1, openingIndex: 1, candidateWhite: true });
  assert.deepEqual(scheduleGame(100, 201), { pair: 100, openingIndex: 0, candidateWhite: false });
});

test('pentanomial gate passes a balanced sample and rejects a loss', () => {
  const split = pentanomialElo(pairGames(100, '1-0', '1-0', true));
  assert.equal(split.pairs, 100);
  assert.equal(split.wins, 100);
  assert.equal(split.losses, 100);
  assert.ok(Math.abs(split.elo) <= 1);
  assert.ok(split.ci95[1] > 0 && split.ci95[1] < 10);
  assert.equal(split.ci95[0], -split.ci95[1]);
  assert.equal(equalVisitGate(split).pass, true);

  const wins = pentanomialElo(pairGames(20, '1-0', '0-1', true));
  assert.equal(wins.losses, 0);
  assert.ok(wins.elo > 0);
  assert.equal(equalVisitGate(wins).pass, true);

  const losses = pentanomialElo(pairGames(20, '0-1', '1-0', true));
  assert.ok(losses.elo < -10);
  assert.ok(losses.ci95[1] < 0);
  assert.equal(equalVisitGate(losses).pass, false);

  const mild = pentanomialElo([...pairGames(30, '0-1', '1-0', true), ...pairGames(20, '1-0', '0-1', true)].map((game, index) => ({ ...game, index })));
  assert.ok(mild.elo < -10);
  assert.equal(equalVisitGate(mild).pass, false);

  const unpaired = pentanomialElo([{ index: 0, result: '1-0', candidateWhite: true }]);
  assert.equal(unpaired.pairs, 0);
  assert.equal(equalVisitGate(unpaired).reason, 'no complete pairs');
  assert.equal(candidatePoints('1/2-1/2', false), 0.5);
});

test('pair likelihood allows correlated outcomes and unobserved pair categories', () => {
  const split = pentanomialElo(pairGames(100, '1-0', '1-0'));
  const sweep = pentanomialElo([...pairGames(50, '1-0', '0-1'), ...pairGames(50, '0-1', '1-0')].map((game, index) => ({ ...game, index })));
  assert.equal(split.score, sweep.score);
  assert.equal(split.elo, 0);
  assert.equal(sweep.elo, 0);
  assert.ok(sweep.ci95[1] > 60, 'correlated sweeps must be wider than split pairs');
  assert.ok(split.ci95[1] > 0, 'zero observed pair variance is not proof of equality');
  const allWins = pentanomialElo(pairGames(100, '1-0', '0-1'));
  assert.ok(allWins.ci95[0] > 600);
  assert.equal(allWins.ci95[1], 2400);
  assert.deepEqual(pentanomialElo([]).ci95, [-2400, 2400]);
});

test('driver rejects unsafe chunks and incompatible checkpoints before launching a browser', () => {
  const dir = mkdtempSync(join(tmpdir(), 'batching-strength-test-'));
  const run = (...args) =>
    spawnSync(process.execPath, ['--experimental-strip-types', 'scripts/batching_strength_match.mjs', '--match', 'b1-v8', '--dir', dir, ...args], {
      encoding: 'utf8',
    });
  try {
    for (const args of [
      ['--minutes', '21'],
      ['--games', '0'],
      ['--max-plies', 'NaN'],
      ['--opening-count', '1'],
      ['--movetime-ms', '0'],
    ]) {
      const result = run(...args);
      assert.equal(result.status, 2, result.stderr);
      assert.doesNotMatch(result.stderr, /open http/);
    }
    writeFileSync(
      join(dir, 'state.json'),
      JSON.stringify({ match: 'b1-v8', build: 'incompatible', generatorVersion: 1, maxPlies: 160, movetimeMs: null, openingCount: 100 }),
    );
    const result = run();
    assert.equal(result.status, 2);
    assert.match(result.stderr, /checkpoint match\/build\/maxPlies\/movetime differs/);
    assert.doesNotMatch(result.stderr, /open http/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('queue parses in zsh and bash without starting a match', () => {
  for (const shell of ['zsh', 'bash']) {
    const result = spawnSync(shell, ['-n', 'scripts/batching_strength_queue.sh'], { encoding: 'utf8' });
    if (result.error?.code === 'ENOENT') continue;
    assert.equal(result.status, 0, `${shell}: ${result.stderr}`);
  }
});

test('final evidence never presents partial or unrun matches as gate results', () => {
  const evidence = JSON.parse(readFileSync(new URL('../docs/batching_strength_2026-10.json', import.meta.url), 'utf8'));
  assert.equal(evidence.defaultsChanged, false);
  const byId = Object.fromEntries(evidence.matches.map((match) => [match.id, match]));
  const b100 = byId['b1-v100'];
  assert.equal(b100.status, 'complete');
  assert.equal(b100.purpose, 'diagnostic-only');
  assert.equal(b100.estimate.games, 200);
  assert.equal(b100.estimate.elo, 6.9);
  assert.deepEqual(b100.estimate.ci95, [-24, 38.5]);
  for (const [id, atCut] of [
    ['lqo-v256', 88],
    ['b1-v400', 26],
  ]) {
    assert.equal(byId[id].status, 'partial');
    assert.equal(byId[id].gate, null);
    assert.equal(byId[id].earlyStop.games, atCut);
    assert.equal(byId[id].earlyStop.cutBy, 'user');
  }
  assert.equal(byId['lqo-time'].status, 'not-run');
  assert.equal(byId['lqo-time'].estimate, null);
  for (const match of evidence.matches) {
    if (match.status !== 'complete') assert.equal(match.gate, null, match.id);
  }
});

test('queen-side points follow the side that still has a queen', () => {
  assert.equal(queenSidePoints('1-0', 'b'), 1);
  assert.equal(queenSidePoints('0-1', 'b'), 0);
  assert.equal(queenSidePoints('0-1', 'w'), 1);
  assert.equal(queenSidePoints('1-0', 'w'), 0);
  assert.equal(queenSidePoints('1/2-1/2', 'w'), 0.5);
  assert.equal(queenSidePoints('1-0', null), null);
});

test('PGN replays UCI from the engine start position', () => {
  const pgn = uciLineToPgn({
    startFen: LQO_BLACK_MISSING_QUEEN_FEN,
    moves: ['e2e4', 'e7e5'],
    white: 'lqo-logical',
    black: 'lqo-serial',
    result: '1/2-1/2',
  });
  assert.match(pgn, /\[SetUp "1"\]/);
  assert.match(pgn, /\[FEN "rnb1kbnr\/pppppppp\/8\/8\/8\/8\/PPPPPPPP\/RNBQKBNR w KQkq - 0 1"\]/);
  assert.match(pgn, /1\. e4 e5 1\/2-1\/2/);
  const illegal = uciLineToPgn({ startFen: START_FEN, moves: ['e2e5'], white: 'w', black: 'b', result: '*' });
  assert.match(illegal, /\{illegal e2e5\}/);
  assert.doesNotMatch(illegal, /\[SetUp /);
});
