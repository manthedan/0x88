#!/usr/bin/env node
// Read-only game audit; no engines or timing benchmarks are run.
// node --experimental-strip-types scripts/analyze_batching_strength.mjs [artifact-root] [output.json]
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { boardToFen, parseFen } from '../src/chess/board.ts';
import { moveToUci } from '../src/chess/moveCodec.ts';
import { legalMoves, makeMove } from '../src/chess/movegen.ts';
import { BATCHING_STRENGTH_MATCHES } from '../src/lc0/batchingStrengthMatch.ts';
import { lqoQueenOddsOpenings, SMALL_NET_EPD_PATH, smallNetOpenings } from '../src/lc0/batchingStrengthOpenings.ts';
import { candidatePoints, distinctOpeningElo, equalVisitGate, pentanomialElo, queenSidePoints } from '../src/lc0/batchingStrengthScore.ts';
import { gameOutcome } from '../src/lc0/engineBattle.ts';

const root = process.argv[2] ?? 'artifacts/batching-strength';
const output = process.argv[3];
const round = (n) => Math.round(n * 100) / 100;
const quantile = (xs, p) => (xs.length ? xs[Math.min(xs.length - 1, Math.floor((xs.length - 1) * p))] : null);
const median = (xs) => (xs.length ? (xs[Math.floor((xs.length - 1) / 2)] + xs[Math.floor(xs.length / 2)]) / 2 : null);
const hash = (file) => createHash('sha256').update(readFileSync(file)).digest('hex');
const evidence = { schema: 3, defaultsChanged: false, matches: [] };
const comparison = join(root, 'model-equivalence.json');
if (existsSync(comparison)) evidence.modelComparison = JSON.parse(readFileSync(comparison, 'utf8'));
for (const spec of Object.values(BATCHING_STRENGTH_MATCHES)) {
  const dir = join(root, spec.id);
  const file = join(dir, 'games.jsonl');
  if (!existsSync(file)) continue;
  const games = readFileSync(file, 'utf8').trim().split('\n').filter(Boolean).map(JSON.parse);
  const estimate = pentanomialElo(games);
  const openingUnitEstimate = distinctOpeningElo(games);
  const repeatedOpeningPairs = estimate.pairs - openingUnitEstimate.pairs;
  const state = JSON.parse(readFileSync(join(dir, 'state.json'), 'utf8'));
  assert.equal(state.generatorVersion, 1, 'audit only supports opening generator 1');
  const openings =
    spec.kind === 'lqo'
      ? lqoQueenOddsOpenings(state.openingCount)
      : smallNetOpenings(readFileSync(join('public', SMALL_NET_EPD_PATH), 'utf8'), state.openingCount);
  const reasons = {};
  const arms = {};
  const colors = { white: { wins: 0, draws: 0, losses: 0 }, black: { wins: 0, draws: 0, losses: 0 } };
  let identicalMovePairs = 0;
  let identicalResultPairs = 0;
  const divergences = [];
  let queenPoints = 0;
  let queenGames = 0;
  for (let i = 0; i < games.length; i++) {
    const game = games[i];
    assert.equal(game.index, i, `${spec.id}: contiguous indexes`);
    assert.equal(game.candidateWhite, i % 2 === 0, `${spec.id}: color schedule`);
    assert.equal(game.pair, Math.floor(i / 2));
    assert.equal(game.whiteName, game.candidateWhite ? spec.candidate.name : spec.baseline.name);
    assert.equal(game.blackName, game.candidateWhite ? spec.baseline.name : spec.candidate.name);
    assert.equal(game.moves.length, game.plies);
    assert.equal(game.whiteVisits.length + game.blackVisits.length, game.plies);
    // Reconstruct the exact opening/history, then independently replay every move.
    const opening = openings[Math.floor(i / 2) % openings.length];
    assert.equal(game.startFen, opening.fen);
    assert.equal(game.opening, opening.name);
    assert.deepEqual(game.bookMoves, opening.moves ?? []);
    let board = parseFen(game.startFen);
    const priorFens = opening.positions.slice(0, -1).map(boardToFen);
    for (const uci of game.moves) {
      assert.equal(gameOutcome(board, priorFens), null, `${spec.id} game ${i}: played past terminal`);
      const move = legalMoves(board).find((m) => moveToUci(m) === uci);
      assert.ok(move, `${spec.id} game ${i}: illegal ${uci}`);
      priorFens.push(boardToFen(board));
      board = makeMove(board, move);
    }
    const terminal = gameOutcome(board, priorFens);
    if (game.reason === 'max plies') {
      assert.equal(terminal, null);
      assert.equal(game.plies, state.maxPlies);
      assert.equal(game.result, '1/2-1/2');
    } else {
      assert.equal(terminal?.reason, game.reason);
      assert.equal(terminal?.result, game.result);
    }
    assert.ok(!game.reason.startsWith('illegal') && game.reason !== 'resigned', `${spec.id}: no forfeits`);
    reasons[game.reason] = (reasons[game.reason] ?? 0) + 1;
    const color = colors[game.candidateWhite ? 'white' : 'black'];
    const point = candidatePoints(game.result, game.candidateWhite);
    color[point === 1 ? 'wins' : point === 0 ? 'losses' : 'draws']++;
    const queenPoint = queenSidePoints(game.result, game.queenLessColor);
    if (queenPoint !== null) {
      queenGames++;
      queenPoints += queenPoint;
    }
    for (const side of ['white', 'black']) {
      const name = game[`${side}Name`];
      arms[name] ??= { ms: [], visits: [] };
      const arm = arms[name];
      arm.ms.push(...game[`${side}MoveMs`]);
      arm.visits.push(...game[`${side}Visits`]);
      assert.equal(game[`${side}MoveMs`].length, game[`${side}Visits`].length);
      if (!state.movetimeMs)
        assert.ok(
          game[`${side}Visits`].every((v) => v === spec.visits),
          `${spec.id}: equal visit counts`,
        );
    }
    if (i % 2 === 1) {
      const prior = games[i - 1];
      assert.equal(game.startFen, prior.startFen);
      assert.deepEqual(game.bookMoves, prior.bookMoves);
      assert.equal(game.opening, prior.opening);
      if (game.result === prior.result) identicalResultPairs++;
      if (JSON.stringify(game.moves) === JSON.stringify(prior.moves)) identicalMovePairs++;
      else {
        let ply = 0;
        while (ply < Math.min(game.moves.length, prior.moves.length) && game.moves[ply] === prior.moves[ply]) ply++;
        divergences.push(ply);
      }
    }
  }
  for (const [name, arm] of Object.entries(arms)) {
    arm.ms.sort((a, b) => a - b);
    arm.visits.sort((a, b) => a - b);
    arms[name] = {
      moves: arm.ms.length,
      medianMs: round(median(arm.ms)),
      p10Ms: round(quantile(arm.ms, 0.1)),
      p90Ms: round(quantile(arm.ms, 0.9)),
      medianVisits: median(arm.visits),
      p10Visits: quantile(arm.visits, 0.1),
      p90Visits: quantile(arm.visits, 0.9),
      minVisits: arm.visits[0],
      maxVisits: arm.visits.at(-1),
    };
  }
  divergences.sort((a, b) => a - b);
  const summary = JSON.parse(readFileSync(join(dir, 'summary.json'), 'utf8'));
  const earlyStop = existsSync(join(dir, 'early-stop.json')) ? JSON.parse(readFileSync(join(dir, 'early-stop.json'), 'utf8')) : null;
  // A user cut (including the chunk's trailing completed games) is descriptive
  // evidence only, never a gate result. Zero games is not an Elo measurement.
  const status = games.length === 0 ? 'not-run' : earlyStop || games.length < state.gamesTarget ? 'partial' : 'complete';
  evidence.matches.push({
    id: spec.id,
    status,
    purpose: spec.id.startsWith('b1-') ? 'diagnostic-only' : 'product-comparison',
    targetGames: state.gamesTarget,
    kind: state.movetimeMs ? 'equal-time' : 'equal-visit',
    visits: spec.visits,
    movetimeMs: state.movetimeMs ?? null,
    build: state.build,
    requestedOpeningCount: state.openingCount,
    actualOpeningCount: openings.length,
    observedOpeningUnits: openingUnitEstimate.pairs,
    repeatedOpeningPairs,
    recordedGames: games.length,
    estimate: games.length ? estimate : null,
    // The all-pair CI assumes independent opening units; don't treat a
    // silently cycling historical suite as a valid gate sample.
    gate: status === 'complete' && repeatedOpeningPairs === 0 ? equalVisitGate(estimate) : null,
    openingUnitEstimate: repeatedOpeningPairs ? openingUnitEstimate : null,
    colors,
    reasons,
    arms,
    identicalMovePairs,
    identicalResultPairs,
    medianFirstDivergencePly: median(divergences),
    queenSideScore: queenGames ? queenPoints / queenGames : null,
    earlyStop,
    backends: summary.backends,
    probes: summary.probes,
    originalSummary: games.length ? { elo: summary.elo, ci95: summary.ci95, method: summary.method } : null,
    files: { games: file, pgn: join(dir, 'games.pgn'), gamesSha256: hash(file), pgnSha256: hash(join(dir, 'games.pgn')) },
  });
}
const json = `${JSON.stringify(evidence, null, 2)}\n`;
if (output) writeFileSync(output, json);
else process.stdout.write(json);
