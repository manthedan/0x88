import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import { boardToFen } from '../src/chess/board.ts';
import { moveToUci } from '../src/chess/moveCodec.ts';
import { legalMoves, makeMove } from '../src/chess/movegen.ts';
import { openingFromUciMoves } from '../src/lc0/arenaOpenings.ts';
import { simulEquivalentGames, simulHelps4 } from '../src/lc0/arenaSimulGate.ts';
import { gameOutcome } from '../src/lc0/engineBattle.ts';

const hash = (value) => createHash('sha256').update(value).digest('hex');
export function spread(values) {
  const sorted = [...values].sort((a, b) => a - b);
  return { median: sorted[Math.floor(sorted.length / 2)], min: sorted[0], max: sorted.at(-1) };
}

export function analyzeSimulRun(run) {
  assert.equal(run.protocol, 'arena-simul-1');
  assert.equal(run.pairs, 3);
  assert.equal(new Set(run.openings.map((opening) => opening.fen)).size, 4);
  assert.equal(run.rows.length, 7);
  const control = run.rows.at(-1);
  assert.equal(control.control, true);
  const rows = run.rows.map((row, index) => {
    assert.equal(row.index, index);
    assert.equal(row.concurrency, index % 2 ? run.n : 1);
    const expectedPhysical = index % 2 || index === 6 ? (run.n <= 4 ? 4 : 8) : 1;
    assert.equal(row.physical, expectedPhysical, 'expected fixed physical batch export');
    assert.equal(row.modelUrl, `/models/lc0/t1-256x10-distilled-swa-2432500.batch${expectedPhysical}.f16.onnx`, 'expected model identity');
    if (index % 2) assert.equal(row.modelUrl, control.modelUrl, 'simul and control share the exact artifact');
    assert.equal(row.completed, 8);
    assert.equal(row.games.length, 8);
    assert.equal(row.physicalRuns, row.broker.batches);
    assert.equal(
      row.broker.evaluations,
      Object.entries(row.broker.batchSizes).reduce((sum, [size, count]) => sum + Number(size) * count, 0),
    );
    assert.ok(Math.abs(row.gamesPerHour - (8 * 3600000) / row.elapsedMs) < 1e-6);
    assert.ok(Math.abs(row.evalsPerSecond - (row.broker.evaluations * 1000) / row.elapsedMs) < 1e-6);
    assert.equal(new Set(row.games.map((game) => game.id)).size, 8);
    for (const game of row.games) {
      const opening = run.openings[Math.floor(game.id / 2)];
      assert.equal(game.opening, opening.name);
      assert.deepEqual(game.bookMoves, opening.moves);
      assert.equal(game.aIsWhite, game.id % 2 === 0);
      assert.equal(game.seed, 1);
      assert.equal(game.moves.length, game.visits.length);
      assert.equal(game.moves.length, game.moveMs.length);
      assert.ok(game.visits.every((visits) => visits === run.visits));
      const history = openingFromUciMoves(opening.name, opening.moves).positions;
      for (const uci of game.moves) {
        const board = history.at(-1);
        assert.equal(gameOutcome(board, history.slice(0, -1).map(boardToFen)), null, 'no moves after a rule ending');
        const move = legalMoves(board).find((candidate) => moveToUci(candidate) === uci);
        assert.ok(move, `illegal move ${uci}`);
        history.push(makeMove(board, move));
      }
      assert.equal(boardToFen(history.at(-1)), game.finalFen);
      const outcome = gameOutcome(history.at(-1), history.slice(0, -1).map(boardToFen));
      if (!outcome) assert.equal(game.moves.length, run.maxPlies);
      assert.deepEqual({ result: game.result, reason: game.reason }, outcome ?? { result: '1/2-1/2', reason: 'max plies' });
    }
    const baseline = run.rows[0].games;
    const matching = (reference) =>
      row.games.filter((game) => {
        const other = reference.find((candidate) => candidate.id === game.id);
        return JSON.stringify(game.moves) === JSON.stringify(other.moves);
      }).length;
    return {
      index,
      concurrency: row.concurrency,
      control: row.control,
      physical: row.physical,
      completed: row.completed,
      elapsedMs: row.elapsedMs,
      gamesPerHour: row.gamesPerHour,
      evalsPerSecond: row.evalsPerSecond,
      visitsPerSecond: row.visitsPerSecond,
      evaluations: row.broker.evaluations,
      searchedPlies: row.games.reduce((sum, game) => sum + game.moves.length, 0),
      occupancy: row.occupancy,
      p95MoveMs: row.p95MoveMs,
      batchSizes: row.broker.batchSizes,
      maxResident: row.broker.maxResident,
      maxObservedQueueMs: row.broker.maxWaitMs,
      memory: row.memory ?? null,
      bufferAllocationBytes: row.bufferAllocationBytes ?? null,
      exportDrift: row.exportDrift,
      rowDrift: row.rowDrift,
      matchingBatch1Games: matching(baseline),
      matchingSameExportGames: matching(control.games),
      sameExportEquivalent: simulEquivalentGames(row.games, control.games),
      games: row.games.map((game) => ({
        id: game.id,
        plies: game.moves.length,
        result: game.result,
        reason: game.reason,
        movesSha256: hash(JSON.stringify(game.moves)),
      })),
    };
  });
  const summarize = (concurrency) => {
    const selected = rows.filter((row) => !row.control && row.concurrency === concurrency);
    return Object.fromEntries(
      ['gamesPerHour', 'evalsPerSecond', 'visitsPerSecond', 'occupancy', 'p95MoveMs'].map((metric) => [metric, spread(selected.map((row) => row[metric]))]),
    );
  };
  return {
    n: run.n,
    visits: run.visits,
    maxPlies: run.maxPlies,
    pairs: run.pairs,
    measuredAt: run.measuredAt,
    userAgent: run.userAgent,
    runtime: run.rows[0].runtime,
    options: run.options,
    openings: run.openings,
    rows,
    serial: summarize(1),
    simul: summarize(run.n),
    pairedGamesPerHourRatios: [0, 2, 4].map((i) => rows[i + 1].gamesPerHour / rows[i].gamesPerHour),
    pairedEvalsPerSecondRatios: [0, 2, 4].map((i) => rows[i + 1].evalsPerSecond / rows[i].evalsPerSecond),
    allowsN8: simulHelps4([run], run.visits, run.maxPlies),
  };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const [output, ...inputs] = process.argv.slice(2);
  assert.ok(output && inputs.length, 'usage: node --experimental-strip-types scripts/analyze_arena_simul.mjs OUTPUT INPUT...');
  const runs = inputs.flatMap((path) => {
    const raw = readFileSync(path);
    const parsed = JSON.parse(raw);
    const exported = Array.isArray(parsed) ? parsed : [parsed];
    return exported.map((run, rawRunIndex) => ({ rawPath: path, rawSha256: hash(raw), rawRunIndex, ...analyzeSimulRun(run) }));
  });
  writeFileSync(output, `${JSON.stringify({ schemaVersion: 1, runs }, null, 2)}\n`);
  for (const run of runs)
    console.log(
      JSON.stringify({
        n: run.n,
        serial: run.serial,
        simul: run.simul,
        pairedRatios: run.pairedGamesPerHourRatios,
        sameExport: run.rows.filter((row) => row.concurrency > 1).map((row) => row.matchingSameExportGames),
        allowsN8: run.allowsN8,
      }),
    );
}
