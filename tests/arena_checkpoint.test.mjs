import assert from 'node:assert/strict';
import { test } from 'node:test';
import { boardToFen, START_FEN } from '../src/chess/board.ts';
import { gameTreeToPgn } from '../src/chess/pgn.ts';
import {
  ARENA_CONTROL_IDS,
  arenaCompletedTree,
  arenaRemainingIndices,
  arenaRunIdentity,
  commitArenaGame,
  createArenaCheckpoint,
  decodeArenaCheckpoint,
  hydrateArenaSchedule,
  loadArenaCheckpoint,
  normalizeArenaControls,
  saveArenaCheckpoint,
} from '../src/lc0/arenaCheckpoint.ts';
import { openingFromUciMoves } from '../src/lc0/arenaOpenings.ts';
import { runArenaSchedule } from '../src/lc0/arenaScheduleRunner.ts';
import { GameTree } from '../src/lc0/gameTree.ts';
import { buildSchedule, tournamentPairings, TournamentStandings } from '../src/lc0/tournament.ts';

function fixture(mode = 'match', openings = [{ name: 'Start', fen: START_FEN }], games = 2) {
  const controls = Object.fromEntries(ARENA_CONTROL_IDS.map((id) => [id, '1']));
  Object.assign(controls, {
    tournamentModeSelect: mode,
    budgetModeSelect: 'movetime',
    movetimeInput: '10',
    gamesInput: String(games),
    startingPositionSelect: 'custom',
    openingText: 'Start',
    delayInput: '0',
    cacheEntriesInput: '2048',
    stockfishThreadsInput: '1',
  });
  const seats = ['0', '1', ...(mode === 'match' ? [] : ['2'])].map((id) => ({ id, name: 'Same engine', family: 'sf', variant: 'lite', strength: 8 }));
  const config = { mode, seats, controls, compatibility: 'bundle-hash:asset-release:runtime' };
  return createArenaCheckpoint(
    config,
    buildSchedule(
      tournamentPairings(
        mode,
        seats.map((seat) => seat.id),
      ),
      openings,
      games,
    ),
  );
}
function completed(checkpoint, scheduleIndex, result = '1/2-1/2') {
  const opening = checkpoint.schedule[scheduleIndex].opening;
  const tree = new GameTree(opening.startFen ?? opening.fen);
  for (const move of opening.moves ?? []) assert.ok(tree.addUci(move));
  return {
    scheduleIndex,
    result,
    reason: 'adjudicated',
    pgn: gameTreeToPgn(tree, { FEN: tree.root.fen, SetUp: '1' }, result),
    rootFen: tree.root.fen,
    moves: opening.moves ?? [],
    charts: [],
    diagnostics: { runtime: 'CPU', search: 'depth 8' },
  };
}
function reidentify(checkpoint) {
  checkpoint.identity = arenaRunIdentity(checkpoint.config, checkpoint.schedule);
  return checkpoint;
}

test('versioned checkpoint round-trips with immutable config and no live BoardStates', () => {
  const opening = openingFromUciMoves('Italian', ['e2e4', 'e7e5', 'g1f3']);
  const checkpoint = fixture('match', [opening]);
  const first = completed(checkpoint, 0, '1-0');
  const saved = commitArenaGame(checkpoint, first);
  const decoded = decodeArenaCheckpoint(JSON.parse(JSON.stringify(saved)));
  assert.deepEqual(decoded, saved);
  assert.equal(decoded.version, 1);
  assert.equal(decoded.schedule[0].opening.positions, undefined);
  assert.equal(checkpoint.completed.length, 0);
  assert.deepEqual(arenaRemainingIndices(decoded), [1]);
  assert.deepEqual(hydrateArenaSchedule(decoded)[0].opening.positions.map(boardToFen), opening.positions.map(boardToFen));
  assert.equal(arenaCompletedTree(first).current.fen, opening.fen);
});

test('effective empty/out-of-range controls create reloadable checkpoints', () => {
  const checkpoint = fixture();
  const raw = {
    ...checkpoint.config.controls,
    gamesInput: '',
    movetimeInput: '',
    delayInput: '99999',
    stockfishThreadsInput: '',
    cacheEntriesInput: '-1',
    lc0BatchSizeInput: '1000',
  };
  const controls = normalizeArenaControls(raw);
  assert.equal(controls.gamesInput, '2');
  assert.equal(controls.movetimeInput, '500');
  assert.equal(controls.delayInput, '3000');
  assert.equal(controls.stockfishThreadsInput, '0');
  assert.equal(controls.cacheEntriesInput, '0');
  assert.equal(controls.lc0BatchSizeInput, '64');
  assert.equal(normalizeArenaControls({ ...raw, gamesInput: '1000' }).gamesInput, '20');
  const config = { ...checkpoint.config, controls };
  const saved = createArenaCheckpoint(config, checkpoint.schedule);
  assert.deepEqual(decodeArenaCheckpoint(structuredClone(saved)), saved);
});

test('unsupported rosters and schemas are rejected before creating a saved run', () => {
  const checkpoint = fixture('round-robin');
  const seats = Array.from({ length: 9 }, (_, i) => ({ ...checkpoint.config.seats[0], id: String(i) }));
  const config = { ...checkpoint.config, seats };
  const schedule = buildSchedule(
    tournamentPairings(
      'round-robin',
      seats.map((seat) => seat.id),
    ),
    [{ name: 'Start', fen: START_FEN }],
    2,
  );
  assert.throws(() => createArenaCheckpoint(config, schedule), /roster/);
  assert.throws(
    () => createArenaCheckpoint({ ...checkpoint.config, controls: { ...checkpoint.config.controls, gamesInput: '' } }, checkpoint.schedule),
    /budget/,
  );
});

test('seat ids survive same-named engines and preserve round robin/gauntlet schedules', () => {
  for (const mode of ['round-robin', 'gauntlet']) {
    const checkpoint = fixture(mode);
    assert.deepEqual(decodeArenaCheckpoint(structuredClone(checkpoint)), checkpoint);
    const standings = new TournamentStandings(checkpoint.config.seats);
    for (const [i, game] of checkpoint.schedule.entries()) {
      const result = completed(checkpoint, i, '1-0');
      standings.record(game.whiteId, game.blackId, result.result);
    }
    assert.equal(standings.table().length, 3);
    assert.equal(standings.totalGames(), checkpoint.schedule.length);
  }
});

test('resume uses the completed-index set, including holes left by concurrent games', async () => {
  let checkpoint = fixture('round-robin', [{ name: 'Start', fen: START_FEN }], 4);
  for (const index of [8, 1, 5]) checkpoint = commitArenaGame(checkpoint, completed(checkpoint, index));
  const pending = arenaRemainingIndices(decodeArenaCheckpoint(structuredClone(checkpoint)));
  assert.deepEqual(pending, [0, 2, 3, 4, 6, 7, 9, 10, 11]);
  const played = [];
  await runArenaSchedule({
    count: pending.length,
    concurrency: 3,
    signal: new AbortController().signal,
    play: async (index) => {
      played.push(pending[index]);
      return completed(checkpoint, pending[index]);
    },
    complete: (_, game) => {
      checkpoint = commitArenaGame(checkpoint, game);
    },
  });
  assert.deepEqual(
    played.sort((a, b) => a - b),
    pending,
  );
  assert.equal(checkpoint.completed.length, 12);
  assert.deepEqual(arenaRemainingIndices(decodeArenaCheckpoint(structuredClone(checkpoint))), []);
});

test('duplicate completion is idempotent and out-of-range completion is rejected', () => {
  const checkpoint = fixture();
  const first = commitArenaGame(checkpoint, completed(checkpoint, 1));
  assert.equal(commitArenaGame(first, completed(checkpoint, 1)), first);
  assert.equal(first.revision, 1);
  assert.throws(() => commitArenaGame(first, { ...completed(checkpoint, 0), scheduleIndex: 5 }), /outside/);
});

test('Stop/cancel commits nothing; the interrupted game restarts from its opening', async () => {
  const checkpoint = fixture('match', [openingFromUciMoves('Sicilian', ['e2e4', 'c7c5'])]);
  const signal = new AbortController();
  await runArenaSchedule({
    count: 2,
    concurrency: 1,
    signal: signal.signal,
    play: async () => {
      signal.abort();
      return completed(checkpoint, 0);
    },
    complete: () => assert.fail('cancelled game must never be scored'),
  });
  assert.deepEqual(arenaRemainingIndices(checkpoint), [0, 1]);
  assert.deepEqual(hydrateArenaSchedule(checkpoint)[0].opening.moves, ['e2e4', 'c7c5']);
});

test('schema rejects unknown versions, bad timestamps, and config tampering', () => {
  for (const mutate of [
    (c) => {
      c.version = 999;
    },
    (c) => {
      c.kind = 'other';
    },
    (c) => {
      c.updatedAt = -1;
    },
    (c) => {
      c.config.controls.movetimeInput = '100';
    },
    (c) => {
      c.schedule[0].whiteId = 'unknown';
    },
    (c) => {
      c.config.seats[1].id = '0';
    },
    (c) => {
      c.completed = null;
    },
  ]) {
    const c = fixture();
    mutate(c);
    assert.throws(() => decodeArenaCheckpoint(c));
  }
});

test('schema independently verifies schedule expansion, suite slots and colour alternation', () => {
  for (const mutate of [
    (c) => {
      c.schedule.pop();
    },
    (c) => {
      c.schedule[1].whiteId = '0';
      c.schedule[1].blackId = '1';
    },
    (c) => {
      c.schedule[0].openingIndex = 3;
    },
    (c) => {
      c.config.controls.gamesInput = '20000';
    },
    (c) => {
      c.schedule[1].opening.name = 'Changed slot';
    },
  ]) {
    const c = structuredClone(fixture());
    mutate(c);
    assert.throws(() => decodeArenaCheckpoint(reidentify(c)));
  }
});

test('corrupt or mismatched opening history is rejected even with a new fingerprint', () => {
  for (const moves of [['e2e5'], ['e2e4']]) {
    const c = fixture();
    for (const game of c.schedule) Object.assign(game.opening, { startFen: START_FEN, moves });
    assert.throws(() => decodeArenaCheckpoint(reidentify(c)));
  }
});

test('duplicate completed indices, invalid results and unfinished PGNs cannot inflate standings', () => {
  const checkpoint = fixture();
  const saved = commitArenaGame(checkpoint, completed(checkpoint, 0));
  for (const mutate of [
    (c) => {
      c.completed.push(c.completed[0]);
      c.revision++;
    },
    (c) => {
      c.completed[0].scheduleIndex = 200;
    },
    (c) => {
      c.completed[0].result = '*';
    },
    (c) => {
      c.completed[0].pgn = '[Result "*"]\n\n*';
    },
    (c) => {
      c.completed[0].moves = ['e2e5'];
    },
    (c) => {
      c.revision = 100;
    },
  ]) {
    const c = structuredClone(saved);
    mutate(c);
    assert.throws(() => decodeArenaCheckpoint(c));
  }
});

test('charts retain per-game engine identities and reject non-finite/corrupt values', () => {
  const checkpoint = fixture('match', [openingFromUciMoves('e4', ['e2e4'])]);
  const game = completed(checkpoint, 0);
  game.charts = [{ engineId: 'arena:0:w:0', engineName: 'Same engine', ply: 0, whiteScore: 0.6, moveMs: 12, nps: 100 }];
  const c = commitArenaGame(checkpoint, game);
  assert.deepEqual(decodeArenaCheckpoint(structuredClone(c)).completed[0].charts, game.charts);
  for (const [key, value] of [
    ['whiteScore', NaN],
    ['whiteScore', 2],
    ['moveMs', Infinity],
    ['nps', -2],
    ['ply', 10],
  ]) {
    const bad = structuredClone(c);
    bad.completed[0].charts[0][key] = value;
    assert.throws(() => decodeArenaCheckpoint(bad));
  }
});

test('schedule lanes await durable completion before starting another game', async () => {
  const events = [];
  await runArenaSchedule({
    count: 2,
    concurrency: 1,
    signal: new AbortController().signal,
    play: async (index) => {
      events.push(`play ${index}`);
      return index;
    },
    complete: async (index) => {
      await new Promise((r) => setTimeout(r, 5));
      events.push(`saved ${index}`);
    },
  });
  assert.deepEqual(events, ['play 0', 'saved 0', 'play 1', 'saved 1']);
});

test('completed neural root chart is plain data and rejects mismatched positions', () => {
  const checkpoint = fixture('match', [openingFromUciMoves('e4', ['e2e4'])]);
  const game = completed(checkpoint, 0);
  game.rootChart = { engineName: 'Lc0', fen: START_FEN, children: [{ uci: 'e2e4', visits: 10, q: 0.1, prior: 0.3 }] };
  const saved = commitArenaGame(checkpoint, game);
  assert.deepEqual(decodeArenaCheckpoint(structuredClone(saved)).completed[0].rootChart, game.rootChart);
  for (const mutate of [
    (c) => {
      c.completed[0].rootChart.children[0].visits = Infinity;
    },
    (c) => {
      c.completed[0].rootChart.fen = openingFromUciMoves('d4', ['d2d4']).fen;
    },
  ]) {
    const c = structuredClone(saved);
    mutate(c);
    assert.throws(() => decodeArenaCheckpoint(c));
  }
});

test('missing IndexedDB rejects explicitly rather than claiming a save', async () => {
  assert.equal(typeof indexedDB, 'undefined');
  await assert.rejects(loadArenaCheckpoint(), /unavailable/);
  await assert.rejects(saveArenaCheckpoint(fixture(), null), /unavailable/);
});
