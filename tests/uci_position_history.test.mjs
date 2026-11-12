import assert from 'node:assert/strict';
import { test } from 'node:test';
import { boardToFen, parseFen, START_FEN } from '../src/chess/board.ts';
import { moveToUci } from '../src/chess/moveCodec.ts';
import { legalMoves, makeMove } from '../src/chess/movegen.ts';
import { playCpuArenaGame } from '../src/lc0/arenaCpuGame.ts';
import { BerserkEngine } from '../src/lc0/berserkEngine.ts';
import { GameTree } from '../src/lc0/gameTree.ts';
import { PlentyChessEngine } from '../src/lc0/plentychessEngine.ts';
import { RecklessEngine } from '../src/lc0/recklessEngine.ts';
import { EngineResourceBroker } from '../src/lc0/resourceBroker.ts';
import { StockfishEngine } from '../src/lc0/stockfishEngine.ts';
import { StormphraxEngine } from '../src/lc0/stormphraxEngine.ts';
import { fenAfterUciMoves, uciGameHistory, uciHistoryFromLastIrreversibleMove } from '../src/lc0/uciGameHistory.ts';
import { ViridithasEngine } from '../src/lc0/viridithasEngine.ts';

function play(fen, moves) {
  const positions = [parseFen(fen)];
  for (const uci of moves) {
    const move = legalMoves(positions.at(-1)).find((candidate) => moveToUci(candidate) === uci);
    assert.ok(move, `${uci} is legal`);
    positions.push(makeMove(positions.at(-1), move));
  }
  return positions;
}

// Kingside castling both sides, a double push answered by en passant, and quiet moves.
const START_MOVES = ['e2e4', 'g8f6', 'e4e5', 'd7d5', 'e5d6', 'e7e6', 'g1f3', 'f8d6', 'f1e2', 'e8g8', 'e1g1'];
const CUSTOM_FEN = 'r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 4 8';
const CUSTOM_MOVES = ['e1c1', 'e8c8', 'h2h3'];
const PROMOTION_FEN = '1n5k/P7/8/8/8/8/6K1/8 w - - 0 1';
const PROMOTION_MOVES = ['a7b8n', 'h8g7', 'b8c6', 'g7f6', 'g2f2'];

test('uciGameHistory recovers the start FEN and UCI moves from a position history', () => {
  for (const [fen, moves] of [
    [START_FEN, []],
    [START_FEN, START_MOVES],
    [CUSTOM_FEN, CUSTOM_MOVES],
    [PROMOTION_FEN, PROMOTION_MOVES],
    ['4k3/8/8/8/8/8/4P3/4K3 w - - 0 1', ['e2e4', 'e8d7', 'e4e5', 'd7c6', 'e5e6', 'c6c5', 'e6e7', 'c5d6', 'e7e8r']],
  ]) {
    assert.deepEqual(uciGameHistory(play(fen, moves)), { fen, moves });
  }
});

test('uciGameHistory starts after a gap in the history instead of inventing moves', () => {
  const first = play(START_FEN, ['e2e4', 'e7e5']);
  const second = play(CUSTOM_FEN, ['h2h3', 'h7h6']);
  assert.deepEqual(uciGameHistory([...first, ...second]), { fen: CUSTOM_FEN, moves: ['h2h3', 'h7h6'] });
  // A position repeated without a move between them is a gap, not a null move.
  const repeated = play(START_FEN, ['g1f3']);
  assert.deepEqual(uciGameHistory([...repeated, repeated.at(-1)]), { fen: boardToFen(repeated.at(-1)), moves: [] });
  assert.throws(() => uciGameHistory([]), /at least one position/);
});

test('fenAfterUciMoves replays legal moves and rejects illegal ones', () => {
  assert.equal(fenAfterUciMoves(START_FEN), START_FEN);
  assert.equal(fenAfterUciMoves(CUSTOM_FEN, CUSTOM_MOVES), boardToFen(play(CUSTOM_FEN, CUSTOM_MOVES).at(-1)));
  assert.throws(() => fenAfterUciMoves(START_FEN, ['e2e5']), /Illegal move e2e5/);
});

const flush = async () => {
  for (let i = 0; i < 10; i += 1) await new Promise((resolve) => setImmediate(resolve));
};

/** String-protocol worker double for Stockfish. */
class StockfishWorker {
  static instances = [];
  messages = [];
  constructor() {
    StockfishWorker.instances.push(this);
  }
  postMessage(message) {
    this.messages.push(message);
    const reply = message === 'uci' ? 'uciok' : message === 'isready' ? 'readyok' : message.startsWith('go ') ? 'bestmove e2e4' : null;
    if (reply) queueMicrotask(() => this.onmessage?.({ data: reply }));
  }
  terminate() {}
}

/** Structured-message worker double for the Emscripten adapters. */
class EmscriptenWorker {
  static instances = [];
  messages = [];
  constructor() {
    EmscriptenWorker.instances.push(this);
  }
  postMessage(message) {
    if (message.command) this.messages.push(message.command);
    queueMicrotask(() => {
      const command = message.command;
      const line = command === 'uci' ? 'uciok' : command === 'isready' ? 'readyok' : command?.startsWith('go ') ? 'bestmove e2e4' : null;
      if (line) this.onmessage?.({ data: { type: 'line', stream: 'stdout', line } });
      this.onmessage?.({ data: { type: message.type === 'init' ? 'ready' : 'commandDone', id: message.id } });
    });
  }
  terminate() {}
}

/** One-shot WASI worker double: each `run` carries the full UCI script. */
class WasiWorker {
  static instances = [];
  messages = [];
  constructor() {
    WasiWorker.instances.push(this);
  }
  postMessage(message) {
    if (message.type !== 'run') return;
    this.messages.push(...message.commands);
    queueMicrotask(() => this.onmessage?.({ data: { type: 'result', id: message.id, stdout: ['bestmove e2e4'], stderr: [], exitCode: 0 } }));
  }
  terminate() {}
}

function install(t, Worker) {
  const previous = { Worker: globalThis.Worker, location: globalThis.location, crossOriginIsolated: globalThis.crossOriginIsolated };
  globalThis.Worker = Worker;
  globalThis.location = { href: 'http://localhost:5181/arena/', origin: 'http://localhost:5181' };
  globalThis.crossOriginIsolated = false;
  Worker.instances = [];
  t.after(() => {
    for (const [key, value] of Object.entries(previous)) {
      if (value === undefined) delete globalThis[key];
      else globalThis[key] = value;
    }
  });
}

const ADAPTERS = [
  ['Stockfish', StockfishWorker, () => new StockfishEngine({ depth: 2 }, '/stockfish/stockfish-19-lite-single.js')],
  ['Berserk', EmscriptenWorker, () => new BerserkEngine({ depth: 2 })],
  ['PlentyChess', EmscriptenWorker, () => new PlentyChessEngine({ depth: 2 })],
  ['Stormphrax', EmscriptenWorker, () => new StormphraxEngine({ depth: 2 })],
  ['Reckless', WasiWorker, () => new RecklessEngine({ depth: 2 }, '/reckless/test.wasm', { forceOneShot: true })],
  ['Viridithas', WasiWorker, () => new ViridithasEngine({ depth: 2 }, '/viridithas/test.wasm', { forceOneShot: true })],
];

for (const [label, Worker, create] of ADAPTERS) {
  test(`${label} bestMove sends the game history as position fen … moves …`, async (t) => {
    install(t, Worker);
    const engine = create();
    t.after(() => engine.dispose());
    assert.equal(await engine.bestMove(START_FEN), 'e2e4');
    assert.equal(await engine.bestMove(START_FEN, undefined, { moves: START_MOVES }), 'e2e4');
    assert.equal(await engine.search(CUSTOM_FEN, undefined, { moves: CUSTOM_MOVES }), 'e2e4');
    await flush();
    const positions = Worker.instances.flatMap((worker) => worker.messages).filter((command) => command.startsWith('position '));
    assert.deepEqual(positions, [
      `position fen ${START_FEN}`,
      `position fen ${START_FEN} moves ${START_MOVES.join(' ')}`,
      `position fen ${CUSTOM_FEN} moves ${CUSTOM_MOVES.join(' ')}`,
    ]);
    await assert.rejects(engine.bestMove(START_FEN, undefined, { moves: ['e2e4;quit'] }), /Invalid UCI move/);
  });
}

test('CPU arena games give each engine the start FEN and the moves played since', async () => {
  const start = '7k/8/6K1/8/8/8/8/5Q2 w - - 0 1';
  const broker = new EngineResourceBroker();
  const calls = [];
  const seat = (id, script) => {
    broker.register(id, { resourceClass: 'cpu', maxThreads: 1 });
    return {
      id,
      name: id,
      engine: {
        prewarm: async () => {},
        newGame: async () => {},
        bestMove: async (fen, _signal, opts) => {
          calls.push([fen, opts?.moves ?? null]);
          return script.shift();
        },
      },
    };
  };
  const white = seat('w', ['f1f2', 'f2f7', 'f7g7']);
  const black = seat('b', ['h8g8', 'g8h8']);
  const game = await playCpuArenaGame({ tree: new GameTree(start), white, black, broker, signal: new AbortController().signal });
  assert.equal(game.result, '1-0');
  assert.deepEqual(calls, [
    [start, []],
    [start, ['f1f2']],
    [start, ['f1f2', 'h8g8']],
    [start, ['f1f2', 'h8g8', 'f2f7']],
    [start, ['f1f2', 'h8g8', 'f2f7', 'g8h8']],
  ]);
});

test('uciHistoryFromLastIrreversibleMove re-roots at the last capture or pawn move', () => {
  const moves = ['g1f3', 'g8f6', 'e2e4', 'f6e4', 'f3g1', 'e4f6', 'g1f3'];
  const rooted = uciHistoryFromLastIrreversibleMove(START_FEN, moves);
  assert.deepEqual(rooted.moves, ['f3g1', 'e4f6', 'g1f3']);
  assert.equal(fenAfterUciMoves(rooted.fen, rooted.moves), fenAfterUciMoves(START_FEN, moves));
  assert.deepEqual(uciHistoryFromLastIrreversibleMove(START_FEN, ['g1f3', 'g8f6']), { fen: START_FEN, moves: ['g1f3', 'g8f6'] });
});

test('Berserk re-roots a position command longer than its 8191-byte input buffer instead of truncating it', async (t) => {
  install(t, EmscriptenWorker);
  const shuffle = Array.from({ length: 425 }, () => ['g1f3', 'g8f6', 'f3g1', 'f6g8']).flat();
  const tail = ['e2e4', 'g8f6', 'g1f3', 'f6g8', 'f3g1'];
  const moves = [...shuffle, ...tail];
  assert.ok(`position fen ${START_FEN} moves ${moves.join(' ')}`.length > 8191);
  const berserk = new BerserkEngine({ depth: 2 });
  const stormphrax = new StormphraxEngine({ depth: 2 });
  t.after(() => {
    berserk.dispose();
    stormphrax.dispose();
  });
  const afterPawn = fenAfterUciMoves(START_FEN, [...shuffle, 'e2e4']);
  assert.equal(await berserk.bestMove(START_FEN, undefined, { moves }), 'e2e4');
  assert.equal(await stormphrax.bestMove(START_FEN, undefined, { moves }), 'e2e4');
  await assert.rejects(berserk.bestMove(START_FEN, undefined, { moves: shuffle }), /Berserk UCI position command is \d+ bytes; the engine reads at most 8191/);
  await flush();
  const [berserkPositions, stormphraxPositions] = EmscriptenWorker.instances.map((worker) =>
    worker.messages.filter((command) => command.startsWith('position ')),
  );
  assert.deepEqual(berserkPositions, [`position fen ${afterPawn} moves ${tail.slice(1).join(' ')}`]);
  assert.deepEqual(stormphraxPositions, [`position fen ${START_FEN} moves ${moves.join(' ')}`], 'engines without a buffer limit get the whole game');
});
