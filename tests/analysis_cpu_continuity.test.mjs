import assert from 'node:assert/strict';
import test from 'node:test';
import { AnalysisCpuSessions, analysisGamePosition } from '../src/lc0/analysisContinuity.ts';
import { uciPositionCommand } from '../src/lc0/browserUciEngine.ts';
import { GameTree } from '../src/lc0/gameTree.ts';
import { StockfishEngine } from '../src/lc0/stockfishEngine.ts';

test('Analysis CPU sessions reset only on changed game, adapter or invalidation', () => {
  const sessions = new AnalysisCpuSessions();
  const engine = {};
  const game = new GameTree();
  assert.equal(sessions.needsReset(engine, game), true);
  sessions.prepared(engine, game);
  game.addUci('e2e4');
  assert.equal(sessions.needsReset(engine, game), false);
  game.back();
  assert.equal(sessions.needsReset(engine, game), false);
  assert.equal(sessions.needsReset({}, game), true);
  assert.equal(sessions.needsReset(engine, new GameTree()), true);
  sessions.invalidate(engine);
  assert.equal(sessions.needsReset(engine, game), true);
});

test('Stockfish analysis API sends root FEN plus moves, without an implicit new game', async () => {
  const previous = globalThis.Worker;
  const commands = [];
  globalThis.Worker = class {
    postMessage(command) {
      commands.push(command);
      if (command === 'uci') queueMicrotask(() => this.onmessage({ data: 'uciok' }));
      if (command === 'isready') queueMicrotask(() => this.onmessage({ data: 'readyok' }));
      if (command.startsWith('go '))
        queueMicrotask(() => {
          this.onmessage({ data: 'info depth 12 score cp 20 pv e7e5' });
          this.onmessage({ data: 'bestmove e7e5' });
        });
    }
    terminate() {}
  };
  const engine = new StockfishEngine({}, '/stockfish/stockfish-19-lite-single.js');
  const game = new GameTree();
  try {
    const lines = await engine.analyze(game.root.fen, { moves: ['e2e4'], depth: 12 });
    assert.ok(commands.includes(`position fen ${game.root.fen} moves e2e4`));
    assert.equal(commands.includes('ucinewgame'), false);
    assert.equal(lines[0].pvUci[0], 'e7e5');
  } finally {
    engine.dispose();
    globalThis.Worker = previous;
  }
});

test('Analysis CPU thread configuration does not resend unchanged options', () => {
  const sessions = new AnalysisCpuSessions();
  const sent = [];
  const engine = { setOptions: (options) => sent.push(options) };
  sessions.setThreads(engine, 2);
  sessions.setThreads(engine, 2);
  sessions.setThreads(engine, 1);
  assert.deepEqual(sent, [{ threads: 2 }, { threads: 1 }]);
});

test('Analysis position history follows variations and explicit review nodes, including repetition', () => {
  const game = new GameTree();
  for (const move of ['g1f3', 'g8f6', 'f3g1', 'f6g8']) game.addUci(move);
  const repeated = game.current;
  assert.deepEqual(analysisGamePosition(game).moves, ['g1f3', 'g8f6', 'f3g1', 'f6g8']);
  game.toStart();
  game.addUci('e2e4');
  assert.deepEqual(analysisGamePosition(game).moves, ['e2e4']);
  assert.deepEqual(analysisGamePosition(game, repeated).moves, ['g1f3', 'g8f6', 'f3g1', 'f6g8']);
  assert.equal(uciPositionCommand(game.root.fen, analysisGamePosition(game, repeated).moves), `position fen ${game.root.fen} moves g1f3 g8f6 f3g1 f6g8`);
  assert.equal(uciPositionCommand(game.root.fen), `position fen ${game.root.fen}`);
  assert.throws(() => uciPositionCommand(game.root.fen, ['e2e4\ngo infinite']), /Invalid UCI/);
});
