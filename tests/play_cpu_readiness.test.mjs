import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import test from 'node:test';
import { boardToFen, parseFen, START_FEN } from '../src/chess/board.ts';
import { moveToUci } from '../src/chess/moveCodec.ts';
import { legalMoves } from '../src/chess/movegen.ts';

const source = readFileSync(new URL('../src/lc0/playBrowser.ts', import.meta.url), 'utf8');
const from = source.indexOf('async function ctxVerifySelectedEngineReady(');
const to = source.indexOf('// ---------------------------------------------------------------------------\n// Game flow', from);
for (const move of [null, 'a1a8', 'e2e4']) {
  test(`CPU readiness requires a legal probe reply: ${move}`, async () => {
    const option = { id: 'sf-lite', family: 'sf', label: 'Stockfish' };
    const notes = [],
      failures = [];
    const ctx = {
      board: parseFen(START_FEN),
      positions: [parseFen(START_FEN)],
      moves: [],
      humanColor: 'w',
      opponent: { active: {} },
      abort: new AbortController(),
      disposed: false,
      enginePreparationGeneration: 0,
      engineReady: false,
    };
    const deps = {
      ctxSelectedEngine: () => option,
      retireUnusedPlayNeuralEngines: async () => {},
      buttonEl: () => ({}),
      ctxSetEngineNote: (text) => notes.push(text),
      ctxRender: () => {},
      ctxCpuEngineFor: async () => ({ setOptions() {}, bestMove: async () => move }),
      ctxSetEngineFailure: (_ctx, error) => failures.push(error),
      ctxHideDownloadProgress: () => {},
      boardToFen,
      legalMoves,
      moveToUci,
    };
    const verify = new Function(...Object.keys(deps), `${stripTypeScriptTypes(source.slice(from, to))}; return ctxVerifySelectedEngineReady;`)(
      ...Object.values(deps),
    );
    await verify(ctx);
    assert.equal(ctx.preparingEngine, false);
    if (move === 'e2e4') {
      assert.equal(ctx.engineReady, true);
      assert.equal(failures.length, 0);
      assert.ok(notes.includes('Stockfish is ready.'));
    } else {
      assert.equal(ctx.engineReady, false, 'null/illegal bestmove is not evidence that this opponent works');
      assert.equal(failures.length, 1, 'the ordinary retry path must expose the failed probe');
      assert.ok(!notes.includes('Stockfish is ready.'));
    }
    assert.equal(ctx.moves.length, 0, 'a readiness probe never plays a move');
  });
}
