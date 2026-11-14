#!/usr/bin/env node
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import { boardToFen, parseFen } from '../src/chess/board.ts';
import { moveToUci } from '../src/chess/moveCodec.ts';
import { legalMoves, makeMove } from '../src/chess/movegen.ts';
import * as ort from '../src/nn/ortRuntime.ts';
import { S4FIX_V_EMA_MODEL_ID, squareformerModelSettings } from '../src/nn/runtimeRegistry.ts';
import { SquareFormerEvaluator } from '../src/nn/squareformerEvaluator.ts';
import { centipawnSearchOptions } from '../src/search/centipawn.ts';
import { chooseMove } from '../src/search/puct.ts';

export async function checkCentipawnSearch({ onPosition = () => {} } = {}) {
  const fixture = JSON.parse(readFileSync(new URL('../tests/fixtures/centipawn_s4fix_search_fixture_n256.json', import.meta.url)));
  const meta = JSON.parse(readFileSync(new URL('../public/models/s4fix_v_ema.meta.json', import.meta.url)));
  const session = await ort.InferenceSession.create(readFileSync(new URL('../public/models/s4fix_v_ema.onnx', import.meta.url)), ort.sessionOptions(['wasm']));
  const evaluator = new SquareFormerEvaluator(session, meta, squareformerModelSettings(S4FIX_V_EMA_MODEL_ID));
  const started = performance.now();
  const rows = [];
  try {
    for (const position of fixture.positions) {
      let board = parseFen(position.root_fen);
      const historyFens = [];
      for (const uci of position.moves) {
        const move = legalMoves(board).find((move) => moveToUci(move) === uci);
        assert.ok(move, `${position.label}: illegal line move ${uci}`);
        historyFens.unshift(boardToFen(board));
        board = makeMove(board, move);
      }
      // FEN encoders do not retain uncapturable EP squares after double pushes.
      const fields = boardToFen(board).split(' ');
      const expectedFields = position.fen.split(' ');
      fields[3] = expectedFields[3];
      assert.equal(fields.join(' '), position.fen, position.label);
      board = parseFen(position.fen);
      const result = await chooseMove(board, evaluator, {
        ...centipawnSearchOptions,
        visits: 256,
        batchSize: 1,
        historyFens,
        yieldEveryMs: 0,
      });
      const siteMove = result.move ? moveToUci(result.move) : null;
      const row = {
        label: position.label,
        nativeMove: position.bestmove,
        nativeVisits: 256,
        nativeMoveVisits: null,
        nativeValue: position.value,
        nativeScoreCp: position.score_cp,
        siteMove,
        siteValue: result.value,
        completedVisits: result.stats.completedVisits,
        agrees: siteMove === position.bestmove,
        siteTop: [...result.policy].sort((a, b) => b.visits - a.visits).slice(0, 5).map((entry) => ({ move: moveToUci(entry.move), visits: entry.visits, q: entry.q })),
      };
      rows.push(row);
      onPosition(row);
    }
  } finally {
    await session.release();
  }
  const agreement = rows.filter((row) => row.agrees).length;
  const decisive = rows.filter((row) => Math.abs(row.nativeValue) >= 0.8);
  return {
    positions: rows.length,
    agreement,
    share: agreement / rows.length,
    decisiveThreshold: 0.8,
    decisiveAgreement: decisive.filter((row) => row.agrees).length,
    decisivePositions: decisive.length,
    elapsedMs: performance.now() - started,
    nativeVisitNote: 'The bundle records 256 total visits, but no per-move native visit counts.',
    passed: agreement / rows.length >= 0.7 && decisive.every((row) => row.agrees),
    disagreements: rows.filter((row) => !row.agrees),
    rows,
  };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const report = await checkCentipawnSearch({ onPosition: (row) => console.error(`${row.agrees ? 'agree' : 'DIFFER'} ${row.label}: ${row.nativeMove} / ${row.siteMove}`) });
  const text = `${JSON.stringify(report, null, 2)}\n`;
  if (process.argv[2]) writeFileSync(process.argv[2], text);
  console.log(text);
  if (!report.passed) process.exitCode = 1;
}
