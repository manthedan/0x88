#!/usr/bin/env node
import { readFileSync } from 'node:fs';
import { buildBoardHistoryFromMoves } from '../src/lc0/history.ts';
import { Lc0PuctSearcher } from '../src/lc0/search.ts';

const model = process.argv[2] ?? '../models/lc0-bestnets/onnx/t1-256x10-distilled-swa-2432500.batch1.f32.onnx';
const fenOrFixture = process.argv[3] ?? 'startpos';
const visits = Number(process.argv[4] ?? 32);
const batchSize = Number(process.argv[5] ?? 1);
const searcher = await Lc0PuctSearcher.create(readFileSync(model));

let input;
if (fenOrFixture.endsWith('.json')) {
  const fixtures = JSON.parse(readFileSync(fenOrFixture, 'utf8'));
  for (const fixture of fixtures) {
    input = fixture.moves ? { positions: buildBoardHistoryFromMoves(fixture.moves, fixture.startFen) } : fixture.fen;
    const result = await searcher.search(input, { visits, batchSize });
    console.log(JSON.stringify({
      id: fixture.id,
      fen: result.fen,
      ...(fixture.moves ? { startFen: fixture.startFen, moves: fixture.moves } : {}),
      visits: result.visits,
      bestMove: result.move,
      value: result.value,
      pv: result.pv,
      topChildren: result.children.slice(0, 10),
      stats: result.search.stats,
    }));
  }
} else {
  input = fenOrFixture === 'startpos' ? undefined : fenOrFixture;
  const result = await searcher.search(input ?? 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1', { visits, batchSize });
  console.log(JSON.stringify({
    fen: result.fen,
    visits: result.visits,
    bestMove: result.move,
    value: result.value,
    topChildren: result.children.slice(0, 10),
    stats: result.search.stats,
  }, null, 2));
}
