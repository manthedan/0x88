#!/usr/bin/env node
import { existsSync, readFileSync } from 'node:fs';
import { boardToFen } from '../src/chess/board.ts';
import { buildBoardHistoryFromMoves } from '../src/lc0/history.ts';
import { Lc0OnnxEvaluator } from '../src/lc0/onnxEvaluator.ts';

const model = process.argv[2] ?? '../models/lc0-bestnets/onnx/t1-256x10-distilled-swa-2432500.batch1.f32.onnx';
const fixturePath = process.argv[3] ?? 'fixtures/lc0/fen_only.json';
if (!existsSync(model)) throw new Error(`Model not found: ${model}`);
const fixtures = JSON.parse(readFileSync(fixturePath, 'utf8'));
const evaluator = await Lc0OnnxEvaluator.create(readFileSync(model));
for (const fixture of fixtures) {
  const positions = fixture.moves ? buildBoardHistoryFromMoves(fixture.moves, fixture.startFen) : undefined;
  const input = positions ? { positions } : fixture.fen;
  const evaluation = await evaluator.evaluate(input);
  console.log(JSON.stringify({
    id: fixture.id,
    fen: fixture.fen ?? boardToFen(positions[positions.length - 1]),
    ...(fixture.moves ? { startFen: fixture.startFen, moves: fixture.moves } : {}),
    bestMove: evaluation.bestMove,
    wdl: evaluation.wdl,
    q: evaluation.q,
    mlh: evaluation.mlh,
    topPriors: evaluation.legalPriors.slice(0, 10).map(({ uci, index, prior }) => ({ uci, index, prior })),
  }));
}
