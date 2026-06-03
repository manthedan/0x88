#!/usr/bin/env node
import { existsSync, readFileSync } from 'node:fs';
import { START_FEN } from '../src/chess/board.ts';
import { Lc0PolicyOnlyPlayer } from '../src/lc0/policyOnlyPlayer.ts';

const model = process.argv[2] ?? '../models/lc0-bestnets/onnx/t1-256x10-distilled-swa-2432500.batch1.f32.onnx';
const fen = process.argv.slice(3).join(' ') || START_FEN;
if (!existsSync(model)) throw new Error(`Model not found: ${model}`);

const player = await Lc0PolicyOnlyPlayer.create(readFileSync(model));
const choice = await player.chooseMove(fen);
console.log(JSON.stringify({
  fen,
  move: choice.move,
  wdl: choice.evaluation.wdl,
  q: choice.evaluation.q,
  mlh: choice.evaluation.mlh,
  topPriors: choice.evaluation.legalPriors.slice(0, 10).map(({ uci, index, prior }) => ({ uci, index, prior })),
}, null, 2));
