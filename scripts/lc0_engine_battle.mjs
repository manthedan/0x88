#!/usr/bin/env node
// EngineBattle harness CLI: play an LC0 fixed-visit search engine against the
// LC0 policy-only engine and report the match score.
//
// Usage: npm run lc0:battle [model] [games] [visits] [maxPlies]
//
// Any other opponent (e.g. Stockfish.wasm) can be dropped in by implementing the
// BattleEngine interface from src/lc0/engineBattle.ts and passing it to runMatch.
import { readFileSync } from 'node:fs';
import { Lc0OnnxEvaluator } from '../src/lc0/onnxEvaluator.ts';
import { Lc0PolicyOnlyPlayer } from '../src/lc0/policyOnlyPlayer.ts';
import { Lc0PuctSearcher } from '../src/lc0/search.ts';
import { lc0PolicyBattleEngine, lc0SearchBattleEngine, runMatch } from '../src/lc0/engineBattle.ts';

const model = process.argv[2] ?? '../models/lc0-bestnets/onnx/t1-256x10-distilled-swa-2432500.batch1.f32.onnx';
const games = Number(process.argv[3] ?? 4);
const visits = Number(process.argv[4] ?? 32);
const maxPlies = Number(process.argv[5] ?? 200);

const evaluator = await Lc0OnnxEvaluator.create(readFileSync(model));
const search = lc0SearchBattleEngine(new Lc0PuctSearcher(evaluator), visits);
const policy = lc0PolicyBattleEngine(new Lc0PolicyOnlyPlayer(evaluator));

const summary = await runMatch(search, policy, games, { maxPlies });

console.log(`# EngineBattle: ${summary.engineA} vs ${summary.engineB}`);
console.log(`games ${summary.games} · ${summary.engineA} ${summary.aWins}W ${summary.bWins}L ${summary.draws}D · score ${summary.aScore}/${summary.games}`);
for (const [i, r] of summary.results.entries()) {
  console.log(`game ${i + 1}: ${r.result} (${r.reason}) · ${r.plies} plies`);
}
