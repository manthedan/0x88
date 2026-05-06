#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { readFileSync, mkdirSync, appendFileSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { ensureRustBin } from './rust_engine.mjs';

function arg(name, fallback = undefined) {
  const prefix = `${name}=`;
  const inline = process.argv.find((value) => value.startsWith(prefix));
  if (inline) return inline.slice(prefix.length);
  const i = process.argv.indexOf(name);
  return i >= 0 ? process.argv[i + 1] : fallback;
}

function readOpenings(path) {
  return readFileSync(path, 'utf8').split(/\r?\n/).map((line) => line.trim()).filter((line) => line && !line.startsWith('#'));
}

function elo(scoreRate) {
  const s = Math.min(0.999, Math.max(0.001, scoreRate));
  return 400 * Math.log10(s / (1 - s));
}

function normal95EloCi(wins, draws, losses) {
  const n = Math.max(1, wins + draws + losses);
  const scores = [1, 0.5, 0];
  const counts = [wins, draws, losses];
  const mean = (wins + 0.5 * draws) / n;
  const variance = counts.reduce((acc, c, i) => acc + c * (scores[i] - mean) ** 2, 0) / Math.max(1, n - 1);
  const se = Math.sqrt(variance / n);
  const lo = Math.min(0.999, Math.max(0.001, mean - 1.96 * se));
  const hi = Math.min(0.999, Math.max(0.001, mean + 1.96 * se));
  return { lo: elo(lo), hi: elo(hi), half: (elo(hi) - elo(lo)) / 2 };
}

function parseMetrics(text) {
  const out = {};
  for (const line of text.split(/\r?\n/)) {
    const m = line.match(/^METRIC\s+([^=]+)=(.+)$/);
    if (m) out[m[1]] = Number(m[2]);
  }
  return out;
}

const candidate = arg('--candidate', 'artifacts/student_distill_benchmark.json');
const baseline = arg('--baseline', 'artifacts/student_distill_benchmark.json');
const openingsPath = arg('--openings-file', 'eval/opening_suite_v1.fen');
const pairs = Number(arg('--pairs', '10'));
const workers = Math.max(1, Number(arg('--workers', String(Math.min(4, pairs)))));
const visits = Number(arg('--visits', '8'));
const maxPlies = Number(arg('--max-plies', '80'));
const adjudicate = arg('--adjudicate', 'terminal');
const stockfish = arg('--stockfish', process.env.STOCKFISH_BIN || 'stockfish');
const stockfishDepth = Number(arg('--stockfish-depth', '8'));
const stockfishDrawCp = Number(arg('--stockfish-draw-cp', '50'));
const outPath = arg('--out', `artifacts/arena/elo_${Date.now()}.jsonl`);
const progressEvery = Number(arg('--progress-every', '999999'));
const bin = ensureRustBin('tiny-leela-rust-arena');
const openings = readOpenings(openingsPath);
if (!openings.length) throw new Error(`no openings in ${openingsPath}`);
mkdirSync(dirname(outPath), { recursive: true });
writeFileSync(outPath, '');

const totalGames = pairs * 2;
const shardCount = Math.min(workers, totalGames);
const shards = [];
for (let i = 0; i < shardCount; i++) {
  const games = Math.floor(totalGames / shardCount) + (i < totalGames % shardCount ? 1 : 0);
  const startGame = shards.reduce((acc, s) => acc + s.games, 0);
  if (games) shards.push({ startGame, games });
}

console.error(`[elo-arena] start games=${totalGames} pairs=${pairs} workers=${shards.length} visits=${visits} openings=${openings.length}`);
const startedAt = Date.now();
const results = await Promise.all(shards.map((shard, shardId) => new Promise((resolve, reject) => {
  const args = [
    `--candidate=${candidate}`,
    `--baseline=${baseline}`,
    `--games=${shard.games}`,
    `--start-game=${shard.startGame}`,
    `--visits=${visits}`,
    `--max-plies=${maxPlies}`,
    `--adjudicate=${adjudicate}`,
    `--stockfish=${stockfish}`,
    `--stockfish-depth=${stockfishDepth}`,
    `--stockfish-draw-cp=${stockfishDrawCp}`,
    `--progress-every=${progressEvery}`,
    `--openings=${openings.join('|')}`,
  ];
  const child = spawn(bin, args, { stdio: ['ignore', 'pipe', 'pipe'] });
  let stdout = '', stderr = '';
  child.stdout.on('data', (b) => { stdout += b; });
  child.stderr.on('data', (b) => { stderr += b; process.stderr.write(`[w${shardId}] ${b}`); });
  child.on('error', reject);
  child.on('close', (code) => {
    appendFileSync(outPath, JSON.stringify({ type: 'shard', shardId, code, startGame: shard.startGame, games: shard.games, stdout, stderr }) + '\n');
    if (code !== 0) reject(new Error(`worker ${shardId} exited ${code}`));
    else resolve(parseMetrics(stdout));
  });
})));

let wins = 0, draws = 0, losses = 0, illegal = 0, pliesWeighted = 0;
for (const r of results) {
  wins += r.arena_wins || 0;
  draws += r.arena_draws || 0;
  losses += r.arena_losses || 0;
  illegal += r.arena_illegal_losses || 0;
  pliesWeighted += (r.arena_avg_plies || 0) * (r.arena_games || 0);
}
const games = wins + draws + losses;
const scoreRate = (wins + 0.5 * draws) / Math.max(1, games);
const ci = normal95EloCi(wins, draws, losses);
const elapsed = (Date.now() - startedAt) / 1000;
const summary = { type: 'summary', candidate, baseline, openingsPath, pairs, workers: shards.length, visits, maxPlies, adjudicate, stockfish: adjudicate === 'stockfish' ? stockfish : undefined, stockfishDepth: adjudicate === 'stockfish' ? stockfishDepth : undefined, stockfishDrawCp: adjudicate === 'stockfish' ? stockfishDrawCp : undefined, games, wins, draws, losses, illegal, scoreRate, elo: elo(scoreRate), elo95Lo: ci.lo, elo95Hi: ci.hi, elapsedSeconds: elapsed };
appendFileSync(outPath, JSON.stringify(summary) + '\n');

console.log(`METRIC elo_arena_games=${games}`);
console.log(`METRIC elo_arena_score_rate=${scoreRate.toFixed(6)}`);
console.log(`METRIC elo_arena_diff=${elo(scoreRate).toFixed(6)}`);
console.log(`METRIC elo_arena_ci95_half=${ci.half.toFixed(6)}`);
console.log(`METRIC elo_arena_wins=${wins}`);
console.log(`METRIC elo_arena_draws=${draws}`);
console.log(`METRIC elo_arena_losses=${losses}`);
console.log(`METRIC elo_arena_illegal_losses=${illegal}`);
console.log(`METRIC elo_arena_avg_plies=${(pliesWeighted / Math.max(1, games)).toFixed(6)}`);
console.log(`METRIC elo_arena_games_per_second=${(games / Math.max(0.001, elapsed)).toFixed(6)}`);
console.log(`METRIC elo_arena_out=${outPath}`);
