#!/usr/bin/env node
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname } from 'node:path';

function arg(name, fallback = undefined) {
  const prefix = `${name}=`;
  const inline = process.argv.find((value) => value.startsWith(prefix));
  if (inline) return inline.slice(prefix.length);
  const i = process.argv.indexOf(name);
  return i >= 0 ? process.argv[i + 1] : fallback;
}
function elo(scoreRate) {
  const s = Math.min(0.999, Math.max(0.001, scoreRate));
  return 400 * Math.log10(s / (1 - s));
}
function addResult(row, score, illegal = false) {
  row.games++; row.score += score;
  if (score === 1) row.wins++; else if (score === 0) row.losses++; else row.draws++;
  if (illegal) row.illegal++;
}

const inputs = arg('--inputs', '').split(',').filter(Boolean);
const out = arg('--out', 'artifacts/search_mode_arena/merged.json');
if (!inputs.length) throw new Error('usage: eval/merge_search_mode_arena_shards.mjs --inputs shard0.json,shard1.json --out merged.json');

const shardDocs = inputs.map((path) => ({ path, doc: JSON.parse(readFileSync(path, 'utf8')) }));
const first = shardDocs[0].doc;
const players = first.protocol?.players ?? [];
const table = Object.fromEntries(players.map((p) => [p.name, { wins: 0, draws: 0, losses: 0, score: 0, games: 0, illegal: 0 }]));
const pairTable = {};
const games = [];

for (let i = 0; i < players.length; i++) for (let j = i + 1; j < players.length; j++) {
  const a = players[i], b = players[j];
  pairTable[`${a.name}__${b.name}`] = { a: a.name, b: b.name, aScore: 0, games: 0, aWdl: [0, 0, 0] };
}

for (const { doc } of shardDocs) {
  for (const game of doc.games ?? []) {
    games.push(game);
    const aScore = game.aScore;
    const bScore = 1 - aScore;
    addResult(table[game.a], aScore, game.illegal === game.a);
    addResult(table[game.b], bScore, game.illegal === game.b);
    const key = `${game.a}__${game.b}`;
    if (!pairTable[key]) pairTable[key] = { a: game.a, b: game.b, aScore: 0, games: 0, aWdl: [0, 0, 0] };
    pairTable[key].aScore += aScore;
    pairTable[key].games++;
    pairTable[key].aWdl[aScore === 1 ? 0 : aScore === 0.5 ? 1 : 2]++;
  }
}

const standings = Object.entries(table)
  .map(([name, r]) => ({ name, ...r, scoreRate: r.score / Math.max(1, r.games), eloVsPool: elo(r.score / Math.max(1, r.games)) }))
  .sort((a, b) => b.scoreRate - a.scoreRate);
const pairs = Object.values(pairTable).map((r) => ({ ...r, aScoreRate: r.aScore / Math.max(1, r.games) }));
const cacheStats = {};
for (const { doc } of shardDocs) {
  for (const [name, stats] of Object.entries(doc.protocol?.cacheStats ?? {})) {
    const row = cacheStats[name] ?? { hits: 0, misses: 0, entries: 0 };
    row.hits += stats.hits ?? 0;
    row.misses += stats.misses ?? 0;
    row.entries += stats.entries ?? 0;
    cacheStats[name] = row;
  }
}
const protocol = {
  ...(first.protocol ?? {}),
  kind: 'search_mode_arena_merged',
  mergedShards: inputs,
  shardCount: inputs.length,
  cacheStats,
  elapsedMs: shardDocs.reduce((sum, s) => sum + Number(s.doc.protocol?.elapsedMs ?? 0), 0),
  createdUtc: new Date().toISOString(),
};
mkdirSync(dirname(out), { recursive: true });
writeFileSync(out, JSON.stringify({ protocol, standings, pairs, games }, null, 2));
writeFileSync(`${out}.protocol.json`, JSON.stringify(protocol, null, 2));
standings.forEach((r, idx) => {
  console.log(`METRIC arena_rank_${idx + 1}_${r.name}_score_rate=${r.scoreRate.toFixed(6)}`);
  console.log(`METRIC arena_${r.name}_games=${r.games}`);
  console.log(`METRIC arena_${r.name}_wdl=${r.wins}_${r.draws}_${r.losses}`);
});
console.log(`METRIC arena_models=${players.length}`);
console.log(`METRIC arena_games=${games.length}`);
console.log(`METRIC arena_shards=${inputs.length}`);
