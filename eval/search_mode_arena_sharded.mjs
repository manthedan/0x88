#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { existsSync, mkdirSync } from 'node:fs';
import { dirname } from 'node:path';

function arg(name, fallback = undefined) {
  const prefix = `${name}=`;
  const inline = process.argv.find((value) => value.startsWith(prefix));
  if (inline) return inline.slice(prefix.length);
  const i = process.argv.indexOf(name);
  return i >= 0 ? process.argv[i + 1] : fallback;
}
function stripArgs(names) {
  const out = [];
  for (let i = 2; i < process.argv.length; i++) {
    const a = process.argv[i];
    const name = names.find((n) => a === n || a.startsWith(`${n}=`));
    if (!name) { out.push(a); continue; }
    if (a === name) i++;
  }
  return out;
}
function run(cmd, args, env, label) {
  return new Promise((resolve, reject) => {
    const child = spawn(cmd, args, { stdio: ['ignore', 'inherit', 'inherit'], env });
    child.on('exit', (code, signal) => code === 0 ? resolve() : reject(new Error(`${label} failed code=${code} signal=${signal}`)));
  });
}

const shards = Math.max(1, Number(arg('--shards', '1')));
const out = arg('--out', 'artifacts/search_mode_arena/arena.json');
const ortThreads = arg('--ort-threads', process.env.ORT_INTRA_OP_NUM_THREADS ?? process.env.ORT_NUM_THREADS ?? '1');
const baseArgs = stripArgs(['--shards', '--ort-threads']);
if (shards <= 1) {
  await run(process.execPath, ['--experimental-strip-types', 'eval/search_mode_arena.mjs', ...baseArgs], process.env, 'arena');
  process.exit(0);
}

const shardDir = `${out}.shards`;
mkdirSync(shardDir, { recursive: true });
mkdirSync(dirname(out), { recursive: true });
const shardOuts = Array.from({ length: shards }, (_, i) => `${shardDir}/shard_${String(i).padStart(3, '0')}.json`);
const env = { ...process.env, ORT_INTRA_OP_NUM_THREADS: String(ortThreads), ORT_NUM_THREADS: String(ortThreads) };
console.error(`[search-mode-arena-sharded] launching ${shards} shards ortThreads=${ortThreads}`);
await Promise.all(shardOuts.map((path, i) => run(process.execPath, [
  '--experimental-strip-types',
  'eval/search_mode_arena.mjs',
  ...baseArgs.filter((a, idx, arr) => !(a === '--out' || arr[idx - 1] === '--out' || a.startsWith('--out='))),
  '--out', path,
  '--shard-count', String(shards),
  '--shard-index', String(i),
], env, `shard ${i}`)));
for (const path of shardOuts) if (!existsSync(path)) throw new Error(`missing shard output: ${path}`);
await run(process.execPath, ['--experimental-strip-types', 'eval/merge_search_mode_arena_shards.mjs', '--inputs', shardOuts.join(','), '--out', out], process.env, 'merge');
console.error(`[search-mode-arena-sharded] merged ${shards} shards -> ${out}`);
