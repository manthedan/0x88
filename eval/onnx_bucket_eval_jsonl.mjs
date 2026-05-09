#!/usr/bin/env node
import { createReadStream, writeFileSync, mkdirSync } from 'node:fs';
import { dirname } from 'node:path';
import { spawn } from 'node:child_process';
import readline from 'node:readline';
import { parseFen } from '../src/chess/board.ts';
import { moveFromUci, moveToActionId } from '../src/chess/moveCodec.ts';
import { legalMoves } from '../src/chess/movegen.ts';
import { OnnxEvaluator } from '../src/nn/onnxEvaluator.ts';
import { SquareFormerEvaluator } from '../src/nn/squareformerEvaluator.ts';
import { readFileSync } from 'node:fs';

function arg(name, fallback = '') { const p = `${name}=`; const x = process.argv.find(v => v.startsWith(p)); if (x) return x.slice(p.length); const i = process.argv.indexOf(name); return i >= 0 ? process.argv[i + 1] : fallback; }
function gid(row) { return String(row.id ?? '').replace(/_[0-9]+$/, ''); }
function ply(row) { const m = String(row.id ?? '').match(/_([0-9]+)$/); return Number(row.ply ?? (m ? m[1] : 0)); }
function phase(row) { const p = ply(row); if (p < 32) return 'opening'; if (p > 80 || material(row.fen) <= 18) return 'endgame'; return 'middlegame'; }
function material(fen) { const vals = { p:1,n:3,b:3,r:5,q:9 }; let s=0; for (const ch of fen.split(' ')[0]) s += vals[ch.toLowerCase()] ?? 0; return s; }
function source(row) { const t = String(row.teacher ?? 'unknown'); if (t.includes('elite')) return 'elite'; if (t.includes('tcec')) return 'tcec'; if (t.includes('lichess')) return 'lichess'; return t; }
function tc(row) { const t = String(row.time_control ?? 'unknown'); const m = t.match(/^(\d+)\+/); if (!m) return 'tc_unknown'; const base = Number(m[1]); if (base < 180) return 'tc_fast'; if (base < 600) return 'tc_blitz'; if (base < 1800) return 'tc_rapid'; return 'tc_classical'; }
function add(stats, key, ce, rank, top1, top4, top8) { const s = stats[key] ??= { rows:0, ce:0, rank:0, top1:0, top4:0, top8:0 }; s.rows++; s.ce += ce; s.rank += rank; s.top1 += top1; s.top4 += top4; s.top8 += top8; }
function openLines(path) { if (path.endsWith('.zst')) { const p = spawn('zstd', ['-dc', path]); return readline.createInterface({ input: p.stdout, crlfDelay: Infinity }); } return readline.createInterface({ input: createReadStream(path), crlfDelay: Infinity }); }

const input = arg('--input');
const model = arg('--model');
const meta = arg('--meta');
const out = arg('--out', 'artifacts/eval/bucket_eval.json');
const maxRowsPerBucket = Number(arg('--max-rows-per-bucket', '5000'));
if (!input || !model || !meta) throw new Error('usage: --input dev.jsonl[.zst] --model x.onnx --meta x.meta.json');
const metaJson = JSON.parse(readFileSync(meta, 'utf8'));
const evaluator = (metaJson.kind === 'squareformer' || metaJson.kind === 'squareformer_v2') ? await SquareFormerEvaluator.create(model, metaJson) : await OnnxEvaluator.create(model, metaJson);
const stats = {}; const counts = {}; let rows = 0, used = 0, bad = 0;
for await (const line of openLines(input)) {
  if (!line.trim()) continue; rows++;
  let row; try { row = JSON.parse(line); } catch { bad++; continue; }
  const keys = [`phase:${phase(row)}`, `source:${source(row)}`, `tc:${tc(row)}`];
  if (keys.every(k => (counts[k] ?? 0) >= maxRowsPerBucket)) continue;
  const moveUci = Object.keys(row.policy ?? {})[0]; if (!moveUci) continue;
  let board, move; try { board = parseFen(row.fen); move = moveFromUci(moveUci); } catch { bad++; continue; }
  const ev = await evaluator.evaluate(board, { historyFens: row.history_fens ?? [] });
  const target = moveToActionId(move);
  const p = Math.max(1e-12, ev.policy.get(target) ?? 0);
  const legal = legalMoves(board).map(m => ({ id: moveToActionId(m), p: ev.policy.get(moveToActionId(m)) ?? 0 })).sort((a,b)=>b.p-a.p);
  const rank = Math.max(1, legal.findIndex(x => x.id === target) + 1 || legal.length + 1);
  const ce = -Math.log(p);
  for (const k of keys) if ((counts[k] ?? 0) < maxRowsPerBucket) { counts[k] = (counts[k] ?? 0) + 1; add(stats, k, ce, rank, rank===1?1:0, rank<=4?1:0, rank<=8?1:0); }
  used++;
}
const protocol = { kind:'onnx_bucket_eval_jsonl', input, model, meta, maxRowsPerBucket, createdUtc:new Date().toISOString() };
const result = { input, model, meta, rows, used, bad, maxRowsPerBucket, protocol, buckets: {} };
for (const [k,s] of Object.entries(stats)) result.buckets[k] = { rows:s.rows, ce:s.ce/s.rows, mean_legal_rank:s.rank/s.rows, top1:s.top1/s.rows, top4:s.top4/s.rows, top8:s.top8/s.rows };
mkdirSync(dirname(out), { recursive:true }); writeFileSync(out, JSON.stringify(result, null, 2)); writeFileSync(`${out}.protocol.json`, JSON.stringify(protocol, null, 2));
for (const [k,s] of Object.entries(result.buckets)) {
  const safe = k.replace(/[^a-zA-Z0-9]+/g, '_');
  console.log(`METRIC bucket_${safe}_rows=${s.rows}`);
  console.log(`METRIC bucket_${safe}_ce=${s.ce.toFixed(6)}`);
  console.log(`METRIC bucket_${safe}_top1=${s.top1.toFixed(6)}`);
  console.log(`METRIC bucket_${safe}_top4=${s.top4.toFixed(6)}`);
  console.log(`METRIC bucket_${safe}_top8=${s.top8.toFixed(6)}`);
  console.log(`METRIC bucket_${safe}_mean_legal_rank=${s.mean_legal_rank.toFixed(6)}`);
}
console.log(`METRIC bucket_eval_rows_seen=${rows}`);
console.log(`METRIC bucket_eval_rows_used=${used}`);
