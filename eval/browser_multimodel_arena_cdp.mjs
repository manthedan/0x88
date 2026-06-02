#!/usr/bin/env node
// Single-tab browser/WebGPU multi-model round-robin runner via Chrome CDP.
// Intended for the Mac mini webroot/Chrome setup. It loads all models once in
// browser-multimodel-arena.html, then writes per-pair JSON + results.json + summary.md.

import fs from 'node:fs/promises';
import path from 'node:path';

function arg(name, fallback = undefined) {
  const idx = process.argv.indexOf(name);
  if (idx >= 0 && idx + 1 < process.argv.length) return process.argv[idx + 1];
  return fallback;
}
function flag(name) { return process.argv.includes(name); }

const cdpBase = arg('--cdp-base', process.env.CDP_BASE ?? 'http://127.0.0.1:9333');
const appBase = arg('--app-base', process.env.APP_BASE ?? 'http://127.0.0.1:8799');
const outDir = arg('--out', process.env.OUT_DIR ?? path.resolve('browser_multimodel_arena_out'));
const timeoutMs = Number(arg('--timeout-ms', process.env.TIMEOUT_MS ?? '7200000'));
const closeTab = !flag('--keep-tab');
const parallel = Math.max(1, Math.floor(Number(arg('--parallel', process.env.PARALLEL ?? '1')) || 1));

const modelBase = arg('--model-base', process.env.MODEL_BASE ?? '/models/tg_wdl_roundrobin_20260529');
const modes = arg('--modes', process.env.MODES ?? 'puct,monty_lc0both');
const visits = arg('--visits', process.env.VISITS ?? '256');
const batch = arg('--batch', process.env.BATCH ?? '64');
const brokerBatch = arg('--broker-batch', process.env.BROKER_BATCH ?? '0');
const brokerWaitMs = arg('--broker-wait-ms', process.env.BROKER_WAIT_MS ?? '0');
const ortEp = arg('--ort-ep', process.env.ORT_EP ?? process.env.TINY_LEELA_ORT_EP ?? 'auto');
const cpuct = arg('--cpuct', process.env.CPUCT ?? '1.5');
const fpu = arg('--fpu', process.env.FPU ?? '0');
const openings = arg('--openings', process.env.OPENINGS ?? '32');
const openingOffset = arg('--opening-offset', process.env.OPENING_OFFSET ?? '0');
const maxPlies = arg('--max-plies', process.env.MAX_PLIES ?? '128');
const openingsUrl = arg('--openings-url', process.env.OPENINGS_URL ?? '/eval/opening_suite_fishtest_uho_128_seed20260527.fen');
const includeMoves = arg('--include-moves', process.env.INCLUDE_MOVES ?? '1');

let modelsJson = arg('--models-json', process.env.MODELS_JSON ?? '');
const modelsJsonFile = arg('--models-json-file', process.env.MODELS_JSON_FILE ?? '');
if (!modelsJson && modelsJsonFile) modelsJson = await fs.readFile(modelsJsonFile, 'utf8');

async function jsonFetch(url, init) {
  const res = await fetch(url, init);
  if (!res.ok) throw new Error(`${init?.method ?? 'GET'} ${url}: ${res.status} ${res.statusText} ${await res.text().catch(() => '')}`);
  return await res.json();
}

class Cdp {
  constructor(wsUrl) {
    this.ws = new WebSocket(wsUrl);
    this.id = 0;
    this.pending = new Map();
    this.handlers = new Map();
    this.ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.id && this.pending.has(msg.id)) {
        const { resolve, reject } = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        msg.error ? reject(new Error(JSON.stringify(msg.error))) : resolve(msg.result);
      } else if (msg.method && this.handlers.has(msg.method)) {
        for (const fn of this.handlers.get(msg.method)) {
          try { fn(msg.params, msg)?.catch?.(() => null); } catch { /* ignore event handler errors */ }
        }
      }
    };
  }
  async open() { await new Promise((resolve, reject) => { this.ws.onopen = resolve; this.ws.onerror = reject; }); }
  on(method, fn) {
    const list = this.handlers.get(method) ?? [];
    list.push(fn);
    this.handlers.set(method, list);
  }
  send(method, params = {}) {
    const id = ++this.id;
    this.ws.send(JSON.stringify({ id, method, params }));
    return new Promise((resolve, reject) => this.pending.set(id, { resolve, reject }));
  }
  close() { this.ws.close(); }
}

const checkpointPrefix = '__TL_ARENA_CHECKPOINT__';

function arenaUrl(overrides = {}) {
  const q = new URLSearchParams({
    modelBase,
    modes,
    visits,
    batch,
    openings: String(overrides.openings ?? openings),
    openingOffset: String(overrides.openingOffset ?? openingOffset),
    maxPlies,
    openingsFile: openingsUrl,
    includeMoves,
    cpuct,
    fpu,
    ortEp,
    brokerBatch,
    brokerWaitMs,
    cacheEntries: process.env.CACHE_ENTRIES ?? '131072',
    yieldEveryMs: process.env.YIELD_EVERY_MS ?? '0',
    cb: `${Date.now()}-${Math.random()}`,
  });
  if (modelsJson) q.set('modelsJson', modelsJson);
  return `${appBase}/browser-multimodel-arena.html?${q.toString()}`;
}

function safeName(s) { return String(s).replace(/[^A-Za-z0-9_.-]+/g, '_'); }
function mdFor(result) {
  const cfg = result.config ?? {};
  const lines = [
    '# Browser WebGPU multi-model arena',
    '',
    `generatedAt=\`${result.finishedAt ?? new Date().toISOString()}\`, visits=\`${cfg.visits}\`, batch=\`${cfg.batchSize}\`, openings=\`${cfg.openingCount}\`, maxPlies=\`${cfg.maxPlies}\`, openingsUrl=\`${cfg.openingsUrl}\`.`,
    '',
  ];
  for (const [mode, m] of Object.entries(result.summary ?? {})) {
    lines.push(`## ${mode}`, '', '| model | score | W-D-L | games |', '|---|---:|---:|---:|');
    for (const r of (m.table ?? [])) lines.push(`| ${r.id} | ${Number(r.scoreRate ?? 0).toFixed(3)} | ${r.wins}-${r.draws}-${r.losses} | ${r.games} |`);
    lines.push('', '| A | B | A score | W-D-L | games | file |', '|---|---|---:|---:|---:|---|');
    for (const p of (m.pairs ?? [])) {
      const file = `${safeName(p.mode)}_${safeName(p.a)}_vs_${safeName(p.b)}.json`;
      lines.push(`| ${p.a} | ${p.b} | ${Number(p.scoreRate ?? 0).toFixed(3)} | ${p.wins}-${p.draws}-${p.losses} | ${p.games} | ${file} |`);
    }
    lines.push('');
  }
  return lines.join('\n') + '\n';
}

function summarizePair(rows, rec) {
  const n = rows.length;
  const aScore = rows.reduce((s, r) => s + Number(r.aScore ?? 0), 0);
  const wins = rows.filter((r) => r.aScore === 1).length;
  const draws = rows.filter((r) => r.aScore === 0.5).length;
  const losses = rows.filter((r) => r.aScore === 0).length;
  const aMoves = rows.reduce((s, r) => s + Number(r.aMoves ?? 0), 0);
  const bMoves = rows.reduce((s, r) => s + Number(r.bMoves ?? 0), 0);
  const aMs = rows.reduce((s, r) => s + Number(r.aThinkMs ?? 0), 0);
  const bMs = rows.reduce((s, r) => s + Number(r.bThinkMs ?? 0), 0);
  const scoreRate = aScore / Math.max(1, n);
  const elo = scoreRate <= 0 ? -Infinity : scoreRate >= 1 ? Infinity : -400 * Math.log10(1 / scoreRate - 1);
  return { mode: rec.mode, a: rec.a?.id ?? rec.summary?.a, b: rec.b?.id ?? rec.summary?.b, games: n, aScore, scoreRate, elo, wins, draws, losses, aVisits: rec.summary?.aVisits, bVisits: rec.summary?.bVisits, aAvgMs: aMs / Math.max(1, aMoves), bAvgMs: bMs / Math.max(1, bMoves), aMoves, bMoves };
}

function emptyStanding() { return { games: 0, score: 0, wins: 0, draws: 0, losses: 0 }; }
function addGame(standings, modelId, score) {
  const s = standings[modelId] ??= emptyStanding();
  s.games += 1; s.score += score;
  if (score === 1) s.wins += 1; else if (score === 0.5) s.draws += 1; else s.losses += 1;
}
function summarizeRoundRobin(results) {
  const byMode = {};
  for (const rec of results) {
    const m = byMode[rec.mode] ??= { standings: {}, pairs: [] };
    for (const g of rec.games ?? []) {
      addGame(m.standings, rec.a.id, Number(g.aScore ?? 0));
      addGame(m.standings, rec.b.id, 1 - Number(g.aScore ?? 0));
    }
    m.pairs.push(rec.summary);
  }
  for (const m of Object.values(byMode)) {
    m.table = Object.entries(m.standings).map(([id, s]) => ({ id, ...s, scoreRate: s.games ? s.score / s.games : 0 })).sort((a, b) => b.scoreRate - a.scoreRate || b.score - a.score);
  }
  return byMode;
}

function aggregateShardResults(shards) {
  if (shards.length === 1) return shards[0];
  const first = shards[0] ?? {};
  const byPair = new Map();
  const errors = [];
  let fatal = false;
  for (const shard of shards) {
    if (shard.fatal) fatal = true;
    if (Array.isArray(shard.errors)) errors.push(...shard.errors);
    for (const rec of shard.results ?? []) {
      const key = `${rec.mode}\u0000${rec.a?.id ?? rec.summary?.a}\u0000${rec.b?.id ?? rec.summary?.b}`;
      let merged = byPair.get(key);
      if (!merged) {
        merged = { ...rec, games: [], startedAt: rec.startedAt, finishedAt: rec.finishedAt };
        byPair.set(key, merged);
      }
      merged.games.push(...(rec.games ?? []));
      if (rec.startedAt && (!merged.startedAt || rec.startedAt < merged.startedAt)) merged.startedAt = rec.startedAt;
      if (rec.finishedAt && (!merged.finishedAt || rec.finishedAt > merged.finishedAt)) merged.finishedAt = rec.finishedAt;
    }
  }
  const results = [...byPair.values()].map((rec) => {
    rec.games = rec.games.map((g, i) => ({ ...g, game: i + 1 }));
    rec.summary = summarizePair(rec.games, rec);
    return rec;
  });
  results.sort((a, b) => String(a.mode).localeCompare(String(b.mode)) || String(a.a?.id).localeCompare(String(b.a?.id)) || String(a.b?.id).localeCompare(String(b.b?.id)));
  return {
    ...first,
    fatal,
    errors,
    startedAt: shards.map((s) => s.startedAt).filter(Boolean).sort()[0] ?? first.startedAt,
    finishedAt: new Date().toISOString(),
    config: {
      ...(first.config ?? {}),
      openingCount: Number(openings),
      openingOffset: Number(openingOffset),
      parallel,
      shards: shards.map((s, i) => ({ shard: i, openingOffset: s.config?.openingOffset, openingCount: s.config?.openingCount, fatal: !!s.fatal })),
    },
    summary: summarizeRoundRobin(results),
    results,
  };
}

function openingShards() {
  const total = Math.max(1, Math.floor(Number(openings) || 1));
  const baseOffset = Math.max(0, Math.floor(Number(openingOffset) || 0));
  const n = Math.min(parallel, total);
  const rows = [];
  for (let i = 0; i < n; i++) {
    const count = Math.floor(total / n) + (i < total % n ? 1 : 0);
    const prior = i * Math.floor(total / n) + Math.min(i, total % n);
    rows.push({ index: i, openingOffset: baseOffset + prior, openings: count });
  }
  return rows.filter((s) => s.openings > 0);
}

await fs.mkdir(outDir, { recursive: true });

async function runShard(shard) {
  const shardDir = parallel === 1 ? outDir : path.join(outDir, 'shards', `shard_${String(shard.index).padStart(2, '0')}`);
  await fs.mkdir(shardDir, { recursive: true });
  const url = arenaUrl({ openings: shard.openings, openingOffset: shard.openingOffset });
  await fs.writeFile(path.join(shardDir, 'run_url.txt'), url + '\n');
  const target = await jsonFetch(`${cdpBase}/json/new`, { method: 'PUT' });
  const cdp = new Cdp(target.webSocketDebuggerUrl);
  await cdp.open();
  const partialDir = path.join(shardDir, 'partial');
  await fs.mkdir(partialDir, { recursive: true });
  let checkpointSeq = 0;
  let checkpointWrites = Promise.resolve();
  function queueCheckpointWrite(payload) {
    checkpointWrites = checkpointWrites.then(async () => {
      checkpointSeq += 1;
      const envelope = { seq: checkpointSeq, shard, receivedAt: new Date().toISOString(), ...payload };
      await fs.appendFile(path.join(partialDir, 'checkpoints.jsonl'), JSON.stringify(envelope) + '\n');
      await fs.writeFile(path.join(partialDir, 'latest_checkpoint.json'), JSON.stringify(envelope, null, 2));
      if (payload.kind === 'game') {
        const pair = `${safeName(payload.mode)}_${safeName(payload.a)}_vs_${safeName(payload.b)}`;
        await fs.writeFile(path.join(partialDir, `${String(checkpointSeq).padStart(6, '0')}_${pair}_game.json`), JSON.stringify(envelope, null, 2));
      } else if (payload.kind === 'pair' && payload.pair) {
        const pairFile = `${safeName(payload.mode)}_${safeName(payload.a)}_vs_${safeName(payload.b)}.partial.json`;
        await fs.writeFile(path.join(partialDir, pairFile), JSON.stringify(payload.pair, null, 2));
      }
    }).catch(async (err) => {
      await fs.appendFile(path.join(partialDir, 'checkpoint_errors.log'), String(err?.stack ?? err) + '\n').catch(() => null);
    });
  }
  cdp.on('Runtime.consoleAPICalled', async (params) => {
    for (const a of params.args ?? []) {
      const value = typeof a.value === 'string' ? a.value : '';
      if (!value.startsWith(checkpointPrefix)) continue;
      queueCheckpointWrite(JSON.parse(value.slice(checkpointPrefix.length)));
    }
  });
  try {
    await cdp.send('Runtime.enable');
    await cdp.send('Page.enable');
    await cdp.send('Page.navigate', { url });
    // Page.navigate returns before the new document's execution context is always
    // ready. Evaluating immediately can race in the old/blank context and yield
    // an undefined CDP result. Give Chrome a short turn to commit the document;
    // the runner-ready poll below still enforces the real readiness condition.
    await new Promise((resolve) => setTimeout(resolve, 3000));
    const ready = await cdp.send('Runtime.evaluate', {
      expression: `new Promise((resolve, reject) => { const start = Date.now(); const tick = () => { const runner = window.tinyLeelaMultiModelArena; if (runner && typeof runner.run === 'function') return resolve(true); if (Date.now() - start > 30000) return reject(new Error('multi-model runner not loaded')); setTimeout(tick, 100); }; tick(); })`,
      awaitPromise: true,
      returnByValue: true,
      timeout: 35000,
    });
    if (ready.exceptionDetails || ready.result?.subtype === 'error') {
      throw new Error(`multi-model runner readiness failed: ${JSON.stringify(ready.exceptionDetails ?? ready.result)}`);
    }
    await cdp.send('Runtime.evaluate', {
      expression: `window.__tinyLeelaArenaCheckpoint = (payload) => { try { console.log(${JSON.stringify(checkpointPrefix)} + JSON.stringify(payload)); } catch (err) { console.warn('[multi-model-arena] checkpoint serialization failed', err); } }; true`,
      returnByValue: true,
      timeout: 5000,
    });
    const evaluated = await cdp.send('Runtime.evaluate', {
      expression: `window.tinyLeelaMultiModelArena.run().then((r) => JSON.stringify(r)).catch((err) => JSON.stringify({ fatal: true, errors: [err && (err.stack || err.message) || String(err)] }))`,
      awaitPromise: true,
      returnByValue: true,
      timeout: timeoutMs,
    });
    const raw = evaluated.result?.value ?? '{"fatal":true,"errors":["no CDP result value"]}';
    const result = JSON.parse(raw);
    result.shard = shard;
    await fs.writeFile(path.join(shardDir, 'results.full.json'), JSON.stringify(result, null, 2));
    await checkpointWrites;
    return result;
  } finally {
    await checkpointWrites.catch(() => null);
    cdp.close();
    if (closeTab) await jsonFetch(`${cdpBase}/json/close/${target.id}`).catch(() => null);
  }
}

const shards = openingShards();
await fs.writeFile(path.join(outDir, 'run_url.txt'), shards.map((s) => arenaUrl({ openings: s.openings, openingOffset: s.openingOffset })).join('\n') + '\n');
await fs.writeFile(path.join(outDir, 'shards.json'), JSON.stringify(shards, null, 2));
const shardResults = await Promise.all(shards.map((s) => runShard(s)));
const result = aggregateShardResults(shardResults);
await fs.writeFile(path.join(outDir, 'results.full.json'), JSON.stringify(result, null, 2));
if (Array.isArray(result.results)) {
  for (const rec of result.results) {
    const file = `${safeName(rec.mode)}_${safeName(rec.a?.id ?? rec.summary?.a)}_vs_${safeName(rec.b?.id ?? rec.summary?.b)}.json`;
    await fs.writeFile(path.join(outDir, file), JSON.stringify(rec, null, 2));
  }
  const slim = { ...result, results: result.results.map((rec) => ({ mode: rec.mode, a: rec.a?.id, b: rec.b?.id, startedAt: rec.startedAt, finishedAt: rec.finishedAt, summary: rec.summary, games: rec.games?.length ?? 0 })) };
  await fs.writeFile(path.join(outDir, 'results.json'), JSON.stringify(slim, null, 2));
  await fs.writeFile(path.join(outDir, 'summary.md'), mdFor(result));
}
await fs.writeFile(path.join(outDir, result.fatal ? 'failed' : 'succeeded'), new Date().toISOString() + '\n');
console.log(JSON.stringify({ outDir, fatal: !!result.fatal, parallel, shards, summary: result.summary, errors: result.errors }, null, 2));
if (result.fatal) process.exitCode = 1;
