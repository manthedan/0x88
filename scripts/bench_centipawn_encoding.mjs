#!/usr/bin/env node
// Run under the machine's benchlock. No GPU, model inference or promise/queue
// latency is timed: measure the actual evaluatePhysicalBatch callback from entry
// until its first VM getFunction call, immediately after preprocessing.
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readFile, unlink, writeFile } from 'node:fs/promises';
import os from 'node:os';
import { performance } from 'node:perf_hooks';
import { parseArgs } from 'node:util';
import { SquareformerTvmjsWebgpuEvaluator as After } from '../src/nn/squareformerTvmjsWebgpuEvaluator.ts';
import { encodingBatch } from '../tests/fixtures/centipawn_encoding.mjs';

const { values } = parseArgs({
  options: {
    'baseline-ref': { type: 'string', default: '5761490' },
    rounds: { type: 'string', default: '12' },
    iterations: { type: 'string', default: '200' },
    warmup: { type: 'string', default: '500' },
    out: { type: 'string' },
  },
});
const rounds = Number(values.rounds);
const iterations = Number(values.iterations);
const warmup = Number(values.warmup);
for (const [name, value] of Object.entries({ rounds, iterations, warmup })) {
  assert.ok(Number.isInteger(value) && value > 0, `${name} must be a positive integer`);
}
const evaluatorPath = 'src/nn/squareformerTvmjsWebgpuEvaluator.ts';
const baselineCommit = execFileSync('git', ['rev-parse', values['baseline-ref']], { encoding: 'utf8' }).trim();
const baselineSource = execFileSync('git', ['show', `${baselineCommit}:${evaluatorPath}`], { encoding: 'utf8' });
// Keep relative dependencies identical to the current evaluator. Temporary file
// lives only in this worktree and is removed even if the measurement fails.
const baselineUrl = new URL(`../src/nn/.centipawn-encoding-baseline-${process.pid}.ts`, import.meta.url);
const metaBytes = await readFile(new URL('../public/models/bt4_soap_rem_c19000_final.meta.json', import.meta.url));
const meta = JSON.parse(metaBytes);
const stop = new Error('preprocessing complete');
function probe(Evaluator) {
  let started = 0;
  let elapsed = 0;
  const evaluator = new Evaluator(
    {},
    { destroy() {} },
    {
      getFunction() {
        elapsed = performance.now() - started;
        throw stop;
      },
    },
    meta,
    16,
  );
  // Replace only serialization; run the unchanged production callback.
  evaluator.runExclusive = (fn) => {
    started = performance.now();
    return fn();
  };
  return async (boards, contexts) => {
    try {
      await evaluator.evaluatePhysicalBatch(boards, contexts);
      throw new Error('VM boundary was not reached');
    } catch (error) {
      if (error !== stop) throw error;
    }
    return elapsed;
  };
}
function summary(samples) {
  const sorted = [...samples].sort((a, b) => a - b);
  const medianMs = (sorted[Math.floor((sorted.length - 1) / 2)] + sorted[Math.floor(sorted.length / 2)]) / 2;
  return { medianMs, minMs: sorted[0], maxMs: sorted.at(-1), samplesMs: samples };
}
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
await writeFile(baselineUrl, baselineSource, { flag: 'wx' });
try {
  const { SquareformerTvmjsWebgpuEvaluator: Before } = await import(baselineUrl.href);
  const probes = { before: probe(Before), after: probe(After) };
  const results = [];
  for (const workload of ['start-white-no-history', 'black-seven-history-masked']) {
    for (const batch of [1, 4, 8, 16]) {
      const { boards, contexts } = encodingBatch(batch, workload === 'start-white-no-history' ? 0 : 1, workload.endsWith('masked'));
      if (workload === 'start-white-no-history') {
        for (const context of contexts) context.historyFens = [];
      } else {
        for (let index = 0; index < contexts.length; index++) {
          // Every real row has seven history plies; b1 is Black to move.
          contexts[index].historyFens = encodingBatch(1, 1).contexts[0].historyFens;
        }
      }
      for (let index = 0; index < warmup; index++) {
        await probes.before(boards, contexts);
        await probes.after(boards, contexts);
      }
      const samples = { before: [], after: [] };
      const orders = [];
      for (let round = 0; round < rounds; round++) {
        const order = round % 2 ? ['after', 'before'] : ['before', 'after'];
        orders.push(order.join('/'));
        for (const variant of order) {
          let total = 0;
          for (let index = 0; index < iterations; index++) total += await probes[variant](boards, contexts);
          samples[variant].push(total / iterations);
        }
      }
      const before = summary(samples.before);
      const after = summary(samples.after);
      const row = { workload, logicalBatch: batch, physicalBatch: 16, orders, before, after, speedup: before.medianMs / after.medianMs };
      results.push(row);
      process.stderr.write(`${workload} b${batch}: ${before.medianMs.toFixed(4)} -> ${after.medianMs.toFixed(4)} ms (${row.speedup.toFixed(2)}x)\n`);
    }
  }
  const report = {
    schema: 'centipawn-padded-encoding-benchmark.v1',
    generatedAt: new Date().toISOString(),
    baselineCommit,
    baselineSourceSha256: sha256(baselineSource),
    afterSourceSha256: sha256(await readFile(new URL(`../${evaluatorPath}`, import.meta.url))),
    metadataSha256: sha256(metaBytes),
    model: 'bt4-soap-rem-c19000-final',
    node: process.version,
    platform: `${os.platform()} ${os.release()} ${os.arch()}`,
    cpu: os.cpus()[0].model,
    rounds,
    iterations,
    warmup,
    timingBoundary:
      'evaluatePhysicalBatch exclusive callback entry through first vmModule.getFunction; includes normalization, allocation, encoding, masks and padding; excludes GPU and queue wait',
    results,
  };
  if (values.out) await writeFile(values.out, `${JSON.stringify(report, null, 2)}\n`);
  console.log(JSON.stringify(report, null, 2));
} finally {
  await unlink(baselineUrl);
}
