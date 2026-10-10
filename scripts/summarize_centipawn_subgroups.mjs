#!/usr/bin/env node
import { readFileSync, writeFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';

export function statistics(values) {
  if (!values.length || values.some((v) => !Number.isFinite(v))) throw new Error('Invalid samples');
  const s = [...values].sort((a, b) => a - b),
    middle = s.length >> 1;
  return {
    count: s.length,
    median: s.length % 2 ? s[middle] : (s[middle - 1] + s[middle]) / 2,
    p10: s[Math.floor((s.length - 1) * 0.1)],
    p90: s[Math.floor((s.length - 1) * 0.9)],
    min: s[0],
    max: s.at(-1),
  };
}
export function summarizeRun(run) {
  const saving = (rows) => statistics(rows.map((r) => 100 * (1 - r.results[1].wallMs / r.results[0].wallMs)));
  const variants = run.invokes.map((rows, k) => ({
    variant: run.artifacts[k].version,
    modelSha256: run.artifacts[k].manifest.models[0].sha256,
    gpuBusyMs: statistics(rows.map((r) => r.busyMs)),
    gpuEnvelopeMs: statistics(rows.map((r) => r.envelopeMs)),
    invokeWallMs: statistics(rows.map((r) => r.wallMs)),
    zeroPasses: statistics(rows.map((r) => r.zeroPasses)),
    searchWallMs: statistics(run.search.map((r) => r.results[k].wallMs)),
    kernels: rows[0].kernels
      ? Object.fromEntries(
          Object.keys(rows[0].kernels).map((name) => [
            name,
            {
              dispatchesPerInvoke: rows[0].kernels[name].count,
              perInvokeGpuMs: statistics(rows.map((r) => r.kernels[name].gpuMs)),
            },
          ]),
        )
      : undefined,
  }));
  return {
    date: run.date,
    visits: run.visits,
    orderOffset: run.orderOffset ?? 0,
    batch: run.batch,
    variants,
    pairedSearchSavingPercent: saving(run.search),
    fixtures: [...new Set(run.search.map((r) => r.fixture))].map((fixture) => {
      const rows = run.search.filter((r) => r.fixture === fixture);
      return { fixture, wallMs: [0, 1].map((k) => statistics(rows.map((r) => r.results[k].wallMs))), pairedSavingPercent: saving(rows) };
    }),
  };
}
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const root = 'lab/centipawn-tvmjs-subgroups-results/';
  const summary = Object.fromEntries([1, 2, 3, 4].map((n) => [`run${n}`, summarizeRun(JSON.parse(readFileSync(`${root}run${n}.json`)))]));
  writeFileSync(`${root}summary.json`, `${JSON.stringify(summary, null, 2)}\n`);
  console.log(JSON.stringify(Object.fromEntries(Object.entries(summary).map(([k, v]) => [k, v.pairedSearchSavingPercent])), null, 2));
}
