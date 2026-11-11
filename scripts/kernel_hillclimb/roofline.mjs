#!/usr/bin/env node
// Join measured per-dispatch GPU profiles (bench pages) with ONNX FLOP/byte
// costs (onnx_cost.py) and the device peaks (peak page) into a per-kernel
// roofline. Output: JSON + a markdown table.
//
// Usage:
//   node scripts/kernel_hillclimb/roofline.mjs \
//     [--results artifacts/kernel-hillclimb/results] [--cost artifacts/kernel-hillclimb/cost] \
//     [--peak peak-run.json[,more.json]] [--out artifacts/kernel-hillclimb/roofline.json] [--md PATH]
//
// Bound classification per kernel group (documented in scripts/kernel_hillclimb/README.md):
//   launch      mean dispatch < LAUNCH_US: fixed per-dispatch cost dominates
//   compute     >= 50% of the attainable roof and AI >= ridge
//   bandwidth   >= 50% of the attainable roof and AI <  ridge
//   underused   neither roof reached: occupancy / kernel-quality limited (WGSL target)
import { existsSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { median } from '../lib/perf_canary_stats.mjs';

const repo = resolve(dirname(fileURLToPath(import.meta.url)), '../..');

export const LAUNCH_US = 8;
export const MATMUL_RE = /^(Gemm|MatMul|FusedMatMul|MatMulNBits|Einsum|Conv|FusedConv)$|matmul|dense|conv2d/i;

/** Which ONNX cost file and search unit belong to each candidate. */
export const PATHS = {
  'ort-lc0-small': { cost: 'lc0-small', shipped: true, unit: 'visit (b1: 1 invoke per NN eval)' },
  'ort-lc0-small-b8': { cost: 'lc0-small-b8', shipped: false, unit: '8 visits' },
  'ort-t3': { cost: 't3', shipped: true, unit: '8 visits' },
  'ort-bt4': { cost: 'bt4', shipped: true, unit: '8 visits' },
  'ort-lqo': { cost: 'lqo', shipped: true, unit: '8 visits' },
  'ort-maia3': { cost: 'maia3', shipped: true, unit: '1 move (1 invoke)' },
  'ort-maia3-grid': { cost: 'maia3-grid', shipped: true, unit: '11-condition Elo grid' },
  'tvmjs-centipawn': { cost: 'centipawn', shipped: true, unit: '16 visits' },
  'tvmjs-centipawn-v3-subgroups': { cost: 'centipawn', shipped: false, unit: '16 visits' },
  'tvmjs-lc0-t1-b8': { cost: 'lc0-small-b8', shipped: false, unit: '8 visits' },
  'tvmjs-lc0-t1-b16': { cost: 'lc0-t1-b16', shipped: false, unit: '16 visits' },
  'tvmjs-lc0-bt4-b8': { cost: 'bt4', shipped: false, unit: '8 visits' },
};

const readJson = (p) => JSON.parse(readFileSync(p, 'utf8'));

export function devicePeaks(peakRun) {
  const runs = peakRun.results ?? [peakRun];
  const pick = (k) => median(runs.map((r) => r.peaks?.[k]).filter(Number.isFinite));
  const square = runs.map((r) => r.peaks?.bestGemmTflops?.['square 2048x2048x2048']).filter(Number.isFinite);
  // Per-shape best GEMM (median over repeats of each repeat's best): plain WGSL
  // (what users' browsers can run today) vs subgroup-matrix (flag-only).
  const perShape = {};
  for (const [ri, r] of runs.entries()) {
    for (const [family, rows] of [
      ['plain', r.results?.gemm ?? []],
      ['sgmatrix', r.results?.subgroupMatrix?.rows ?? []],
    ]) {
      for (const row of rows) {
        if (!Number.isFinite(row.tflops)) continue;
        if (!perShape[row.shape]) perShape[row.shape] = { plain: [], sgmatrix: [] };
        const s = perShape[row.shape];
        s[family][ri] = Math.max(s[family][ri] ?? 0, row.tflops);
      }
    }
  }
  const gemmByShape = Object.fromEntries(
    Object.entries(perShape).map(([shape, v]) => [
      shape,
      {
        plainTflops: v.plain.length ? median(v.plain.filter(Number.isFinite)) : null,
        sgmatrixTflops: v.sgmatrix.length ? median(v.sgmatrix.filter(Number.isFinite)) : null,
      },
    ]),
  );
  const plainGemm = Math.max(0, ...Object.values(gemmByShape).map((v) => v.plainTflops ?? 0));
  const flops = Math.max(pick('fmaF32Tflops'), pick('fmaF16Tflops')) * 1e12;
  const bw = Math.max(pick('readGBps'), pick('copyGBps')) * 1e9;
  return { flops, bw, ridge: flops / bw, gemmCeilingSgMatrix: square.length ? median(square) * 1e12 : null, gemmCeilingPlain: plainGemm * 1e12, gemmByShape };
}

/** Greedy byte match of matmul dispatches to ONNX matmul nodes. */
export function matchMatmuls(dispatches, nodes) {
  const pool = nodes.filter((n) => MATMUL_RE.test(n.op)).map((n) => ({ ...n, used: false }));
  const out = [];
  for (const d of dispatches) {
    let best = null;
    let bestErr = Infinity;
    const einsum = /einsum/i.test(d.kname ?? '');
    for (const n of pool) {
      if (n.used || (n.op === 'Einsum') !== einsum) continue;
      const err = Math.abs(Math.log((d.bindBytes || 1) / (n.bytes || 1)));
      if (err < bestErr) {
        bestErr = err;
        best = n;
      }
    }
    if (best) best.used = true;
    out.push({ ...d, node: best, matchErr: best ? Math.exp(bestErr) - 1 : null });
  }
  return { matched: out, unmatchedNodes: pool.filter((n) => !n.used).length };
}

function probeMacs(manifestPath) {
  // TVMJS shape-k16 probes record per-function MACs; find one next to the manifest.
  if (!manifestPath) return null;
  const dir = join(repo, 'public', dirname(manifestPath));
  if (!existsSync(dir)) return null;
  const probe = readdirSync(dir).find((f) => f.endsWith('.probe.json'));
  if (!probe) return null;
  const fns = readJson(join(dir, probe)).dlight_matmul_dispatch?.functions ?? {};
  const macs = {};
  for (const [name, v] of Object.entries(fns)) if (v?.shape?.macs) macs[name] = v.shape.macs;
  return macs;
}

export function classify({ meanUs, achievedFlops, achievedBw, ai }, peaks) {
  const attainable = Math.min(peaks.flops, ai * peaks.bw);
  const util = attainable > 0 ? Math.max(achievedFlops, 0) / attainable : 0;
  const bwUtil = achievedBw / peaks.bw;
  if (meanUs < LAUNCH_US) return { bound: 'launch', util: Math.max(util, bwUtil) };
  if (achievedFlops > 0 && util >= 0.5) return { bound: ai >= peaks.ridge ? 'compute' : 'bandwidth', util };
  if (bwUtil >= 0.5) return { bound: 'bandwidth', util: bwUtil };
  return { bound: 'underused', util: Math.max(util, bwUtil) };
}

export function analyzeCandidate(name, run, cost, peaks) {
  const results = run.results.filter((r) => r.profile?.sequence?.length);
  if (!results.length) return { name, error: 'no profile' };
  const meta = PATHS[name] ?? {};
  // Group dispatches by (name, grid, bound bytes); average over repeats.
  const groups = new Map();
  let macsByFn = null;
  if (results[0].candidate?.runtime === 'tvmjs-webgpu') macsByFn = probeMacs(results[0].candidate.manifest);
  for (const [ri, r] of results.entries()) {
    const seq = r.profile.sequence.map(([kname, us, wg, bindBytes], i) => ({ i, kname, us, wg, bindBytes }));
    const mm = seq.filter((d) => MATMUL_RE.test(d.kname));
    const isOrt = r.candidate?.runtime === 'ort-webgpu';
    const { matched } = isOrt && cost ? matchMatmuls(mm, cost.nodes) : { matched: mm.map((d) => ({ ...d, node: null })) };
    const flopsAt = new Map(matched.map((d) => [d.i, { flops: d.node?.flops ?? null, mnk: d.node?.mnk ?? null, matchErr: d.matchErr ?? null }]));
    for (const d of seq) {
      const key = `${d.kname}|${d.wg}|${d.bindBytes}`;
      const g = groups.get(key) ?? {
        kernel: d.kname,
        grid: d.wg,
        bindBytes: d.bindBytes,
        samples: [],
        perRepeat: new Map(),
        flops: null,
        mnk: null,
        matchErr: null,
      };
      g.samples.push(d.us);
      g.perRepeat.set(ri, (g.perRepeat.get(ri) ?? 0) + 1);
      const f = flopsAt.get(d.i);
      if (f?.flops != null) {
        g.flops = f.flops;
        g.mnk = f.mnk;
        g.matchErr = f.matchErr;
      } else if (macsByFn) {
        const fn = d.kname.replace(/_kernel(_\d+)?$/, '');
        if (macsByFn[fn]) g.flops = 2 * macsByFn[fn];
      }
      groups.set(key, g);
    }
  }
  const rows = [...groups.values()].map((g) => {
    const perInvoke = median([...g.perRepeat.values()]);
    const meanUs = median(g.samples); // median: robust to contention spikes on a shared GPU
    const msPerInvoke = (meanUs * perInvoke) / 1000;
    const bytes = g.bindBytes;
    const achievedBw = bytes / (meanUs * 1e-6);
    const achievedFlops = g.flops ? g.flops / (meanUs * 1e-6) : 0;
    const ai = g.flops ? g.flops / bytes : 0;
    const cls = classify({ meanUs, achievedFlops, achievedBw, ai }, peaks);
    return {
      kernel: g.kernel,
      grid: g.grid,
      dispatchesPerInvoke: perInvoke,
      meanUs,
      msPerInvoke,
      flops: g.flops,
      mnk: g.mnk,
      bytes,
      tflops: achievedFlops / 1e12,
      gbps: achievedBw / 1e9,
      pctPeakFlops: (100 * achievedFlops) / peaks.flops,
      pctPeakBw: (100 * achievedBw) / peaks.bw,
      ai,
      ...cls,
      matchErr: g.matchErr,
    };
  });
  rows.sort((a, b) => b.msPerInvoke - a.msPerInvoke);
  const splitBusyMs = median(results.map((r) => r.profile.kernelMsPerInvoke));
  // Pass mode (runtime's own passes timestamped, no split) is the true busy time
  // when present; split-mode sums carry ~1 µs/dispatch + serialization inflation.
  const pm = results.map((r) => r.profile.passMode).filter(Boolean);
  // Attribution basis is the split-mode kernel sum. Non-kernel wall: for runtimes
  // with few passes (ORT batches ~16 dispatches/pass) use pass mode on the same
  // invokes (wall - busy); when every dispatch is its own pass (TVMJS), pass
  // timestamps themselves inflate the wall (~25-50 us/pass on Apple), so use the
  // unprofiled wall minus the split kernel sum instead.
  const busyMs = splitBusyMs;
  const passEnvelopeMs = pm.length ? median(pm.map((p) => p.envelopeMs)) : null;
  const passGapMs = pm.length ? median(pm.map((p) => p.gapMs)) : null;
  // Decompose the wall of the same invokes the pass timestamps came from.
  const pmWallMs = pm.length ? median(pm.map((p) => p.wallMs)) : null;
  const pmBusyMs = pm.length ? median(pm.map((p) => p.busyMs)) : null;
  const pmPasses = pm.length ? median(pm.map((p) => p.passesPerInvoke)) : null;
  const usePassMode = pm.length && pmPasses <= 100;
  const nonKernelMs = usePassMode ? pmWallMs - pmBusyMs : null;
  const wallMs = median(results.map((r) => r.timing.runMs.median));
  const envelopeMs = median(results.map((r) => median(r.profile.perInvoke.map((p) => p.envelopeMs))));
  const dispatches = median(results.map((r) => median(r.profile.perInvoke.map((p) => p.dispatches))));
  const byBound = {};
  for (const r of rows) byBound[r.bound] = (byBound[r.bound] ?? 0) + r.msPerInvoke;
  const rowSum = rows.reduce((s, r) => s + r.msPerInvoke, 0);
  // Scale group sums (invoke-1 samples) to the all-invoke busy mean.
  const scale = rowSum > 0 ? busyMs / rowSum : 1;
  for (const k of Object.keys(byBound)) byBound[k] *= scale;
  const mmRows = rows.filter((r) => MATMUL_RE.test(r.kernel));
  const mmMs = mmRows.reduce((s, r) => s + r.msPerInvoke, 0) * scale;
  const mmFlopsCost = cost ? cost.nodes.filter((n) => MATMUL_RE.test(n.op)).reduce((s, n) => s + n.flops, 0) : null;
  const totalFlops = cost?.totals.flops ?? null;
  const batch = results[0].candidate?.batch ?? null;
  return {
    name,
    shipped: meta.shipped ?? null,
    runtime: results[0].candidate?.runtime,
    model: results[0].candidate?.model ?? results[0].candidate?.manifest,
    batch,
    searchUnit: meta.unit ?? null,
    perInvoke: {
      wallMs,
      wallMsSamples: results.map((r) => r.timing.runMs.median),
      kernelBusyMs: busyMs,
      splitBusyMs,
      splitEnvelopeMs: envelopeMs,
      gpuEnvelopeMs: passEnvelopeMs,
      gpuIdleInEnvelopeMs: passGapMs,
      passModeWallMs: pmWallMs,
      outsideGpuEnvelopeMs: passEnvelopeMs != null ? pmWallMs - passEnvelopeMs : null,
      nonKernelMs: nonKernelMs ?? wallMs - busyMs,
      nonKernelMethod: usePassMode ? 'pass-mode wall - pass-mode busy (same invokes)' : 'unprofiled wall - split kernel sum',
      passModeBusyMs: pmBusyMs,
      passesPerInvoke: pm.length ? median(pm.map((p) => p.passesPerInvoke)) : null,
      dispatches,
      launchFloorMs: (dispatches * 0.85) / 1000,
      submits: median(results.map((r) => r.profile.submitsPerInvoke)),
      gflops: totalFlops ? totalFlops / 1e9 : null,
      achievedTflopsWall: totalFlops ? totalFlops / (wallMs * 1e-3) / 1e12 : null,
      pctPeakWall: totalFlops ? (100 * totalFlops) / (wallMs * 1e-3) / peaks.flops : null,
      matmulMs: mmMs,
      matmulTflops: mmFlopsCost && mmMs ? mmFlopsCost / (mmMs * 1e-3) / 1e12 : null,
      matmulPctPlainGemmCeiling: mmFlopsCost && mmMs ? (100 * mmFlopsCost) / (mmMs * 1e-3) / peaks.gemmCeilingPlain : null,
      weightMB: cost ? cost.totals.weightBytes / 1e6 : null,
      timeByBoundMs: { ...byBound, nonKernel: nonKernelMs ?? wallMs - busyMs },
    },
    perPositionMs: batch ? wallMs / batch : null,
    parity: results.map((r) => r.parity),
    ok: run.ok,
    loads: run.loads,
    uptime: run.host?.uptime,
    kernels: rows,
  };
}

function fmt(v, d = 2) {
  return v == null || !Number.isFinite(v) ? '-' : v.toFixed(d);
}

export function toMarkdown(report) {
  const p = report.peaks;
  const lines = [];
  lines.push(
    `Peaks: FMA ${fmt(p.flops / 1e12)} TFLOP/s, BW ${fmt(p.bw / 1e9, 0)} GB/s, ridge ${fmt(p.ridge, 1)} FLOP/B, plain-WGSL GEMM ceiling ${fmt(p.gemmCeilingPlain / 1e12)} TFLOP/s, subgroup-matrix GEMM ${fmt((p.gemmCeilingSgMatrix ?? NaN) / 1e12)} TFLOP/s (flag-only).`,
    '',
  );
  lines.push(
    '| path | shipped | batch | wall ms | GPU kernel ms | non-kernel ms | GPU idle in envelope ms | dispatches | passes | GFLOP | TFLOP/s (wall) | % FMA peak | matmul ms | matmul TFLOP/s | launch ms | underused ms | bw ms | compute ms |',
  );
  lines.push('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|');
  for (const c of report.candidates) {
    if (c.error) continue;
    const q = c.perInvoke;
    const b = q.timeByBoundMs;
    lines.push(
      `| ${c.name} | ${c.shipped ? 'yes' : 'no'} | ${c.batch} | ${fmt(q.wallMs)} | ${fmt(q.kernelBusyMs)} | ${fmt(q.nonKernelMs)} | ${fmt(q.gpuIdleInEnvelopeMs)} | ${q.dispatches} | ${q.passesPerInvoke ?? '-'} | ${fmt(q.gflops)} | ${fmt(q.achievedTflopsWall)} | ${fmt(q.pctPeakWall, 1)} | ${fmt(q.matmulMs)} | ${fmt(q.matmulTflops)} | ${fmt(b.launch)} | ${fmt(b.underused)} | ${fmt(b.bandwidth)} | ${fmt(b.compute)} |`,
    );
  }
  for (const c of report.candidates) {
    if (c.error) continue;
    lines.push('', `### ${c.name} (${c.model}, b${c.batch})`, '');
    lines.push('| kernel | grid | n/inv | µs | ms/inv | M×N×K (batch) | MFLOP | KB bound | TFLOP/s | GB/s | %FLOP | %BW | AI | bound |');
    lines.push('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|');
    for (const r of c.kernels.slice(0, 14)) {
      const mnk = r.mnk?.einsum ?? (r.mnk ? `${r.mnk.M}×${r.mnk.N}×${r.mnk.K}${r.mnk.batch > 1 ? ` (${r.mnk.batch})` : ''}` : '');
      lines.push(
        `| ${r.kernel.slice(0, 48)} | ${r.grid} | ${r.dispatchesPerInvoke} | ${fmt(r.meanUs, 1)} | ${fmt(r.msPerInvoke, 3)} | ${mnk} | ${r.flops ? fmt(r.flops / 1e6, 1) : '-'} | ${fmt(r.bytes / 1024, 0)} | ${fmt(r.tflops, 3)} | ${fmt(r.gbps, 1)} | ${fmt(r.pctPeakFlops, 1)} | ${fmt(r.pctPeakBw, 1)} | ${fmt(r.ai, 1)} | ${r.bound} |`,
      );
    }
  }
  return lines.join('\n');
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const { values: opt } = parseArgs({
    options: {
      results: { type: 'string', default: 'artifacts/kernel-hillclimb/results' },
      cost: { type: 'string', default: 'artifacts/kernel-hillclimb/cost' },
      peak: { type: 'string', default: 'artifacts/kernel-hillclimb/results/peak-dev.json,artifacts/kernel-hillclimb/results/peak-dev-r3.json' },
      out: { type: 'string', default: 'artifacts/kernel-hillclimb/roofline.json' },
      md: { type: 'string' },
    },
  });
  // --peak may list several bench-run files (comma-separated); repeats are pooled.
  const peaks = devicePeaks({ results: opt.peak.split(',').flatMap((f) => readJson(resolve(repo, f)).results) });
  const dir = resolve(repo, opt.results);
  const candidates = [];
  for (const name of Object.keys(PATHS)) {
    const file = join(dir, `${name}.json`);
    if (!existsSync(file)) continue;
    const costFile = join(resolve(repo, opt.cost), `${PATHS[name].cost}.json`);
    const cost = existsSync(costFile) ? readJson(costFile) : null;
    candidates.push(analyzeCandidate(name, readJson(file), cost, peaks));
  }
  const report = { schema: 'kernel-hillclimb.roofline.v1', generatedAt: new Date().toISOString(), peaks, launchUs: LAUNCH_US, candidates };
  writeFileSync(resolve(repo, opt.out), `${JSON.stringify(report, null, 1)}\n`);
  const md = toMarkdown(report);
  if (opt.md) writeFileSync(resolve(repo, opt.md), `${md}\n`);
  console.log(md);
}
