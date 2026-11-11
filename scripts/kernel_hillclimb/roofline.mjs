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
import { existsSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
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
  // Failed peak runs (validation errors, failed GEMM output checks) carry no usable peaks.
  const runs = (peakRun.results ?? [peakRun]).filter((r) => r.ok !== false);
  if (!runs.length) throw new Error('devicePeaks: no ok peak runs');
  // Median over repeats of the values that exist: an adapter without
  // shader-f16 has no f16 FMA figure, which is not an error.
  const pick = (k) => {
    const v = runs.map((r) => r.peaks?.[k]).filter(Number.isFinite);
    return v.length ? median(v) : null;
  };
  // Per-shape best GEMM (median over repeats of each repeat's best): plain WGSL
  // (what users' browsers can run today) vs subgroup-matrix (flag-only).
  const perShape = {};
  for (const [ri, r] of runs.entries()) {
    for (const [family, rows] of [
      ['plain', r.results?.gemm ?? []],
      ['sgmatrix', r.results?.subgroupMatrix?.rows ?? []],
    ]) {
      for (const row of rows) {
        if (!Number.isFinite(row.tflops) || row.check?.valid === false) continue;
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
  const fma = [pick('fmaF32Tflops'), pick('fmaF16Tflops')].filter((v) => v != null);
  const bws = [pick('readGBps'), pick('copyGBps')].filter((v) => v != null);
  if (!fma.length || !bws.length) throw new Error('devicePeaks: peak runs carry no FMA or bandwidth figure');
  const flops = Math.max(...fma) * 1e12;
  const bw = Math.max(...bws) * 1e9;
  // Subgroup-matrix ceiling: its own rows only (bestGemmTflops mixes in plain
  // WGSL, which reported a "subgroup-matrix" ceiling on adapters without it).
  const sgSquare = gemmByShape['square 2048x2048x2048']?.sgmatrixTflops ?? null;
  return { flops, bw, ridge: flops / bw, gemmCeilingSgMatrix: sgSquare != null ? sgSquare * 1e12 : null, gemmCeilingPlain: plainGemm * 1e12, gemmByShape };
}

// ORT WebGPU tiles C at 32x32 (vec4 lanes, 8x8 threads); 4..128 leaves headroom
// while rejecting 1-row tiles that would let any grid fit any small M.
const POW2_TILES = [4, 8, 16, 32, 64, 128];
const fitsTile = (n, g) => POW2_TILES.some((t) => Math.ceil(n / t) === g);

/**
 * Is a dispatch grid "XxYxZ" plausible for a node's output shape? ORT WebGPU
 * tiles C as x over N and y over rows; a 3-D batched MatMul puts the batch in
 * z, a 2-D (or batch-flattened) one folds it into the rows. Einsum and
 * indirect dispatches carry no usable shape, so they are not filtered.
 */
export function gridConsistent(grid, mnk, { exactTile = null } = {}) {
  if (!mnk || mnk.einsum || !grid || grid === 'indirect') return true;
  const [x, y, z] = String(grid).split('x').map(Number);
  if (![x, y, z].every(Number.isFinite)) return true;
  const { M, N, batch = 1 } = mnk;
  const fits = exactTile ? (n, g) => Math.ceil(n / exactTile) === g : fitsTile;
  if (z > 1) return batch === z && fits(N, x) && fits(M, y);
  return fits(N, x) && fits(M * batch, y);
}

/**
 * Attach ONNX matmul FLOPs to ORT matmul dispatches (ORT names kernels only by
 * op type). Dispatches are grouped into classes (kernel, grid, bound bytes).
 * Bound bytes are bound BUFFER sizes, which ORT's allocator rounds up into
 * buckets (Maia3 grid: 4.19 MB bound for a 3.21 MB matmul), so bytes are weak
 * evidence: a node is a candidate when its byte footprint is between
 * minRatio and 1+relTol of the bound bytes (plus absTol). The grid is the
 * strong evidence: ORT WebGPU tiles C at 32x32 (x over N, y over rows, z =
 * batch for 3-D MatMul); candidates that fit that exactly win, else any
 * power-of-two tile in 4..128 is accepted. If the candidates still disagree
 * on FLOPs, a class whose dispatch count equals exactly one candidate's node
 * count takes it (or, failing that, the only candidate with enough nodes to
 * cover the class's dispatches); otherwise the class stays unassigned (flops null,
 * ambiguous) rather than guessed: equal byte footprints can differ 16x in
 * FLOPs (BT4: 512x1024x1024 vs 256-batch 64x64x32, both 4,194,304 B).
 */
export function matchMatmuls(dispatches, nodes, { relTol = 0.1, minRatio = 0.5, absTol = 65536 } = {}) {
  const nodeClasses = new Map();
  for (const n of nodes) {
    if (!MATMUL_RE.test(n.op)) continue;
    const key = `${n.op === 'Einsum'}|${n.bytes}|${n.flops}|${JSON.stringify(n.mnk ?? null)}`;
    const c = nodeClasses.get(key) ?? { einsum: n.op === 'Einsum', bytes: n.bytes, flops: n.flops, mnk: n.mnk ?? null, count: 0, node: n };
    c.count += 1;
    nodeClasses.set(key, c);
  }
  const dispatchClasses = new Map();
  for (const d of dispatches) {
    const key = `${d.kname}|${d.wg}|${d.bindBytes}`;
    const c = dispatchClasses.get(key) ?? { kname: d.kname, wg: d.wg, bindBytes: d.bindBytes, count: 0 };
    c.count += 1;
    dispatchClasses.set(key, c);
  }
  const resolved = new Map();
  for (const [key, dc] of dispatchClasses) {
    const einsum = /einsum/i.test(dc.kname ?? '');
    const bind = dc.bindBytes || 0;
    const byBytes = [...nodeClasses.values()]
      .filter((n) => n.einsum === einsum && n.bytes >= minRatio * bind - absTol && n.bytes <= (1 + relTol) * bind + absTol)
      .sort((a, b) => Math.abs(bind - a.bytes) - Math.abs(bind - b.bytes));
    const exact = byBytes.filter((n) => !n.mnk?.einsum && n.mnk && gridConsistent(dc.wg, n.mnk, { exactTile: 32 }));
    const cands = exact.length ? exact : byBytes.filter((n) => gridConsistent(dc.wg, n.mnk));
    const distinct = (list) => new Set(list.map((n) => n.flops)).size;
    let pick = null;
    let resolvedBy = null;
    if (cands.length && distinct(cands) === 1) {
      pick = cands[0];
      resolvedBy = 'bytes+grid';
    } else if (cands.length) {
      // Count evidence: the graph has exactly as many such nodes as the class
      // has dispatches, or only one candidate has enough nodes to cover them.
      const byCount = cands.filter((n) => n.count === dc.count);
      const byCapacity = cands.filter((n) => n.count >= dc.count);
      if (byCount.length && distinct(byCount) === 1) {
        pick = byCount[0];
        resolvedBy = 'bytes+grid+count';
      } else if (byCapacity.length && distinct(byCapacity) === 1) {
        pick = byCapacity[0];
        resolvedBy = 'bytes+grid+capacity';
      }
    }
    resolved.set(key, {
      node: pick?.node ?? null,
      matchErr: pick ? Math.abs(bind - pick.bytes) / Math.max(pick.bytes, 1) : null,
      ambiguous: !pick && cands.length > 1,
      resolvedBy,
      candidates: cands.map((n) => ({ flops: n.flops, bytes: n.bytes, mnk: n.mnk, count: n.count })),
    });
  }
  const matched = dispatches.map((d) => ({ ...d, ...resolved.get(`${d.kname}|${d.wg}|${d.bindBytes}`) }));
  return {
    matched,
    ambiguousClasses: [...resolved.entries()].filter(([, r]) => r.ambiguous).map(([k, r]) => ({ class: k, candidates: r.candidates })),
    unmatchedClasses: [...resolved.entries()].filter(([, r]) => !r.node && !r.ambiguous).map(([k]) => k),
  };
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
  // A zero / missing duration (quantized or dropped timestamps) has no rate.
  if (!(meanUs > 0)) return { bound: 'unresolved', util: null };
  const attainable = Math.min(peaks.flops, ai * peaks.bw);
  const util = attainable > 0 ? Math.max(achievedFlops, 0) / attainable : 0;
  const bwUtil = achievedBw / peaks.bw;
  if (meanUs < LAUNCH_US) return { bound: 'launch', util: Math.max(util, bwUtil) };
  if (achievedFlops > 0 && util >= 0.5) return { bound: ai >= peaks.ridge ? 'compute' : 'bandwidth', util };
  if (bwUtil >= 0.5) return { bound: 'bandwidth', util: bwUtil };
  return { bound: 'underused', util: Math.max(util, bwUtil) };
}

/** Why a repeat is unusable for per-kernel attribution (null = usable). */
export function rejectReason(r) {
  if (!r.ok) return 'run not ok (gate failed or errors)';
  if (!r.profile?.sequence?.length) return 'no profile';
  if (r.profile.timestampsQuantized) return 'timestamps quantized (65.5 us)';
  if (r.profile.dropped > 0 || r.profile.complete === false) return 'profile dropped dispatches';
  return null;
}

/** Whole-report rejection: a failed --command-gate invalidates every repeat. */
export function runRejectReason(run) {
  const cg = run.gate?.commandGate;
  if (cg && cg.pass === false) return `command gate failed (${cg.command})`;
  return null;
}

export function analyzeCandidate(name, run, cost, peaks) {
  const whole = runRejectReason(run);
  if (whole) return { name, error: whole, rejected: run.results.map((_, i) => ({ repeat: i, reason: whole })) };
  const rejected = run.results.map((r, i) => ({ repeat: i, reason: rejectReason(r) })).filter((x) => x.reason);
  const results = run.results.filter((r) => !rejectReason(r));
  if (!results.length) return { name, error: `no usable profile: ${rejected.map((x) => x.reason).join('; ')}`, rejected };
  const meta = PATHS[name] ?? {};
  // Group dispatches by (name, grid, bound bytes); average over repeats.
  const groups = new Map();
  let macsByFn = null;
  if (results[0].candidate?.runtime === 'tvmjs-webgpu') macsByFn = probeMacs(results[0].candidate.manifest);
  const ambiguous = new Map();
  for (const [ri, r] of results.entries()) {
    const seq = r.profile.sequence.map(([kname, us, wg, bindBytes], i) => ({ i, kname, us, wg, bindBytes }));
    const mm = seq.filter((d) => MATMUL_RE.test(d.kname));
    const isOrt = r.candidate?.runtime === 'ort-webgpu';
    const m = isOrt && cost ? matchMatmuls(mm, cost.nodes) : { matched: mm.map((d) => ({ ...d, node: null })), ambiguousClasses: [] };
    for (const a of m.ambiguousClasses) ambiguous.set(a.class, a.candidates);
    const flopsAt = new Map(
      m.matched.map((d) => [d.i, { flops: d.node?.flops ?? null, mnk: d.node?.mnk ?? null, matchErr: d.matchErr ?? null, ambiguous: !!d.ambiguous }]),
    );
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
        ambiguous: false,
      };
      g.samples.push(d.us);
      g.perRepeat.set(ri, (g.perRepeat.get(ri) ?? 0) + 1);
      const f = flopsAt.get(d.i);
      if (f?.ambiguous) g.ambiguous = true;
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
    const timed = meanUs > 0;
    const achievedBw = timed ? bytes / (meanUs * 1e-6) : null;
    const achievedFlops = timed && g.flops ? g.flops / (meanUs * 1e-6) : timed ? 0 : null;
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
      tflops: achievedFlops == null ? null : achievedFlops / 1e12,
      gbps: achievedBw == null ? null : achievedBw / 1e9,
      pctPeakFlops: achievedFlops == null ? null : (100 * achievedFlops) / peaks.flops,
      pctPeakBw: achievedBw == null ? null : (100 * achievedBw) / peaks.bw,
      ai,
      ...cls,
      matchErr: g.matchErr,
      ambiguous: g.ambiguous,
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
  // FLOPs the matcher could attach per invoke vs the graph's matmul FLOPs: a
  // coverage check on the byte matcher (ambiguous classes stay unassigned).
  const mmFlopsAttached = rows.filter((r) => MATMUL_RE.test(r.kernel) && r.flops).reduce((s, r) => s + r.flops * r.dispatchesPerInvoke, 0);
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
      matmulFlopsAttachedPct: mmFlopsCost ? (100 * mmFlopsAttached) / mmFlopsCost : null,
      matmulTflops: mmFlopsCost && mmMs ? mmFlopsCost / (mmMs * 1e-3) / 1e12 : null,
      matmulPctPlainGemmCeiling: mmFlopsCost && mmMs ? (100 * mmFlopsCost) / (mmMs * 1e-3) / peaks.gemmCeilingPlain : null,
      weightMB: cost ? cost.totals.weightBytes / 1e6 : null,
      timeByBoundMs: { ...byBound, nonKernel: nonKernelMs ?? wallMs - busyMs },
    },
    perPositionMs: batch ? wallMs / batch : null,
    parity: results.map((r) => r.parity),
    repeatsUsed: results.length,
    rejected,
    ambiguousMatches: [...ambiguous.entries()].map(([k, c]) => ({ class: k, candidates: c })),
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
  const failed = report.candidates.filter((c) => c.error);
  if (failed.length) {
    lines.push('', 'Rejected (not in the table):');
    for (const c of failed) lines.push(`- ${c.name}: REJECTED, ${c.error}`);
  }
  for (const c of report.candidates) {
    if (c.error) continue;
    lines.push('', `### ${c.name} (${c.model}, b${c.batch})`, '');
    const notes = [`repeats used ${c.repeatsUsed}/${c.repeatsUsed + c.rejected.length}`];
    for (const x of c.rejected) notes.push(`repeat ${x.repeat} rejected: ${x.reason}`);
    if (c.perInvoke.matmulFlopsAttachedPct != null) notes.push(`matmul FLOPs attached to kernels: ${fmt(c.perInvoke.matmulFlopsAttachedPct, 1)}%`);
    if (c.ambiguousMatches.length) notes.push(`ambiguous (left unassigned): ${c.ambiguousMatches.map((a) => a.class).join(', ')}`);
    lines.push(notes.join('; '), '');
    lines.push('| kernel | grid | n/inv | µs | ms/inv | M×N×K (batch) | MFLOP | KB bound | TFLOP/s | GB/s | %FLOP | %BW | AI | bound |');
    lines.push('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|');
    for (const r of c.kernels.slice(0, 14)) {
      const mnk =
        r.ambiguous && !r.flops
          ? 'ambiguous'
          : (r.mnk?.einsum ?? (r.mnk ? `${r.mnk.M}×${r.mnk.N}×${r.mnk.K}${r.mnk.batch > 1 ? ` (${r.mnk.batch})` : ''}` : ''));
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
      peak: { type: 'string', default: 'artifacts/kernel-hillclimb/results/peak-dev.json' },
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
