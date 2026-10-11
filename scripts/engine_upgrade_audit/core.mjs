import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { inspectWasmSimd } from '../inspect_wasm_simd.mjs';
import { compareBench, hasFailures, simdStatus } from './parsers.mjs';
import { loadProvenance, provenanceGates, upstreamGate } from './provenance.mjs';
import { parityGate } from './parity.mjs';
import { Unavailable } from './source.mjs';
import stockfish from './stockfish.mjs';
import berserk from './berserk.mjs';
import viridithas from './viridithas.mjs';
import plentychess from './plentychess.mjs';
import stormphrax from './stormphrax.mjs';
import reckless from './reckless.mjs';

export const adapters = { stockfish, berserk, viridithas, plentychess, stormphrax, reckless };
export const usage = `Usage: npm run engine:upgrade-audit -- <engine> [options]
Engines: ${Object.keys(adapters).join(', ')}
  --artifacts <public/engine>   read-only candidate directory (uses its sibling lock)
  --lock <path> --manifest <path> --bench-source <path>   explicit provenance/corpus
  --native-cmd "<cmd>"          same full bench through trusted shell command
  --native-bench N              explicit native node total (no per-position reference)
  --native-lite-cmd "<cmd>"     Stockfish.js lite reference (different network)
  --native-lite-bench N         Stockfish lite explicit depth-10 node total
  --upstream                   allow git tag/locked commit queries (temp clone only)
  --baseline-diverged          record bench DIFF as WARN, never hide it
  --parity                     existing rotated-FEN harness after correctness gates
  --json <path> --timeout <ms>  gate matrix; per-process timeout (default 600000)
  --nnue <path>                Reckless external-WASI network
  --help`;

export function parseArgs(args) {
  const options = { timeout: 600000 };
  const flags = { '--upstream': 'upstream', '--baseline-diverged': 'baselineDiverged', '--parity': 'parity', '--help': 'help' };
  const values = { '--artifacts': 'artifacts', '--lock': 'lock', '--manifest': 'manifest', '--bench-source': 'benchSource',
    '--native-cmd': 'nativeCmd', '--native-lite-cmd': 'nativeLiteCmd', '--native-bench': 'nativeBench',
    '--native-lite-bench': 'nativeLiteBench', '--json': 'json', '--timeout': 'timeout', '--nnue': 'nnue' };
  let engine;
  for (let i = 0; i < args.length; i += 1) {
    const arg = args[i];
    if (flags[arg]) options[flags[arg]] = true;
    else if (values[arg]) {
      const value = args[++i];
      if (!value || value.startsWith('--')) throw new Error(`Missing value for ${arg}`);
      options[values[arg]] = value;
    } else if (!engine && !arg.startsWith('-')) engine = arg;
    else throw new Error(`Unknown argument: ${arg}`);
  }
  if (!options.help && !adapters[engine]) throw new Error(`Unknown engine: ${engine ?? '(missing)'}`);
  for (const key of ['nativeBench', 'nativeLiteBench', 'timeout']) {
    if (options[key] === undefined) continue;
    options[key] = Number(options[key]);
    if (!Number.isSafeInteger(options[key]) || options[key] <= 0) throw new Error(`Invalid ${key}: expected a positive safe integer`);
  }
  return { engine, options };
}

const skip = (name, detail, data = {}) => ({ name, status: 'SKIP', detail, data });
const fail = (name, error, data = {}) => ({ name, status: 'FAIL', detail: error.message, data });

export function nativeUnavailable(error) {
  // Only OS executable-lookup failures are known to be missing executables.
  // Engine diagnostics (e.g. "network not found") are runtime failures. A
  // supplied remote/shell command exiting 127 isn't an attested lookup error.
  return error.code === 'ENOENT' && String(error.syscall ?? '').startsWith('spawn ');
}

export function selectReference(commandReference, explicit, commitBench, commitCompatible = true, explicitSource = '--native-bench') {
  if (commandReference) return commandReference;
  if (explicit !== undefined) return { nodes: explicit, source: explicitSource };
  if (commitBench && commitCompatible) return { nodes: commitBench, source: 'locked commit Bench:' };
  return null;
}

export async function audit(engine, options = {}) {
  options = { timeout: 600000, ...options };
  const report = { engine, ref: null, runtime: process.version, platform: `${process.platform}/${process.arch}`, gates: [] };
  const adapter = adapters[engine];
  if (!adapter) throw new Error(`Unknown engine: ${engine}`);
  const context = { engine, options, artifacts: resolve(options.artifacts ?? `public/${engine}`), tmp: mkdtempSync(join(tmpdir(), 'engine-upgrade-audit-')) };
  const gates = report.gates;
  try {
    let tiers;
    try {
      context.provenance = loadProvenance(engine, context.artifacts, options);
      report.ref = context.provenance.upstream.ref;
      report.artifacts = context.artifacts;
      gates.push(...provenanceGates(context));
      tiers = adapter.tiers(context);
      if (!tiers.length) throw new Error('No WASM tiers found');
    } catch (error) {
      gates.push(fail('provenance', error), skip('simd', 'blocked by provenance failure'), skip('bench', 'blocked by provenance failure'), skip('parity', 'blocked by provenance failure'));
      return report;
    }
    // Cheap correctness gates complete before native/WASM execution. SIMD and
    // provenance errors block all slow stages; bench DIFFs still collect EVERY
    // tier so the matrix doesn't hide an inherited scalar/SIMD split.
    for (const tier of tiers) {
      try {
        if (!existsSync(tier.wasm)) throw new Error(`Missing WASM artifact: ${tier.wasm}`);
        const result = inspectWasmSimd(readFileSync(tier.wasm));
        const status = simdStatus(`${tier.name}${tier.requiresSimd ? '-simd' : ''}`, result);
        gates.push({ name: 'simd', status,
          detail: `SIMD=${result.total} relaxed=${result.relaxedTotal} dot=${result.relaxedDot} f32(madd/nmadd/min/max)=${Object.values(result.relaxedF32).join('/')}${status === 'FAIL' ? '; silent scalar tier' : status === 'WARN' ? '; relaxed f32 FMA may be fused or unfused by CPU' : ''}`,
          data: { tier: tier.name, artifact: tier.wasm, requiresSimd: Boolean(tier.requiresSimd), ...result } });
      } catch (error) { gates.push(fail('simd', error, { tier: tier.name, artifact: tier.wasm })); }
    }
    if (!hasFailures(gates)) {
      try { gates.push(await upstreamGate(context)); }
      catch (error) { gates.push(fail('provenance', error, { check: 'upstream' })); }
    } else gates.push(skip('provenance', 'upstream query blocked by cheap correctness failure', { check: 'upstream' }));
    let benchPrepared = false;
    const completedBenches = new Set();
    if (hasFailures(gates)) {
      gates.push(skip('bench', 'blocked by provenance/SIMD FAIL'));
    } else {
      let preparationError;
      try { await adapter.prepare?.(context); benchPrepared = true; } catch (error) { preparationError = error; }
      const nativeByFlavor = new Map();
      for (const tier of tiers) {
        const data = { tier: tier.name, artifact: tier.wasm };
        if (preparationError) {
          gates.push(preparationError instanceof Unavailable ? skip('bench', preparationError.message, data) : fail('bench', preparationError, data));
          continue;
        }
        const flavor = tier.flavor ?? 'default';
        if (!nativeByFlavor.has(flavor)) {
          let reference = null;
          let nativeError = null;
          try { reference = await adapter.native?.(context, tier); }
          catch (error) { nativeError = error; }
          const explicit = tier.flavor === 'lite' ? options.nativeLiteBench : options.nativeBench;
          // A failed supplied command must not be overridden by a convenient N.
          if (!nativeError) reference = selectReference(reference, explicit, context.commitBench, adapter.commitBenchCompatible !== false,
            tier.flavor === 'lite' ? '--native-lite-bench' : '--native-bench');
          nativeByFlavor.set(flavor, { reference, nativeError });
        }
        const { reference, nativeError } = nativeByFlavor.get(flavor);
        if (nativeError) {
          // Missing native executables are a precise SKIP; malformed bench,
          // timeout or other runtime failures are not permission to pass.
          const unavailable = nativeUnavailable(nativeError);
          gates.push(unavailable ? skip('bench', `native unavailable: ${nativeError.message}; build: ${adapter.nativeBuild}`, data) : fail('bench', nativeError, data));
          continue;
        }
        try {
          console.error(`audit: ${engine} ${tier.name} full bench...`);
          const candidate = await adapter.bench(context, tier);
          if (candidate.skip) { gates.push(skip('bench', candidate.skip, { ...data, ...candidate })); continue; }
          const comparison = compareBench(candidate, reference, options.baselineDiverged);
          // A measured bench with no native reference still records its total;
          // cross-WASM parity is useful, but the native comparison stays SKIP.
          completedBenches.add(tier.name);
          if (!reference) comparison.detail += `${context.commitBench && adapter.commitBenchCompatible === false ? '; commit Bench: is a different depth/protocol, not a replay reference' : ''}; build: ${adapter.nativeBuild}`;
          gates.push({ name: 'bench', ...comparison, data: { ...data, ...comparison.data } });
        } catch (error) { gates.push(fail('bench', error, data)); }
      }
    }
    if (!options.parity) gates.push(skip('parity', 'not requested (use --parity)'));
    else if (hasFailures(gates)) gates.push(skip('parity', 'blocked by earlier correctness FAIL'));
    else if (!benchPrepared) gates.push(skip('parity', 'blocked: required full-bench corpus unavailable'));
    else if (tiers.some((tier) => !tier.api && !completedBenches.has(tier.name))) {
      gates.push(skip('parity', 'blocked: required full bench not completed', {
        tiers: tiers.filter((tier) => !tier.api && !completedBenches.has(tier.name)).map((tier) => tier.name),
      }));
    } else {
      try { gates.push(await parityGate(context, adapter, tiers)); }
      catch (error) { gates.push(fail('parity', error)); }
    }
    return report;
  } finally { rmSync(context.tmp, { recursive: true, force: true }); }
}

export function printReport(report) {
  console.log(`\n${report.engine} | ref=${report.ref ?? 'unknown'} | ${report.runtime ?? ''} ${report.platform ?? ''}`);
  console.log('| Gate | Tier/check | Status | Detail |');
  console.log('| --- | --- | --- | --- |');
  for (const gate of report.gates) {
    const detail = gate.detail.replaceAll('|', '/').replace(/\s+/g, ' ');
    console.log(`| ${gate.name} | ${gate.data?.tier ?? gate.data?.check ?? '—'} | ${gate.status} | ${detail} |`);
  }
}

export function saveReport(report, path) {
  mkdirSync(dirname(resolve(path)), { recursive: true });
  writeFileSync(path, `${JSON.stringify(report, null, 2)}\n`);
}
