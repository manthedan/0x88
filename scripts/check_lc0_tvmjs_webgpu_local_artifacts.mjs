#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { readFile, stat } from 'node:fs/promises';
import { dirname, join } from 'node:path';

const DEFAULT_MANIFEST = 'public/runtimes/lc0-tvmjs-webgpu/t1-256x10-distilled-swa-2432500/f16/v1/manifest.json';
const DEFAULT_EVIDENCE = 'artifacts/tvm/lc0_tvmjs_webgpu_search_smoke_summary.json';

function usage() {
  console.log(`Usage: node scripts/check_lc0_tvmjs_webgpu_local_artifacts.mjs [options]\n\nChecks local generated TVMJS/WebGPU browser artifacts and evidence summaries.\nThese artifacts are intentionally ignored/local unless a release policy says otherwise.\n\nOptions:\n  --manifest PATH              Staged runtime manifest (default ${DEFAULT_MANIFEST})\n  --evidence PATH              Evidence summary JSON (default ${DEFAULT_EVIDENCE})\n  --no-evidence                Check only the staged manifest/files; useful before a new-family evidence summary exists\n  --min-search-rows N          Minimum evidence search rows (default 94)\n  --min-fixed-suite-reports N  Minimum fixed-suite-style report count (default 0)\n  --min-stockfish-scored-runs N\n                                Minimum Stockfish-scored evidence runs (default 0)\n  --require-all-matches        Fail unless evidence moveMatches equals searchRows (default behavior)\n  --expected-model-family NAME Require manifest.modelFamily\n  --expected-dtype NAME        Require manifest.dtype, e.g. f16\n  --expected-version NAME      Require manifest.version when present in newly staged manifests\n  --expected-batches LIST      Require exactly these manifest model batches, e.g. 1,4,8\n  -h, --help                   Show help\n`);
}

function parseBatches(raw) {
  const tokens = String(raw).split(',').map((item) => item.trim());
  const batches = [];
  for (const token of tokens) {
    if (!/^[-+]?\d+$/.test(token)) throw new Error(`Invalid positive integer batch token '${token}' in --expected-batches=${raw}`);
    const value = Number(token);
    if (!Number.isSafeInteger(value) || value <= 0) throw new Error(`Invalid positive integer batch token '${token}' in --expected-batches=${raw}`);
    batches.push(value);
  }
  return batches;
}

function sameNumberList(a, b) {
  if (a.length !== b.length) return false;
  return a.every((value, index) => value === b[index]);
}

function parseArgs(argv) {
  const args = { manifest: DEFAULT_MANIFEST, evidence: DEFAULT_EVIDENCE, checkEvidence: true, minSearchRows: 94, minFixedSuiteReports: 0, minStockfishScoredRuns: 0, requireAllMatches: true, expectedModelFamily: '', expectedDtype: '', expectedVersion: '', expectedBatches: undefined };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const next = () => { if (i + 1 >= argv.length) throw new Error(`${arg} requires a value`); return argv[++i]; };
    if (arg === '--manifest') args.manifest = next();
    else if (arg === '--evidence') args.evidence = next();
    else if (arg === '--no-evidence') args.checkEvidence = false;
    else if (arg === '--min-search-rows') args.minSearchRows = Number(next());
    else if (arg === '--min-fixed-suite-reports') args.minFixedSuiteReports = Number(next());
    else if (arg === '--min-stockfish-scored-runs') args.minStockfishScoredRuns = Number(next());
    else if (arg === '--require-all-matches') args.requireAllMatches = true;
    else if (arg === '--expected-model-family') args.expectedModelFamily = next();
    else if (arg === '--expected-dtype') args.expectedDtype = next();
    else if (arg === '--expected-version') args.expectedVersion = next();
    else if (arg === '--expected-batches') args.expectedBatches = parseBatches(next());
    else if (arg === '-h' || arg === '--help') args.help = true;
    else throw new Error(`Unknown option: ${arg}`);
  }
  if (!Number.isFinite(args.minSearchRows) || args.minSearchRows < 0) throw new Error(`Invalid --min-search-rows ${args.minSearchRows}`);
  if (!Number.isFinite(args.minFixedSuiteReports) || args.minFixedSuiteReports < 0) throw new Error(`Invalid --min-fixed-suite-reports ${args.minFixedSuiteReports}`);
  if (!Number.isFinite(args.minStockfishScoredRuns) || args.minStockfishScoredRuns < 0) throw new Error(`Invalid --min-stockfish-scored-runs ${args.minStockfishScoredRuns}`);
  return args;
}

async function fileSize(path) {
  return (await stat(path)).size;
}

async function fileSha256(path) {
  return createHash('sha256').update(await readFile(path)).digest('hex');
}

async function loadJson(path) {
  return JSON.parse(await readFile(path, 'utf8'));
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) return usage();
  const failures = [];
  const out = {
    schema: 'lc0_browser.tvmjs_webgpu_local_artifact_check.v1',
    generatedAt: new Date().toISOString(),
    manifest: args.manifest,
    evidence: args.evidence,
    ok: true,
  };
  try {
    const manifest = await loadJson(args.manifest);
    const manifestDir = dirname(args.manifest);
    out.manifestSchema = manifest.schema;
    out.modelFamily = manifest.modelFamily;
    out.dtype = manifest.dtype;
    out.version = manifest.version;
    out.requiredFeatures = manifest.requiredFeatures ?? [];
    if (args.expectedModelFamily && manifest.modelFamily !== args.expectedModelFamily) failures.push(`manifest modelFamily ${manifest.modelFamily ?? '<missing>'} != ${args.expectedModelFamily}`);
    if (args.expectedDtype && manifest.dtype !== args.expectedDtype) failures.push(`manifest dtype ${manifest.dtype ?? '<missing>'} != ${args.expectedDtype}`);
    if (args.expectedVersion && manifest.version !== args.expectedVersion) failures.push(`manifest version ${manifest.version ?? '<missing>'} != ${args.expectedVersion}`);
    if (!Array.isArray(manifest.requiredFeatures) || !manifest.requiredFeatures.includes('webgpu')) failures.push('manifest requiredFeatures missing webgpu');
    if (!Array.isArray(manifest.requiredFeatures) || !manifest.requiredFeatures.includes('shader-f16')) failures.push('manifest requiredFeatures missing shader-f16');
    out.models = manifest.models?.map((model) => ({ batch: model.batch, wasm: model.wasm, bytes: model.bytes, sha256: model.sha256 })) ?? [];
    const actualBatches = out.models.map((model) => model.batch).sort((a, b) => a - b);
    if (args.expectedBatches && !sameNumberList(actualBatches, [...args.expectedBatches].sort((a, b) => a - b))) failures.push(`manifest batches ${actualBatches.join(',')} != expected ${args.expectedBatches.join(',')}`);
    out.files = [];
    const files = manifest.files ?? [];
    const filesByPath = new Map(files.map((file) => [file.path, file]));
    for (const file of files) {
      if (!file.path || !file.sha256 || !Number.isFinite(file.bytes)) {
        failures.push(`manifest file ${file.path ?? '<missing>'} missing path/bytes/sha256`);
        continue;
      }
      const path = join(manifestDir, file.path);
      try {
        const bytes = await fileSize(path);
        const sha256 = await fileSha256(path);
        out.files.push({ path: file.path, bytes, sha256 });
        if (bytes !== file.bytes) failures.push(`manifest file ${file.path} bytes ${bytes} != ${file.bytes}`);
        if (sha256 !== file.sha256) failures.push(`manifest file ${file.path} sha256 ${sha256} != ${file.sha256}`);
      } catch (error) {
        failures.push(`manifest file ${file.path} check failed: ${error.message ?? error}`);
      }
    }
    for (const model of manifest.models ?? []) {
      if (!model.wasm || !model.probe || !model.sha256 || !Number.isFinite(model.bytes)) failures.push(`manifest model batch ${model.batch} missing wasm/probe/bytes/sha256`);
      const wasmEntry = filesByPath.get(model.wasm);
      if (!wasmEntry) failures.push(`manifest model batch ${model.batch} wasm ${model.wasm} missing from files[]`);
      else {
        if (wasmEntry.bytes !== model.bytes) failures.push(`manifest model batch ${model.batch} bytes ${model.bytes} != files[] ${wasmEntry.bytes}`);
        if (wasmEntry.sha256 !== model.sha256) failures.push(`manifest model batch ${model.batch} sha256 ${model.sha256} != files[] ${wasmEntry.sha256}`);
      }
      if (model.probe && !filesByPath.has(model.probe)) failures.push(`manifest model batch ${model.batch} probe ${model.probe} missing from files[]`);
    }
  } catch (error) {
    failures.push(`manifest check failed: ${error.message ?? error}`);
  }
  if (args.checkEvidence) {
    try {
      const evidence = await loadJson(args.evidence);
      out.evidenceSchema = evidence.schema;
      out.searchRows = evidence.searchRows;
      out.moveMatches = evidence.moveMatches;
      out.stockfishScoredRuns = evidence.stockfishScoredRuns?.length ?? 0;
      out.fixedSuiteStyleReports = evidence.fixedSuiteStyleReports?.length ?? 0;
      if ((evidence.searchRows ?? 0) < args.minSearchRows) failures.push(`evidence search rows ${evidence.searchRows ?? 0} < ${args.minSearchRows}`);
      if (out.fixedSuiteStyleReports < args.minFixedSuiteReports) failures.push(`evidence fixed-suite reports ${out.fixedSuiteStyleReports} < ${args.minFixedSuiteReports}`);
      if (out.stockfishScoredRuns < args.minStockfishScoredRuns) failures.push(`evidence Stockfish-scored runs ${out.stockfishScoredRuns} < ${args.minStockfishScoredRuns}`);
      if (args.requireAllMatches && evidence.moveMatches !== evidence.searchRows) failures.push(`evidence move matches ${evidence.moveMatches}/${evidence.searchRows}`);
    } catch (error) {
      failures.push(`evidence check failed: ${error.message ?? error}`);
    }
  } else {
    out.evidence = null;
    out.evidenceSkipped = true;
  }
  try {
    out.manifestBytes = await fileSize(args.manifest);
    if (args.checkEvidence) out.evidenceBytes = await fileSize(args.evidence);
  } catch { /* already covered above */ }
  out.ok = failures.length === 0;
  if (failures.length) out.failures = failures;
  process.stdout.write(`${JSON.stringify(out, null, 2)}\n`);
  if (failures.length) process.exitCode = 1;
}

main().catch((error) => {
  console.error(error?.stack ?? error?.message ?? String(error));
  process.exit(1);
});
