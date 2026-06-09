#!/usr/bin/env node
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';

const DEFAULT_OUT = 'artifacts/tvm/lc0_tvmjs_startup_amortization_summary.json';

function usage() {
  console.log(`Usage: node scripts/summarize_lc0_tvmjs_startup_amortization.mjs --in MATRIX.json [options]\n\nSummarizes startup/init and simple amortization telemetry from a TVMJS-vs-hybrid matrix artifact.\nThis is research evidence only; ORT startup/init is reported only if present in the input artifact.\n\nOptions:\n  --in PATH      TVMJS-vs-hybrid matrix artifact\n  --out PATH     Output JSON summary (default ${DEFAULT_OUT})\n  -h, --help     Show help\n`);
}

function parseArgs(argv) {
  const args = { out: DEFAULT_OUT };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const next = () => { if (i + 1 >= argv.length) throw new Error(`${arg} requires a value`); return argv[++i]; };
    if (arg === '--in') args.in = next();
    else if (arg === '--out') args.out = next();
    else if (arg === '-h' || arg === '--help') args.help = true;
    else throw new Error(`Unknown option: ${arg}`);
  }
  if (!args.help && !args.in) throw new Error('--in is required');
  return args;
}

function numberOrUndefined(value) {
  return Number.isFinite(value) ? Number(value) : undefined;
}

function sumKnown(fields, names) {
  let total = 0;
  const included = {};
  for (const name of names) {
    const value = numberOrUndefined(fields?.[name]);
    if (value === undefined) continue;
    included[name] = value;
    total += value;
  }
  return { totalMs: Number(total.toFixed(6)), included };
}

function amortizedMs(coldMs, searchMeanMs, rows) {
  const cold = numberOrUndefined(coldMs);
  const search = numberOrUndefined(searchMeanMs);
  const count = Number.isFinite(rows) && rows > 0 ? Math.floor(rows) : undefined;
  if (cold === undefined || search === undefined || count === undefined) return undefined;
  return Number((search + cold / count).toFixed(6));
}

function ratio(a, b) {
  if (!Number.isFinite(a) || !Number.isFinite(b) || b === 0) return undefined;
  return Number((a / b).toFixed(6));
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) return usage();
  const artifact = JSON.parse(await readFile(args.in, 'utf8'));
  const tvmjs = artifact.summary?.tvmjs ?? {};
  const hybrid = artifact.summary?.hybrid ?? {};
  const headToHead = artifact.summary?.headToHead ?? {};
  const startup = tvmjs.startupTimings ?? {};
  const tvmjsCold = sumKnown(startup, [
    'manifestFetchMs',
    'manifestJsonMs',
    'tvmjsBundleLoadMs',
    'requestAdapterMs',
    'requestDeviceMs',
    'wasmFetchVerifyMs',
    'tvmjsInstantiateMs',
    'tvmInitWebGpuMs',
    'systemLibMs',
    'webgpuPipelinePrebuildMs',
    'createVirtualMachineMs',
    'inputTensorAllocMs',
    'inputUploadMs',
  ]);
  const rows = numberOrUndefined(tvmjs.searchRows) ?? numberOrUndefined(headToHead.comparableRows);
  const summary = {
    schema: 'lc0_browser.tvmjs_startup_amortization_summary.v1',
    generatedAt: new Date().toISOString(),
    sourceArtifact: args.in,
    ok: artifact.ok === true,
    caveat: 'Research summary only. TVMJS startup fields are a sum of observed phase timings, not proof of a strictly serialized critical path. ORT startup/init is unknown unless the source artifact explicitly includes it.',
    parameters: artifact.parameters,
    rows,
    tvmjs: {
      startupTimings: startup,
      coldStartupKnown: tvmjsCold,
      searchMeanMs: numberOrUndefined(tvmjs.tvmSearchMeanMs),
      amortizedMeanMsPerSearch: amortizedMs(tvmjsCold.totalMs, tvmjs.tvmSearchMeanMs, rows),
      gpuBufferAllocation: tvmjs.gpuBufferAllocation,
    },
    ortF16: {
      startupKnownMs: undefined,
      searchMeanMs: numberOrUndefined(tvmjs.ortSearchMeanMs),
      note: 'ORT f16 search timing is available from the TVMJS smoke comparison; ORT session/device startup is not separated in this matrix artifact.',
    },
    hybrid: {
      workerInitMs: numberOrUndefined(hybrid.workerInitMs),
      searchMeanMs: numberOrUndefined(hybrid.searchMeanElapsedMs),
      amortizedMeanMsPerSearch: amortizedMs(hybrid.workerInitMs, hybrid.searchMeanElapsedMs, rows),
      gpuBufferAllocation: hybrid.gpuBufferAllocation,
    },
    comparisons: {
      tvmjsSearchVsOrtSearch: ratio(tvmjs.tvmSearchMeanMs, tvmjs.ortSearchMeanMs),
      tvmjsSearchVsHybridSearch: ratio(tvmjs.tvmSearchMeanMs, hybrid.searchMeanElapsedMs),
      tvmjsAmortizedVsHybridAmortized: ratio(
        amortizedMs(tvmjsCold.totalMs, tvmjs.tvmSearchMeanMs, rows),
        amortizedMs(hybrid.workerInitMs, hybrid.searchMeanElapsedMs, rows),
      ),
    },
  };
  await mkdir(dirname(args.out), { recursive: true });
  await writeFile(args.out, `${JSON.stringify(summary, null, 2)}\n`);
  console.log(JSON.stringify(summary, null, 2));
}

main().catch((error) => {
  console.error(error?.stack || error?.message || String(error));
  process.exit(1);
});
