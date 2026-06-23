#!/usr/bin/env node
import { stat } from 'node:fs/promises';
import { join, relative } from 'node:path';

const ROOT = process.cwd();
const PUBLIC_ROOT = join(ROOT, 'public');

const ASSET_GROUPS = [
  {
    family: 'lc0',
    label: 'Lc0 BT4 analysis model',
    status: 'optional-gated',
    command: 'npm run lc0:prepare-model-assets',
    docs: 'docs/engine_catalog.md#lc0-family',
    assets: ['/models/lc0/BT4-1024x15x32h-swa-6147500-policytune-332.batch4.f16.onnx'],
  },
  {
    family: 'reckless',
    label: 'Reckless WASI/browser variants',
    status: 'experimental-selectable',
    command: 'npm run reckless:build-production && npm run reckless:build-browser-api && npm run reckless:build-browser-api-simd && npm run reckless:build-browser-api-simd-external && npm run reckless:build-lite-wasi',
    docs: 'docs/engine_catalog.md#reckless-family',
    assets: [
      '/reckless/reckless.wasm',
      '/reckless/reckless-simd128.wasm',
      '/reckless/reckless-relaxed-simd128.wasm',
      '/reckless/reckless-browser-api.wasm',
      '/reckless/reckless-browser-api-simd128.wasm',
      '/reckless/reckless-browser-api-simd128-external.wasm',
      '/reckless/reckless-v60-7f587dfb.nnue',
      '/reckless/reckless-v53-l1-512.wasm',
    ],
  },
  {
    family: 'viridithas',
    label: 'Viridithas WASI variants',
    status: 'experimental-selectable',
    command: 'npm run viridithas:build-wasi && npm run viridithas:build-simd-wasi && npm run viridithas:build-relaxed-simd-wasi',
    docs: 'docs/engine_catalog.md#viridithas-family',
    assets: ['/viridithas/viridithas.wasm', '/viridithas/viridithas-simd128.wasm', '/viridithas/viridithas-relaxed-simd128.wasm'],
  },
  {
    family: 'berserk',
    label: 'Berserk Emscripten worker',
    status: 'experimental-selectable',
    command: 'npm run berserk:build-emscripten && npm run berserk:build-simd-emscripten && npm run berserk:build-relaxed-simd-emscripten',
    docs: 'docs/engine_catalog.md#berserk-family',
    assets: [
      '/berserk/berserk-emscripten.js',
      '/berserk/berserk-emscripten.wasm',
      '/berserk/berserk-emscripten.data',
      '/berserk/berserk-emscripten-simd128.js',
      '/berserk/berserk-emscripten-simd128.wasm',
      '/berserk/berserk-emscripten-simd128.data',
      '/berserk/berserk-emscripten-relaxed-simd128.js',
      '/berserk/berserk-emscripten-relaxed-simd128.wasm',
      '/berserk/berserk-emscripten-relaxed-simd128.data',
    ],
  },
  {
    family: 'plentychess',
    label: 'PlentyChess Emscripten worker',
    status: 'experimental-selectable',
    command: 'npm run plentychess:build-emscripten && npm run plentychess:build-sse41-emscripten && npm run plentychess:build-relaxed-simd-emscripten',
    docs: 'docs/engine_catalog.md#plentychess-family',
    assets: [
      '/plentychess/plentychess-emscripten.js',
      '/plentychess/plentychess-emscripten.wasm',
      '/plentychess/plentychess-emscripten.data',
      '/plentychess/plentychess-emscripten-sse41.js',
      '/plentychess/plentychess-emscripten-sse41.wasm',
      '/plentychess/plentychess-emscripten-sse41.data',
      '/plentychess/plentychess-emscripten-relaxed-simd128.js',
      '/plentychess/plentychess-emscripten-relaxed-simd128.wasm',
      '/plentychess/plentychess-emscripten-relaxed-simd128.data',
    ],
  },
];

function usage() {
  console.log(`Usage: node scripts/check_browser_engine_assets.mjs [options]\n\nChecks local public/ browser engine assets used by /app/analysis and /app/arena, then prints the prep/build command for each missing family.\n\nOptions:\n  --only LIST       Comma-separated family ids to check (default all)\n  --allow-missing   Exit 0 even when assets are missing\n  --json            Print JSON only\n  -h, --help        Show this help\n`);
}

function parseArgs(argv) {
  const args = { only: undefined, allowMissing: false, json: false, help: false };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const next = () => {
      if (i + 1 >= argv.length) throw new Error(`${arg} requires a value`);
      return argv[++i];
    };
    if (arg === '--only') args.only = new Set(next().split(',').map((value) => value.trim()).filter(Boolean));
    else if (arg === '--allow-missing') args.allowMissing = true;
    else if (arg === '--json') args.json = true;
    else if (arg === '-h' || arg === '--help') args.help = true;
    else throw new Error(`Unknown option: ${arg}`);
  }
  return args;
}

function filePathForUrl(urlPath) {
  if (!urlPath.startsWith('/')) throw new Error(`expected same-origin absolute URL path, got ${urlPath}`);
  return join(PUBLIC_ROOT, urlPath.slice(1));
}

async function checkAsset(urlPath) {
  const filePath = filePathForUrl(urlPath);
  try {
    const info = await stat(filePath);
    return { url: urlPath, path: relative(ROOT, filePath), ok: info.isFile(), bytes: info.isFile() ? info.size : 0 };
  } catch (error) {
    if (error?.code !== 'ENOENT') throw error;
    return { url: urlPath, path: relative(ROOT, filePath), ok: false, bytes: 0 };
  }
}

async function checkGroup(group) {
  const assets = await Promise.all(group.assets.map(checkAsset));
  const missing = assets.filter((asset) => !asset.ok).map((asset) => asset.url);
  const bytes = assets.reduce((sum, asset) => sum + (asset.bytes ?? 0), 0);
  return { ...group, ok: missing.length === 0, missing, bytes, assets };
}

function textReport(report) {
  const lines = [];
  lines.push(`Browser engine assets: ${report.ok ? 'ok' : `${report.missingFamilies.length} family group(s) missing assets`}`);
  for (const group of report.groups) {
    lines.push(`\n${group.ok ? '✓' : '✗'} ${group.family} — ${group.label} (${group.status})`);
    lines.push(`  docs: ${group.docs}`);
    lines.push(`  bytes present: ${group.bytes}`);
    for (const asset of group.assets) lines.push(`  ${asset.ok ? 'ok ' : 'miss'} ${asset.url}${asset.ok ? ` (${asset.bytes} bytes)` : ''}`);
    if (!group.ok) lines.push(`  prepare: ${group.command}`);
  }
  if (!report.ok) {
    lines.push('\nMissing asset prep commands:');
    for (const command of report.nextCommands) lines.push(`  ${command}`);
  }
  return lines.join('\n');
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) return usage();
  const selected = ASSET_GROUPS.filter((group) => !args.only || args.only.has(group.family));
  if (args.only) {
    const known = new Set(ASSET_GROUPS.map((group) => group.family));
    const unknown = [...args.only].filter((family) => !known.has(family));
    if (unknown.length) throw new Error(`Unknown family id(s): ${unknown.join(', ')}`);
  }
  const groups = await Promise.all(selected.map(checkGroup));
  const missingFamilies = groups.filter((group) => !group.ok).map((group) => group.family);
  const nextCommands = [...new Set(groups.filter((group) => !group.ok).map((group) => group.command))];
  const report = { status: 'BROWSER_ENGINE_ASSET_CHECK_DONE', ok: missingFamilies.length === 0, missingFamilies, nextCommands, groups };
  if (args.json) console.log(JSON.stringify(report, null, 2));
  else console.log(textReport(report));
  if (!report.ok && !args.allowMissing) process.exitCode = 1;
}

main().catch((error) => {
  console.error(error.stack ?? error.message);
  process.exit(1);
});
