#!/usr/bin/env node
import { stat } from 'node:fs/promises';
import { join, relative } from 'node:path';
import { BROWSER_ENGINE_ASSET_GROUPS } from './engine_artifact_registry.mjs';

const ROOT = process.cwd();
const PUBLIC_ROOT = join(ROOT, 'public');

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
  const optionalAssets = await Promise.all((group.optionalAssets ?? []).map(checkAsset));
  const missing = assets.filter((asset) => !asset.ok).map((asset) => asset.url);
  const optionalMissing = optionalAssets.filter((asset) => !asset.ok).map((asset) => asset.url);
  const bytes = [...assets, ...optionalAssets].reduce((sum, asset) => sum + (asset.bytes ?? 0), 0);
  return { ...group, ok: missing.length === 0, missing, optionalMissing, bytes, assets, optionalAssets };
}

function textReport(report) {
  const lines = [];
  lines.push(`Browser engine assets: ${report.ok ? 'ok' : `${report.missingFamilies.length} family group(s) missing assets`}`);
  for (const group of report.groups) {
    lines.push(`\n${group.ok ? '✓' : '✗'} ${group.family} — ${group.label} (${group.status})`);
    lines.push(`  docs: ${group.docs}`);
    lines.push(`  bytes present: ${group.bytes}`);
    for (const asset of group.assets) lines.push(`  ${asset.ok ? 'ok ' : 'miss'} ${asset.url}${asset.ok ? ` (${asset.bytes} bytes)` : ''}`);
    for (const asset of group.optionalAssets) lines.push(`  ${asset.ok ? 'ok ' : 'opt '} ${asset.url}${asset.ok ? ` (${asset.bytes} bytes)` : ' (optional, not staged)'}`);
    if (!group.ok || group.optionalMissing.length) lines.push(`  prepare: ${group.command}`);
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
  const selected = BROWSER_ENGINE_ASSET_GROUPS.filter((group) => !args.only || args.only.has(group.family));
  if (args.only) {
    const known = new Set(BROWSER_ENGINE_ASSET_GROUPS.map((group) => group.family));
    const unknown = [...args.only].filter((family) => !known.has(family));
    if (unknown.length) throw new Error(`Unknown family id(s): ${unknown.join(', ')}`);
  }
  const groups = await Promise.all(selected.map(checkGroup));
  // A family whose artifacts are deliberately not distributed (Berserk: the
  // upstream NNUE has no resolved license) is expected to be absent on a clean
  // checkout. Report it so `nextCommands` still tells you how to build it, but
  // do not fail the readiness check over an intentional absence.
  const buildLocally = (group) => group.status === 'build-locally-not-distributed';
  const missingFamilies = groups.filter((group) => !group.ok && !buildLocally(group)).map((group) => group.family);
  const optionalMissingFamilies = groups.filter((group) => !group.ok && buildLocally(group)).map((group) => group.family);
  const nextCommands = [...new Set(groups.filter((group) => !group.ok).map((group) => group.command))];
  const report = { status: 'BROWSER_ENGINE_ASSET_CHECK_DONE', ok: missingFamilies.length === 0, missingFamilies, optionalMissingFamilies, nextCommands, groups };
  if (args.json) console.log(JSON.stringify(report, null, 2));
  else console.log(textReport(report));
  if (!report.ok && !args.allowMissing) process.exitCode = 1;
}

main().catch((error) => {
  console.error(error.stack ?? error.message);
  process.exit(1);
});
