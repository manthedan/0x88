#!/usr/bin/env node
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { maia3MoveIndex } from '../src/lc0/maia3.ts';

const DEFAULT_UPSTREAM = process.env.MAIA_PLATFORM_FRONTEND_DIR ?? '/tmp/maia-platform-frontend';

function usage() {
  console.log(`Usage: node --experimental-strip-types scripts/maia3_upstream_move_map_parity.mjs [options]\n\nCompares this project's algorithmic Maia3 4352-move indexer against the upstream CSSLab maia-platform-frontend JSON move maps. It reads upstream files from a local clone and does not vendor/copy them.\n\nOptions:\n  --upstream-dir PATH   Local maia-platform-frontend checkout (default ${DEFAULT_UPSTREAM})\n  --out PATH            Optional JSON artifact path\n  -h, --help            Show this help\n`);
}

function parseArgs(argv) {
  const args = { upstreamDir: DEFAULT_UPSTREAM };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const next = () => {
      if (i + 1 >= argv.length) throw new Error(`${arg} requires a value`);
      return argv[++i];
    };
    if (arg === '--upstream-dir') args.upstreamDir = next();
    else if (arg === '--out') args.out = next();
    else if (arg === '-h' || arg === '--help') args.help = true;
    else throw new Error(`Unknown option: ${arg}`);
  }
  return args;
}

async function readJson(path) {
  return JSON.parse(await readFile(path, 'utf8'));
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) { usage(); return; }
  const dataDir = join(args.upstreamDir, 'src/lib/engine/data');
  const forwardPath = join(dataDir, 'all_moves_maia3.json');
  const reversePath = join(dataDir, 'all_moves_maia3_reversed.json');
  const forward = await readJson(forwardPath);
  const reverse = await readJson(reversePath);
  const mismatches = [];
  for (const [uci, index] of Object.entries(forward)) {
    const local = maia3MoveIndex(uci);
    if (local !== index) mismatches.push({ uci, upstream: index, local });
    const reversed = reverse[String(index)];
    if (reversed !== uci) mismatches.push({ index, upstreamReverse: reversed, expected: uci });
  }
  for (let index = 0; index < 4352; index++) {
    const uci = reverse[String(index)];
    if (typeof uci !== 'string') mismatches.push({ index, missingReverse: true });
    else if (forward[uci] !== index) mismatches.push({ index, reverseUci: uci, forwardIndex: forward[uci] });
  }
  const result = {
    ok: mismatches.length === 0,
    upstreamDir: args.upstreamDir,
    forwardPath,
    reversePath,
    entries: Object.keys(forward).length,
    reverseEntries: Object.keys(reverse).length,
    mismatches,
  };
  if (args.out) {
    await mkdir(dirname(args.out), { recursive: true });
    await writeFile(args.out, `${JSON.stringify(result, null, 2)}\n`);
  }
  if (!result.ok) {
    console.error(JSON.stringify(result, null, 2));
    process.exitCode = 1;
    return;
  }
  console.log(JSON.stringify({ ok: true, entries: result.entries, reverseEntries: result.reverseEntries, out: args.out }, null, 2));
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : error);
  process.exitCode = 1;
});
