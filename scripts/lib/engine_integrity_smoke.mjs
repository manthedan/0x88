import { resolve } from 'node:path';
import { parseScriptArgs } from './cli.mjs';

export function parseIntegritySmokeArgs(argv) {
  const args = parseScriptArgs(argv, {
    options: { 'ui-only': { type: 'boolean', default: false } },
    allowPositionals: true,
    usage: 'Usage: node scripts/engine_integrity_browser_smoke.mjs [dist-directory] [--ui-only]',
  });
  if (args.positionals.length > 1) throw new Error('Expected at most one dist directory');
  return { dist: resolve(args.positionals[0] ?? 'dist-client'), uiOnly: args.uiOnly };
}
