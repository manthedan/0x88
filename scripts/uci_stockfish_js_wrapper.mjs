#!/usr/bin/env node
// Minimal stdin/stdout UCI wrapper for the stockfish.js WASM package.
// Usage: node scripts/uci_stockfish_js_wrapper.mjs [lite-single|full-single|lite|full]
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import readline from 'node:readline';

const require = createRequire(import.meta.url);
const repoRoot = dirname(dirname(fileURLToPath(import.meta.url)));
const variant = (process.argv[2] || process.env.STOCKFISH_JS_VARIANT || 'lite-single').toLowerCase();
const suffix = variant === 'full-single' || variant === 'single' ? 'single'
  : variant === 'full' ? ''
  : variant === 'lite' ? 'lite'
  : variant === 'lite-single' || variant === 'single-lite' ? 'lite-single'
  : variant;
const jsName = suffix ? `stockfish-18-${suffix}.js` : 'stockfish-18.js';
const wasmName = suffix ? `stockfish-18-${suffix}.wasm` : 'stockfish-18.wasm';
const binDir = join(repoRoot, 'node_modules', 'stockfish', 'bin');
const jsPath = join(binDir, jsName);
const wasmPath = join(binDir, wasmName);
const INIT = require(jsPath);

let engine;
function command(s) {
  if (!engine || !s) return;
  try {
    engine.ccall('command', null, ['string'], [s], { async: /^go\b/.test(s) });
  } catch (err) {
    process.stderr.write(`[stockfish-js-wrapper] command failed: ${s}: ${err?.stack || err}\n`);
  }
}
function shutdown() {
  try { command('quit'); } catch {}
  setTimeout(() => process.exit(0), 20).unref?.();
}

engine = { locateFile: (p) => p.includes('.wasm') ? wasmPath : jsPath, listener: (line) => process.stdout.write(`${line}\n`) };
await INIT()(engine);
while (engine._isReady && !engine._isReady()) await new Promise((resolve) => setTimeout(resolve, 10));

const rl = readline.createInterface({ input: process.stdin, terminal: false });
rl.on('line', (line) => {
  const s = line.trim();
  if (!s) return;
  command(s);
  if (s === 'quit') setTimeout(() => process.exit(0), 20).unref?.();
});
rl.on('close', shutdown);
process.on('SIGTERM', shutdown);
process.on('SIGINT', shutdown);
