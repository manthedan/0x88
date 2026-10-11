#!/usr/bin/env node
// Delegate to the pinned package's own CLI loader: SF19 single builds export
// command(), but pthread builds now export uci(). The upstream loader owns
// that distinction, search queuing, readiness and pthread termination.
// Usage: node scripts/uci_stockfish_js_wrapper.mjs [lite-single|full-single|lite|full|path]
import { spawn } from 'node:child_process';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const raw = process.argv[2] || process.env.STOCKFISH_JS_VARIANT || 'lite-single';
const aliases = { 'full-single': 'single', 'single-lite': 'lite-single' };
const variant = aliases[raw] || raw;
const path = require('stockfish').findEngine(variant);
const child = spawn(process.execPath, [path], { stdio: 'inherit' });
for (const signal of ['SIGTERM', 'SIGINT']) process.on(signal, () => child.kill(signal));
child.on('error', (error) => { console.error(error); process.exitCode = 1; });
child.on('exit', (code, signal) => {
  process.exitCode = code ?? (signal ? 1 : 0);
});
