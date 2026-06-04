#!/usr/bin/env node
import { copyFileSync, existsSync, mkdirSync, readFileSync, statSync } from 'node:fs';
import { basename, dirname, join, resolve } from 'node:path';

const source = resolve(process.env.RECKLESS_SOURCE_DIR ?? '.local_engines/reckless-wasi-src');
const explicitEvalfile = process.env.RECKLESS_EVALFILE ? resolve(process.env.RECKLESS_EVALFILE) : '';
const out = resolve(process.env.RECKLESS_NNUE_OUT ?? 'public/reckless/reckless-v60-7f587dfb.nnue');

function defaultNetworkPath() {
  const buildScript = join(source, 'build', 'build.rs');
  if (!existsSync(buildScript)) throw new Error(`Reckless build script missing: ${buildScript}`);
  const text = readFileSync(buildScript, 'utf8');
  const match = text.match(/const\s+NETWORK_NAME:\s*&str\s*=\s*"([^"]+)"/);
  if (!match) throw new Error(`Could not find NETWORK_NAME in ${buildScript}`);
  return join(source, 'networks', match[1]);
}

const nnue = explicitEvalfile || defaultNetworkPath();
if (!existsSync(nnue)) {
  throw new Error(`Reckless NNUE asset missing: ${nnue}. Run npm run reckless:build-wasi first or set RECKLESS_EVALFILE.`);
}

mkdirSync(dirname(out), { recursive: true });
copyFileSync(nnue, out);
const bytes = statSync(out).size;
console.log(`Wrote ${out} (${bytes} bytes) from ${basename(nnue)}`);
