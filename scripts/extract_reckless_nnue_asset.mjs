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

function sourceL1Size() {
  const nnueSource = join(source, 'src', 'nnue.rs');
  if (!existsSync(nnueSource)) throw new Error(`Reckless NNUE source missing: ${nnueSource}`);
  const text = readFileSync(nnueSource, 'utf8');
  return Number(text.match(/const L1_SIZE: usize = (\d+);/)?.[1] ?? '768');
}

const l1Size = sourceL1Size();
if (l1Size !== 768 && process.env.RECKLESS_BROWSER_API_ALLOW_CUSTOM_NNUE !== '1') {
  throw new Error(`External Reckless NNUE asset extraction expects the full v60 NNUE shape (L1_SIZE=768), but ${source} has L1_SIZE=${l1Size}. Run npm run reckless:build-wasi first, or set RECKLESS_SOURCE_DIR to a full Reckless source tree.`);
}

const nnue = explicitEvalfile || defaultNetworkPath();
if (!existsSync(nnue)) {
  throw new Error(`Reckless NNUE asset missing: ${nnue}. Run npm run reckless:build-wasi first or set RECKLESS_EVALFILE.`);
}

mkdirSync(dirname(out), { recursive: true });
copyFileSync(nnue, out);
const bytes = statSync(out).size;
console.log(`Wrote ${out} (${bytes} bytes) from ${basename(nnue)}`);
