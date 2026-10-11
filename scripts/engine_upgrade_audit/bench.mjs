import { copyFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { basename, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseBenchOutput } from './parsers.mjs';
import { runProcess } from './process.mjs';
import { wasiBenchProtocol } from './wasi_protocol.mjs';

const worker = (name) => fileURLToPath(new URL(name, import.meta.url));
let serial = 0;
const builtinRows = { berserk: 50, reckless: 50 };
// Each runner requests ONE bench. Separately captured output streams cannot
// establish chronological summary precedence, so conflicting totals fail.
const builtinParserOptions = (engine) => ({ expectedRows: builtinRows[engine], requireConsistentSummaries: true });
function workspace(context, label) {
  const dir = join(context.tmp, `${label}-${serial++}`);
  mkdirSync(dir);
  return dir;
}

export function stageEmscripten(context, tier, stockfish = false) {
  const dir = workspace(context, 'emscripten');
  const base = basename(tier.wasm, '.wasm');
  const js = join(dir, `${base}.js`);
  writeFileSync(join(dir, 'package.json'), '{"type":"commonjs"}');
  copyFileSync(tier.js, js);
  copyFileSync(tier.wasm, join(dir, `${base}.wasm`));
  if (!stockfish) {
    const variant = join(context.artifacts, `${base}.data`);
    const canonical = join(context.artifacts, `${context.engine}-emscripten.data`);
    const data = existsSync(variant) ? variant : canonical;
    copyFileSync(data, join(dir, `${base}.data`));
    // Some glue hardcodes the shared URL rather than its own variant basename.
    if (`${context.engine}-emscripten.data` !== `${base}.data`) copyFileSync(data, join(dir, `${context.engine}-emscripten.data`));
  }
  return { label: tier.name, command: process.execPath,
    args: stockfish ? [js] : [worker('emscripten_worker.mjs'), js] };
}

export async function runReplay(context, cell, defaults, depth, nativeBench = false, parity = false) {
  const dir = workspace(context, 'replay');
  const spec = join(dir, 'spec.json');
  const result = join(dir, 'result.json');
  writeFileSync(spec, JSON.stringify({ cell: { ...cell, nativeBench },
    options: { defaults, benchDepth: depth, depth: 9, threads: 1, positions: parity ? 20 : 0, timeout: context.options.timeout } }));
  await runProcess(process.execPath, [worker('replay_worker.mjs'), spec, result], { timeout: context.options.timeout });
  return JSON.parse(readFileSync(result, 'utf8'));
}

export function nativeCell(command, label = 'native') {
  // User-owned trusted command, including ssh ... bash -lc ...; don't split or
  // rewrite its quoting. The outer process group bounds shell grandchildren.
  return { label, command: 'bash', args: ['-lc', command] };
}

export async function runNativeBuiltin(context) {
  const command = context.options.nativeCmd;
  if (!command) return null;
  // Send the same UCI bench command through stdin. Appending argv to a quoted
  // ssh ... bash -lc command would set bash's $0, not the engine's bench arg.
  const output = await runProcess('bash', ['-lc', command], { input: 'bench\nquit\n', timeout: context.options.timeout });
  return { ...parseBenchOutput(`${output.stdout}\n${output.stderr}`, builtinParserOptions(context.engine)), source: '--native-cmd', command };
}

export function externalNetworkPath(context) {
  const files = [...new Set(context.verifiedNetworkFiles ?? [])];
  if (files.length !== 1) throw new Error('External WASI requires one provenance-verified network file; supply --nnue and a matching locked digest');
  return files[0];
}

export async function runWasiBench(context, tier) {
  const dir = workspace(context, 'wasi');
  writeFileSync(join(dir, 'stdin'), wasiBenchProtocol(context.engine).stdin);
  if (tier.external) {
    const network = externalNetworkPath(context);
    copyFileSync(network, join(dir, 'reckless.nnue'));
  }
  await runProcess(process.execPath, [worker('wasi_worker.mjs'), context.engine, tier.wasm, dir], { timeout: context.options.timeout });
  const stdout = readFileSync(join(dir, 'stdout'), 'utf8');
  const stderr = readFileSync(join(dir, 'stderr'), 'utf8');
  return { ...parseBenchOutput(`${stdout}\n${stderr}`, builtinParserOptions(context.engine)), method: 'built-in bench (node:wasi preview1)' };
}

export async function runEmscriptenBuiltin(context, tier) {
  const cell = stageEmscripten(context, tier);
  const output = await runProcess(cell.command, cell.args, { input: 'bench\n', timeout: context.options.timeout, allowFailure: true });
  if (output.timedOut) throw new Error('command("bench") timed out; supported built-in bench did not complete');
  if (output.code !== 0) throw new Error(`bench process exit ${output.code}: ${output.stderr.slice(-1000)}`);
  const text = `${output.stdout}\n${output.stderr}`;
  if (/unknown command[^\n]*bench/i.test(text)) return { skip: 'facade explicitly rejects command("bench"); replay not validated in v1', diagnostic: text.slice(-500) };
  return { ...parseBenchOutput(text, builtinParserOptions(context.engine)), method: 'command("bench")' };
}
