#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { writeFile } from 'node:fs/promises';

const DEFAULT_HOST = '127.0.0.1';
const DEFAULT_PORT = 5292;
const DEFAULT_TIMEOUT_MS = 300_000;

function usage() {
  console.log(`Usage: node scripts/centipawn_tvmjs_webgpu_bench.mjs [options]

Options:
  --manifest PATH   Staged TVMJS manifest path
  --ort-model PATH  Production ONNX path
  --batch N         Fixed TVMJS batch (default 16)
  --warmup N        Warmup invocations per runtime (default 5)
  --repeats N       Timed invocations per runtime (default 20)
  --host HOST       Vite host (default ${DEFAULT_HOST})
  --port N          Vite port (default ${DEFAULT_PORT})
  --timeout MS      Overall timeout (default ${DEFAULT_TIMEOUT_MS})
  --agent-browser   Browser automation binary (default agent-browser)
  --out PATH        JSON evidence output
  -h, --help        Show help
`);
}

function parseArgs(argv) {
  const args = {
    manifest: '/runtimes/centipawn-tvmjs-webgpu/bt4-soap-rem-c19000-final/f32/v1-baseline/manifest.json',
    ortModel: '/models/bt4_soap_rem_c19000_final.onnx',
    batch: 16,
    warmup: 5,
    repeats: 20,
    host: DEFAULT_HOST,
    port: DEFAULT_PORT,
    timeoutMs: DEFAULT_TIMEOUT_MS,
    agentBrowser: process.env.AGENT_BROWSER_BIN ?? 'agent-browser',
  };
  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index];
    const next = () => {
      if (index + 1 >= argv.length) throw new Error(`${arg} requires a value`);
      return argv[++index];
    };
    if (arg === '--manifest') args.manifest = next();
    else if (arg === '--ort-model') args.ortModel = next();
    else if (arg === '--batch') args.batch = Number(next());
    else if (arg === '--warmup') args.warmup = Number(next());
    else if (arg === '--repeats') args.repeats = Number(next());
    else if (arg === '--host') args.host = next();
    else if (arg === '--port') args.port = Number(next());
    else if (arg === '--timeout') args.timeoutMs = Number(next());
    else if (arg === '--agent-browser') args.agentBrowser = next();
    else if (arg === '--out') args.out = next();
    else if (arg === '-h' || arg === '--help') args.help = true;
    else throw new Error(`Unknown option: ${arg}`);
  }
  for (const key of ['batch', 'repeats', 'port', 'timeoutMs']) {
    if (!Number.isFinite(args[key]) || args[key] <= 0) throw new Error(`Invalid --${key}: ${args[key]}`);
  }
  if (!Number.isFinite(args.warmup) || args.warmup < 0) throw new Error(`Invalid --warmup: ${args.warmup}`);
  return args;
}

function spawnCapture(command, commandArgs, options = {}) {
  return new Promise((resolve, reject) => {
    const { timeoutMs, stdin, ...spawnOptions } = options;
    const child = spawn(command, commandArgs, { stdio: ['pipe', 'pipe', 'pipe'], ...spawnOptions });
    const stdout = [];
    const stderr = [];
    let settled = false;
    const finish = (callback, value) => {
      if (settled) return;
      settled = true;
      if (timer) clearTimeout(timer);
      callback(value);
    };
    const timer = timeoutMs
      ? setTimeout(() => {
          child.kill('SIGKILL');
          finish(reject, new Error(`${command} timed out after ${timeoutMs}ms`));
        }, timeoutMs)
      : undefined;
    child.stdout.on('data', (chunk) => stdout.push(chunk));
    child.stderr.on('data', (chunk) => stderr.push(chunk));
    child.on('error', (error) => finish(reject, error));
    child.on('close', (status) => {
      const output = Buffer.concat(stdout).toString('utf8');
      const errors = Buffer.concat(stderr).toString('utf8');
      if (status !== 0) return finish(reject, new Error(`${command} failed with ${status}: ${errors || output}`));
      return finish(resolve, output);
    });
    child.stdin.end(stdin);
  });
}

function parseAgentJson(output) {
  const parsed = JSON.parse(output.trim());
  if (parsed?.success === false) throw new Error(parsed.error ?? output);
  return parsed?.data ?? parsed;
}

async function runAgent(args, session, commandArgs, stdin) {
  const output = await spawnCapture(
    args.agentBrowser,
    ['--json', '--session', session, ...commandArgs],
    { timeoutMs: args.timeoutMs, stdin },
  );
  return parseAgentJson(output);
}

function startServer(args) {
  const server = spawn('npm', ['run', 'web:client', '--', '--host', args.host, '--port', String(args.port), '--strictPort'], {
    env: { ...process.env, LC0_TVMJS_LAB: '1' },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  let output = '';
  server.ready = new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`Vite did not become ready: ${output}`)), 30_000);
    const inspect = (chunk) => {
      output += chunk.toString('utf8');
      const plain = output.replace(/\u001b\[[0-?]*[ -/]*[@-~]/g, '');
      if (/ready in \d+\s*ms/.test(plain) || plain.includes(`:${args.port}/`)) {
        clearTimeout(timer);
        resolve();
      }
    };
    server.stdout.on('data', (chunk) => {
      process.stderr.write(`[vite] ${chunk}`);
      inspect(chunk);
    });
    server.stderr.on('data', (chunk) => {
      process.stderr.write(`[vite] ${chunk}`);
      inspect(chunk);
    });
    server.on('exit', (status, signal) => {
      clearTimeout(timer);
      reject(new Error(`Vite exited before ready (${status ?? signal}): ${output}`));
    });
  });
  return server;
}

async function waitForResult(args, session) {
  const deadline = Date.now() + args.timeoutMs;
  let last;
  while (Date.now() < deadline) {
    last = await runAgent(
      args,
      session,
      ['eval', '--stdin'],
      `(() => ({
        result: window.centipawnTvmjsLastResult ?? null,
        error: window.centipawnTvmjsLastError ?? null,
        log: document.getElementById('log')?.textContent ?? ''
      }))()`,
    );
    const value = last?.value ?? (
      last && 'result' in last && !('error' in last) && !('log' in last) ? last.result : last
    );
    if (value?.error) throw new Error(value.error);
    if (value?.result) return value.result;
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  throw new Error(`Timed out waiting for benchmark result: ${JSON.stringify(last)}`);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    usage();
    return;
  }
  const server = startServer(args);
  const session = `centipawn-tvmjs-${process.pid}`;
  try {
    await server.ready;
    const url = new URL(`http://${args.host}:${args.port}/lab/centipawn-tvmjs-webgpu-bench.html`);
    url.searchParams.set('manifest', args.manifest);
    url.searchParams.set('ortModel', args.ortModel);
    url.searchParams.set('batch', String(Math.floor(args.batch)));
    url.searchParams.set('warmup', String(Math.floor(args.warmup)));
    url.searchParams.set('repeats', String(Math.floor(args.repeats)));
    url.searchParams.set('autorun', '1');
    process.stderr.write(`[centipawn-tvmjs-bench] ${url}\n`);
    await runAgent(args, session, ['open', String(url)]);
    const result = await waitForResult(args, session);
    const artifact = {
      schema: 'lc0_browser.centipawn_tvmjs_webgpu_evidence.v1',
      generatedAt: new Date().toISOString(),
      url: String(url),
      result,
    };
    if (args.out) await writeFile(args.out, `${JSON.stringify(artifact, null, 2)}\n`);
    console.log(JSON.stringify(artifact, null, 2));
    if (!result.ok) process.exitCode = 1;
  } finally {
    try {
      await runAgent({ ...args, timeoutMs: 5000 }, session, ['close']);
    } catch {
      // Best effort.
    }
    server.kill('SIGTERM');
  }
}

main().catch((error) => {
  console.error(error.stack ?? error.message);
  process.exit(1);
});
