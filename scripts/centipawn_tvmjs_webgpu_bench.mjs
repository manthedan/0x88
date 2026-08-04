#!/usr/bin/env node
import { writeFile } from 'node:fs/promises';
import { parseScriptArgs } from './lib/cli.mjs';
import { runAgent } from './lib/process.mjs';
import { startViteServer } from './lib/server.mjs';

const DEFAULT_HOST = '127.0.0.1';
const DEFAULT_PORT = 5292;
const DEFAULT_TIMEOUT_MS = 300_000;

const USAGE = `Usage: node scripts/centipawn_tvmjs_webgpu_bench.mjs [options]

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
`;

function parseArgs(argv) {
  const args = parseScriptArgs(argv, {
    options: {
      manifest: { type: 'string', default: '/runtimes/centipawn-tvmjs-webgpu/bt4-soap-rem-c19000-final/f32/v1-baseline/manifest.json' },
      'ort-model': { type: 'string', default: '/models/bt4_soap_rem_c19000_final.onnx' },
      batch: { type: 'string', default: '16' },
      warmup: { type: 'string', default: '5' },
      repeats: { type: 'string', default: '20' },
      host: { type: 'string', default: DEFAULT_HOST },
      port: { type: 'string', default: String(DEFAULT_PORT) },
      timeout: { type: 'string', default: String(DEFAULT_TIMEOUT_MS) },
      'agent-browser': { type: 'string', default: process.env.AGENT_BROWSER_BIN ?? 'agent-browser' },
      out: { type: 'string' },
    },
    usage: USAGE,
  });
  args.batch = Number(args.batch);
  args.warmup = Number(args.warmup);
  args.repeats = Number(args.repeats);
  args.port = Number(args.port);
  args.timeoutMs = Number(args.timeout);
  delete args.timeout;
  for (const key of ['batch', 'repeats', 'port', 'timeoutMs']) {
    if (!Number.isFinite(args[key]) || args[key] <= 0) throw new Error(`Invalid --${key}: ${args[key]}`);
  }
  if (!Number.isFinite(args.warmup) || args.warmup < 0) throw new Error(`Invalid --warmup: ${args.warmup}`);
  return args;
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
    const value = last?.value ?? (last && 'result' in last && !('error' in last) && !('log' in last) ? last.result : last);
    if (value?.error) throw new Error(value.error);
    if (value?.result) return value.result;
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  throw new Error(`Timed out waiting for benchmark result: ${JSON.stringify(last)}`);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const server = startViteServer(args, { env: { LC0_TVMJS_LAB: '1' } });
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
