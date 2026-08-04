import { spawn } from 'node:child_process';
import { setTimeout as delay } from 'node:timers/promises';

// Matches ANSI escape sequences (e.g. vite's colored "ready in 123ms" output).
// biome-ignore lint/suspicious/noControlCharactersInRegex: intentionally strips ANSI escape sequences from child process output
const ANSI_PATTERN = /\[[0-?]*[ -/]*[@-~]/g;

/**
 * Watch a spawned child's stdout+stderr until `match` passes on the
 * ANSI-stripped accumulated output. Resolves with the plain output; rejects on
 * timeout or if the child exits first. Shared replacement for the per-script
 * wait-for-server polling loops.
 *
 * waitForOutput(child, { match: /ready in \d+\s*ms/ or (text) => bool, timeoutMs, label })
 */
export function waitForOutput(child, { match, timeoutMs = 30_000, label = 'process' } = {}) {
  const matches = typeof match === 'function' ? match : (text) => match.test(text);
  return new Promise((resolvePromise, rejectPromise) => {
    let output = '';
    const plain = () => output.replace(ANSI_PATTERN, '');
    const cleanup = () => {
      clearTimeout(timer);
      child.stdout?.off('data', onData);
      child.stderr?.off('data', onData);
      child.off('exit', onExit);
    };
    const timer = setTimeout(() => {
      cleanup();
      rejectPromise(new Error(`${label} did not produce expected output within ${timeoutMs}ms:\n${plain()}`));
    }, timeoutMs);
    const onData = (chunk) => {
      output += chunk.toString('utf8');
      if (matches(plain())) {
        cleanup();
        resolvePromise(plain());
      }
    };
    const onExit = (code) => {
      cleanup();
      rejectPromise(new Error(`${label} exited with code ${code} before producing expected output:\n${plain()}`));
    };
    child.stdout?.on('data', onData);
    child.stderr?.on('data', onData);
    child.once('exit', onExit);
  });
}

/** Start the repository Vite server and expose a readiness promise on it. */
export function startViteServer(args, { env, strictPort = true, timeoutMs = 30_000 } = {}) {
  if (args.noServer || args.explicitBaseUrl) return null;
  const commandArgs = ['run', 'web:client', '--', '--host', args.host, '--port', String(args.port), ...(strictPort ? ['--strictPort'] : [])];
  const server = spawn('npm', commandArgs, {
    env: env ? { ...process.env, ...env } : process.env,
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  const echoOutput = (chunk) => process.stderr.write(`[vite] ${chunk}`);
  server.stdout.on('data', echoOutput);
  server.stderr.on('data', echoOutput);
  server.ready = waitForOutput(server, {
    match: (text) => /ready in \d+\s*ms/.test(text) || text.includes(`:${args.port}/`),
    timeoutMs,
    label: `Vite dev server (port ${args.port})`,
  });
  return server;
}

/** Poll an HTTP route until the server responds successfully. */
export async function waitForHttp(baseUrl, { path = '/', timeoutMs = 30_000, label = 'Vite dev server' } = {}) {
  const deadline = Date.now() + timeoutMs;
  let lastError;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(new URL(path, baseUrl), { cache: 'no-store' });
      if (response.ok) return;
      lastError = new Error(`HTTP ${response.status}`);
    } catch (error) {
      lastError = error;
    }
    await delay(250);
  }
  throw new Error(`${label} did not become ready at ${baseUrl}: ${lastError?.message ?? 'timeout'}`);
}
