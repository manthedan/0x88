import { spawn } from 'node:child_process';

/**
 * Spawn a command, capture stdout/stderr, and resolve with trimmed stdout.
 * Rejects on non-zero exit (with stderr context), spawn errors, or timeout
 * (SIGKILL). Shared replacement for the per-script spawnCapture copies.
 *
 * Options:
 * - timeoutMs: kill after this many ms (default 30_000); 0 = no timeout
 * - cwd, env: passed through to spawn (env merges over process.env)
 * - input: string written to stdin and closed (stdio[0] becomes 'pipe')
 * - echoStderr: also stream stderr to process.stderr live (long-running cells)
 */
export function spawnCapture(command, commandArgs, { timeoutMs = 30_000, cwd, env, input, echoStderr = false } = {}) {
  return new Promise((resolvePromise, rejectPromise) => {
    const child = spawn(command, commandArgs, {
      stdio: [input === undefined ? 'ignore' : 'pipe', 'pipe', 'pipe'],
      ...(cwd ? { cwd } : {}),
      ...(env ? { env: { ...process.env, ...env } } : {}),
    });
    const stdout = [];
    const stderr = [];
    let settled = false;
    const finish = (fn, value) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      fn(value);
    };
    const timer =
      timeoutMs > 0
        ? setTimeout(() => {
            child.kill('SIGKILL');
            finish(rejectPromise, new Error(`${command} ${commandArgs.join(' ')} timed out after ${timeoutMs}ms`));
          }, timeoutMs)
        : undefined;
    child.stdout.on('data', (chunk) => stdout.push(chunk));
    child.stderr.on('data', (chunk) => {
      stderr.push(chunk);
      if (echoStderr) process.stderr.write(chunk);
    });
    child.on('error', (error) => finish(rejectPromise, error));
    child.on('close', (status) => {
      const output = Buffer.concat(stdout).toString('utf8').trim();
      const errors = Buffer.concat(stderr).toString('utf8').trim();
      if (status !== 0) return finish(rejectPromise, new Error(`${command} ${commandArgs.join(' ')} failed with ${status}: ${errors || output}`));
      finish(resolvePromise, output);
    });
    if (input !== undefined) {
      child.stdin.write(input);
      child.stdin.end();
    }
  });
}

/**
 * Run agent-browser in JSON mode. Supports both historical call shapes while
 * scripts migrate:
 * - runAgent(args, commandArgs, timeoutMs?, session?, input?)
 * - runAgent(args, session, commandArgs, timeoutMsOrInput?)
 */
export async function runAgent(args, commandArgsOrSession, timeoutOrCommandArgs = 30_000, sessionOrTimeout, input) {
  let commandArgs;
  let timeoutMs;
  let session;
  let commandInput = input;
  if (Array.isArray(commandArgsOrSession)) {
    commandArgs = commandArgsOrSession;
    timeoutMs = typeof timeoutOrCommandArgs === 'number' ? timeoutOrCommandArgs : 30_000;
    session = typeof sessionOrTimeout === 'string' ? sessionOrTimeout : args.session;
  } else {
    session = commandArgsOrSession;
    commandArgs = timeoutOrCommandArgs;
    if (typeof sessionOrTimeout === 'number') timeoutMs = sessionOrTimeout;
    else {
      timeoutMs = args.timeoutMs ?? 30_000;
      commandInput = sessionOrTimeout;
    }
  }
  const fullArgs = ['--json', ...(session ? ['--session', session] : []), ...commandArgs];
  const stdout = await spawnCapture(args.agentBrowser, fullArgs, {
    timeoutMs,
    input: commandInput,
    echoStderr: args.agentBrowserEchoStderr === true,
  });
  if (!stdout) return null;
  const parsed = JSON.parse(stdout);
  if (parsed && typeof parsed === 'object' && 'success' in parsed) {
    if (parsed.success === false) throw new Error(`${args.agentBrowser} ${commandArgs.join(' ')} failed: ${parsed.error ?? stdout}`);
    return parsed.data ?? parsed;
  }
  return parsed;
}
