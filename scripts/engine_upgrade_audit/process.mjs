import { spawn } from 'node:child_process';

let interrupted = false;

// Own a process group so a synchronous WASM deadlock or shell grandchild can
// be terminated too. No detached/background benchmarks survive this promise.
export function runProcess(command, args, { cwd, input, env, timeout = 600000, allowFailure = false } = {}) {
  if (interrupted) return Promise.reject(new Error('audit interrupted; remaining subprocesses cancelled'));
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { cwd, env, detached: process.platform !== 'win32', stdio: ['pipe', 'pipe', 'pipe'] });
    let stdout = '';
    let stderr = '';
    let timedOut = false;
    const kill = () => {
      try { process.kill(process.platform === 'win32' ? child.pid : -child.pid, 'SIGKILL'); } catch {}
    };
    const onSignal = () => { interrupted = true; kill(); };
    for (const signal of ['SIGINT', 'SIGTERM']) process.once(signal, onSignal);
    const cleanup = () => {
      clearTimeout(timer);
      for (const signal of ['SIGINT', 'SIGTERM']) process.removeListener(signal, onSignal);
    };
    const timer = setTimeout(() => { timedOut = true; kill(); }, timeout);
    child.stdout.setEncoding('utf8').on('data', (chunk) => { stdout += chunk; });
    child.stderr.setEncoding('utf8').on('data', (chunk) => { stderr += chunk; });
    child.stdin.on('error', () => {}); // EPIPE is reported through the exit status.
    child.stdin.end(input);
    child.on('error', (error) => { kill(); cleanup(); reject(error); });
    child.on('close', (code, signal) => {
      // A launcher may exit after forking a descendant with redirected stdio.
      // The group remains ours even when its leader/streams have closed.
      kill();
      cleanup();
      const result = { code, signal, timedOut, stdout, stderr };
      if (interrupted || (!allowFailure && (code !== 0 || timedOut))) reject(Object.assign(new Error(`${command}: ${interrupted ? 'audit interrupted' : timedOut ? 'timeout' : `exit ${code} (${signal})`}: ${stderr.slice(-1500)}`), { result }));
      else resolve(result);
    });
  });
}

export const shellQuote = (text) => `'${String(text).replaceAll("'", "'\\''")}'`;
