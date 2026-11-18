import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { existsSync, mkdirSync, readFileSync, rmdirSync, rmSync, writeFileSync } from 'node:fs';
import { hostname } from 'node:os';
import { dirname, join } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';

export function claimOutput(dir) {
  mkdirSync(dirname(dir), { recursive: true });
  try {
    mkdirSync(dir);
  } catch (error) {
    if (error.code === 'EEXIST') throw Error(`Use a new --out directory; already exists: ${dir}`);
    throw error;
  }
}

// One coordinator per output directory, held from inspection through final reporting, so a resume
// cannot pick the same pending sessions as a live run. Never taken over automatically: two resumes
// racing to replace a stale lock could both win. Only SIGKILL/power loss leave it behind.
export function lockRun(dir) {
  const file = join(dir, 'coordinator.lock');
  const owner = `${hostname()} ${process.pid}\n`;
  try {
    writeFileSync(file, owner, { flag: 'wx' });
  } catch (error) {
    if (error.code !== 'EEXIST') throw error;
    throw Error(`Another coordinator (${readFileSync(file, 'utf8').trim()}) owns ${dir}; if that process is gone, remove ${file}`);
  }
  return () => {
    try {
      if (readFileSync(file, 'utf8') === owner) rmSync(file);
    } catch (error) {
      if (error.code !== 'ENOENT') throw error;
    }
  };
}

export function watchShutdown() {
  const controller = new AbortController();
  const stop = (signal) => controller.abort(Object.assign(Error(`Interrupted by ${signal}`), { exitCode: signal === 'SIGINT' ? 130 : 143 }));
  process.on('SIGINT', stop);
  process.on('SIGTERM', stop);
  return {
    signal: controller.signal,
    dispose() {
      process.off('SIGINT', stop);
      process.off('SIGTERM', stop);
    },
  };
}

export async function acquireLock(dir, signal) {
  const owner = `${hostname()} ${process.pid} ${new Date().toISOString()}\n`;
  const file = join(dir, 'owner');
  for (;;) {
    signal.throwIfAborted();
    try {
      mkdirSync(dir);
      break;
    } catch (error) {
      if (error.code !== 'EEXIST') throw error;
    }
    console.log(`Waiting for shared mutex ${dir}; no stale/foreign lock is removed`);
    await shutdownDelay(5000, signal);
  }
  try {
    writeFileSync(file, owner, { flag: 'wx' });
  } catch (error) {
    if (error.code !== 'EEXIST') rmdirSync(dir);
    throw error;
  }
  return () => {
    try {
      if (readFileSync(file, 'utf8') === owner) rmSync(dir, { recursive: true });
    } catch (error) {
      if (error.code !== 'ENOENT') throw error;
    }
  };
}

export const cleanupMarker = (resultFile) => `${resultFile}.cleanup-pending`;
export const cancellationMarker = (resultFile) => `${resultFile}.cancelled`;

export function checkSessionCancellation(resultFile, signal) {
  signal.throwIfAborted();
  if (existsSync(cancellationMarker(resultFile))) throw Error('Interrupted before browser startup');
}

function refusedConnect(error) {
  if (error?.code === 'ECONNREFUSED' && error.syscall === 'connect') return true;
  return error instanceof AggregateError && error.errors.length > 0 && error.errors.every(refusedConnect);
}

export const safariCreationRefused = (error) => !!error.webDriverError || refusedConnect(error.cause);

export function releaseAfterSession(resultFile, release) {
  if (existsSync(cleanupMarker(resultFile))) return false;
  release();
  return true;
}

export async function waitForCleanup(resultFile, retry, pause = () => delay(5000)) {
  const marker = cleanupMarker(resultFile);
  while (existsSync(marker)) {
    await pause();
    if (!retry || !existsSync(marker)) continue;
    try {
      await retry();
      rmSync(marker, { force: true });
    } catch (error) {
      console.error(`Owned Safari teardown still unconfirmed: ${error.message}`);
    }
  }
}

export async function shutdownDelay(ms, signal) {
  await delay(ms, undefined, { signal }).catch((error) => {
    signal.throwIfAborted();
    throw error;
  });
}

export async function runOwnedSession(bin, argv, signal, wrappedResultFile) {
  signal.throwIfAborted();
  // Separate from terminal Ctrl-C; don't interrupt a benchlock wrapper that holds a live session's mutex.
  const child = spawn(bin, argv, { stdio: 'inherit', detached: true });
  const stop = () => {
    if (!wrappedResultFile) child.kill('SIGTERM');
    else {
      try {
        writeFileSync(cancellationMarker(wrappedResultFile), `${signal.reason}\n`);
      } catch (error) {
        console.error(`Unable to cancel queued session: ${error.message}`);
      }
    }
  };
  signal.addEventListener('abort', stop, { once: true });
  try {
    const [status] = await once(child, 'close');
    return status;
  } finally {
    signal.removeEventListener('abort', stop);
  }
}
