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

export function releaseAfterSession(resultFile, release) {
  if (existsSync(cleanupMarker(resultFile))) return false;
  release();
  return true;
}

export async function shutdownDelay(ms, signal) {
  await delay(ms, undefined, { signal }).catch((error) => {
    signal.throwIfAborted();
    throw error;
  });
}

export async function runOwnedSession(bin, argv, signal, wrapped) {
  signal.throwIfAborted();
  // Separate from terminal Ctrl-C; don't interrupt a benchlock wrapper that holds a live session's mutex.
  const child = spawn(bin, argv, { stdio: 'inherit', detached: true });
  const stop = () => {
    if (!wrapped) child.kill('SIGTERM');
  };
  signal.addEventListener('abort', stop, { once: true });
  try {
    const [status] = await once(child, 'close');
    return status;
  } finally {
    signal.removeEventListener('abort', stop);
  }
}
