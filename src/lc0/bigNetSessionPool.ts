import { BIG_NETS, type BigNetConfig, Bt4WorkerSearcher } from './bt4Engine.ts';
import { isMemoryConstrainedDevice, phoneReclaimSettings } from './deviceMemoryClass.ts';
import { awaitEngineRetirements, awaitWorkerReclaim, hasEngineRetirements, trackEngineRetirement } from './engineRetirement.ts';
import { ExclusiveQueue } from './exclusiveQueue.ts';
import { untilAborted } from './modelCache.ts';
import { PHONE_LARGE_NET_REASON, phoneBigNetBlocked } from './phoneLargeNetPolicy.ts';

export type BigNetKey = BigNetConfig['key'];

export const BIG_NET_SESSION_IDLE_TTL_MS = 3 * 60 * 1000;

type TimeoutHandle = ReturnType<typeof setTimeout>;
type TimerScheduler = {
  setTimeout: typeof setTimeout;
  clearTimeout: typeof clearTimeout;
};

// Browsers throw "Illegal invocation" when setTimeout is called as a method of
// another object, so the default scheduler calls the globals directly.
const defaultScheduler: TimerScheduler = {
  setTimeout: ((handler: () => void, timeout?: number) => setTimeout(handler, timeout)) as typeof setTimeout,
  clearTimeout: (handle) => clearTimeout(handle),
};

type DisposableSearcher = {
  dispose(): void | Promise<void>;
};

type PoolRecord<T extends DisposableSearcher> = {
  searcher: T;
  timer: TimeoutHandle | null;
};

export class BigNetSessionPool<T extends DisposableSearcher> {
  private readonly records = new Map<BigNetKey, PoolRecord<T>>();
  private readonly retiring = new Set<Promise<void>>();
  private readonly acquisitions = new ExclusiveQueue();
  private readonly factory: (config: BigNetConfig) => T;
  private readonly scheduler: TimerScheduler;
  private readonly defaultIdleTtlMs: number;
  private readonly constrained: () => boolean;

  constructor(
    factory: (config: BigNetConfig) => T,
    scheduler: TimerScheduler = defaultScheduler,
    defaultIdleTtlMs = BIG_NET_SESSION_IDLE_TTL_MS,
    constrained = isMemoryConstrainedDevice,
  ) {
    this.factory = factory;
    this.scheduler = scheduler;
    this.defaultIdleTtlMs = defaultIdleTtlMs;
    this.constrained = constrained;
  }

  acquire(key: BigNetKey, exclusive = false): T {
    // Never retain retired nets alongside a new one on phones. Arena also
    // requests exclusive residency; other callers (Analysis's parallel init)
    // still own their active sessions and must not be evicted behind their back.
    if (this.constrained()) {
      for (const [other, record] of this.records) {
        if (other !== key && (exclusive || record.timer !== null)) this.disposeNow(other);
      }
    }
    let record = this.records.get(key);
    if (!record) {
      record = { searcher: this.factory(BIG_NETS[key]), timer: null };
      this.records.set(key, record);
    }
    if (record.timer) {
      this.scheduler.clearTimeout(record.timer);
      record.timer = null;
    }
    return record.searcher;
  }

  /** Await bounded worker cleanup before constructing a replacement on phones.
   * Shared acquisitions still preserve explicitly active Analysis rows.
   */
  acquireAsync(key: BigNetKey, exclusive = false, signal?: AbortSignal): Promise<T> {
    return this.acquisitions.run(async () => {
      if (this.constrained()) {
        do {
          for (const [other, record] of this.records) {
            if (other !== key && (exclusive || record.timer !== null)) this.disposeNow(other);
          }
          await Promise.all(this.retiring);
          signal?.throwIfAborted();
        } while (this.retiring.size > 0);
      }
      signal?.throwIfAborted();
      return this.acquire(key, exclusive);
    }, signal);
  }

  release(key: BigNetKey, idleTtlMs = this.defaultIdleTtlMs): void {
    const record = this.records.get(key);
    if (!record) return;
    if (record.timer) this.scheduler.clearTimeout(record.timer);
    if (this.constrained() || idleTtlMs <= 0) {
      this.disposeNow(key);
      return;
    }
    record.timer = this.scheduler.setTimeout(() => this.disposeNow(key), idleTtlMs);
  }

  releaseUnused(activeKeys: Iterable<BigNetKey>, idleTtlMs = this.defaultIdleTtlMs): void {
    const active = new Set(activeKeys);
    for (const key of this.records.keys()) {
      if (!active.has(key)) this.release(key, idleTtlMs);
    }
  }

  /** Transfer an exact record out of the pool without destroying its in-flight worker. */
  detach(key: BigNetKey, expected: T): boolean {
    const record = this.records.get(key);
    if (!record || record.searcher !== expected) return false;
    if (record.timer) this.scheduler.clearTimeout(record.timer);
    this.records.delete(key);
    return true;
  }

  disposeNow(key: BigNetKey): void {
    const record = this.records.get(key);
    if (!record) return;
    if (record.timer) this.scheduler.clearTimeout(record.timer);
    this.records.delete(key);
    const disposal = record.searcher.dispose();
    if (disposal) {
      const pending = trackEngineRetirement(Promise.resolve(disposal));
      this.retiring.add(pending);
      // Synchronous page/pool owners remain supported without unhandled rejection.
      void pending.finally(() => this.retiring.delete(pending)).catch(() => undefined);
    }
  }

  disposeAllNow(): void {
    for (const key of [...this.records.keys()]) this.disposeNow(key);
  }

  has(key: BigNetKey): boolean {
    return this.records.has(key);
  }

  peek(key: BigNetKey): T | null {
    return this.records.get(key)?.searcher ?? null;
  }
}

const defaultPool = new BigNetSessionPool<Bt4WorkerSearcher>((config) => new Bt4WorkerSearcher(config));
let pagehideDisposalInstalled = false;

function installPagehideDisposal(): void {
  if (pagehideDisposalInstalled || typeof window === 'undefined') return;
  pagehideDisposalInstalled = true;
  window.addEventListener('pagehide', (event) => {
    if ((event as PageTransitionEvent).persisted) return;
    defaultPool.disposeAllNow();
  });
}

export function acquireBigNetSearcher(key: BigNetKey, exclusive = false): Bt4WorkerSearcher {
  if (phoneBigNetBlocked(key)) throw new Error(PHONE_LARGE_NET_REASON);
  installPagehideDisposal();
  return defaultPool.acquire(key, exclusive);
}

export async function acquireBigNetSearcherAsync(key: BigNetKey, exclusive = false, signal?: AbortSignal): Promise<Bt4WorkerSearcher> {
  if (phoneBigNetBlocked(key)) throw new Error(PHONE_LARGE_NET_REASON);
  installPagehideDisposal();
  const searcher = await defaultPool.acquireAsync(key, exclusive, signal);
  // Consent can be revoked while a previous worker drains.
  if (phoneBigNetBlocked(key)) {
    defaultPool.disposeNow(key);
    throw new Error(PHONE_LARGE_NET_REASON);
  }
  return searcher;
}

/** Cold phone loads only; callers must not add an await on the desktop/warm path. */
export function preparePhoneBigNet(
  key: BigNetKey,
  evalCacheEntries: number,
  signal: AbortSignal,
  retireCpu: () => void,
  onWait: () => void,
  exclusive = false,
): Promise<void> | undefined {
  if (!isMemoryConstrainedDevice() || phoneBigNetBlocked(key)) return;
  signal.throwIfAborted();
  if (exclusive) defaultPool.releaseUnused([key], 0);
  const existing = defaultPool.peek(key);
  if (existing?.isInitializedFor({ evalCacheEntries })) return;
  // A differently configured warm session would otherwise retire inside init(),
  // after the reclaim gate. Leave same-owner in-flight cold loads alone.
  if (existing?.loaded) defaultPool.disposeNow(key);
  const settings = phoneReclaimSettings();
  if (settings.retireCpu) retireCpu();
  return (async () => {
    do {
      if (hasEngineRetirements()) await untilAborted(awaitEngineRetirements(), signal);
      await awaitWorkerReclaim(settings.reclaimGapMs, signal, onWait);
    } while (hasEngineRetirements());
  })();
}

export function releaseBigNetSearcher(key: BigNetKey, idleTtlMs = BIG_NET_SESSION_IDLE_TTL_MS): void {
  defaultPool.release(key, idleTtlMs);
}

export function releaseUnusedBigNetSearchers(activeKeys: Iterable<BigNetKey>, idleTtlMs = BIG_NET_SESSION_IDLE_TTL_MS): void {
  defaultPool.releaseUnused(activeKeys, idleTtlMs);
}

export function detachBigNetSearcher(key: BigNetKey, expected: Bt4WorkerSearcher): boolean {
  return defaultPool.detach(key, expected);
}

export function disposeBigNetSearcherNow(key: BigNetKey): void {
  defaultPool.disposeNow(key);
}

export function disposeAllBigNetSearchersNow(): void {
  defaultPool.disposeAllNow();
}

export function hasBigNetSearcher(key: BigNetKey): boolean {
  return defaultPool.has(key);
}

export function peekBigNetSearcher(key: BigNetKey): Bt4WorkerSearcher | null {
  return defaultPool.peek(key);
}
