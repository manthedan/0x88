import { BIG_NETS, type BigNetConfig, Bt4WorkerSearcher } from './bt4Engine.ts';
import { isMemoryConstrainedDevice } from './deviceMemoryClass.ts';

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
  dispose(): void;
};

type PoolRecord<T extends DisposableSearcher> = {
  searcher: T;
  timer: TimeoutHandle | null;
};

export class BigNetSessionPool<T extends DisposableSearcher> {
  private readonly records = new Map<BigNetKey, PoolRecord<T>>();
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

  acquire(key: BigNetKey): T {
    // Phones have one resident slot. Terminate the old worker before even
    // constructing its replacement, including sessions retained by the TTL.
    if (this.constrained()) {
      for (const other of this.records.keys()) {
        if (other !== key) this.disposeNow(other);
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
    record.searcher.dispose();
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

export function acquireBigNetSearcher(key: BigNetKey): Bt4WorkerSearcher {
  installPagehideDisposal();
  return defaultPool.acquire(key);
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
