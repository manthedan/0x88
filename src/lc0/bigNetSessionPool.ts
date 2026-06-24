import { BIG_NETS, Bt4WorkerSearcher, type BigNetConfig } from './bt4Engine.ts';

export type BigNetKey = BigNetConfig['key'];

export const BIG_NET_SESSION_IDLE_TTL_MS = 3 * 60 * 1000;

type TimeoutHandle = ReturnType<typeof setTimeout>;
type TimerScheduler = {
  setTimeout: typeof setTimeout;
  clearTimeout: typeof clearTimeout;
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

  constructor(
    factory: (config: BigNetConfig) => T,
    scheduler: TimerScheduler = { setTimeout, clearTimeout },
    defaultIdleTtlMs = BIG_NET_SESSION_IDLE_TTL_MS,
  ) {
    this.factory = factory;
    this.scheduler = scheduler;
    this.defaultIdleTtlMs = defaultIdleTtlMs;
  }

  acquire(key: BigNetKey): T {
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
    if (idleTtlMs <= 0) {
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
