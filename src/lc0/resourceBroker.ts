/**
 * EngineResourceBroker: per-page CPU/GPU budget arbitration for browser
 * engines. Engines acquire a lease per search, apply the granted thread
 * count via UCI `Threads`, and release when the search settles. Policies:
 * `exclusive` (arena turn-taking: one CPU lease at a time, full budget) and
 * `shared` (analysis: deterministic weighted split over registered CPU
 * participants, never blocking). GPU leases never queue and never draw down
 * the CPU ledger; GPU contention is arbitrated by the shared evaluator
 * service instead. See docs/engine_resource_broker_design.md.
 */

export type EngineResourceClass = 'cpu' | 'gpu';
export type ResourceBrokerPolicy = 'exclusive' | 'shared';
export type PerformanceDial = 'eco' | 'balanced' | 'max';

export interface EngineResourceProfile {
  resourceClass: EngineResourceClass;
  /** Upper bound the engine build supports; broker grants never exceed it. */
  maxThreads: number;
  /** Relative share under the `shared` policy. Default 1; 0 excludes the engine from shares. */
  weight?: number;
}

export interface ResourceBrokerEnvironment {
  hardwareConcurrency: number;
  crossOriginIsolated: boolean;
  /** Measured usable-thread count from the calibration probe, when present. */
  calibratedThreads?: number;
}

export interface ResourceLeaseRequest {
  engineId: string;
  signal?: AbortSignal;
}

export interface ResourceLease {
  engineId: string;
  resourceClass: EngineResourceClass;
  /** Granted UCI thread count (already clamped to profile and budget). */
  threads: number;
  release(): void;
}

export interface ResourceBrokerSnapshot {
  policy: ResourceBrokerPolicy;
  dial: PerformanceDial;
  cpuBudget: number;
  activeLeases: { engineId: string; resourceClass: EngineResourceClass; threads: number }[];
  queuedEngineIds: string[];
}

export function defaultBrowserEnvironment(): ResourceBrokerEnvironment {
  const nav = typeof navigator !== 'undefined' ? navigator : undefined;
  return {
    hardwareConcurrency: Math.max(1, Math.floor(nav?.hardwareConcurrency ?? 1)),
    crossOriginIsolated: typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated === true,
  };
}

/**
 * CPU thread budget for a dial. Base prefers the calibrated probe result over
 * `hardwareConcurrency` (hybrid chips overstate usable cores). Two threads
 * stay reserved for the main thread/UI and GPU-engine overhead in `balanced`;
 * `max` reserves one; `eco` halves the base for thermals/battery. Without
 * cross-origin isolation, threaded WASM builds cannot run at all, so the
 * budget is 1.
 */
export function cpuThreadBudget(env: ResourceBrokerEnvironment, dial: PerformanceDial): number {
  if (!env.crossOriginIsolated) return 1;
  const base = Math.max(1, Math.floor(env.calibratedThreads ?? env.hardwareConcurrency));
  if (dial === 'eco') return Math.max(1, Math.floor(base / 2));
  if (dial === 'max') return Math.max(1, base - 1);
  return Math.max(1, base - 2);
}

function clampThreads(granted: number, profile: EngineResourceProfile): number {
  return Math.max(1, Math.min(Math.max(1, Math.floor(profile.maxThreads)), Math.floor(granted)));
}

function abortError(): Error {
  const error = new Error('Resource lease request aborted');
  error.name = 'AbortError';
  return error;
}

interface QueuedRequest {
  engineId: string;
  grant: (lease: ResourceLease) => void;
  reject: (error: Error) => void;
  cleanup: () => void;
}

export interface EngineResourceBrokerOptions {
  policy?: ResourceBrokerPolicy;
  dial?: PerformanceDial;
  /** Environment provider; injected for tests, defaults to the live browser globals. */
  environment?: () => ResourceBrokerEnvironment;
}

export class EngineResourceBroker {
  private policy: ResourceBrokerPolicy;
  private dial: PerformanceDial;
  private readonly environment: () => ResourceBrokerEnvironment;
  private readonly profiles = new Map<string, EngineResourceProfile>();
  private readonly active = new Map<symbol, { engineId: string; resourceClass: EngineResourceClass; threads: number }>();
  private readonly queue: QueuedRequest[] = [];

  constructor(options: EngineResourceBrokerOptions = {}) {
    this.policy = options.policy ?? 'exclusive';
    this.dial = options.dial ?? 'balanced';
    this.environment = options.environment ?? defaultBrowserEnvironment;
  }

  /** Register or update an engine's resource profile. Idempotent. */
  register(engineId: string, profile: EngineResourceProfile): void {
    this.profiles.set(engineId, { weight: 1, ...profile });
  }

  unregister(engineId: string): void {
    this.profiles.delete(engineId);
  }

  setPolicy(policy: ResourceBrokerPolicy): void {
    this.policy = policy;
    this.drainQueue();
  }

  setDial(dial: PerformanceDial): void {
    this.dial = dial;
  }

  getDial(): PerformanceDial {
    return this.dial;
  }

  cpuBudget(): number {
    return cpuThreadBudget(this.environment(), this.dial);
  }

  private activeCpuLeases(): number {
    let count = 0;
    for (const lease of this.active.values()) if (lease.resourceClass === 'cpu') count += 1;
    return count;
  }

  private sharedGrant(profile: EngineResourceProfile): number {
    const budget = this.cpuBudget();
    let totalWeight = 0;
    for (const candidate of this.profiles.values()) {
      if (candidate.resourceClass === 'cpu') totalWeight += Math.max(0, candidate.weight ?? 1);
    }
    const weight = Math.max(0, profile.weight ?? 1);
    if (totalWeight <= 0 || weight <= 0) return 1;
    return clampThreads(Math.floor((budget * weight) / totalWeight), profile);
  }

  private makeLease(engineId: string, profile: EngineResourceProfile, threads: number): ResourceLease {
    const key = Symbol(engineId);
    this.active.set(key, { engineId, resourceClass: profile.resourceClass, threads });
    let released = false;
    return {
      engineId,
      resourceClass: profile.resourceClass,
      threads,
      release: () => {
        if (released) return;
        released = true;
        this.active.delete(key);
        this.drainQueue();
      },
    };
  }

  private drainQueue(): void {
    while (this.queue.length) {
      if (this.policy === 'exclusive' && this.activeCpuLeases() > 0) return;
      const next = this.queue.shift()!;
      next.cleanup();
      const profile = this.profiles.get(next.engineId)!;
      const threads = this.policy === 'exclusive'
        ? clampThreads(this.cpuBudget(), profile)
        : this.sharedGrant(profile);
      next.grant(this.makeLease(next.engineId, profile, threads));
    }
  }

  /**
   * Acquire a lease for one search. GPU leases and `shared`-policy CPU leases
   * resolve immediately; `exclusive` CPU leases queue FIFO behind the current
   * holder and reject with AbortError if the request's signal fires first.
   */
  async acquire(request: ResourceLeaseRequest): Promise<ResourceLease> {
    const profile = this.profiles.get(request.engineId);
    if (!profile) throw new Error(`Engine ${request.engineId} is not registered with the resource broker`);
    if (request.signal?.aborted) throw abortError();

    if (profile.resourceClass === 'gpu') {
      // GPU leases are informational (snapshot/diagnostics): CPU-side overhead
      // for GPU engines lives in the standing budget reserve.
      return this.makeLease(request.engineId, profile, 1);
    }

    if (this.policy === 'shared') {
      return this.makeLease(request.engineId, profile, this.sharedGrant(profile));
    }

    if (this.activeCpuLeases() === 0 && this.queue.length === 0) {
      return this.makeLease(request.engineId, profile, clampThreads(this.cpuBudget(), profile));
    }

    return new Promise<ResourceLease>((resolve, reject) => {
      const entry: QueuedRequest = {
        engineId: request.engineId,
        grant: resolve,
        reject,
        cleanup: () => request.signal?.removeEventListener('abort', onAbort),
      };
      const onAbort = () => {
        const index = this.queue.indexOf(entry);
        if (index >= 0) this.queue.splice(index, 1);
        entry.cleanup();
        reject(abortError());
      };
      request.signal?.addEventListener('abort', onAbort, { once: true });
      this.queue.push(entry);
    });
  }

  snapshot(): ResourceBrokerSnapshot {
    return {
      policy: this.policy,
      dial: this.dial,
      cpuBudget: this.cpuBudget(),
      activeLeases: [...this.active.values()].map((lease) => ({ ...lease })),
      queuedEngineIds: this.queue.map((entry) => entry.engineId),
    };
  }
}

const DIAL_STORAGE_KEY = 'tinyLeela.performanceDial';
const CALIBRATED_THREADS_KEY = 'tinyLeela.calibratedThreads';

interface StorageLike {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

export function loadPerformanceDial(storage: StorageLike | undefined, fallback: PerformanceDial = 'balanced'): PerformanceDial {
  const raw = storage?.getItem(DIAL_STORAGE_KEY);
  return raw === 'eco' || raw === 'balanced' || raw === 'max' ? raw : fallback;
}

export function savePerformanceDial(storage: StorageLike | undefined, dial: PerformanceDial): void {
  storage?.setItem(DIAL_STORAGE_KEY, dial);
}

export function loadCalibratedThreads(storage: StorageLike | undefined): number | undefined {
  const raw = Number(storage?.getItem(CALIBRATED_THREADS_KEY));
  return Number.isFinite(raw) && raw >= 1 ? Math.floor(raw) : undefined;
}

export function saveCalibratedThreads(storage: StorageLike | undefined, threads: number): void {
  if (Number.isFinite(threads) && threads >= 1) storage?.setItem(CALIBRATED_THREADS_KEY, String(Math.floor(threads)));
}
