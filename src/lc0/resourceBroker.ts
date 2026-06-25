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
export type ResourceBrokerPolicy = 'exclusive' | 'shared' | 'bounded';
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

export interface SharedParticipant {
  engineId: string;
  maxThreads: number;
  weight?: number;
}

/**
 * Deterministic shared-policy allocation: every participant with weight > 0
 * gets a 1-thread baseline, then the remaining budget is dealt one thread at
 * a time in weight-descending round-robin order to engines still under their
 * `maxThreads` cap. Surplus stranded by capped engines (e.g. single-threaded
 * WASI builds) therefore flows to engines that can use it.
 */
export function allocateSharedThreads(budget: number, participants: SharedParticipant[]): Map<string, number> {
  const allocation = new Map<string, number>();
  const eligible = participants.filter((participant) => (participant.weight ?? 1) > 0);
  if (!eligible.length) return allocation;
  for (const participant of eligible) allocation.set(participant.engineId, 1);
  let remaining = Math.max(0, Math.floor(budget)) - eligible.length;
  const order = [...eligible].sort((a, b) => (b.weight ?? 1) - (a.weight ?? 1) || a.engineId.localeCompare(b.engineId));
  while (remaining > 0) {
    let dealt = false;
    for (const participant of order) {
      if (remaining <= 0) break;
      const current = allocation.get(participant.engineId)!;
      if (current >= Math.max(1, Math.floor(participant.maxThreads))) continue;
      allocation.set(participant.engineId, current + 1);
      remaining -= 1;
      dealt = true;
    }
    if (!dealt) break;
  }
  return allocation;
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
  /** Max concurrent CPU engine leases under the 'bounded' policy. Default 2. */
  maxConcurrentCpu?: number;
}

export class EngineResourceBroker {
  private policy: ResourceBrokerPolicy;
  private dial: PerformanceDial;
  private readonly maxConcurrentCpu: number;
  private readonly environment: () => ResourceBrokerEnvironment;
  private readonly profiles = new Map<string, EngineResourceProfile>();
  private readonly active = new Map<symbol, { engineId: string; resourceClass: EngineResourceClass; threads: number }>();
  private readonly queue: QueuedRequest[] = [];

  constructor(options: EngineResourceBrokerOptions = {}) {
    this.policy = options.policy ?? 'exclusive';
    this.dial = options.dial ?? 'balanced';
    this.maxConcurrentCpu = Math.max(1, Math.floor(options.maxConcurrentCpu ?? 2));
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

  private sharedGrant(engineId: string, profile: EngineResourceProfile): number {
    const participants: SharedParticipant[] = [];
    for (const [id, candidate] of this.profiles) {
      if (candidate.resourceClass !== 'cpu') continue;
      participants.push({ engineId: id, maxThreads: candidate.maxThreads, weight: candidate.weight ?? 1 });
    }
    const allocation = allocateSharedThreads(this.cpuBudget(), participants);
    return clampThreads(allocation.get(engineId) ?? 1, profile);
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
      if (this.policy === 'bounded' && this.activeCpuLeases() >= this.maxConcurrentCpu) return;
      const next = this.queue.shift()!;
      next.cleanup();
      const profile = this.profiles.get(next.engineId)!;
      const threads = this.policy === 'exclusive'
        ? clampThreads(this.cpuBudget(), profile)
        : this.sharedGrant(next.engineId, profile);
      next.grant(this.makeLease(next.engineId, profile, threads));
    }
  }

  /**
   * Acquire a lease for one search. GPU leases and `shared`-policy CPU leases
   * resolve immediately; `exclusive` CPU leases queue FIFO behind the current
   * holder; `bounded` CPU leases resolve immediately up to maxConcurrentCpu and
   * queue beyond that. All queued requests reject with AbortError if the
   * request's signal fires first.
   */
  async acquire(request: ResourceLeaseRequest): Promise<ResourceLease> {
    const profile = this.profiles.get(request.engineId);
    if (!profile) throw new Error(`Engine ${request.engineId} is not registered with the resource broker`);
    if (request.signal?.aborted) throw abortError();

    if (profile.resourceClass === 'gpu') {
      return this.makeLease(request.engineId, profile, 1);
    }

    if (this.policy === 'shared') {
      return this.makeLease(request.engineId, profile, this.sharedGrant(request.engineId, profile));
    }

    const currentActive = this.activeCpuLeases();
    const hasQueue = this.queue.length > 0;
    const canStart = this.policy === 'exclusive'
      ? currentActive === 0 && !hasQueue
      : currentActive < this.maxConcurrentCpu && !hasQueue;

    if (canStart) {
      const threads = this.policy === 'exclusive'
        ? clampThreads(this.cpuBudget(), profile)
        : this.sharedGrant(request.engineId, profile);
      return this.makeLease(request.engineId, profile, threads);
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
