export type AssetProbeStatus = 'present' | 'missing' | 'unknown';

type ProbeFailure = {
  attempts: number;
  retryAt: number;
};
type NotificationRound = {
  listeners: Map<() => void, () => void>;
  dueAt?: number;
};
const failures = new Map<string, ProbeFailure>();
const subscribers = new Map<string, NotificationRound>();
const maxAttempts = 3;
// Fresh closures from repeated renders replace old subscriptions instead of
// growing the next notification round. Allow several independent UI consumers.
// N keys over K attempts notify at most 4*N*(2*K-1) times: one completion
// per attempt and one cooldown expiry between attempts, regardless of renders.
const maxSubscribersPerKey = 4;
let notificationTimer: ReturnType<typeof setTimeout> | undefined;
let notificationAt = Infinity;

function scheduleNotifications(): void {
  let dueAt = Infinity;
  for (const round of subscribers.values()) dueAt = Math.min(dueAt, round.dueAt ?? Infinity);
  if (dueAt === notificationAt) return;
  if (notificationTimer !== undefined) clearTimeout(notificationTimer);
  notificationTimer = undefined;
  notificationAt = dueAt;
  if (dueAt === Infinity) return;
  notificationTimer = setTimeout(
    () => {
      notificationTimer = undefined;
      notificationAt = Infinity;
      const listeners = new Set<() => void>();
      // Detach ALL due rounds before calling any render. A re-subscription must
      // belong to the next round, even when several cooldowns expire together.
      const due = [...subscribers].filter(([, round]) => (round.dueAt ?? Infinity) <= Date.now());
      for (const [key] of due) subscribers.delete(key);
      for (const [, round] of due) {
        for (const [listener, unsubscribe] of round.listeners) {
          listeners.add(listener);
          unsubscribe();
        }
      }
      scheduleNotifications();
      for (const listener of listeners) listener();
    },
    Math.max(0, dueAt - Date.now()),
  );
  (notificationTimer as ReturnType<typeof setTimeout> & { unref?: () => void }).unref?.();
}

/** One-shot subscription for the next state change. At most four callbacks per
 * URL group are retained; the oldest is replaced when a caller renders again. */
export function subscribeAssetProbeChange(urls: readonly string[], listener?: () => void, signal?: AbortSignal): () => void {
  if (!listener || signal?.aborted) return () => {};
  const key = JSON.stringify(urls);
  let round = subscribers.get(key);
  if (!round) {
    round = { listeners: new Map() };
    subscribers.set(key, round);
  }
  const listeners = round.listeners;
  const existing = listeners.get(listener);
  if (existing) return existing;
  const unsubscribe = () => {
    listeners.delete(listener);
    signal?.removeEventListener('abort', unsubscribe);
    if (!listeners.size && subscribers.get(key) === round) subscribers.delete(key);
    scheduleNotifications();
  };
  listeners.set(listener, unsubscribe);
  signal?.addEventListener('abort', unsubscribe, { once: true });
  if (listeners.size > maxSubscribersPerKey) listeners.values().next().value?.();
  return unsubscribe;
}

/** Batch all state changes in this tick, deduplicating callbacks across keys.
 * Repeated notifications without a state change must not be scheduled by callers. */
export function notifyAssetProbeChange(urls: readonly string[], listener?: () => void): void {
  subscribeAssetProbeChange(urls, listener);
  const round = subscribers.get(JSON.stringify(urls));
  if (round) round.dueAt = Date.now();
  scheduleNotifications();
}

/** One-shot cooldown notification. Unsubscribe (or abort) when the listener
 * goes away; without listeners no retry timer or callback is retained. */
export function subscribeAssetProbeRetry(urls: readonly string[], listener: () => void, signal?: AbortSignal): () => void {
  const key = JSON.stringify(urls);
  const failure = failures.get(key);
  if (!failure || failure.attempts >= maxAttempts || signal?.aborted) return () => {};
  const unsubscribe = subscribeAssetProbeChange(urls, listener, signal);
  const round = subscribers.get(key);
  if (round) round.dueAt = Math.min(round.dueAt ?? Infinity, failure.retryAt);
  scheduleNotifications();
  return unsubscribe;
}

/** Completion callbacks may render and probe again. Remain silent during the
 * cooldown, then wake those renders once so recovery needs no unrelated UI event. */
export function assetProbeRetryAllowed(urls: readonly string[], onRetry?: () => void): boolean {
  if (Date.now() >= (failures.get(JSON.stringify(urls))?.retryAt ?? 0)) {
    subscribeAssetProbeChange(urls, onRetry);
    return true;
  }
  if (onRetry) subscribeAssetProbeRetry(urls, onRetry);
  return false;
}

/** Only HTTP success and 404/410 are definitive. A bounded transient failure
 * leaves availability unknown until the third attempt, then settles missing for
 * the session so existing pickers stop displaying checking indefinitely. */
export async function probeAssetUrls(
  urls: readonly string[],
  { timeoutMs = 8_000, signal }: { timeoutMs?: number; signal?: AbortSignal } = {},
): Promise<AssetProbeStatus> {
  const key = JSON.stringify(urls);
  if ((failures.get(key)?.attempts ?? 0) >= maxAttempts) {
    notifyAssetProbeChange(urls);
    return 'missing';
  }
  if (!assetProbeRetryAllowed(urls)) return 'unknown';
  const controller = new AbortController();
  let finish!: (status: AssetProbeStatus) => void;
  const cancelled = new Promise<AssetProbeStatus>((resolve) => {
    finish = resolve;
  });
  const abort = () => {
    controller.abort();
    finish('unknown');
  };
  signal?.addEventListener('abort', abort, { once: true });
  const timer = setTimeout(abort, Math.max(1, timeoutMs));
  (timer as ReturnType<typeof setTimeout> & { unref?: () => void }).unref?.();
  if (signal?.aborted) abort();
  try {
    const statuses = Promise.all(
      urls.map(async (url): Promise<AssetProbeStatus> => {
        try {
          const response = await fetch(url, { method: 'HEAD', cache: 'no-store', signal: controller.signal });
          if (response.ok) return 'present';
          if (response.status === 404 || response.status === 410) {
            finish('missing');
            controller.abort();
            return 'missing';
          }
          return 'unknown';
        } catch {
          return 'unknown';
        }
      }),
    ).then((results): AssetProbeStatus => (results.includes('missing') ? 'missing' : results.every((status) => status === 'present') ? 'present' : 'unknown'));
    const status = await Promise.race([statuses, cancelled]);
    if (status === 'unknown') {
      const previous = failures.get(key);
      const attempts = (previous?.attempts ?? 0) + 1;
      if (attempts >= maxAttempts) {
        failures.set(key, { attempts, retryAt: 0 });
        return 'missing';
      }
      failures.set(key, { attempts, retryAt: Date.now() + 1_000 * 2 ** (attempts - 1) });
    } else {
      failures.delete(key);
    }
    return status;
  } finally {
    notifyAssetProbeChange(urls);
    clearTimeout(timer);
    signal?.removeEventListener('abort', abort);
  }
}
