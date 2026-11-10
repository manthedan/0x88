export type AssetProbeStatus = 'present' | 'missing' | 'unknown';

type ProbeFailure = {
  attempts: number;
  retryAt: number;
  timer?: ReturnType<typeof setTimeout>;
  listeners: Map<() => void, () => void>;
};
const failures = new Map<string, ProbeFailure>();
const maxAttempts = 3;

function clearRetry(failure: ProbeFailure): void {
  clearTimeout(failure.timer);
  failure.timer = undefined;
  for (const unsubscribe of failure.listeners.values()) unsubscribe();
}

/** One-shot cooldown notification. Unsubscribe (or abort) when the listener
 * goes away; without listeners no retry timer is retained. */
export function subscribeAssetProbeRetry(urls: readonly string[], listener: () => void, signal?: AbortSignal): () => void {
  const failure = failures.get(JSON.stringify(urls));
  if (!failure || failure.attempts >= maxAttempts || signal?.aborted) return () => {};
  const existing = failure.listeners.get(listener);
  if (existing) return existing;
  const unsubscribe = () => {
    failure.listeners.delete(listener);
    signal?.removeEventListener('abort', unsubscribe);
    if (!failure.listeners.size) {
      clearTimeout(failure.timer);
      failure.timer = undefined;
    }
  };
  failure.listeners.set(listener, unsubscribe);
  signal?.addEventListener('abort', unsubscribe, { once: true });
  if (failure.timer === undefined) {
    failure.timer = setTimeout(
      () => {
        const listeners = [...failure.listeners.keys()];
        clearRetry(failure);
        for (const notify of listeners) notify();
      },
      Math.max(0, failure.retryAt - Date.now()),
    );
    (failure.timer as ReturnType<typeof setTimeout> & { unref?: () => void }).unref?.();
  }
  return unsubscribe;
}

/** Completion callbacks may render and probe again. Remain silent during the
 * cooldown, then wake those renders once so recovery needs no unrelated UI event. */
export function assetProbeRetryAllowed(urls: readonly string[], onRetry?: () => void): boolean {
  if (Date.now() >= (failures.get(JSON.stringify(urls))?.retryAt ?? 0)) return true;
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
  if ((failures.get(key)?.attempts ?? 0) >= maxAttempts) return 'missing';
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
      if (previous) clearRetry(previous);
      if (attempts >= maxAttempts) {
        failures.set(key, { attempts, retryAt: 0, listeners: new Map() });
        return 'missing';
      }
      failures.set(key, { attempts, retryAt: Date.now() + 1_000 * 2 ** (attempts - 1), listeners: new Map() });
    } else {
      const previous = failures.get(key);
      if (previous) clearRetry(previous);
      failures.delete(key);
    }
    return status;
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener('abort', abort);
  }
}
