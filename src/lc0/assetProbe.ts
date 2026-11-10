export type AssetProbeStatus = 'present' | 'missing' | 'unknown';

const failures = new Map<string, { attempts: number; retryAt: number }>();

/** Completion callbacks may render and probe again. During cooldown callers
 * must return unknown without publishing another completion notification. */
export function assetProbeRetryAllowed(urls: readonly string[]): boolean {
  return Date.now() >= (failures.get(JSON.stringify(urls))?.retryAt ?? 0);
}

/** Only HTTP success and 404/410 are definitive. A bounded transient failure
 * leaves availability unknown, so a later caller can retry without a reload. */
export async function probeAssetUrls(
  urls: readonly string[],
  { timeoutMs = 8_000, signal }: { timeoutMs?: number; signal?: AbortSignal } = {},
): Promise<AssetProbeStatus> {
  if (!assetProbeRetryAllowed(urls)) return 'unknown';
  const key = JSON.stringify(urls);
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
      const attempts = Math.min(6, (failures.get(key)?.attempts ?? 0) + 1);
      failures.set(key, { attempts, retryAt: Date.now() + Math.min(30_000, 1_000 * 2 ** (attempts - 1)) });
    } else failures.delete(key);
    return status;
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener('abort', abort);
  }
}
