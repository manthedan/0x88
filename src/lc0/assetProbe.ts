export type AssetProbeStatus = 'present' | 'missing' | 'unknown';

/** Only HTTP success and 404/410 are definitive. A bounded transient failure
 * leaves availability unknown, so a later caller can retry without a reload. */
export async function probeAssetUrls(
  urls: readonly string[],
  { timeoutMs = 8_000, signal }: { timeoutMs?: number; signal?: AbortSignal } = {},
): Promise<AssetProbeStatus> {
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
          return response.status === 404 || response.status === 410 ? 'missing' : 'unknown';
        } catch {
          return 'unknown';
        }
      }),
    ).then((results): AssetProbeStatus => (results.includes('missing') ? 'missing' : results.every((status) => status === 'present') ? 'present' : 'unknown'));
    return await Promise.race([statuses, cancelled]);
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener('abort', abort);
  }
}
