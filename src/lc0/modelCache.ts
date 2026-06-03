export type Lc0ModelCacheMode = 'url' | 'cache';

export interface Lc0ModelManifestEntry {
  file: string;
  url: string;
  bytes?: number;
  sha256?: string;
}

export interface Lc0ModelManifest {
  models?: Lc0ModelManifestEntry[];
}

export interface Lc0ModelLoadResult {
  model: string | ArrayBuffer;
  url: string;
  mode: Lc0ModelCacheMode;
  cacheStatus: 'disabled' | 'unavailable' | 'hit' | 'miss';
  bytes?: number;
  expectedBytes?: number;
  elapsedMs: number;
}

export interface Lc0ModelLoadOptions {
  cache?: boolean;
  cacheName?: string;
  manifestUrl?: string;
}

const DEFAULT_CACHE_NAME = 'lc0-browser-models-v1';
const DEFAULT_MANIFEST_URL = '/models/lc0/manifest.json';

function nowMs(): number {
  return typeof performance === 'undefined' ? Date.now() : performance.now();
}

function cacheApiAvailable(): boolean {
  return typeof caches !== 'undefined' && typeof fetch !== 'undefined' && typeof Response !== 'undefined';
}

async function fetchManifestEntry(modelUrl: string, manifestUrl: string): Promise<Lc0ModelManifestEntry | undefined> {
  try {
    const response = await fetch(manifestUrl, { cache: 'no-cache' });
    if (!response.ok) return undefined;
    const manifest = await response.json() as Lc0ModelManifest;
    const absolute = new URL(modelUrl, location.href).href;
    return manifest.models?.find((entry) => {
      const entryUrl = new URL(entry.url, location.href).href;
      return entry.url === modelUrl || entryUrl === absolute || entry.file === modelUrl.split('/').pop();
    });
  } catch {
    return undefined;
  }
}

async function responseToArrayBuffer(response: Response, expectedBytes?: number): Promise<ArrayBuffer> {
  const bytes = await response.arrayBuffer();
  if (expectedBytes !== undefined && bytes.byteLength !== expectedBytes) {
    throw new Error(`Cached LC0 model byte length mismatch: got ${bytes.byteLength}, expected ${expectedBytes}`);
  }
  return bytes;
}

export async function loadLc0ModelForOrt(modelUrl: string, options: Lc0ModelLoadOptions = {}): Promise<Lc0ModelLoadResult> {
  const started = nowMs();
  if (!options.cache) {
    return { model: modelUrl, url: modelUrl, mode: 'url', cacheStatus: 'disabled', elapsedMs: nowMs() - started };
  }
  if (!cacheApiAvailable()) {
    return { model: modelUrl, url: modelUrl, mode: 'url', cacheStatus: 'unavailable', elapsedMs: nowMs() - started };
  }

  const manifestEntry = await fetchManifestEntry(modelUrl, options.manifestUrl ?? DEFAULT_MANIFEST_URL);
  const expectedBytes = manifestEntry?.bytes;
  const request = new Request(modelUrl, { cache: 'force-cache' });
  const cache = await caches.open(options.cacheName ?? DEFAULT_CACHE_NAME);
  const cached = await cache.match(request);
  if (cached) {
    const model = await responseToArrayBuffer(cached, expectedBytes);
    return { model, url: modelUrl, mode: 'cache', cacheStatus: 'hit', bytes: model.byteLength, expectedBytes, elapsedMs: nowMs() - started };
  }

  const response = await fetch(request);
  if (!response.ok) throw new Error(`LC0 model fetch failed for ${modelUrl}: ${response.status}`);
  await cache.put(request, response.clone());
  const model = await responseToArrayBuffer(response, expectedBytes);
  return { model, url: modelUrl, mode: 'cache', cacheStatus: 'miss', bytes: model.byteLength, expectedBytes, elapsedMs: nowMs() - started };
}

export interface Lc0ModelCacheClearResult {
  cleared: boolean;
  removedEntries: number;
}

/**
 * Delete the LC0 model Cache Storage bucket. Returns how many cached entries
 * were removed so the UI can report a meaningful result. Safe to call when the
 * Cache API is unavailable or the cache was never created.
 */
export async function clearLc0ModelCache(cacheName: string = DEFAULT_CACHE_NAME): Promise<Lc0ModelCacheClearResult> {
  if (!cacheApiAvailable()) return { cleared: false, removedEntries: 0 };
  const has = await caches.has(cacheName);
  if (!has) return { cleared: false, removedEntries: 0 };
  const cache = await caches.open(cacheName);
  const removedEntries = (await cache.keys()).length;
  const cleared = await caches.delete(cacheName);
  return { cleared, removedEntries };
}

export function describeLc0ModelLoad(result: Lc0ModelLoadResult): string {
  const mb = result.bytes === undefined ? '' : ` · ${(result.bytes / 1_000_000).toFixed(1)} MB`;
  const timing = ` · ${result.elapsedMs.toFixed(0)} ms`;
  if (result.cacheStatus === 'disabled') return `disabled${timing}`;
  if (result.cacheStatus === 'unavailable') return `unavailable${timing}`;
  return `${result.cacheStatus}${mb}${timing}`;
}
