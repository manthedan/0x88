import { resolvePublicAssetUrl } from './assetUrls.ts';

export type Lc0ModelCacheMode = 'url' | 'cache' | 'memory';
export type Lc0ModelLoadSource = 'url' | 'memory' | 'cache-storage' | 'network';

export interface Lc0ModelManifestEntry {
  file: string;
  url: string;
  bytes?: number;
  sha256?: string;
}

export interface Lc0ModelManifest {
  models?: Lc0ModelManifestEntry[];
}

export interface Lc0ModelLoadTelemetry {
  /** Where the bytes/reference handed to ORT came from. */
  source: Lc0ModelLoadSource;
  /** Request.cache used for the model fetch, when this loader owned a fetch. */
  requestCache?: RequestCache;
  manifestMs?: number;
  cacheReadMs?: number;
  downloadMs?: number;
  hashMs?: number;
  cacheWriteMs?: number;
  totalMs: number;
  /** True when a streamed download used one preallocated output buffer. */
  preallocatedDownload?: boolean;
}

export interface Lc0ModelLoadResult {
  model: string | ArrayBuffer;
  url: string;
  mode: Lc0ModelCacheMode;
  cacheStatus: 'disabled' | 'unavailable' | 'hit' | 'miss';
  bytes?: number;
  expectedBytes?: number;
  /** Computed sha256 of the loaded bytes, when SubtleCrypto was available. */
  sha256?: string;
  expectedSha256?: string;
  /** true/false when sha256 was checked against the manifest; undefined when not checkable. */
  sha256Valid?: boolean;
  /** Set when a stale/corrupt cached entry was evicted and refetched from the network. */
  revalidated?: boolean;
  elapsedMs: number;
  telemetry: Lc0ModelLoadTelemetry;
}

export interface Lc0ModelLoadOptions {
  cache?: boolean;
  cacheName?: string;
  manifestUrl?: string;
  /**
   * Network download progress. Providing it forces the load to fetch the bytes
   * itself (streamed) even when cache=false, so the caller gets bytes in
   * memory ('memory' mode) instead of a URL for the runtime to fetch opaquely.
   * Not called for cache hits. `total` comes from Content-Length or the
   * manifest and may be undefined.
   */
  onProgress?: (loadedBytes: number, totalBytes?: number) => void;
}

const DEFAULT_CACHE_NAME = 'lc0-browser-models-v1';
const DEFAULT_MANIFEST_URL = resolvePublicAssetUrl('/models/lc0/manifest.json');

function defaultManifestUrlForModel(modelUrl: string): string {
  try {
    const url = new URL(modelUrl, location.href);
    if (url.pathname.startsWith('/models/lc0/')) return new URL('/models/lc0/manifest.json', url).href;
    if (url.pathname.startsWith('/models/maia3/')) return new URL('/models/maia3/manifest.json', url).href;
  } catch {
    // Fall through to the configured local/default manifest.
  }
  return DEFAULT_MANIFEST_URL;
}

function nowMs(): number {
  return typeof performance === 'undefined' ? Date.now() : performance.now();
}

function cacheApiAvailable(): boolean {
  return typeof caches !== 'undefined' && typeof fetch !== 'undefined' && typeof Response !== 'undefined';
}

function telemetry(started: number, source: Lc0ModelLoadSource, values: Omit<Lc0ModelLoadTelemetry, 'source' | 'totalMs'> = {}): Lc0ModelLoadTelemetry {
  return { source, ...values, totalMs: nowMs() - started };
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

async function fetchManifestEntryWithTiming(modelUrl: string, manifestUrl: string): Promise<{ entry?: Lc0ModelManifestEntry; elapsedMs: number }> {
  const started = nowMs();
  const entry = await fetchManifestEntry(modelUrl, manifestUrl);
  return { entry, elapsedMs: nowMs() - started };
}

export interface Lc0ModelBytesExpectation {
  expectedBytes?: number;
  expectedSha256?: string;
}

export interface Lc0ModelBytesCheck {
  ok: boolean;
  byteLength: number;
  /** Lowercase hex sha256, present only when it was actually computed. */
  sha256?: string;
  /** true when the sha256 was compared against an expected value. */
  sha256Checked: boolean;
  /** Time spent hashing, present only when sha256 was computed. */
  hashMs?: number;
  reason?: string;
}

/** Compute the lowercase-hex sha256 of model bytes using SubtleCrypto. */
export async function sha256Hex(
  bytes: ArrayBuffer | Uint8Array,
  subtle: SubtleCrypto | undefined = globalThis.crypto?.subtle,
): Promise<string> {
  if (!subtle) throw new Error('SubtleCrypto unavailable for sha256');
  const source = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  // The DOM lib types digest()'s arg as ArrayBuffer-backed; a plain Uint8Array
  // is structurally fine at runtime, so cast past the SharedArrayBuffer generic.
  const digest = await subtle.digest('SHA-256', source as unknown as BufferSource);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('');
}

/**
 * Validate model bytes against the manifest's byte length and sha256. Byte
 * length is a cheap first gate; sha256 is the real integrity check. When
 * SubtleCrypto is unavailable (e.g. insecure context) sha256 cannot be checked,
 * so length-validated bytes are accepted with sha256Checked=false.
 */
export async function verifyLc0ModelBytes(
  bytes: ArrayBuffer,
  expectation: Lc0ModelBytesExpectation = {},
): Promise<Lc0ModelBytesCheck> {
  const byteLength = bytes.byteLength;
  if (expectation.expectedBytes !== undefined && byteLength !== expectation.expectedBytes) {
    return { ok: false, byteLength, sha256Checked: false, reason: `byte length mismatch: got ${byteLength}, expected ${expectation.expectedBytes}` };
  }
  if (!expectation.expectedSha256) return { ok: true, byteLength, sha256Checked: false };
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) return { ok: true, byteLength, sha256Checked: false };
  const hashStarted = nowMs();
  const sha256 = await sha256Hex(bytes, subtle);
  const hashMs = nowMs() - hashStarted;
  if (sha256 !== expectation.expectedSha256.toLowerCase()) {
    return { ok: false, byteLength, sha256, sha256Checked: true, hashMs, reason: `sha256 mismatch: got ${sha256}, expected ${expectation.expectedSha256.toLowerCase()}` };
  }
  return { ok: true, byteLength, sha256, sha256Checked: true, hashMs };
}

interface FetchedModelBytes {
  bytes: ArrayBuffer;
  downloadMs: number;
  preallocated: boolean;
}

/** Fetch model bytes, streaming chunks through onProgress when possible. */
async function fetchModelBytes(
  request: Request,
  modelUrl: string,
  expectedBytes: number | undefined,
  onProgress: Lc0ModelLoadOptions['onProgress'],
): Promise<FetchedModelBytes> {
  const started = nowMs();
  const response = await fetch(request);
  if (!response.ok) throw new Error(`LC0 model fetch failed for ${modelUrl}: ${response.status}`);
  if (!response.body) {
    const bytes = await response.arrayBuffer();
    return { bytes, downloadMs: nowMs() - started, preallocated: false };
  }

  const headerLength = Number(response.headers.get('content-length') ?? '');
  // A content-encoded response's Content-Length counts compressed bytes, not
  // the decoded stream measured here; fall back to the manifest size then.
  const encoded = !!response.headers.get('content-encoding');
  const total = !encoded && Number.isFinite(headerLength) && headerLength > 0 ? headerLength : expectedBytes;
  const reader = response.body.getReader();
  let target = total !== undefined ? new Uint8Array(total) : undefined;
  let chunks: Uint8Array[] = target ? [] : [];
  let loaded = 0;
  onProgress?.(0, total);
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (target && loaded + value.byteLength <= target.byteLength) {
      target.set(value, loaded);
    } else {
      if (target) {
        chunks = [target.subarray(0, loaded)];
        target = undefined;
      }
      chunks.push(value);
    }
    loaded += value.byteLength;
    onProgress?.(loaded, total);
  }

  if (target) {
    const exact = loaded === target.byteLength;
    const bytes = exact ? target.buffer : target.slice(0, loaded).buffer;
    return { bytes, downloadMs: nowMs() - started, preallocated: exact };
  }

  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return { bytes: bytes.buffer, downloadMs: nowMs() - started, preallocated: false };
}

export async function loadLc0ModelForOrt(modelUrl: string, options: Lc0ModelLoadOptions = {}): Promise<Lc0ModelLoadResult> {
  const started = nowMs();
  if (!options.cache) {
    if (!options.onProgress) {
      return { model: modelUrl, url: modelUrl, mode: 'url', cacheStatus: 'disabled', elapsedMs: nowMs() - started, telemetry: telemetry(started, 'url') };
    }
    // Progress requires owning the fetch: download (with normal HTTP caching),
    // validate, and hand the bytes over without persisting to Cache Storage.
    const manifest = await fetchManifestEntryWithTiming(modelUrl, options.manifestUrl ?? defaultManifestUrlForModel(modelUrl));
    const expectation: Lc0ModelBytesExpectation = { expectedBytes: manifest.entry?.bytes, expectedSha256: manifest.entry?.sha256 };
    const fetched = await fetchModelBytes(new Request(modelUrl), modelUrl, expectation.expectedBytes, options.onProgress);
    const check = await verifyLc0ModelBytes(fetched.bytes, expectation);
    if (!check.ok) throw new Error(`LC0 model validation failed for ${modelUrl}: ${check.reason}`);
    return {
      model: fetched.bytes, url: modelUrl, mode: 'memory', cacheStatus: 'disabled', bytes: fetched.bytes.byteLength,
      expectedBytes: expectation.expectedBytes, sha256: check.sha256, expectedSha256: expectation.expectedSha256,
      sha256Valid: check.sha256Checked ? true : undefined, elapsedMs: nowMs() - started,
      telemetry: telemetry(started, 'memory', {
        manifestMs: manifest.elapsedMs,
        downloadMs: fetched.downloadMs,
        hashMs: check.hashMs,
        preallocatedDownload: fetched.preallocated,
      }),
    };
  }
  if (!cacheApiAvailable()) {
    return { model: modelUrl, url: modelUrl, mode: 'url', cacheStatus: 'unavailable', elapsedMs: nowMs() - started, telemetry: telemetry(started, 'url') };
  }

  const manifest = await fetchManifestEntryWithTiming(modelUrl, options.manifestUrl ?? defaultManifestUrlForModel(modelUrl));
  const expectation: Lc0ModelBytesExpectation = { expectedBytes: manifest.entry?.bytes, expectedSha256: manifest.entry?.sha256 };
  const cacheKey = new Request(modelUrl);
  const fetchRequest = new Request(modelUrl, { cache: 'force-cache' });
  const cache = await caches.open(options.cacheName ?? DEFAULT_CACHE_NAME);

  const cached = await cache.match(cacheKey);
  if (cached) {
    const cacheReadStarted = nowMs();
    const bytes = await cached.arrayBuffer();
    const cacheReadMs = nowMs() - cacheReadStarted;
    const check = await verifyLc0ModelBytes(bytes, expectation);
    if (check.ok) {
      return {
        model: bytes, url: modelUrl, mode: 'cache', cacheStatus: 'hit', bytes: bytes.byteLength,
        expectedBytes: expectation.expectedBytes, sha256: check.sha256, expectedSha256: expectation.expectedSha256,
        sha256Valid: check.sha256Checked ? true : undefined, elapsedMs: nowMs() - started,
        telemetry: telemetry(started, 'cache-storage', {
          manifestMs: manifest.elapsedMs,
          cacheReadMs,
          hashMs: check.hashMs,
        }),
      };
    }
    // A stale or corrupt cache entry was evicted. The recovery fetch must bypass
    // the browser HTTP cache; otherwise an unchanged mutable URL can return the
    // same obsolete bytes and fail integrity until the HTTP entry expires.
    await cache.delete(cacheKey);
    const reloadRequest = new Request(modelUrl, { cache: 'reload' });
    const result = await fetchAndCacheModel(cache, cacheKey, reloadRequest, modelUrl, expectation, started, options.onProgress, {
      manifestMs: manifest.elapsedMs,
      cacheReadMs,
      hashMs: check.hashMs,
    });
    return { ...result, revalidated: true };
  }

  return fetchAndCacheModel(cache, cacheKey, fetchRequest, modelUrl, expectation, started, options.onProgress, { manifestMs: manifest.elapsedMs });
}

async function fetchAndCacheModel(
  cache: Cache,
  cacheKey: Request,
  fetchRequest: Request,
  modelUrl: string,
  expectation: Lc0ModelBytesExpectation,
  started: number,
  onProgress: Lc0ModelLoadOptions['onProgress'],
  inheritedTelemetry: Pick<Lc0ModelLoadTelemetry, 'manifestMs' | 'cacheReadMs' | 'hashMs'> = {},
): Promise<Lc0ModelLoadResult> {
  const fetched = await fetchModelBytes(fetchRequest, modelUrl, expectation.expectedBytes, onProgress);
  const check = await verifyLc0ModelBytes(fetched.bytes, expectation);
  if (!check.ok) throw new Error(`LC0 model validation failed for ${modelUrl}: ${check.reason}`);
  // Only validated bytes are written to the cache, so a corrupt download is
  // never persisted for future loads.
  const cacheWriteStarted = nowMs();
  await cache.put(cacheKey, new Response(fetched.bytes));
  const cacheWriteMs = nowMs() - cacheWriteStarted;
  return {
    model: fetched.bytes, url: modelUrl, mode: 'cache', cacheStatus: 'miss', bytes: fetched.bytes.byteLength,
    expectedBytes: expectation.expectedBytes, sha256: check.sha256, expectedSha256: expectation.expectedSha256,
    sha256Valid: check.sha256Checked ? true : undefined, elapsedMs: nowMs() - started,
    telemetry: telemetry(started, 'network', {
      ...inheritedTelemetry,
      requestCache: fetchRequest.cache,
      downloadMs: fetched.downloadMs,
      hashMs: (inheritedTelemetry.hashMs ?? 0) + (check.hashMs ?? 0) || undefined,
      cacheWriteMs,
      preallocatedDownload: fetched.preallocated,
    }),
  };
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
  const integrity = result.expectedSha256 === undefined
    ? ''
    : result.sha256Valid === true ? ' · sha256 ok'
    : result.sha256Valid === false ? ' · sha256 BAD'
    : ' · sha256 unchecked';
  const revalidated = result.revalidated ? ' · revalidated' : '';
  return `${result.cacheStatus}${mb}${integrity}${revalidated}${timing}`;
}
