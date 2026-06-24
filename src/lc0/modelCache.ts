type ImportMetaWithEnv = ImportMeta & { env?: Record<string, string | undefined> };
const env = (import.meta as ImportMetaWithEnv).env ?? {};

export type Lc0ModelCacheMode = 'url' | 'cache' | 'memory';
export type Lc0ModelLoadSource = 'url' | 'memory' | 'cache-storage' | 'network';

export interface Lc0ModelManifestEntry {
  file: string;
  url: string;
  /** Optional immutable/content-addressed URL for the bytes named by `url`. */
  artifactUrl?: string;
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
  /** URL handed to ORT or used for byte fetches; may be content-addressed. */
  url: string;
  /** Stable/logical URL requested by the caller when different from `url`. */
  logicalUrl?: string;
  mode: Lc0ModelCacheMode;
  cacheStatus: 'disabled' | 'unavailable' | 'quota-limited' | 'hit' | 'miss';
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
  /** Optional channel manifest whose release manifest maps stable URLs to immutable artifact URLs. */
  channelUrl?: string | null;
  /**
   * Network download progress. Providing it forces the load to fetch the bytes
   * itself (streamed) even when cache=false, so the caller gets bytes in
   * memory ('memory' mode) instead of a URL for the runtime to fetch opaquely.
   * Not called for cache hits. `total` comes from Content-Length or the
   * manifest and may be undefined.
   */
  onProgress?: (loadedBytes: number, totalBytes?: number) => void;
  /**
   * Ask the browser for persistent storage before admitting a model into Cache
   * Storage. Browsers may still deny it, so callers must tolerate fallback.
   */
  requestPersistentStorage?: boolean;
  /**
   * Minimum free quota to leave after caching this model. If the browser reports
   * less, the loader falls back to URL mode instead of materializing bytes.
   */
  minimumFreeBytesAfterCache?: number;
}

const DEFAULT_CACHE_NAME = 'lc0-browser-models-v1';
const DEFAULT_MANIFEST_URL = '/models/lc0/manifest.json';
const DEFAULT_CACHE_FREE_BYTES_RESERVE = 64 * 1024 * 1024;

function defaultManifestUrlForModel(modelUrl: string): string {
  try {
    const url = new URL(modelUrl, location.href);
    // Model bytes may be served from the configured R2 asset origin, but the
    // small manifests are part of the Netlify app shell. Keeping manifest
    // lookups same-origin avoids guaranteed 404s on assets.0x88.app while the
    // release-channel resolver below still maps logical model paths to R2.
    if (url.pathname.startsWith('/models/lc0/')) return '/models/lc0/manifest.json';
    if (url.pathname.startsWith('/models/maia3/')) return '/models/maia3/manifest.json';
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

async function cacheQuotaAllows(expectedBytes: number | undefined, options: Lc0ModelLoadOptions): Promise<boolean> {
  if (expectedBytes === undefined || expectedBytes <= 0) return true;
  if (typeof navigator === 'undefined') return true;
  const storage = (navigator as Navigator & {
    storage?: {
      estimate?: () => Promise<{ usage?: number; quota?: number }>;
      persist?: () => Promise<boolean>;
    };
  }).storage;
  if (!storage?.estimate) return true;
  if (options.requestPersistentStorage && storage.persist) {
    try { await storage.persist(); } catch { /* best effort only */ }
  }
  try {
    const estimate = await storage.estimate();
    const quota = Number(estimate.quota);
    const usage = Number(estimate.usage);
    if (!Number.isFinite(quota) || !Number.isFinite(usage) || quota <= 0) return true;
    const reserve = Math.max(0, Math.floor(Number(options.minimumFreeBytesAfterCache ?? DEFAULT_CACHE_FREE_BYTES_RESERVE) || 0));
    return quota - usage - expectedBytes >= reserve;
  } catch {
    return true;
  }
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

async function fetchManifestEntryWithTiming(modelUrl: string, manifestUrl: string): Promise<{ entry?: Lc0ModelManifestEntry; manifestUrl: string; elapsedMs: number }> {
  const started = nowMs();
  const entry = await fetchManifestEntry(modelUrl, manifestUrl);
  return { entry, manifestUrl, elapsedMs: nowMs() - started };
}

function manifestArtifactUrl(entry: Lc0ModelManifestEntry | undefined, manifestUrl: string): string | undefined {
  if (!entry?.artifactUrl) return undefined;
  return new URL(entry.artifactUrl, new URL(manifestUrl, location.href)).href;
}

function logicalUrlField(modelUrl: string, resolvedUrl: string): string | undefined {
  return resolvedUrl === modelUrl ? undefined : modelUrl;
}

interface Lc0ArtifactChannelManifest {
  releaseManifestUrl?: string;
}

interface Lc0ArtifactReleaseManifest {
  artifacts?: Array<{ logicalUrl: string; artifactUrl: string; bytes?: number; sha256?: string }>;
}

interface ResolvedModelArtifact {
  url: string;
  expectedBytes?: number;
  expectedSha256?: string;
}

function cleanChannelUrl(raw: string | null | undefined): string | undefined {
  const trimmed = raw?.trim();
  if (!trimmed) return undefined;
  try {
    return new URL(trimmed, location.href).href;
  } catch {
    return undefined;
  }
}

function configuredChannelUrl(): string | undefined {
  const globals = globalThis as { LC0_BROWSER_ARTIFACT_CHANNEL_URL?: string };
  // Channel manifests are a deployment control plane. Do not accept query-param
  // overrides here: URL-mode callers hand the resolved artifact URL to ORT, so
  // the channel source must come from trusted build/global configuration.
  return cleanChannelUrl(globals.LC0_BROWSER_ARTIFACT_CHANNEL_URL)
    ?? cleanChannelUrl(env.VITE_LC0_ARTIFACT_CHANNEL_URL);
}

function logicalPathForReleaseLookup(modelUrl: string): string | undefined {
  try {
    return new URL(modelUrl, location.href).pathname;
  } catch {
    return undefined;
  }
}

async function resolveChannelArtifactUrl(modelUrl: string, channelUrl: string | undefined): Promise<ResolvedModelArtifact | undefined> {
  if (!channelUrl) return undefined;
  const logicalPath = logicalPathForReleaseLookup(modelUrl);
  if (!logicalPath) return undefined;
  try {
    const channelResponse = await fetch(channelUrl, { cache: 'no-cache' });
    if (!channelResponse.ok) return undefined;
    const channel = await channelResponse.json() as Lc0ArtifactChannelManifest;
    if (!channel.releaseManifestUrl) return undefined;
    const releaseUrl = new URL(channel.releaseManifestUrl, channelUrl).href;
    const releaseResponse = await fetch(releaseUrl, { cache: 'force-cache' });
    if (!releaseResponse.ok) return undefined;
    const release = await releaseResponse.json() as Lc0ArtifactReleaseManifest;
    const artifact = release.artifacts?.find((entry) => entry.logicalUrl === logicalPath);
    return artifact?.artifactUrl ? {
      url: new URL(artifact.artifactUrl, releaseUrl).href,
      expectedBytes: artifact.bytes,
      expectedSha256: artifact.sha256,
    } : undefined;
  } catch {
    return undefined;
  }
}

async function resolveArtifactUrl(modelUrl: string, manifest: { entry?: Lc0ModelManifestEntry; manifestUrl: string }, channelUrl: string | undefined): Promise<ResolvedModelArtifact> {
  const manifestUrl = manifestArtifactUrl(manifest.entry, manifest.manifestUrl);
  if (manifestUrl) return { url: manifestUrl, expectedBytes: manifest.entry?.bytes, expectedSha256: manifest.entry?.sha256 };
  return await resolveChannelArtifactUrl(modelUrl, channelUrl)
    ?? { url: modelUrl, expectedBytes: manifest.entry?.bytes, expectedSha256: manifest.entry?.sha256 };
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
  const channelUrl = options.channelUrl === null ? undefined : cleanChannelUrl(options.channelUrl) ?? configuredChannelUrl();
  if (!options.cache) {
    const manifest = await fetchManifestEntryWithTiming(modelUrl, options.manifestUrl ?? defaultManifestUrlForModel(modelUrl));
    const resolved = await resolveArtifactUrl(modelUrl, manifest, channelUrl);
    if (!options.onProgress) {
      return {
        model: resolved.url,
        url: resolved.url,
        logicalUrl: logicalUrlField(modelUrl, resolved.url),
        mode: 'url',
        cacheStatus: 'disabled',
        elapsedMs: nowMs() - started,
        telemetry: telemetry(started, 'url', { manifestMs: manifest.elapsedMs }),
      };
    }
    // Progress requires owning the fetch: download (with normal HTTP caching),
    // validate, and hand the bytes over without persisting to Cache Storage.
    const expectation: Lc0ModelBytesExpectation = { expectedBytes: resolved.expectedBytes, expectedSha256: resolved.expectedSha256 };
    const fetched = await fetchModelBytes(new Request(resolved.url), resolved.url, expectation.expectedBytes, options.onProgress);
    const check = await verifyLc0ModelBytes(fetched.bytes, expectation);
    if (!check.ok) throw new Error(`LC0 model validation failed for ${modelUrl}: ${check.reason}`);
    return {
      model: fetched.bytes, url: resolved.url, logicalUrl: logicalUrlField(modelUrl, resolved.url), mode: 'memory', cacheStatus: 'disabled', bytes: fetched.bytes.byteLength,
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
  const manifest = await fetchManifestEntryWithTiming(modelUrl, options.manifestUrl ?? defaultManifestUrlForModel(modelUrl));
  const resolved = await resolveArtifactUrl(modelUrl, manifest, channelUrl);
  if (!cacheApiAvailable()) {
    return {
      model: resolved.url,
      url: resolved.url,
      logicalUrl: logicalUrlField(modelUrl, resolved.url),
      mode: 'url',
      cacheStatus: 'unavailable',
      elapsedMs: nowMs() - started,
      telemetry: telemetry(started, 'url', { manifestMs: manifest.elapsedMs }),
    };
  }
  const expectation: Lc0ModelBytesExpectation = { expectedBytes: resolved.expectedBytes, expectedSha256: resolved.expectedSha256 };
  const cacheKey = new Request(resolved.url);
  const fetchRequest = new Request(resolved.url, { cache: 'force-cache' });
  const cache = await caches.open(options.cacheName ?? DEFAULT_CACHE_NAME);

  const cached = await cache.match(cacheKey);
  if (cached) {
    const cacheReadStarted = nowMs();
    const bytes = await cached.arrayBuffer();
    const cacheReadMs = nowMs() - cacheReadStarted;
    const check = await verifyLc0ModelBytes(bytes, expectation);
    if (check.ok) {
      return {
        model: bytes, url: resolved.url, logicalUrl: logicalUrlField(modelUrl, resolved.url), mode: 'cache', cacheStatus: 'hit', bytes: bytes.byteLength,
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
    if (!(await cacheQuotaAllows(expectation.expectedBytes, options))) {
      return {
        model: resolved.url,
        url: resolved.url,
        logicalUrl: logicalUrlField(modelUrl, resolved.url),
        mode: 'url',
        cacheStatus: 'quota-limited',
        elapsedMs: nowMs() - started,
        telemetry: telemetry(started, 'url', {
          manifestMs: manifest.elapsedMs,
          cacheReadMs,
          hashMs: check.hashMs,
        }),
        revalidated: true,
      };
    }
    const reloadRequest = new Request(resolved.url, { cache: 'reload' });
    const result = await fetchAndCacheModel(cache, cacheKey, reloadRequest, modelUrl, resolved.url, expectation, started, options.onProgress, {
      manifestMs: manifest.elapsedMs,
      cacheReadMs,
      hashMs: check.hashMs,
    });
    return { ...result, revalidated: true };
  }

  if (!(await cacheQuotaAllows(expectation.expectedBytes, options))) {
    return {
      model: resolved.url,
      url: resolved.url,
      logicalUrl: logicalUrlField(modelUrl, resolved.url),
      mode: 'url',
      cacheStatus: 'quota-limited',
      elapsedMs: nowMs() - started,
      telemetry: telemetry(started, 'url', { manifestMs: manifest.elapsedMs }),
    };
  }

  return fetchAndCacheModel(cache, cacheKey, fetchRequest, modelUrl, resolved.url, expectation, started, options.onProgress, { manifestMs: manifest.elapsedMs });
}

async function fetchAndCacheModel(
  cache: Cache,
  cacheKey: Request,
  fetchRequest: Request,
  logicalModelUrl: string,
  fetchModelUrl: string,
  expectation: Lc0ModelBytesExpectation,
  started: number,
  onProgress: Lc0ModelLoadOptions['onProgress'],
  inheritedTelemetry: Pick<Lc0ModelLoadTelemetry, 'manifestMs' | 'cacheReadMs' | 'hashMs'> = {},
): Promise<Lc0ModelLoadResult> {
  const fetched = await fetchModelBytes(fetchRequest, fetchModelUrl, expectation.expectedBytes, onProgress);
  const check = await verifyLc0ModelBytes(fetched.bytes, expectation);
  if (!check.ok) throw new Error(`LC0 model validation failed for ${fetchModelUrl}: ${check.reason}`);
  // Only validated bytes are written to the cache, so a corrupt download is
  // never persisted for future loads.
  const cacheWriteStarted = nowMs();
  await cache.put(cacheKey, new Response(fetched.bytes));
  const cacheWriteMs = nowMs() - cacheWriteStarted;
  return {
    model: fetched.bytes, url: fetchModelUrl, logicalUrl: logicalUrlField(logicalModelUrl, fetchModelUrl), mode: 'cache', cacheStatus: 'miss', bytes: fetched.bytes.byteLength,
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
  if (result.cacheStatus === 'quota-limited') return `quota-limited${timing}`;
  const integrity = result.expectedSha256 === undefined
    ? ''
    : result.sha256Valid === true ? ' · sha256 ok'
    : result.sha256Valid === false ? ' · sha256 BAD'
    : ' · sha256 unchecked';
  const revalidated = result.revalidated ? ' · revalidated' : '';
  return `${result.cacheStatus}${mb}${integrity}${revalidated}${timing}`;
}
