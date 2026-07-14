type ImportMetaWithEnv = ImportMeta & { env?: Record<string, string | undefined> };
const env = (import.meta as ImportMetaWithEnv).env ?? {};

export type Lc0ModelCacheMode = 'url' | 'cache' | 'memory';
export type Lc0ModelLoadSource = 'url' | 'memory' | 'cache-storage' | 'network';
export type Lc0ModelVerificationMode = 'hashed' | 'metadata' | 'length-only' | 'unchecked';

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
  source: Lc0ModelLoadSource;
  requestCache?: RequestCache;
  manifestMs?: number;
  cacheReadMs?: number;
  downloadMs?: number;
  hashMs?: number;
  cacheWriteMs?: number;
  totalMs: number;
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
  /** Computed or metadata-trusted sha256 of the loaded bytes. */
  sha256?: string;
  expectedSha256?: string;
  /** true/false when sha256 was checked against the manifest; undefined when not checkable. */
  sha256Valid?: boolean;
  verification?: Lc0ModelVerificationMode;
  /** Set when a stale/corrupt cached entry was evicted and refetched from the network. */
  revalidated?: boolean;
  /** Storage persistence state after an explicit persistence request. */
  storagePersistent?: boolean;
  cacheReadMs?: number;
  downloadMs?: number;
  hashMs?: number;
  elapsedMs: number;
  telemetry: Lc0ModelLoadTelemetry;
}

export interface Lc0ModelVerificationMetadata {
  key: string;
  schemaVersion: number;
  runtimeVersion: string;
  cacheName: string;
  url: string;
  expectedBytes?: number;
  expectedSha256?: string;
  observedBytes: number;
  sha256?: string;
  verifiedAt: number;
  lastUsedAt: number;
}

export interface Lc0ModelVerificationMetadataStore {
  get(key: string): Promise<Lc0ModelVerificationMetadata | undefined>;
  put(entry: Lc0ModelVerificationMetadata): Promise<void>;
  delete(key: string): Promise<void>;
  list(cacheName: string): Promise<Lc0ModelVerificationMetadata[]>;
  clear(cacheName: string): Promise<void>;
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
   * Not called for cache hits. `total` is the decoded Content-Length or the
   * manifest byte count and may be undefined.
   */
  onProgress?: (loadedBytes: number, totalBytes?: number) => void;
  /** Request persistent browser storage after an explicit user cache opt-in. */
  requestPersistentStorage?: boolean;
  /** Minimum free quota retained after caching. Defaults to 64 MiB. */
  minimumFreeBytesAfterCache?: number;
  /** Rehash metadata-trusted entries after this age. Defaults to 30 days. */
  rehashAfterMs?: number;
  /** Model-level LRU bounds. Set to Infinity to disable a bound. */
  maxCacheEntries?: number;
  maxCacheBytes?: number;
  /** Injectable for tests or non-browser hosts; IndexedDB is used by default. */
  metadataStore?: Lc0ModelVerificationMetadataStore;
}

const DEFAULT_CACHE_NAME = 'lc0-browser-models-v1';
const DEFAULT_MANIFEST_URL = '/models/lc0/manifest.json';
const DEFAULT_CACHE_FREE_BYTES_RESERVE = 64 * 1024 * 1024;
const METADATA_DB_NAME = 'lc0-browser-model-cache-metadata';
const METADATA_STORE_NAME = 'verified-models';
const METADATA_SCHEMA_VERSION = 1;
const METADATA_RUNTIME_VERSION = 'lc0-model-cache-v2';
const DEFAULT_REHASH_AFTER_MS = 30 * 24 * 60 * 60 * 1000;
const DEFAULT_MAX_CACHE_ENTRIES = 8;
const DEFAULT_MAX_CACHE_BYTES = 1_000_000_000;

function defaultManifestUrlForModel(modelUrl: string): string {
  try {
    const url = new URL(modelUrl, location.href);
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

function epochMs(): number {
  return Date.now();
}

function absoluteUrl(url: string): string {
  try {
    return new URL(url, typeof location === 'undefined' ? 'http://localhost/' : location.href).href;
  } catch {
    return url;
  }
}

function cacheApiAvailable(): boolean {
  return typeof caches !== 'undefined' && typeof fetch !== 'undefined' && typeof Response !== 'undefined';
}

async function cacheQuotaAllows(expectedBytes: number | undefined, options: Lc0ModelLoadOptions): Promise<boolean> {
  if (expectedBytes === undefined || expectedBytes <= 0 || typeof navigator === 'undefined') return true;
  const storage = navigator.storage;
  if (!storage?.estimate) return true;
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

function metadataKey(cacheName: string, modelUrl: string): string {
  return `${cacheName}\n${absoluteUrl(modelUrl)}`;
}

function requestResult<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error('IndexedDB request failed'));
  });
}

function transactionDone(transaction: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    transaction.oncomplete = () => resolve();
    transaction.onabort = () => reject(transaction.error ?? new Error('IndexedDB transaction aborted'));
    transaction.onerror = () => reject(transaction.error ?? new Error('IndexedDB transaction failed'));
  });
}

let metadataDbPromise: Promise<IDBDatabase> | undefined;

function openMetadataDb(): Promise<IDBDatabase> {
  if (metadataDbPromise) return metadataDbPromise;
  metadataDbPromise = new Promise<IDBDatabase>((resolve, reject) => {
    const request = indexedDB.open(METADATA_DB_NAME, METADATA_SCHEMA_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(METADATA_STORE_NAME)) {
        const store = db.createObjectStore(METADATA_STORE_NAME, { keyPath: 'key' });
        store.createIndex('cacheName', 'cacheName', { unique: false });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error('Unable to open model metadata IndexedDB'));
    request.onblocked = () => reject(new Error('Model metadata IndexedDB upgrade was blocked'));
  }).catch((error) => {
    metadataDbPromise = undefined;
    throw error;
  });
  return metadataDbPromise as Promise<IDBDatabase>;
}

class IndexedDbModelMetadataStore implements Lc0ModelVerificationMetadataStore {
  async get(key: string): Promise<Lc0ModelVerificationMetadata | undefined> {
    const db = await openMetadataDb();
    const transaction = db.transaction(METADATA_STORE_NAME, 'readonly');
    const result = await requestResult(transaction.objectStore(METADATA_STORE_NAME).get(key));
    await transactionDone(transaction);
    return result as Lc0ModelVerificationMetadata | undefined;
  }

  async put(entry: Lc0ModelVerificationMetadata): Promise<void> {
    const db = await openMetadataDb();
    const transaction = db.transaction(METADATA_STORE_NAME, 'readwrite');
    transaction.objectStore(METADATA_STORE_NAME).put(entry);
    await transactionDone(transaction);
  }

  async delete(key: string): Promise<void> {
    const db = await openMetadataDb();
    const transaction = db.transaction(METADATA_STORE_NAME, 'readwrite');
    transaction.objectStore(METADATA_STORE_NAME).delete(key);
    await transactionDone(transaction);
  }

  async list(cacheName: string): Promise<Lc0ModelVerificationMetadata[]> {
    const db = await openMetadataDb();
    const transaction = db.transaction(METADATA_STORE_NAME, 'readonly');
    const store = transaction.objectStore(METADATA_STORE_NAME);
    const index = store.indexNames.contains('cacheName') ? store.index('cacheName') : undefined;
    const result = index
      ? await requestResult(index.getAll(IDBKeyRange.only(cacheName)))
      : (await requestResult(store.getAll())).filter((entry) => (entry as Lc0ModelVerificationMetadata).cacheName === cacheName);
    await transactionDone(transaction);
    return result as Lc0ModelVerificationMetadata[];
  }

  async clear(cacheName: string): Promise<void> {
    const entries = await this.list(cacheName);
    if (!entries.length) return;
    const db = await openMetadataDb();
    const transaction = db.transaction(METADATA_STORE_NAME, 'readwrite');
    const store = transaction.objectStore(METADATA_STORE_NAME);
    for (const entry of entries) store.delete(entry.key);
    await transactionDone(transaction);
  }
}

let defaultMetadataStore: Lc0ModelVerificationMetadataStore | undefined;

function browserMetadataStore(): Lc0ModelVerificationMetadataStore | undefined {
  if (typeof indexedDB === 'undefined') return undefined;
  defaultMetadataStore ??= new IndexedDbModelMetadataStore();
  return defaultMetadataStore;
}

async function ignoreMetadataFailure<T>(operation: (() => Promise<T>) | undefined): Promise<T | undefined> {
  if (!operation) return undefined;
  try {
    return await operation();
  } catch {
    // Cache Storage remains a correct, hash-validated fallback when IndexedDB
    // is disabled, blocked, full, or unavailable in a worker/browser mode.
    return undefined;
  }
}

const manifestFlights = new Map<string, Promise<Lc0ModelManifest | undefined>>();

async function fetchModelManifest(manifestUrl: string): Promise<Lc0ModelManifest | undefined> {
  const key = absoluteUrl(manifestUrl);
  const existing = manifestFlights.get(key);
  if (existing) return existing;
  const flight = (async () => {
    try {
      const response = await fetch(manifestUrl, { cache: 'no-cache' });
      if (!response.ok) return undefined;
      return await response.json() as Lc0ModelManifest;
    } catch {
      return undefined;
    }
  })();
  manifestFlights.set(key, flight);
  try {
    return await flight;
  } finally {
    if (manifestFlights.get(key) === flight) manifestFlights.delete(key);
  }
}

async function fetchManifestEntry(modelUrl: string, manifestUrl: string): Promise<Lc0ModelManifestEntry | undefined> {
  const manifest = await fetchModelManifest(manifestUrl);
  const absolute = absoluteUrl(modelUrl);
  return manifest?.models?.find((entry) => {
    const entryUrl = absoluteUrl(entry.url);
    return entry.url === modelUrl || entryUrl === absolute || entry.file === modelUrl.split('/').pop();
  });
}

async function fetchManifestEntryWithTiming(modelUrl: string, manifestUrl: string): Promise<{ entry?: Lc0ModelManifestEntry; manifestUrl: string; elapsedMs: number }> {
  const started = nowMs();
  const entry = await fetchManifestEntry(modelUrl, manifestUrl);
  return { entry, manifestUrl, elapsedMs: nowMs() - started };
}

function manifestArtifactUrl(entry: Lc0ModelManifestEntry | undefined, manifestUrl: string): string | undefined {
  if (!entry?.artifactUrl) return undefined;
  return new URL(entry.artifactUrl, new URL(manifestUrl, typeof location === 'undefined' ? 'http://localhost/' : location.href)).href;
}

function logicalUrlField(modelUrl: string, resolvedUrl: string): string | undefined {
  return absoluteUrl(resolvedUrl) === absoluteUrl(modelUrl) ? undefined : modelUrl;
}

interface Lc0ArtifactChannelManifest { releaseManifestUrl?: string }
interface Lc0ArtifactReleaseManifest {
  artifacts?: Array<{ logicalUrl: string; artifactUrl: string; bytes?: number; sha256?: string }>;
}
interface ResolvedModelArtifact { url: string; expectedBytes?: number; expectedSha256?: string }

function cleanChannelUrl(raw: string | null | undefined): string | undefined {
  const trimmed = raw?.trim();
  if (!trimmed) return undefined;
  try { return new URL(trimmed, typeof location === 'undefined' ? 'http://localhost/' : location.href).href; }
  catch { return undefined; }
}

function configuredChannelUrl(): string | undefined {
  const globals = globalThis as { LC0_BROWSER_ARTIFACT_CHANNEL_URL?: string };
  return cleanChannelUrl(globals.LC0_BROWSER_ARTIFACT_CHANNEL_URL)
    ?? cleanChannelUrl(env.VITE_LC0_ARTIFACT_CHANNEL_URL);
}

async function resolveChannelArtifactUrl(modelUrl: string, channelUrl: string | undefined): Promise<ResolvedModelArtifact | undefined> {
  if (!channelUrl) return undefined;
  let logicalPath: string;
  try { logicalPath = new URL(modelUrl, typeof location === 'undefined' ? 'http://localhost/' : location.href).pathname; }
  catch { return undefined; }
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
  const artifactUrl = manifestArtifactUrl(manifest.entry, manifest.manifestUrl);
  if (artifactUrl) return { url: artifactUrl, expectedBytes: manifest.entry?.bytes, expectedSha256: manifest.entry?.sha256 };
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
  /** Lowercase hex sha256, present only when it was computed or trusted from metadata. */
  sha256?: string;
  /** true when the sha256 was compared against an expected value. */
  sha256Checked: boolean;
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
  const digest = await subtle.digest('SHA-256', source as unknown as BufferSource);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('');
}

/** Validate model bytes against the manifest's decoded byte length and SHA-256. */
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

function metadataMatches(
  metadata: Lc0ModelVerificationMetadata | undefined,
  bytes: ArrayBuffer,
  expectation: Lc0ModelBytesExpectation,
  rehashAfterMs: number,
): metadata is Lc0ModelVerificationMetadata {
  if (!metadata || metadata.schemaVersion !== METADATA_SCHEMA_VERSION || metadata.runtimeVersion !== METADATA_RUNTIME_VERSION) return false;
  if (!expectation.expectedSha256 || metadata.sha256 !== expectation.expectedSha256.toLowerCase()) return false;
  if (metadata.observedBytes !== bytes.byteLength) return false;
  if (expectation.expectedBytes !== undefined && metadata.observedBytes !== expectation.expectedBytes) return false;
  return rehashAfterMs === Infinity || epochMs() - metadata.verifiedAt <= Math.max(0, rehashAfterMs);
}

interface FetchModelBytesResult {
  bytes: ArrayBuffer;
  downloadMs: number;
  preallocated: boolean;
}

/** Fetch model bytes and preallocate the decoded target buffer when its size is known. */
async function fetchModelBytes(
  request: Request,
  modelUrl: string,
  expectedBytes: number | undefined,
  onProgress: Lc0ModelLoadOptions['onProgress'],
): Promise<FetchModelBytesResult> {
  const started = nowMs();
  const response = await fetch(request);
  if (!response.ok) throw new Error(`LC0 model fetch failed for ${modelUrl}: ${response.status}`);
  if (!response.body) return { bytes: await response.arrayBuffer(), downloadMs: nowMs() - started, preallocated: false };

  const headerLength = Number(response.headers.get('content-length') ?? '');
  const encoded = !!response.headers.get('content-encoding');
  const total = !encoded && Number.isFinite(headerLength) && headerLength > 0 ? headerLength : expectedBytes;
  const reader = response.body.getReader();
  let target = total && Number.isSafeInteger(total) && total > 0 ? new Uint8Array(total) : new Uint8Array(0);
  let loaded = 0;
  onProgress?.(0, total);
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    const required = loaded + value.byteLength;
    if (required > target.byteLength) {
      const grown = new Uint8Array(Math.max(required, target.byteLength ? target.byteLength * 2 : 64 * 1024));
      grown.set(target.subarray(0, loaded));
      target = grown;
    }
    target.set(value, loaded);
    loaded = required;
    onProgress?.(loaded, total);
  }
  const preallocated = Boolean(total && loaded === target.byteLength);
  const bytes = loaded === target.byteLength ? target.buffer : target.buffer.slice(0, loaded);
  return { bytes, downloadMs: nowMs() - started, preallocated };
}

function verificationMode(check: Lc0ModelBytesCheck): Lc0ModelVerificationMode {
  if (check.sha256Checked) return 'hashed';
  return check.byteLength >= 0 ? 'length-only' : 'unchecked';
}

function verifiedMetadata(
  key: string,
  cacheName: string,
  modelUrl: string,
  expectation: Lc0ModelBytesExpectation,
  check: Lc0ModelBytesCheck,
): Lc0ModelVerificationMetadata {
  const time = epochMs();
  return {
    key,
    schemaVersion: METADATA_SCHEMA_VERSION,
    runtimeVersion: METADATA_RUNTIME_VERSION,
    cacheName,
    url: absoluteUrl(modelUrl),
    expectedBytes: expectation.expectedBytes,
    expectedSha256: expectation.expectedSha256?.toLowerCase(),
    observedBytes: check.byteLength,
    sha256: check.sha256,
    verifiedAt: time,
    lastUsedAt: time,
  };
}

async function requestPersistentStorage(enabled: boolean | undefined): Promise<boolean | undefined> {
  if (!enabled || typeof navigator === 'undefined' || !navigator.storage) return undefined;
  try {
    if (await navigator.storage.persisted?.()) return true;
    return await navigator.storage.persist?.();
  } catch {
    return undefined;
  }
}

async function enforceModelCacheBounds(
  cache: Cache,
  cacheName: string,
  keepUrl: string,
  store: Lc0ModelVerificationMetadataStore | undefined,
  options: Lc0ModelLoadOptions,
): Promise<void> {
  const maxEntries = options.maxCacheEntries ?? DEFAULT_MAX_CACHE_ENTRIES;
  const maxBytes = options.maxCacheBytes ?? DEFAULT_MAX_CACHE_BYTES;
  if (maxEntries === Infinity && maxBytes === Infinity) return;

  const keys = await cache.keys();
  const records = await ignoreMetadataFailure(store ? () => store.list(cacheName) : undefined) ?? [];
  const byUrl = new Map(records.map((entry) => [entry.url, entry]));
  const candidates = keys.map((request, order) => {
    const url = absoluteUrl(request.url);
    const metadata = byUrl.get(url);
    return { request, url, order, bytes: metadata?.observedBytes ?? 0, lastUsedAt: metadata?.lastUsedAt ?? 0 };
  });
  let entries = candidates.length;
  let bytes = candidates.reduce((sum, entry) => sum + entry.bytes, 0);
  const oldestFirst = candidates
    .filter((entry) => entry.url !== absoluteUrl(keepUrl))
    .sort((a, b) => a.lastUsedAt - b.lastUsedAt || a.order - b.order);
  for (const entry of oldestFirst) {
    if (entries <= maxEntries && bytes <= maxBytes) break;
    if (await cache.delete(entry.request)) {
      entries -= 1;
      bytes -= entry.bytes;
      await ignoreMetadataFailure(store ? () => store.delete(metadataKey(cacheName, entry.url)) : undefined);
    }
  }
}

interface ModelLoadFlight {
  promise: Promise<Lc0ModelLoadResult>;
  progress: Set<NonNullable<Lc0ModelLoadOptions['onProgress']>>;
  latestProgress?: { loaded: number; total?: number };
}

const modelLoadFlights = new Map<string, ModelLoadFlight>();

function modelLoadFlightKey(modelUrl: string, options: Lc0ModelLoadOptions): string | undefined {
  // Injectable stores intentionally opt out so tests and isolated callers do not
  // accidentally share a load across different persistence domains.
  if (!options.cache || options.metadataStore) return undefined;
  return JSON.stringify([
    absoluteUrl(modelUrl),
    options.cacheName ?? DEFAULT_CACHE_NAME,
    absoluteUrl(options.manifestUrl ?? defaultManifestUrlForModel(modelUrl)),
    options.channelUrl ?? configuredChannelUrl() ?? null,
    options.rehashAfterMs ?? DEFAULT_REHASH_AFTER_MS,
    options.requestPersistentStorage ?? false,
    options.minimumFreeBytesAfterCache ?? DEFAULT_CACHE_FREE_BYTES_RESERVE,
    options.maxCacheEntries ?? DEFAULT_MAX_CACHE_ENTRIES,
    options.maxCacheBytes ?? DEFAULT_MAX_CACHE_BYTES,
  ]);
}

/** Share concurrent same-realm model fetch/read/hash work without retaining bytes after settlement. */
export function loadLc0ModelForOrt(modelUrl: string, options: Lc0ModelLoadOptions = {}): Promise<Lc0ModelLoadResult> {
  const key = modelLoadFlightKey(modelUrl, options);
  if (!key) return loadLc0ModelForOrtUnshared(modelUrl, options);
  const existing = modelLoadFlights.get(key);
  if (existing) {
    if (options.onProgress) {
      existing.progress.add(options.onProgress);
      if (existing.latestProgress) options.onProgress(existing.latestProgress.loaded, existing.latestProgress.total);
    }
    return existing.promise;
  }
  const progress = new Set<NonNullable<Lc0ModelLoadOptions['onProgress']>>();
  if (options.onProgress) progress.add(options.onProgress);
  const entry = {} as ModelLoadFlight;
  const sharedOptions: Lc0ModelLoadOptions = {
    ...options,
    onProgress: (loaded, total) => {
      entry.latestProgress = { loaded, total };
      for (const callback of entry.progress) callback(loaded, total);
    },
  };
  entry.progress = progress;
  entry.promise = loadLc0ModelForOrtUnshared(modelUrl, sharedOptions);
  modelLoadFlights.set(key, entry);
  void entry.promise.finally(() => {
    if (modelLoadFlights.get(key) === entry) modelLoadFlights.delete(key);
    entry.progress.clear();
  }).catch(() => undefined);
  return entry.promise;
}

async function loadLc0ModelForOrtUnshared(modelUrl: string, options: Lc0ModelLoadOptions): Promise<Lc0ModelLoadResult> {
  const started = nowMs();
  const channelUrl = options.channelUrl === null ? undefined : cleanChannelUrl(options.channelUrl) ?? configuredChannelUrl();
  const manifest = await fetchManifestEntryWithTiming(modelUrl, options.manifestUrl ?? defaultManifestUrlForModel(modelUrl));
  const resolved = await resolveArtifactUrl(modelUrl, manifest, channelUrl);
  const expectation: Lc0ModelBytesExpectation = { expectedBytes: resolved.expectedBytes, expectedSha256: resolved.expectedSha256 };

  if (!options.cache) {
    if (!options.onProgress) {
      return {
        model: resolved.url, url: resolved.url, logicalUrl: logicalUrlField(modelUrl, resolved.url), mode: 'url',
        cacheStatus: 'disabled', verification: 'unchecked', elapsedMs: nowMs() - started,
        telemetry: telemetry(started, 'url', { manifestMs: manifest.elapsedMs }),
      };
    }
    const download = await fetchModelBytes(new Request(resolved.url), resolved.url, expectation.expectedBytes, options.onProgress);
    const check = await verifyLc0ModelBytes(download.bytes, expectation);
    if (!check.ok) throw new Error(`LC0 model validation failed for ${modelUrl}: ${check.reason}`);
    return {
      model: download.bytes, url: resolved.url, logicalUrl: logicalUrlField(modelUrl, resolved.url), mode: 'memory', cacheStatus: 'disabled', bytes: download.bytes.byteLength,
      expectedBytes: expectation.expectedBytes, sha256: check.sha256, expectedSha256: expectation.expectedSha256,
      sha256Valid: check.sha256Checked ? true : undefined, verification: verificationMode(check),
      downloadMs: download.downloadMs, hashMs: check.hashMs ?? 0, elapsedMs: nowMs() - started,
      telemetry: telemetry(started, 'memory', { manifestMs: manifest.elapsedMs, downloadMs: download.downloadMs, hashMs: check.hashMs, preallocatedDownload: download.preallocated }),
    };
  }

  if (!cacheApiAvailable()) {
    return {
      model: resolved.url, url: resolved.url, logicalUrl: logicalUrlField(modelUrl, resolved.url), mode: 'url',
      cacheStatus: 'unavailable', verification: 'unchecked', elapsedMs: nowMs() - started,
      telemetry: telemetry(started, 'url', { manifestMs: manifest.elapsedMs }),
    };
  }

  const cacheName = options.cacheName ?? DEFAULT_CACHE_NAME;
  const store = options.metadataStore ?? browserMetadataStore();
  const key = metadataKey(cacheName, resolved.url);
  const cacheKey = new Request(resolved.url);
  const fetchRequest = new Request(resolved.url, { cache: 'force-cache' });
  const cache = await caches.open(cacheName);
  const storagePersistent = await requestPersistentStorage(options.requestPersistentStorage);
  const cached = await cache.match(cacheKey);

  if (cached) {
    const cacheReadStarted = nowMs();
    const bytes = await cached.arrayBuffer();
    const cacheReadMs = nowMs() - cacheReadStarted;
    const metadata = await ignoreMetadataFailure(store ? () => store.get(key) : undefined);
    if (metadataMatches(metadata, bytes, expectation, options.rehashAfterMs ?? DEFAULT_REHASH_AFTER_MS)) {
      await ignoreMetadataFailure(store ? () => store.put({ ...metadata, lastUsedAt: epochMs() }) : undefined);
      return {
        model: bytes, url: resolved.url, logicalUrl: logicalUrlField(modelUrl, resolved.url), mode: 'cache', cacheStatus: 'hit', bytes: bytes.byteLength,
        expectedBytes: expectation.expectedBytes, sha256: metadata.sha256, expectedSha256: expectation.expectedSha256,
        sha256Valid: true, verification: 'metadata', cacheReadMs, hashMs: 0, storagePersistent, elapsedMs: nowMs() - started,
        telemetry: telemetry(started, 'cache-storage', { manifestMs: manifest.elapsedMs, cacheReadMs, hashMs: 0 }),
      };
    }

    const check = await verifyLc0ModelBytes(bytes, expectation);
    if (check.ok) {
      await ignoreMetadataFailure(store ? () => store.put(verifiedMetadata(key, cacheName, resolved.url, expectation, check)) : undefined);
      return {
        model: bytes, url: resolved.url, logicalUrl: logicalUrlField(modelUrl, resolved.url), mode: 'cache', cacheStatus: 'hit', bytes: bytes.byteLength,
        expectedBytes: expectation.expectedBytes, sha256: check.sha256, expectedSha256: expectation.expectedSha256,
        sha256Valid: check.sha256Checked ? true : undefined, verification: verificationMode(check), cacheReadMs, hashMs: check.hashMs ?? 0,
        storagePersistent, elapsedMs: nowMs() - started,
        telemetry: telemetry(started, 'cache-storage', { manifestMs: manifest.elapsedMs, cacheReadMs, hashMs: check.hashMs }),
      };
    }

    await cache.delete(cacheKey);
    await ignoreMetadataFailure(store ? () => store.delete(key) : undefined);
    if (!(await cacheQuotaAllows(expectation.expectedBytes, options))) {
      return {
        model: resolved.url, url: resolved.url, logicalUrl: logicalUrlField(modelUrl, resolved.url), mode: 'url', cacheStatus: 'quota-limited',
        verification: 'unchecked', storagePersistent, revalidated: true, elapsedMs: nowMs() - started,
        telemetry: telemetry(started, 'url', { manifestMs: manifest.elapsedMs, cacheReadMs, hashMs: check.hashMs }),
      };
    }
    const reloadRequest = new Request(resolved.url, { cache: 'reload' });
    const result = await fetchAndCacheModel(cache, cacheKey, reloadRequest, modelUrl, resolved.url, expectation, started, cacheName, key, store, options, storagePersistent, {
      manifestMs: manifest.elapsedMs, cacheReadMs, hashMs: check.hashMs,
    });
    return { ...result, revalidated: true };
  }

  if (!(await cacheQuotaAllows(expectation.expectedBytes, options))) {
    return {
      model: resolved.url, url: resolved.url, logicalUrl: logicalUrlField(modelUrl, resolved.url), mode: 'url', cacheStatus: 'quota-limited',
      verification: 'unchecked', storagePersistent, elapsedMs: nowMs() - started,
      telemetry: telemetry(started, 'url', { manifestMs: manifest.elapsedMs }),
    };
  }

  return fetchAndCacheModel(cache, cacheKey, fetchRequest, modelUrl, resolved.url, expectation, started, cacheName, key, store, options, storagePersistent, { manifestMs: manifest.elapsedMs });
}

async function fetchAndCacheModel(
  cache: Cache,
  cacheKey: Request,
  fetchRequest: Request,
  logicalModelUrl: string,
  fetchModelUrl: string,
  expectation: Lc0ModelBytesExpectation,
  started: number,
  cacheName: string,
  key: string,
  store: Lc0ModelVerificationMetadataStore | undefined,
  options: Lc0ModelLoadOptions,
  storagePersistent: boolean | undefined,
  inheritedTelemetry: Pick<Lc0ModelLoadTelemetry, 'manifestMs' | 'cacheReadMs' | 'hashMs'> = {},
): Promise<Lc0ModelLoadResult> {
  const download = await fetchModelBytes(fetchRequest, fetchModelUrl, expectation.expectedBytes, options.onProgress);
  const check = await verifyLc0ModelBytes(download.bytes, expectation);
  if (!check.ok) throw new Error(`LC0 model validation failed for ${fetchModelUrl}: ${check.reason}`);

  const cacheWriteStarted = nowMs();
  await cache.put(cacheKey, new Response(download.bytes));
  const cacheWriteMs = nowMs() - cacheWriteStarted;
  await ignoreMetadataFailure(store ? () => store.put(verifiedMetadata(key, cacheName, fetchModelUrl, expectation, check)) : undefined);
  await enforceModelCacheBounds(cache, cacheName, fetchModelUrl, store, options);
  const hashMs = check.hashMs ?? 0;
  return {
    model: download.bytes, url: fetchModelUrl, logicalUrl: logicalUrlField(logicalModelUrl, fetchModelUrl), mode: 'cache', cacheStatus: 'miss', bytes: download.bytes.byteLength,
    expectedBytes: expectation.expectedBytes, sha256: check.sha256, expectedSha256: expectation.expectedSha256,
    sha256Valid: check.sha256Checked ? true : undefined, verification: verificationMode(check), storagePersistent,
    downloadMs: download.downloadMs, hashMs, elapsedMs: nowMs() - started,
    telemetry: telemetry(started, 'network', {
      ...inheritedTelemetry,
      requestCache: fetchRequest.cache,
      downloadMs: download.downloadMs,
      hashMs: (inheritedTelemetry.hashMs ?? 0) + hashMs || undefined,
      cacheWriteMs,
      preallocatedDownload: download.preallocated,
    }),
  };
}

export interface Lc0ModelCacheClearResult {
  cleared: boolean;
  removedEntries: number;
}

/** Delete one cached model and its verification metadata. */
export async function clearLc0ModelCacheEntry(
  modelUrl: string,
  cacheName: string = DEFAULT_CACHE_NAME,
  metadataStore: Lc0ModelVerificationMetadataStore | undefined = browserMetadataStore(),
): Promise<boolean> {
  if (!cacheApiAvailable() || !await caches.has(cacheName)) return false;
  const cache = await caches.open(cacheName);
  const removed = await cache.delete(new Request(modelUrl));
  await ignoreMetadataFailure(metadataStore ? () => metadataStore.delete(metadataKey(cacheName, modelUrl)) : undefined);
  return removed;
}

/** Delete the LC0 model Cache Storage bucket and matching IndexedDB metadata. */
export async function clearLc0ModelCache(
  cacheName: string = DEFAULT_CACHE_NAME,
  metadataStore: Lc0ModelVerificationMetadataStore | undefined = browserMetadataStore(),
): Promise<Lc0ModelCacheClearResult> {
  if (!cacheApiAvailable()) return { cleared: false, removedEntries: 0 };
  const has = await caches.has(cacheName);
  if (!has) {
    await ignoreMetadataFailure(metadataStore ? () => metadataStore.clear(cacheName) : undefined);
    return { cleared: false, removedEntries: 0 };
  }
  const cache = await caches.open(cacheName);
  const removedEntries = (await cache.keys()).length;
  const cleared = await caches.delete(cacheName);
  await ignoreMetadataFailure(metadataStore ? () => metadataStore.clear(cacheName) : undefined);
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
    : result.sha256Valid === true ? result.verification === 'metadata' ? ' · sha256 trusted' : ' · sha256 ok'
    : result.sha256Valid === false ? ' · sha256 BAD'
    : ' · sha256 unchecked';
  const revalidated = result.revalidated ? ' · revalidated' : '';
  return `${result.cacheStatus}${mb}${integrity}${revalidated}${timing}`;
}
