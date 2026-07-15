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

/**
 * Research-only manifest for reconstructing one decoded ONNX file from
 * independently cached, content-addressed shards. Production model loading
 * does not consult this schema unless a caller explicitly uses
 * loadResumableLc0ModelForOrt().
 */
export interface Lc0ResumableModelShardManifest {
  schema: 'lc0_browser.resumable_model_shards.v1';
  chunkBytes: number;
  decoded: {
    bytes: number;
    sha256: string;
  };
  shards: Array<{
    bytes: number;
    sha256: string;
    url: string;
  }>;
}

export interface Lc0ModelShardStore {
  /** Stable identity for stores sharing one persistence domain. Omit to isolate by store object identity. */
  persistenceDomainKey?: string | object;
  get(sha256: string): Promise<ArrayBuffer | undefined>;
  getBounded?(sha256: string, expectedBytes: number, signal?: AbortSignal): Promise<ArrayBuffer | undefined>;
  put(sha256: string, bytes: ArrayBuffer): Promise<void>;
  delete(sha256: string): Promise<void>;
  touch?(sha256: string): Promise<void>;
  list?(): Promise<Array<{ sha256: string; bytes: number; lastUsedAt?: number }>>;
  clear?(): Promise<number>;
}

export interface Lc0ResumableModelProgress {
  phase: 'download' | 'reconstruct';
  completedBytes: number;
  totalBytes: number;
  completedShards: number;
  totalShards: number;
}

export interface Lc0ResumableModelLoadOptions {
  /** Required explicit opt-in. This API throws unless set to true. */
  researchOnly: true;
  cacheName?: string;
  concurrency?: number;
  corruptionRetries?: number;
  signal?: AbortSignal;
  fetchFn?: typeof fetch;
  shardStore?: Lc0ModelShardStore;
  /** Maximum decoded manifest bytes. Defaults to 1 MiB; set to Infinity to disable. */
  maxManifestBytes?: number;
  /** Persistent research shard-cache bounds. Set to Infinity to disable a bound. */
  maxCacheEntries?: number;
  maxCacheBytes?: number;
  /** Maximum decoded model bytes. Defaults to 1,000,000,000; set to Infinity to disable. */
  maxDecodedBytes?: number;
  /** Maximum ordered shard references. Defaults to 64; set to Infinity to disable. */
  maxShardReferences?: number;
  /** Minimum browser storage quota retained after shard caching. Defaults to 64 MiB. */
  minimumFreeBytesAfterCache?: number;
  /** Optional host-specific shard URL policy. Browser callers normally use standard URL resolution. */
  resolveShardUrl?: (shardUrl: string, manifestBaseUrl: string) => string;
  onProgress?: (progress: Lc0ResumableModelProgress) => void;
}

export interface Lc0ResumableModelLoadResult {
  /** Byte-identical, fully hash-validated decoded ONNX for ORT session creation. */
  model: ArrayBuffer;
  manifestUrl: string;
  sha256: string;
  bytes: number;
  downloadedBytes: number;
  reusedBytes: number;
  downloadedShards: number;
  reusedShards: number;
  corruptShardsEvicted: number;
  corruptionRetries: number;
  uniqueShardCount: number;
  deduplicatedReferences: number;
  peakTemporaryBytes: number;
  elapsedMs: number;
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

interface Lc0ArtifactChannelManifest { releaseManifestUrl?: string; releaseUrl?: string }
interface Lc0ArtifactRepresentation {
  encoding: 'identity' | 'br';
  url: string;
  bytes: number;
  sha256: string;
}
interface Lc0ArtifactReleaseEntry {
  logicalUrl?: string;
  name?: string;
  file?: string;
  artifactUrl?: string;
  bytes?: number;
  sha256?: string;
  raw?: { bytes: number; sha256: string };
  representations?: Lc0ArtifactRepresentation[];
}
interface Lc0ArtifactReleaseManifest { artifacts?: Lc0ArtifactReleaseEntry[] }
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
    const releasePath = channel.releaseManifestUrl ?? channel.releaseUrl;
    if (!releasePath) return undefined;
    const releaseUrl = new URL(releasePath, channelUrl).href;
    const releaseResponse = await fetch(releaseUrl, { cache: 'force-cache' });
    if (!releaseResponse.ok) return undefined;
    const release = await releaseResponse.json() as Lc0ArtifactReleaseManifest;
    const artifacts = release.artifacts ?? [];
    const exactArtifact = artifacts.find((entry) => entry.logicalUrl === logicalPath);
    const logicalFile = logicalPath.split('/').pop();
    const fallbackArtifacts = exactArtifact ? [] : artifacts.filter((entry) => !entry.logicalUrl
      && (entry.file === logicalFile || entry.name === logicalFile));
    const artifact = exactArtifact ?? (fallbackArtifacts.length === 1 ? fallbackArtifacts[0] : undefined);
    if (!artifact) return undefined;
    if (artifact.raw && artifact.representations?.length) {
      // Full model loads do not use Range, so prefer the immutable Brotli body.
      // Fetch transparently decodes it; integrity remains the decoded raw hash.
      const representation = artifact.representations.find((entry) => entry.encoding === 'br')
        ?? artifact.representations.find((entry) => entry.encoding === 'identity');
      if (!representation) return undefined;
      return {
        url: new URL(representation.url, releaseUrl).href,
        expectedBytes: artifact.raw.bytes,
        expectedSha256: artifact.raw.sha256,
      };
    }
    return artifact.artifactUrl ? {
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

const MIN_RESUMABLE_SHARD_BYTES = 16 * 1024 * 1024;
const MAX_RESUMABLE_SHARD_BYTES = 32 * 1024 * 1024;
const DEFAULT_RESUMABLE_SHARD_CACHE_NAME = 'lc0-browser-model-shards-research-v1';
const DEFAULT_RESUMABLE_SHARD_CONCURRENCY = 3;
const DEFAULT_RESUMABLE_CORRUPTION_RETRIES = 1;
const DEFAULT_RESUMABLE_MAX_CACHE_ENTRIES = 64;
const DEFAULT_RESUMABLE_MAX_CACHE_BYTES = 1_000_000_000;
const DEFAULT_RESUMABLE_MAX_MANIFEST_BYTES = 1024 * 1024;
const DEFAULT_RESUMABLE_MAX_DECODED_BYTES = 1_000_000_000;
const DEFAULT_RESUMABLE_MAX_SHARD_REFERENCES = 64;
const SHA256_PATTERN = /^[0-9a-f]{64}$/i;

function abortError(): Error {
  if (typeof DOMException !== 'undefined') return new DOMException('The operation was aborted', 'AbortError');
  const error = new Error('The operation was aborted');
  error.name = 'AbortError';
  return error;
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted) throw abortError();
}

function resumableShardCacheKey(sha256: string): Request {
  const origin = typeof location === 'undefined' ? 'http://localhost/' : location.href;
  return new Request(new URL(`/__lc0-resumable-model-shards__/sha256/${sha256}`, origin).href);
}

function resumableShardMetadataKey(sha256: string): Request {
  const origin = typeof location === 'undefined' ? 'http://localhost/' : location.href;
  return new Request(new URL(`/__lc0-resumable-model-shards__/metadata/sha256/${sha256}`, origin).href);
}

class CacheStorageModelShardStore implements Lc0ModelShardStore {
  private readonly cacheName: string;

  constructor(cacheName: string) {
    this.cacheName = cacheName;
  }

  private async cache(): Promise<Cache> {
    if (!cacheApiAvailable()) throw new Error('Cache Storage is unavailable for resumable model shards');
    return caches.open(this.cacheName);
  }

  async get(sha256: string): Promise<ArrayBuffer | undefined> {
    return this.getBounded(sha256, MAX_RESUMABLE_SHARD_BYTES);
  }

  async getBounded(sha256: string, expectedBytes: number, signal?: AbortSignal): Promise<ArrayBuffer | undefined> {
    const response = await (await this.cache()).match(resumableShardCacheKey(sha256));
    return response ? readBoundedShardResponse(response, expectedBytes, signal, () => {}) : undefined;
  }

  async put(sha256: string, bytes: ArrayBuffer): Promise<void> {
    const cache = await this.cache();
    await cache.put(resumableShardCacheKey(sha256), new Response(bytes, {
      headers: {
        'x-lc0-shard-bytes': String(bytes.byteLength),
      },
    }));
    try {
      await cache.put(resumableShardMetadataKey(sha256), new Response(JSON.stringify({
        bytes: bytes.byteLength,
        lastUsedAt: epochMs(),
      }), {
        headers: { 'content-type': 'application/json' },
      }));
    } catch {
      // Retention metadata is optional. The persisted body remains valid and
      // list() can recover its byte count from x-lc0-shard-bytes.
    }
  }

  async delete(sha256: string): Promise<void> {
    const cache = await this.cache();
    await Promise.all([
      cache.delete(resumableShardCacheKey(sha256)),
      cache.delete(resumableShardMetadataKey(sha256)),
    ]);
  }

  async touch(sha256: string): Promise<void> {
    const cache = await this.cache();
    const response = await cache.match(resumableShardCacheKey(sha256));
    if (!response) return;
    const headerBytes = Number(response.headers.get('x-lc0-shard-bytes'));
    await cache.put(resumableShardMetadataKey(sha256), new Response(JSON.stringify({
      bytes: Number.isSafeInteger(headerBytes) && headerBytes >= 0 ? headerBytes : undefined,
      lastUsedAt: epochMs(),
    }), {
      headers: { 'content-type': 'application/json' },
    }));
  }

  async list(): Promise<Array<{ sha256: string; bytes: number; lastUsedAt?: number }>> {
    const cache = await this.cache();
    const keys = await cache.keys();
    const entries = await Promise.all(keys.map(async (request) => {
      const pathname = new URL(request.url).pathname;
      if (pathname.includes('/metadata/')) return undefined;
      const match = /\/sha256\/([0-9a-f]{64})$/i.exec(pathname);
      if (!match) return undefined;
      const response = await cache.match(request);
      if (!response) return undefined;
      const headerBytes = Number(response.headers.get('x-lc0-shard-bytes'));
      let metadata: { bytes?: number; lastUsedAt?: number } | undefined;
      try {
        const metadataResponse = await cache.match(resumableShardMetadataKey(match[1]));
        metadata = metadataResponse ? await metadataResponse.json() as { bytes?: number; lastUsedAt?: number } : undefined;
      } catch {
        // Missing or invalid retention metadata falls back to body headers.
      }
      const metadataBytes = Number(metadata?.bytes);
      const metadataLastUsedAt = Number(metadata?.lastUsedAt);
      return {
        sha256: match[1].toLowerCase(),
        bytes: Number.isSafeInteger(metadataBytes) && metadataBytes >= 0
          ? metadataBytes
          : Number.isSafeInteger(headerBytes) && headerBytes >= 0
            ? headerBytes
            : Infinity,
        lastUsedAt: Number.isFinite(metadataLastUsedAt) ? metadataLastUsedAt : undefined,
      };
    }));
    return entries.filter((entry): entry is NonNullable<typeof entry> => entry !== undefined);
  }

  async clear(): Promise<number> {
    const cache = await this.cache();
    const removedEntries = (await cache.keys())
      .filter((request) => /\/sha256\/[0-9a-f]{64}$/i.test(new URL(request.url).pathname)
        && !new URL(request.url).pathname.includes('/metadata/'))
      .length;
    await caches.delete(this.cacheName);
    return removedEntries;
  }
}

function normalizedResumableCacheBound(value: number | undefined, fallback: number): number {
  if (value === Infinity) return Infinity;
  const normalized = Math.floor(value ?? fallback);
  if (!Number.isFinite(normalized) || normalized < 0) throw new Error('Resumable model shard cache bounds must be non-negative integers or Infinity');
  return normalized;
}

function normalizedResumableManifestLimit(
  value: number | undefined,
  fallback: number,
  label: string,
): number {
  const normalized = value ?? fallback;
  if (normalized === Infinity) return Infinity;
  if (!Number.isSafeInteger(normalized) || normalized < 0) {
    throw new Error(`${label} must be a non-negative safe integer or Infinity`);
  }
  return normalized;
}

async function enforceResumableShardCacheBounds(
  store: Lc0ModelShardStore,
  maxEntries: number,
  maxBytes: number,
  protectedHashes: ReadonlySet<string>,
): Promise<void> {
  if (!store.list || (maxEntries === Infinity && maxBytes === Infinity)) return;
  const entries = await store.list();
  let entryCount = entries.length;
  let knownByteCount = 0;
  let unknownByteCount = 0;
  for (const entry of entries) {
    if (Number.isFinite(entry.bytes)) knownByteCount += entry.bytes;
    else unknownByteCount += 1;
  }
  const candidates = entries
    .filter((entry) => !protectedHashes.has(entry.sha256))
    .sort((a, b) => (a.lastUsedAt ?? 0) - (b.lastUsedAt ?? 0) || a.sha256.localeCompare(b.sha256));
  for (const entry of candidates) {
    const bytesWithinBound = maxBytes === Infinity
      || (unknownByteCount === 0 && knownByteCount <= maxBytes);
    if (entryCount <= maxEntries && bytesWithinBound) break;
    await store.delete(entry.sha256);
    entryCount -= 1;
    if (Number.isFinite(entry.bytes)) knownByteCount -= entry.bytes;
    else unknownByteCount -= 1;
  }
}

async function bestEffortEnforceResumableShardCacheBounds(
  store: Lc0ModelShardStore,
  maxEntries: number,
  maxBytes: number,
  protectedHashes: ReadonlySet<string>,
): Promise<void> {
  try {
    await enforceResumableShardCacheBounds(store, maxEntries, maxBytes, protectedHashes);
  } catch {
    // Retention is best-effort. Integrity and reconstruction remain correct
    // when cache enumeration or deletion is unavailable.
  }
}

async function resumableShardQuotaAllows(expectedBytes: number, minimumFreeBytesAfterCache: number): Promise<boolean> {
  if (typeof navigator === 'undefined' || !navigator.storage?.estimate) return true;
  try {
    const estimate = await navigator.storage.estimate();
    const quota = Number(estimate.quota);
    const usage = Number(estimate.usage);
    if (!Number.isFinite(quota) || !Number.isFinite(usage) || quota <= 0) return true;
    return quota - usage - expectedBytes >= minimumFreeBytesAfterCache;
  } catch {
    return true;
  }
}

async function ensureResumableShardQuota(
  store: Lc0ModelShardStore,
  expectedBytes: number,
  minimumFreeBytesAfterCache: number,
  protectedHashes: ReadonlySet<string>,
): Promise<void> {
  if (await resumableShardQuotaAllows(expectedBytes, minimumFreeBytesAfterCache)) return;
  if (store.list) {
    const candidates = (await store.list())
      .filter((entry) => !protectedHashes.has(entry.sha256))
      .sort((a, b) => (a.lastUsedAt ?? 0) - (b.lastUsedAt ?? 0) || a.sha256.localeCompare(b.sha256));
    for (const entry of candidates) {
      await store.delete(entry.sha256);
      if (await resumableShardQuotaAllows(expectedBytes, minimumFreeBytesAfterCache)) return;
    }
  }
  if (!await resumableShardQuotaAllows(expectedBytes, minimumFreeBytesAfterCache)) {
    throw new Error(`Insufficient storage quota for resumable model shard (${expectedBytes} bytes)`);
  }
}

interface ResumableShardPersistenceDomain {
  activeHashes: Map<string, number>;
  writeTail: Promise<void>;
}

const resumableShardCacheStorageDomains = new Map<string, ResumableShardPersistenceDomain>();
const resumableShardCustomStoreDomains = new WeakMap<Lc0ModelShardStore, ResumableShardPersistenceDomain>();
const resumableShardStringDomains = new Map<string, ResumableShardPersistenceDomain>();
const resumableShardObjectDomains = new WeakMap<object, ResumableShardPersistenceDomain>();

function newResumableShardPersistenceDomain(): ResumableShardPersistenceDomain {
  return {
    activeHashes: new Map(),
    writeTail: Promise.resolve(),
  };
}

function resumableShardPersistenceDomain(
  cacheName: string,
  customStore: Lc0ModelShardStore | undefined,
): ResumableShardPersistenceDomain {
  if (customStore) {
    const persistenceDomainKey = customStore.persistenceDomainKey;
    if (typeof persistenceDomainKey === 'string') {
      let domain = resumableShardStringDomains.get(persistenceDomainKey);
      if (!domain) {
        domain = newResumableShardPersistenceDomain();
        resumableShardStringDomains.set(persistenceDomainKey, domain);
      }
      return domain;
    }
    if (persistenceDomainKey && typeof persistenceDomainKey === 'object') {
      let domain = resumableShardObjectDomains.get(persistenceDomainKey);
      if (!domain) {
        domain = newResumableShardPersistenceDomain();
        resumableShardObjectDomains.set(persistenceDomainKey, domain);
      }
      return domain;
    }
    let domain = resumableShardCustomStoreDomains.get(customStore);
    if (!domain) {
      domain = newResumableShardPersistenceDomain();
      resumableShardCustomStoreDomains.set(customStore, domain);
    }
    return domain;
  }
  let domain = resumableShardCacheStorageDomains.get(cacheName);
  if (!domain) {
    domain = newResumableShardPersistenceDomain();
    resumableShardCacheStorageDomains.set(cacheName, domain);
  }
  return domain;
}

async function serializeResumableShardCacheWrite<T>(
  domain: ResumableShardPersistenceDomain,
  operation: () => Promise<T>,
): Promise<T> {
  const previous = domain.writeTail;
  let release!: () => void;
  domain.writeTail = new Promise<void>((resolve) => {
    release = resolve;
  });
  await previous;
  try {
    return await operation();
  } finally {
    release();
  }
}

function retainActiveResumableShardHashes(
  domain: ResumableShardPersistenceDomain,
  hashes: ReadonlySet<string>,
): () => void {
  const counts = domain.activeHashes;
  for (const hash of hashes) counts.set(hash, (counts.get(hash) ?? 0) + 1);
  let released = false;
  return () => {
    if (released) return;
    released = true;
    for (const hash of hashes) {
      const count = counts.get(hash);
      if (count === undefined || count <= 1) counts.delete(hash);
      else counts.set(hash, count - 1);
    }
  };
}

function activeResumableShardHashes(domain: ResumableShardPersistenceDomain): ReadonlySet<string> {
  return new Set(domain.activeHashes.keys());
}

async function readBoundedShardResponse(
  response: Response,
  expectedBytes: number,
  signal: AbortSignal | undefined,
  onBytes: (bytes: number) => void,
): Promise<ArrayBuffer> {
  const contentLengthHeader = response.headers.get('content-length');
  const contentLength = Number(contentLengthHeader);
  if (response.headers.get('content-encoding') === null
    && contentLengthHeader !== null
    && Number.isFinite(contentLength)
    && contentLength > expectedBytes) {
    try {
      await response.body?.cancel(`Shard response exceeds expected ${expectedBytes} bytes`);
    } catch {
      // The response is already rejected as oversized even if cancellation fails.
    }
    throw new Error(`Resumable model shard response exceeded expected length ${expectedBytes}`);
  }
  if (!response.body) throw new Error('Resumable model shard response body is unavailable');
  const reader = response.body.getReader();
  const onAbort = (): void => {
    void reader.cancel('The operation was aborted').catch(() => {});
  };
  signal?.addEventListener('abort', onAbort, { once: true });
  const target = new Uint8Array(expectedBytes);
  let loaded = 0;
  try {
    for (;;) {
      throwIfAborted(signal);
      const { done, value } = await reader.read();
      if (done) break;
      onBytes(value.byteLength);
      if (loaded + value.byteLength > expectedBytes) {
        try {
          await reader.cancel(`Shard response exceeds expected ${expectedBytes} bytes`);
        } catch {
          // The response is already rejected as oversized even if cancellation fails.
        }
        throw new Error(`Resumable model shard response exceeded expected length ${expectedBytes}`);
      }
      target.set(value, loaded);
      loaded += value.byteLength;
    }
    throwIfAborted(signal);
  } finally {
    signal?.removeEventListener('abort', onAbort);
    reader.releaseLock();
  }
  return loaded === expectedBytes ? target.buffer : target.buffer.slice(0, loaded);
}

async function readBoundedResumableManifest(
  response: Response,
  maxBytes: number,
  signal: AbortSignal | undefined,
): Promise<unknown> {
  const contentLengthHeader = response.headers.get('content-length');
  const contentLength = Number(contentLengthHeader);
  if (response.headers.get('content-encoding') === null
    && contentLengthHeader !== null
    && Number.isFinite(contentLength)
    && contentLength > maxBytes) {
    try {
      await response.body?.cancel(`Manifest response exceeds configured limit ${maxBytes} bytes`);
    } catch {
      // The response is already rejected as oversized even if cancellation fails.
    }
    throw new Error(`Resumable model shard manifest exceeded configured limit ${maxBytes} bytes`);
  }
  if (!response.body) throw new Error('Resumable model shard manifest response body is unavailable');
  const reader = response.body.getReader();
  const onAbort = (): void => {
    void reader.cancel('The operation was aborted').catch(() => {});
  };
  signal?.addEventListener('abort', onAbort, { once: true });
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  try {
    for (;;) {
      throwIfAborted(signal);
      const { done, value } = await reader.read();
      if (done) break;
      if (loaded + value.byteLength > maxBytes) {
        try {
          await reader.cancel(`Manifest response exceeds configured limit ${maxBytes} bytes`);
        } catch {
          // The response is already rejected as oversized even if cancellation fails.
        }
        throw new Error(`Resumable model shard manifest exceeded configured limit ${maxBytes} bytes`);
      }
      chunks.push(value);
      loaded += value.byteLength;
    }
    throwIfAborted(signal);
  } finally {
    signal?.removeEventListener('abort', onAbort);
    reader.releaseLock();
  }
  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return JSON.parse(new TextDecoder('utf-8').decode(bytes)) as unknown;
}

function isOversizedResumableShardResponse(error: unknown): boolean {
  return error instanceof Error
    && error.message.startsWith('Resumable model shard response exceeded expected length');
}

function validateResumableModelShardManifest(
  value: unknown,
  maxDecodedBytes: number,
  maxShardReferences: number,
): Lc0ResumableModelShardManifest {
  const manifest = value as Partial<Lc0ResumableModelShardManifest>;
  if (manifest?.schema !== 'lc0_browser.resumable_model_shards.v1') {
    throw new Error('Invalid resumable model shard manifest schema');
  }
  if (!Number.isSafeInteger(manifest.chunkBytes)
    || manifest.chunkBytes! < MIN_RESUMABLE_SHARD_BYTES
    || manifest.chunkBytes! > MAX_RESUMABLE_SHARD_BYTES) {
    throw new Error('Invalid resumable model shard chunkBytes: expected 16-32 MiB');
  }
  if (!manifest.decoded
    || !Number.isSafeInteger(manifest.decoded.bytes)
    || manifest.decoded.bytes <= 0
    || !SHA256_PATTERN.test(manifest.decoded.sha256)) {
    throw new Error('Invalid resumable model shard decoded metadata');
  }
  if (manifest.decoded.bytes > maxDecodedBytes) {
    throw new Error(`Resumable model decoded bytes exceed configured limit ${maxDecodedBytes}`);
  }
  if (!Array.isArray(manifest.shards) || manifest.shards.length === 0) {
    throw new Error('Invalid resumable model shard list');
  }
  if (manifest.shards.length > maxShardReferences) {
    throw new Error(`Resumable model shard references exceed configured limit ${maxShardReferences}`);
  }
  let totalBytes = 0;
  const descriptors = new Map<string, { bytes: number; url: string }>();
  manifest.shards.forEach((shard, index) => {
    if (!shard || !Number.isSafeInteger(shard.bytes) || shard.bytes <= 0 || shard.bytes > manifest.chunkBytes!) {
      throw new Error(`Invalid resumable model shard length at index ${index}`);
    }
    if (index < manifest.shards!.length - 1 && shard.bytes !== manifest.chunkBytes) {
      throw new Error(`Invalid non-final resumable model shard length at index ${index}`);
    }
    if (!SHA256_PATTERN.test(shard.sha256) || typeof shard.url !== 'string' || !shard.url) {
      throw new Error(`Invalid resumable model shard descriptor at index ${index}`);
    }
    const hash = shard.sha256.toLowerCase();
    const previous = descriptors.get(hash);
    if (previous && (previous.bytes !== shard.bytes || previous.url !== shard.url)) {
      throw new Error(`Conflicting resumable model shard descriptor for ${hash}`);
    }
    descriptors.set(hash, { bytes: shard.bytes, url: shard.url });
    totalBytes += shard.bytes;
  });
  if (totalBytes !== manifest.decoded.bytes) {
    throw new Error(`Invalid resumable model decoded length: shards total ${totalBytes}, expected ${manifest.decoded.bytes}`);
  }
  return {
    schema: manifest.schema,
    chunkBytes: manifest.chunkBytes!,
    decoded: { bytes: manifest.decoded.bytes, sha256: manifest.decoded.sha256.toLowerCase() },
    shards: manifest.shards.map((shard) => ({
      bytes: shard.bytes,
      sha256: shard.sha256.toLowerCase(),
      url: shard.url,
    })),
  };
}

async function verifiedShardBytes(
  store: Lc0ModelShardStore,
  sha256: string,
  expectedBytes: number,
  refreshLastUsed: boolean = true,
  accountTemporaryBytes?: (bytes: number) => () => void,
  signal?: AbortSignal,
): Promise<{ bytes?: ArrayBuffer; corrupt: boolean; releaseTemporaryBytes?: () => void }> {
  let bytes: ArrayBuffer | undefined;
  const boundedReleaseTemporaryBytes = store.getBounded
    ? accountTemporaryBytes?.(expectedBytes)
    : undefined;
  try {
    bytes = store.getBounded
      ? await store.getBounded(sha256, expectedBytes, signal)
      : await store.get(sha256);
  } catch (error) {
    boundedReleaseTemporaryBytes?.();
    if (!isOversizedResumableShardResponse(error)) throw error;
    await store.delete(sha256);
    return { corrupt: true };
  }
  if (!bytes) {
    boundedReleaseTemporaryBytes?.();
    return { corrupt: false };
  }
  const releaseTemporaryBytes = boundedReleaseTemporaryBytes
    ?? accountTemporaryBytes?.(bytes.byteLength);
  try {
    if (bytes.byteLength !== expectedBytes || await sha256Hex(bytes) !== sha256) {
      await store.delete(sha256);
      releaseTemporaryBytes?.();
      return { corrupt: true };
    }
    if (refreshLastUsed) {
      try {
        await store.touch?.(sha256);
      } catch {
        // Retention metadata is best-effort and must not invalidate verified bytes.
      }
    }
    return { bytes, corrupt: false, releaseTemporaryBytes };
  } catch (error) {
    releaseTemporaryBytes?.();
    throw error;
  }
}

/**
 * Explicit research path for resumable model shards.
 *
 * Every shard is independently persisted only after length/hash validation.
 * Cached corruption is evicted, missing or evicted shards are redownloaded,
 * and corrupt network responses are retried without disturbing valid shards.
 * The ordered ONNX is reconstructed and its final decoded hash is validated
 * before the bytes are returned to any ORT session caller.
 */
export async function loadResumableLc0ModelForOrt(
  manifestUrl: string,
  options: Lc0ResumableModelLoadOptions,
): Promise<Lc0ResumableModelLoadResult> {
  if (options?.researchOnly !== true) throw new Error('Resumable model shards require explicit researchOnly: true opt-in');
  const started = nowMs();
  const fetchFn = options.fetchFn ?? globalThis.fetch;
  if (!fetchFn) throw new Error('fetch is unavailable for resumable model shards');
  const concurrency = Math.max(1, Math.min(8, Math.floor(options.concurrency ?? DEFAULT_RESUMABLE_SHARD_CONCURRENCY)));
  const retryLimit = Math.max(0, Math.min(3, Math.floor(options.corruptionRetries ?? DEFAULT_RESUMABLE_CORRUPTION_RETRIES)));
  const maxCacheEntries = normalizedResumableCacheBound(options.maxCacheEntries, DEFAULT_RESUMABLE_MAX_CACHE_ENTRIES);
  const maxCacheBytes = normalizedResumableCacheBound(options.maxCacheBytes, DEFAULT_RESUMABLE_MAX_CACHE_BYTES);
  const maxManifestBytes = normalizedResumableManifestLimit(
    options.maxManifestBytes,
    DEFAULT_RESUMABLE_MAX_MANIFEST_BYTES,
    'maxManifestBytes',
  );
  const maxDecodedBytes = normalizedResumableManifestLimit(
    options.maxDecodedBytes,
    DEFAULT_RESUMABLE_MAX_DECODED_BYTES,
    'maxDecodedBytes',
  );
  const maxShardReferences = normalizedResumableManifestLimit(
    options.maxShardReferences,
    DEFAULT_RESUMABLE_MAX_SHARD_REFERENCES,
    'maxShardReferences',
  );
  const minimumFreeBytesAfterCache = normalizedResumableCacheBound(
    options.minimumFreeBytesAfterCache,
    DEFAULT_CACHE_FREE_BYTES_RESERVE,
  );
  const cacheName = options.cacheName ?? DEFAULT_RESUMABLE_SHARD_CACHE_NAME;
  const customStore = options.shardStore;
  const store = customStore ?? new CacheStorageModelShardStore(cacheName);
  const persistenceDomain = resumableShardPersistenceDomain(cacheName, customStore);
  throwIfAborted(options.signal);

  const manifestResponse = await fetchFn(manifestUrl, { cache: 'no-cache', signal: options.signal });
  if (!manifestResponse.ok) throw new Error(`Resumable model shard manifest fetch failed: ${manifestResponse.status}`);
  const manifest = validateResumableModelShardManifest(
    await readBoundedResumableManifest(manifestResponse, maxManifestBytes, options.signal),
    maxDecodedBytes,
    maxShardReferences,
  );
  const manifestBaseUrl = manifestResponse.url || manifestUrl;
  const resolvedShardUrls = new Map<string, string>();
  const unique = new Map<string, (typeof manifest.shards)[number]>();
  const referenceCounts = new Map<string, number>();
  for (const shard of manifest.shards) {
    resolvedShardUrls.set(
      shard.sha256,
      options.resolveShardUrl
        ? options.resolveShardUrl(shard.url, manifestBaseUrl)
        : new URL(shard.url, manifestBaseUrl).href,
    );
    unique.set(shard.sha256, shard);
  }
  for (const shard of manifest.shards) referenceCounts.set(shard.sha256, (referenceCounts.get(shard.sha256) ?? 0) + 1);
  const manifestHashes = new Set(unique.keys());
  const releaseActiveHashes = retainActiveResumableShardHashes(persistenceDomain, manifestHashes);
  let finalBoundsEnforced = false;
  try {
    await bestEffortEnforceResumableShardCacheBounds(
      store,
      maxCacheEntries,
      maxCacheBytes,
      activeResumableShardHashes(persistenceDomain),
    );
    throwIfAborted(options.signal);

    let downloadedBytes = 0;
    let reusedBytes = 0;
    let downloadedShards = 0;
    let reusedShards = 0;
    let corruptShardsEvicted = 0;
    let corruptionRetries = 0;
    let completedBytes = 0;
    let completedShards = 0;
    let activeTemporaryBytes = 0;
    let peakTemporaryBytes = 0;
    const downloadedShardHashes = new Set<string>();
    const reusedShardHashes = new Set<string>();
    const markShardDownloaded = (shard: (typeof manifest.shards)[number]): void => {
      if (!downloadedShardHashes.has(shard.sha256)) {
        downloadedShardHashes.add(shard.sha256);
        downloadedShards += 1;
      }
      if (reusedShardHashes.delete(shard.sha256)) {
        reusedShards -= 1;
        reusedBytes -= shard.bytes;
      }
    };
    const markShardReused = (shard: (typeof manifest.shards)[number]): void => {
      if (downloadedShardHashes.has(shard.sha256) || reusedShardHashes.has(shard.sha256)) return;
      reusedShardHashes.add(shard.sha256);
      reusedShards += 1;
      reusedBytes += shard.bytes;
    };
    const accountTemporaryBytes = (bytes: number): (() => void) => {
      activeTemporaryBytes += bytes;
      peakTemporaryBytes = Math.max(peakTemporaryBytes, activeTemporaryBytes);
      let released = false;
      return () => {
        if (released) return;
        released = true;
        activeTemporaryBytes -= bytes;
      };
    };
    const jobs = [...unique.values()];
    let nextJob = 0;

    const report = (phase: Lc0ResumableModelProgress['phase']): void => options.onProgress?.({
      phase,
      completedBytes,
      totalBytes: manifest.decoded.bytes,
      completedShards,
      totalShards: phase === 'download' ? unique.size : manifest.shards.length,
    });
    report('download');

    const workerAbort = new AbortController();
    const onCallerAbort = (): void => workerAbort.abort();
    options.signal?.addEventListener('abort', onCallerAbort, { once: true });
    if (options.signal?.aborted) workerAbort.abort();
    const workerSignal = workerAbort.signal;
    let workerFailure: unknown;
    const downloadAndPersistShard = async (
      shard: (typeof manifest.shards)[number],
      signal: AbortSignal | undefined,
    ): Promise<void> => {
      for (let attempt = 0; attempt <= retryLimit; attempt += 1) {
        throwIfAborted(signal);
        const shardUrl = resolvedShardUrls.get(shard.sha256)!;
        const response = await fetchFn(shardUrl, {
          cache: attempt === 0 ? 'force-cache' : 'reload',
          signal,
        });
        if (!response.ok) throw new Error(`Resumable model shard fetch failed for ${shard.sha256}: ${response.status}`);
        const releaseTemporaryBytes = accountTemporaryBytes(shard.bytes);
        try {
          const bytes = await readBoundedShardResponse(response, shard.bytes, signal, (chunkBytes) => {
            downloadedBytes += chunkBytes;
          });
          const valid = bytes.byteLength === shard.bytes && await sha256Hex(bytes) === shard.sha256;
          if (valid) {
            throwIfAborted(signal);
            let reusedConcurrentWrite = false;
            await serializeResumableShardCacheWrite(persistenceDomain, async () => {
              throwIfAborted(signal);
              const existing = await verifiedShardBytes(
                store,
                shard.sha256,
                shard.bytes,
                true,
                accountTemporaryBytes,
                signal,
              );
              if (existing.corrupt) corruptShardsEvicted += 1;
              if (existing.bytes) {
                existing.releaseTemporaryBytes?.();
                reusedConcurrentWrite = true;
                return;
              }
              await ensureResumableShardQuota(
                store,
                shard.bytes,
                minimumFreeBytesAfterCache,
                activeResumableShardHashes(persistenceDomain),
              );
              throwIfAborted(signal);
              await store.put(shard.sha256, bytes);
              await bestEffortEnforceResumableShardCacheBounds(
                store,
                maxCacheEntries,
                maxCacheBytes,
                activeResumableShardHashes(persistenceDomain),
              );
            });
            if (reusedConcurrentWrite) {
              markShardReused(shard);
            } else {
              markShardDownloaded(shard);
            }
            return;
          }
        } catch (error) {
          if (signal?.aborted || (error instanceof Error && error.name === 'AbortError')) throw error;
          if (!isOversizedResumableShardResponse(error)) throw error;
        } finally {
          releaseTemporaryBytes();
        }
        if (attempt < retryLimit) corruptionRetries += 1;
      }
      throw new Error(`Resumable model shard corruption persisted after ${retryLimit + 1} attempts: ${shard.sha256}`);
    };
    const workers = Array.from({ length: Math.min(concurrency, jobs.length) }, async () => {
      for (;;) {
        throwIfAborted(workerSignal);
        const jobIndex = nextJob;
        nextJob += 1;
        if (jobIndex >= jobs.length) return;
        const shard = jobs[jobIndex];
        const cached = await verifiedShardBytes(
          store,
          shard.sha256,
          shard.bytes,
          true,
          accountTemporaryBytes,
          workerSignal,
        );
        if (cached.corrupt) corruptShardsEvicted += 1;
        if (cached.bytes) {
          try {
            markShardReused(shard);
          } finally {
            cached.releaseTemporaryBytes?.();
          }
        } else {
          await downloadAndPersistShard(shard, workerSignal);
        }
        completedBytes += shard.bytes * (referenceCounts.get(shard.sha256) ?? 1);
        completedShards += 1;
        report('download');
      }
    });
    const guardedWorkers = workers.map(async (worker) => {
      try {
        await worker;
      } catch (error) {
        if (workerFailure === undefined && !options.signal?.aborted) workerFailure = error;
        workerAbort.abort();
        throw error;
      }
    });
    const workerSettlements = await Promise.allSettled(guardedWorkers);
    options.signal?.removeEventListener('abort', onCallerAbort);
    if (options.signal?.aborted) throw abortError();
    if (workerFailure !== undefined) throw workerFailure;
    const rejectedWorker = workerSettlements.find((settlement): settlement is PromiseRejectedResult => settlement.status === 'rejected');
    if (rejectedWorker) throw rejectedWorker.reason;

    throwIfAborted(options.signal);
    const model = new Uint8Array(manifest.decoded.bytes);
    completedBytes = 0;
    completedShards = 0;
    report('reconstruct');
    let offset = 0;
    for (const shard of manifest.shards) {
      throwIfAborted(options.signal);
      let stored = await verifiedShardBytes(
        store,
        shard.sha256,
        shard.bytes,
        false,
        accountTemporaryBytes,
        options.signal,
      );
      if (stored.corrupt) corruptShardsEvicted += 1;
      if (!stored.bytes) {
        await downloadAndPersistShard(shard, options.signal);
        stored = await verifiedShardBytes(
          store,
          shard.sha256,
          shard.bytes,
          false,
          accountTemporaryBytes,
          options.signal,
        );
        if (stored.corrupt) corruptShardsEvicted += 1;
        if (!stored.bytes) {
          throw new Error(`Resumable model shard was evicted during reconstruction after recovery: ${shard.sha256}`);
        }
      }
      try {
        model.set(new Uint8Array(stored.bytes), offset);
      } finally {
        stored.releaseTemporaryBytes?.();
      }
      offset += shard.bytes;
      completedBytes += shard.bytes;
      completedShards += 1;
      report('reconstruct');
    }
    throwIfAborted(options.signal);
    const decodedSha256 = await sha256Hex(model);
    throwIfAborted(options.signal);
    if (decodedSha256 !== manifest.decoded.sha256) {
      throw new Error(`Resumable model decoded sha256 mismatch: got ${decodedSha256}, expected ${manifest.decoded.sha256}`);
    }
    releaseActiveHashes();
    await bestEffortEnforceResumableShardCacheBounds(
      store,
      maxCacheEntries,
      maxCacheBytes,
      activeResumableShardHashes(persistenceDomain),
    );
    finalBoundsEnforced = true;
    throwIfAborted(options.signal);
    const result: Lc0ResumableModelLoadResult = {
      model: model.buffer,
      manifestUrl,
      sha256: decodedSha256,
      bytes: model.byteLength,
      downloadedBytes,
      reusedBytes,
      downloadedShards,
      reusedShards,
      corruptShardsEvicted,
      corruptionRetries,
      uniqueShardCount: unique.size,
      deduplicatedReferences: manifest.shards.length - unique.size,
      peakTemporaryBytes,
      elapsedMs: nowMs() - started,
    };
    throwIfAborted(options.signal);
    return result;
  } finally {
    releaseActiveHashes();
    if (!finalBoundsEnforced) {
      await bestEffortEnforceResumableShardCacheBounds(
        store,
        maxCacheEntries,
        maxCacheBytes,
        activeResumableShardHashes(persistenceDomain),
      );
    }
  }
}

export interface Lc0ResumableShardCacheClearOptions {
  /** Required explicit opt-in. This API throws unless set to true. */
  researchOnly: true;
  cacheName?: string;
  shardStore?: Lc0ModelShardStore;
}

/** Best-effort cleanup for the research-only persistent resumable shard cache. */
export async function clearResumableLc0ModelShardCache(
  options: Lc0ResumableShardCacheClearOptions,
): Promise<{ removedEntries: number }> {
  if (options?.researchOnly !== true) throw new Error('Resumable model shard cache cleanup requires explicit researchOnly: true opt-in');
  const store = options.shardStore ?? new CacheStorageModelShardStore(options.cacheName ?? DEFAULT_RESUMABLE_SHARD_CACHE_NAME);
  if (store.clear) return { removedEntries: await store.clear() };
  if (!store.list) return { removedEntries: 0 };
  const entries = await store.list();
  await Promise.all(entries.map((entry) => store.delete(entry.sha256)));
  return { removedEntries: entries.length };
}
