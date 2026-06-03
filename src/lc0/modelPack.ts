import { sha256Hex } from './modelCache.ts';

export interface Lc0WebValueInfo {
  name: string;
  dtype: string;
  onnxDtype?: string;
  shape: Array<number | string | null>;
}

export interface Lc0WebNodeInfo {
  name?: string;
  opType: string;
  domain?: string;
  inputs: string[];
  outputs: string[];
  attributes?: Record<string, unknown>;
}

export interface Lc0WebShardInfo {
  file: string;
  bytes: number;
  sha256: string;
}

export interface Lc0WebTensorInfo {
  name: string;
  dtype: string;
  onnxDtype?: string;
  shape: number[];
  shard: string;
  byteOffset: number;
  byteLength: number;
  sha256?: string;
}

export interface Lc0WebModelPackManifest {
  format: 'lc0web';
  version: number;
  packSha256?: string;
  model: {
    name: string;
    family: 'lc0' | string;
    sourceFormat?: string;
    sourceFile?: string;
    sourceSha256?: string;
    architecture?: string;
    recommendedRuntime?: 'ort-webgpu' | 'custom-webgpu' | 'hybrid' | string;
    layout?: string;
  };
  graph: {
    name?: string;
    opsets?: Array<{ domain: string; version: number }>;
    inputs: Lc0WebValueInfo[];
    outputs: Lc0WebValueInfo[];
    nodes?: Lc0WebNodeInfo[];
    opHistogram?: Record<string, number>;
  };
  weights: {
    shardBytesTarget?: number;
    alignmentBytes?: number;
    totalTensorBytes: number;
    tensorCount: number;
    dtypeHistogram?: Record<string, number>;
    shards: Lc0WebShardInfo[];
    tensors: Lc0WebTensorInfo[];
  };
}

export interface Lc0WebTensorView {
  info: Lc0WebTensorInfo;
  bytes: Uint8Array;
}

export interface Lc0WebLoadedPack {
  manifestUrl: string;
  manifest: Lc0WebModelPackManifest;
  elapsedMs: number;
  verifiedShards: Array<{ file: string; bytes: number; sha256: string }>;
  tensors: Map<string, Lc0WebTensorView>;
}

export interface Lc0WebLoadOptions {
  /** Defaults to global fetch. Useful for tests and worker-owned cache layers. */
  fetchFn?: typeof fetch;
  /** Defaults to true when weights are loaded. */
  verifyShards?: boolean;
  /** Defaults to true. Set false to inspect only JSON graph/weight metadata. */
  loadWeights?: boolean;
  /** Load only these tensors and their containing shards. Defaults to all tensors. */
  tensorNames?: string[];
}

function nowMs(): number {
  return typeof performance === 'undefined' ? Date.now() : performance.now();
}

function normalizeManifestUrl(href: string): string {
  // Some dev servers tolerate/echo `model.lc0web.json/`. Treat pack URLs as
  // files so relative shard URLs resolve beside the manifest, not below it.
  return href.replace(/\.json\/$/i, '.json');
}

function absoluteUrl(url: string): string {
  if (/^[a-z][a-z0-9+.-]*:/i.test(url)) return normalizeManifestUrl(url);
  if (typeof location !== 'undefined') return normalizeManifestUrl(new URL(url, location.href).href);
  throw new Error(`Relative lc0web pack URL requires browser location or an absolute URL: ${url}`);
}

function shardUrl(manifestUrl: string, file: string): string {
  return new URL(file, absoluteUrl(manifestUrl)).href;
}

function assertFiniteNonNegativeInteger(value: unknown, label: string): asserts value is number {
  if (!Number.isInteger(value) || (value as number) < 0) {
    throw new Error(`Invalid lc0web ${label}: ${value}`);
  }
}

export function validateLc0WebModelPackManifest(manifest: unknown): asserts manifest is Lc0WebModelPackManifest {
  const m = manifest as Lc0WebModelPackManifest;
  if (!m || typeof m !== 'object') throw new Error('Invalid lc0web manifest: not an object');
  if (m.format !== 'lc0web') throw new Error(`Invalid lc0web manifest format: ${String(m.format)}`);
  if (m.version !== 1) throw new Error(`Unsupported lc0web manifest version: ${String(m.version)}`);
  if (!m.model || typeof m.model.name !== 'string') throw new Error('Invalid lc0web manifest: missing model.name');
  if (!m.graph || !Array.isArray(m.graph.inputs) || !Array.isArray(m.graph.outputs)) {
    throw new Error('Invalid lc0web manifest: missing graph inputs/outputs');
  }
  if (!m.weights || !Array.isArray(m.weights.shards) || !Array.isArray(m.weights.tensors)) {
    throw new Error('Invalid lc0web manifest: missing weights shards/tensors');
  }
  assertFiniteNonNegativeInteger(m.weights.totalTensorBytes, 'weights.totalTensorBytes');
  assertFiniteNonNegativeInteger(m.weights.tensorCount, 'weights.tensorCount');
  if (m.weights.tensorCount !== m.weights.tensors.length) {
    throw new Error(`Invalid lc0web manifest: tensorCount ${m.weights.tensorCount} != tensors.length ${m.weights.tensors.length}`);
  }

  const shardNames = new Set<string>();
  for (const shard of m.weights.shards) {
    if (!shard || typeof shard.file !== 'string' || !shard.file) throw new Error('Invalid lc0web shard file');
    if (shardNames.has(shard.file)) throw new Error(`Duplicate lc0web shard: ${shard.file}`);
    shardNames.add(shard.file);
    assertFiniteNonNegativeInteger(shard.bytes, `shard.bytes ${shard.file}`);
    if (typeof shard.sha256 !== 'string' || !/^[0-9a-f]{64}$/i.test(shard.sha256)) {
      throw new Error(`Invalid lc0web shard sha256 for ${shard.file}`);
    }
  }

  const tensorNames = new Set<string>();
  for (const tensor of m.weights.tensors) {
    if (!tensor || typeof tensor.name !== 'string' || !tensor.name) throw new Error('Invalid lc0web tensor name');
    if (tensorNames.has(tensor.name)) throw new Error(`Duplicate lc0web tensor: ${tensor.name}`);
    tensorNames.add(tensor.name);
    if (!Array.isArray(tensor.shape) || !tensor.shape.every((dim) => Number.isInteger(dim) && dim >= 0)) {
      throw new Error(`Invalid lc0web tensor shape for ${tensor.name}`);
    }
    if (typeof tensor.dtype !== 'string' || !tensor.dtype) throw new Error(`Invalid lc0web tensor dtype for ${tensor.name}`);
    if (!shardNames.has(tensor.shard)) throw new Error(`Tensor ${tensor.name} references missing shard ${tensor.shard}`);
    assertFiniteNonNegativeInteger(tensor.byteOffset, `tensor.byteOffset ${tensor.name}`);
    assertFiniteNonNegativeInteger(tensor.byteLength, `tensor.byteLength ${tensor.name}`);
    const shard = m.weights.shards.find((s) => s.file === tensor.shard)!;
    if (tensor.byteOffset + tensor.byteLength > shard.bytes) {
      throw new Error(`Tensor ${tensor.name} range exceeds shard ${tensor.shard}`);
    }
    if (tensor.sha256 !== undefined && !/^[0-9a-f]{64}$/i.test(tensor.sha256)) {
      throw new Error(`Invalid lc0web tensor sha256 for ${tensor.name}`);
    }
  }
}

async function fetchArrayBuffer(fetchFn: typeof fetch, url: string): Promise<ArrayBuffer> {
  const response = await fetchFn(url);
  if (!response.ok) throw new Error(`lc0web fetch failed for ${url}: ${response.status}`);
  return response.arrayBuffer();
}

export async function loadLc0WebModelPack(manifestUrlInput: string, options: Lc0WebLoadOptions = {}): Promise<Lc0WebLoadedPack> {
  const started = nowMs();
  const manifestUrl = absoluteUrl(manifestUrlInput);
  const fetchFn = options.fetchFn ?? fetch;
  const loadWeights = options.loadWeights ?? true;
  const verifyShards = options.verifyShards ?? loadWeights;
  const manifestResponse = await fetchFn(manifestUrl);
  if (!manifestResponse.ok) throw new Error(`lc0web manifest fetch failed for ${manifestUrl}: ${manifestResponse.status}`);
  const manifest = await manifestResponse.json() as unknown;
  validateLc0WebModelPackManifest(manifest);

  if (!loadWeights) {
    return { manifestUrl, manifest, elapsedMs: nowMs() - started, verifiedShards: [], tensors: new Map() };
  }

  const wantedTensorNames = options.tensorNames ? new Set(options.tensorNames) : undefined;
  const selectedTensors = wantedTensorNames
    ? manifest.weights.tensors.filter((tensor) => wantedTensorNames.has(tensor.name))
    : manifest.weights.tensors;
  if (wantedTensorNames && selectedTensors.length !== wantedTensorNames.size) {
    const found = new Set(selectedTensors.map((tensor) => tensor.name));
    const missing = [...wantedTensorNames].filter((name) => !found.has(name));
    throw new Error(`lc0web pack missing requested tensors: ${missing.join(', ')}`);
  }

  const selectedShardFiles = new Set(selectedTensors.map((tensor) => tensor.shard));
  const selectedShards = manifest.weights.shards.filter((shard) => selectedShardFiles.has(shard.file));
  const shardBytes = new Map<string, Uint8Array>();
  const verifiedShards: Array<{ file: string; bytes: number; sha256: string }> = [];
  for (const shard of selectedShards) {
    const bytes = new Uint8Array(await fetchArrayBuffer(fetchFn, shardUrl(manifestUrl, shard.file)));
    if (bytes.byteLength !== shard.bytes) {
      throw new Error(`lc0web shard ${shard.file} byte length mismatch: got ${bytes.byteLength}, expected ${shard.bytes}`);
    }
    if (verifyShards) {
      const actualSha = await sha256Hex(bytes);
      if (actualSha !== shard.sha256.toLowerCase()) {
        throw new Error(`lc0web shard ${shard.file} sha256 mismatch: got ${actualSha}, expected ${shard.sha256.toLowerCase()}`);
      }
      verifiedShards.push({ file: shard.file, bytes: bytes.byteLength, sha256: actualSha });
    }
    shardBytes.set(shard.file, bytes);
  }

  const tensors = new Map<string, Lc0WebTensorView>();
  for (const tensor of selectedTensors) {
    const shard = shardBytes.get(tensor.shard);
    if (!shard) throw new Error(`lc0web internal error: shard not loaded for tensor ${tensor.name}`);
    const view = shard.subarray(tensor.byteOffset, tensor.byteOffset + tensor.byteLength);
    if (tensor.sha256 && verifyShards) {
      const actualSha = await sha256Hex(view);
      if (actualSha !== tensor.sha256.toLowerCase()) {
        throw new Error(`lc0web tensor ${tensor.name} sha256 mismatch: got ${actualSha}, expected ${tensor.sha256.toLowerCase()}`);
      }
    }
    tensors.set(tensor.name, { info: tensor, bytes: view });
  }

  return { manifestUrl, manifest, elapsedMs: nowMs() - started, verifiedShards, tensors };
}
