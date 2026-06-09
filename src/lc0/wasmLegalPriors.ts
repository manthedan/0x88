import type { Lc0LegalPrior } from './onnxEvaluator.ts';
import { LC0_POLICY_SIZE } from './policyMap.ts';

export type Lc0WasmLegalPriorsSource = string | URL | ArrayBuffer | Uint8Array | WebAssembly.Module;

export interface Lc0WasmLegalPriorTiming {
  textEncodeMs: number;
  inputWriteMs: number;
  logitsWriteMs: number;
  wasmRunMs: number;
  outputReadMs: number;
  totalMs: number;
  bridgeCopyMs: number;
}

export interface Lc0WasmLegalPriorResult {
  legalPriors: Lc0LegalPrior[];
  bestMove?: string;
  timing: Lc0WasmLegalPriorTiming;
}

type Lc0LegalPriorsExports = WebAssembly.Exports & {
  memory: WebAssembly.Memory;
  lc0_legal_input_buffer_ptr: () => number;
  lc0_legal_input_buffer_len: () => number;
  lc0_legal_logits_buffer_ptr: () => number;
  lc0_legal_logits_len: () => number;
  lc0_legal_indices_ptr: () => number;
  lc0_legal_priors_ptr: () => number;
  lc0_legal_logits_out_ptr: () => number;
  lc0_legal_uci_ptr: () => number;
  lc0_legal_promo_ptr: () => number;
  lc0_legal_count: () => number;
  lc0_legal_last_error: () => number;
  lc0_legal_priors_from_fen: (len: number, temperature: number, topK: number) => number;
};

function nowMs(): number {
  return typeof performance === 'undefined' ? Date.now() : performance.now();
}

function packedUciToString(packed: number, promo: number): string {
  const text = String.fromCharCode(packed & 0xff, (packed >>> 8) & 0xff, (packed >>> 16) & 0xff, (packed >>> 24) & 0xff);
  return promo ? `${text}${String.fromCharCode(promo)}` : text;
}

const WASM_LEGAL_ERROR_LABELS = new Map<number, string>([
  [1, 'FEN exceeds input buffer'],
  [2, 'invalid piece-placement rank separator'],
  [3, 'too many squares in FEN rank'],
  [4, 'piece after full FEN rank'],
  [5, 'invalid FEN piece character'],
  [6, 'incomplete FEN piece placement'],
  [8, 'invalid FEN active color'],
  [10, 'invalid FEN castling field'],
  [12, 'truncated FEN en-passant field'],
  [13, 'invalid FEN en-passant square'],
  [14, 'invalid trailing FEN en-passant field'],
]);

function wasmLegalErrorDetail(status: number, lastError: number): string {
  const code = lastError || status;
  const label = WASM_LEGAL_ERROR_LABELS.get(code) ?? 'unknown error';
  return status === lastError ? `${code} (${label})` : `status=${status}, last_error=${lastError} (${label})`;
}

export class Lc0WasmLegalPriors {
  private readonly exports: Lc0LegalPriorsExports;
  private readonly textEncoder = new TextEncoder();

  constructor(instance: WebAssembly.Instance) {
    this.exports = instance.exports as Lc0LegalPriorsExports;
    if (!(this.exports.memory instanceof WebAssembly.Memory)) throw new Error('LC0 WASM legal-prior module did not export memory');
    if (this.exports.lc0_legal_logits_len() !== LC0_POLICY_SIZE) throw new Error(`Unexpected LC0 WASM policy size: ${this.exports.lc0_legal_logits_len()}`);
  }

  evaluateFen(fen: string, mappedPolicy: ArrayLike<number>, options: { temperature?: number; topK?: number } = {}): Lc0WasmLegalPriorResult {
    if (mappedPolicy.length < LC0_POLICY_SIZE) throw new Error(`LC0 WASM legal-prior logits too short: ${mappedPolicy.length} < ${LC0_POLICY_SIZE}`);
    const totalStarted = nowMs();
    const textEncodeStarted = nowMs();
    const bytes = this.textEncoder.encode(fen);
    const textEncodeMs = nowMs() - textEncodeStarted;
    const inputWriteStarted = nowMs();
    const inputPtr = this.exports.lc0_legal_input_buffer_ptr();
    const inputCap = this.exports.lc0_legal_input_buffer_len();
    if (bytes.byteLength > inputCap) throw new Error(`LC0 WASM legal-prior FEN too large: ${bytes.byteLength} > ${inputCap}`);
    new Uint8Array(this.exports.memory.buffer, inputPtr, bytes.byteLength).set(bytes);
    const inputWriteMs = nowMs() - inputWriteStarted;
    const logitsWriteStarted = nowMs();
    new Float32Array(this.exports.memory.buffer, this.exports.lc0_legal_logits_buffer_ptr(), LC0_POLICY_SIZE).set(mappedPolicy);
    const logitsWriteMs = nowMs() - logitsWriteStarted;
    const wasmRunStarted = nowMs();
    const status = this.exports.lc0_legal_priors_from_fen(bytes.byteLength, options.temperature ?? 1.359, options.topK ?? 0);
    const wasmRunMs = nowMs() - wasmRunStarted;
    if (status !== 0) throw new Error(`LC0 WASM legal-prior failed: ${wasmLegalErrorDetail(status, this.exports.lc0_legal_last_error())}`);
    const outputReadStarted = nowMs();
    const count = this.exports.lc0_legal_count();
    const indices = new Uint16Array(this.exports.memory.buffer, this.exports.lc0_legal_indices_ptr(), count);
    const priors = new Float32Array(this.exports.memory.buffer, this.exports.lc0_legal_priors_ptr(), count);
    const logits = new Float32Array(this.exports.memory.buffer, this.exports.lc0_legal_logits_out_ptr(), count);
    const ucis = new Uint32Array(this.exports.memory.buffer, this.exports.lc0_legal_uci_ptr(), count);
    const promos = new Uint8Array(this.exports.memory.buffer, this.exports.lc0_legal_promo_ptr(), count);
    const legalPriors = Array.from({ length: count }, (_, i) => ({
      uci: packedUciToString(ucis[i], promos[i]),
      index: indices[i],
      logit: logits[i],
      prior: priors[i],
    }));
    const outputReadMs = nowMs() - outputReadStarted;
    const totalMs = nowMs() - totalStarted;
    return { legalPriors, bestMove: legalPriors[0]?.uci, timing: { textEncodeMs, inputWriteMs, logitsWriteMs, wasmRunMs, outputReadMs, totalMs, bridgeCopyMs: textEncodeMs + inputWriteMs + logitsWriteMs + outputReadMs } };
  }
}

export async function createLc0WasmLegalPriors(source: Lc0WasmLegalPriorsSource = '/lc0/lc0_legal_priors.wasm'): Promise<Lc0WasmLegalPriors> {
  if (source instanceof WebAssembly.Module) return new Lc0WasmLegalPriors(await WebAssembly.instantiate(source, {}));
  if (source instanceof ArrayBuffer || ArrayBuffer.isView(source)) {
    const bytes = source instanceof ArrayBuffer ? source : source.buffer.slice(source.byteOffset, source.byteOffset + source.byteLength);
    const result = await WebAssembly.instantiate(bytes, {}) as WebAssembly.Instance | WebAssembly.WebAssemblyInstantiatedSource;
    return new Lc0WasmLegalPriors(result instanceof WebAssembly.Instance ? result : result.instance);
  }
  const response = await fetch(source);
  if (!response.ok) throw new Error(`Failed to fetch LC0 WASM legal-prior module ${source}: HTTP ${response.status}`);
  const result = await WebAssembly.instantiate(await response.arrayBuffer(), {}) as WebAssembly.Instance | WebAssembly.WebAssemblyInstantiatedSource;
  return new Lc0WasmLegalPriors(result instanceof WebAssembly.Instance ? result : result.instance);
}
