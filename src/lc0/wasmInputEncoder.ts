import { LC0_BOARD_SQUARES, LC0_CLASSICAL_112_PLANES, type Lc0EncodedPlanes112 } from './encoder112.ts';

export type Lc0WasmInputEncoderSource = string | URL | ArrayBuffer | Uint8Array | WebAssembly.Module;

type Lc0InputEncoderExports = WebAssembly.Exports & {
  memory: WebAssembly.Memory;
  lc0_input_buffer_ptr: () => number;
  lc0_input_buffer_len: () => number;
  lc0_planes_buffer_ptr: () => number;
  lc0_planes_len: () => number;
  lc0_masks_buffer_ptr: () => number;
  lc0_values_buffer_ptr: () => number;
  lc0_last_error: () => number;
  lc0_encode_fen: (len: number, historyFill: number) => number;
  lc0_encode_fen_history: (len: number) => number;
};

export class Lc0WasmInputEncoder {
  private readonly exports: Lc0InputEncoderExports;
  private readonly textEncoder = new TextEncoder();

  constructor(instance: WebAssembly.Instance) {
    this.exports = instance.exports as Lc0InputEncoderExports;
    if (!(this.exports.memory instanceof WebAssembly.Memory)) throw new Error('LC0 WASM input encoder did not export memory');
  }

  encodeFen(fen: string, options: { historyFill?: boolean } = {}): Lc0EncodedPlanes112 {
    const bytes = this.textEncoder.encode(fen);
    this.writeInput(bytes);
    const status = this.exports.lc0_encode_fen(bytes.byteLength, options.historyFill === false ? 0 : 1);
    this.assertOk(status);
    return this.readPlanes();
  }

  encodeFenHistory(fens: readonly string[]): Lc0EncodedPlanes112 {
    const bytes = this.textEncoder.encode(fens.join('\n'));
    this.writeInput(bytes);
    const status = this.exports.lc0_encode_fen_history(bytes.byteLength);
    this.assertOk(status);
    return this.readPlanes();
  }

  private writeInput(bytes: Uint8Array): void {
    const ptr = this.exports.lc0_input_buffer_ptr();
    const cap = this.exports.lc0_input_buffer_len();
    if (bytes.byteLength > cap) throw new Error(`LC0 WASM input too large: ${bytes.byteLength} > ${cap}`);
    new Uint8Array(this.exports.memory.buffer, ptr, bytes.byteLength).set(bytes);
  }

  private readPlanes(): Lc0EncodedPlanes112 {
    const ptr = this.exports.lc0_planes_buffer_ptr();
    const len = this.exports.lc0_planes_len();
    const planes = new Float32Array(len);
    planes.set(new Float32Array(this.exports.memory.buffer, ptr, len));
    if (len !== LC0_CLASSICAL_112_PLANES * LC0_BOARD_SQUARES) throw new Error(`Unexpected LC0 WASM plane count: ${len}`);
    const masks = Array.from(new BigUint64Array(this.exports.memory.buffer, this.exports.lc0_masks_buffer_ptr(), LC0_CLASSICAL_112_PLANES));
    const values = Array.from(new Float64Array(this.exports.memory.buffer, this.exports.lc0_values_buffer_ptr(), LC0_CLASSICAL_112_PLANES));
    return { planes, shape: [1, 112, 8, 8], masks, values };
  }

  private assertOk(status: number): void {
    if (status !== 0) throw new Error(`LC0 WASM input encoder failed: ${status || this.exports.lc0_last_error()}`);
  }
}

export async function createLc0WasmInputEncoder(source: Lc0WasmInputEncoderSource = '/lc0/lc0_input_encoder.wasm'): Promise<Lc0WasmInputEncoder> {
  if (source instanceof WebAssembly.Module) return new Lc0WasmInputEncoder(await WebAssembly.instantiate(source, {}));
  if (source instanceof ArrayBuffer || ArrayBuffer.isView(source)) {
    const bytes = source instanceof ArrayBuffer ? source : source.buffer.slice(source.byteOffset, source.byteOffset + source.byteLength);
    const result = await WebAssembly.instantiate(bytes, {}) as WebAssembly.Instance | WebAssembly.WebAssemblyInstantiatedSource;
    return new Lc0WasmInputEncoder(result instanceof WebAssembly.Instance ? result : result.instance);
  }
  const response = await fetch(source);
  if (!response.ok) throw new Error(`Failed to fetch LC0 WASM input encoder ${source}: HTTP ${response.status}`);
  const result = await WebAssembly.instantiate(await response.arrayBuffer(), {}) as WebAssembly.Instance | WebAssembly.WebAssemblyInstantiatedSource;
  return new Lc0WasmInputEncoder(result instanceof WebAssembly.Instance ? result : result.instance);
}
