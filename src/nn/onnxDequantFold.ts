/**
 * Folds weight-only DequantizeLinear nodes into plain initializers before ORT sees the model.
 *
 * The shipped LC0 models store MatMul weights as int8 (per output column) or int4 (blocked)
 * with f16 scales and dequantize them in-graph. ORT WebGPU runs every one of those
 * DequantizeLinear kernels on every inference (about a third of the eval time on T1 batch 1),
 * although the inputs are constants. This pass rewrites the model bytes once at load time:
 * each DequantizeLinear whose inputs are all initializers is replaced by an initializer holding
 * the dequantized weights (in the scale's dtype), and the now-unused quantized tensors are
 * dropped. The download stays small; the graph ORT compiles is the plain f16/f32 one.
 *
 * Works on the raw protobuf so no ONNX/protobuf library is needed: only the ModelProto ->
 * GraphProto -> node/initializer fields are re-encoded, everything else is copied verbatim.
 * Supported: x in {int8, uint8, int4, uint4} stored as raw_data (or int32_data for int8/uint8),
 * scale in {float32, float16}, optional zero point, per-tensor / per-axis / blocked layouts
 * (opset 21). Anything else is left in place for ORT.
 */

import { f16BitsToF32, f32ToF16Bits } from './float16.ts';

const WIRE_VARINT = 0;
const WIRE_FIXED64 = 1;
const WIRE_LEN = 2;
const WIRE_FIXED32 = 5;

// TensorProto.DataType
export const ONNX_FLOAT = 1;
export const ONNX_UINT8 = 2;
export const ONNX_INT8 = 3;
export const ONNX_INT32 = 6;
export const ONNX_FLOAT16 = 10;
export const ONNX_UINT4 = 21;
export const ONNX_INT4 = 22;

type Field = {
  field: number;
  wire: number;
  /** Byte range of the whole field (tag + value). */
  start: number;
  end: number;
  /** Byte range of the value (payload for length-delimited fields). */
  valueStart: number;
  valueEnd: number;
  varint?: number;
};

/**
 * Decodes a protobuf varint as a two's-complement int64. The low and high 32-bit halves are
 * accumulated exactly, so negative int32/int64 values (10-byte varints near 2^64, e.g. axis=-1
 * or negative int32_data entries) come back as the intended negative number instead of being
 * rounded to 2^64 by float arithmetic. Magnitudes above 2^53 lose precision; nothing in the
 * ONNX metadata we read gets there.
 */
function readVarint(buf: Uint8Array, pos: number): [number, number] {
  let lo = 0;
  let hi = 0;
  let shift = 0;
  let p = pos;
  for (;;) {
    if (p >= buf.length) throw new Error('onnx: truncated varint');
    const b = buf[p++];
    const bits = b & 0x7f;
    if (shift < 32) {
      lo = (lo | (bits << shift)) >>> 0;
      if (shift > 25) hi = (hi | (bits >>> (32 - shift))) >>> 0;
    } else {
      hi = (hi | (bits << (shift - 32))) >>> 0;
    }
    if ((b & 0x80) === 0) break;
    shift += 7;
    if (shift > 63) throw new Error('onnx: varint too long');
  }
  const signedHi = hi >= 0x8000_0000 ? hi - 0x1_0000_0000 : hi;
  return [signedHi * 0x1_0000_0000 + lo, p];
}

export function parseProtoFields(buf: Uint8Array, start = 0, end = buf.length): Field[] {
  const out: Field[] = [];
  let p = start;
  while (p < end) {
    const fieldStart = p;
    const [tag, afterTag] = readVarint(buf, p);
    p = afterTag;
    const field = Math.floor(tag / 8);
    const wire = tag & 7;
    let valueStart = p;
    let valueEnd: number;
    let varint: number | undefined;
    if (wire === WIRE_VARINT) {
      const [v, next] = readVarint(buf, p);
      varint = v;
      valueEnd = next;
    } else if (wire === WIRE_FIXED64) {
      valueEnd = p + 8;
    } else if (wire === WIRE_FIXED32) {
      valueEnd = p + 4;
    } else if (wire === WIRE_LEN) {
      const [len, next] = readVarint(buf, p);
      valueStart = next;
      valueEnd = next + len;
      if (len < 0 || valueEnd > buf.length) throw new Error('onnx: invalid protobuf length');
    } else {
      throw new Error(`onnx: unsupported protobuf wire type ${wire} for field ${field}`);
    }
    if (valueEnd > end) throw new Error('onnx: truncated protobuf field');
    out.push({ field, wire, start: fieldStart, end: valueEnd, valueStart, valueEnd, varint });
    p = valueEnd;
  }
  return out;
}

const textDecoder = new TextDecoder();
const textEncoder = new TextEncoder();

function fieldString(buf: Uint8Array, f: Field): string {
  return textDecoder.decode(buf.subarray(f.valueStart, f.valueEnd));
}

export class ProtoWriter {
  private chunks: Uint8Array[] = [];
  private scratch = new Uint8Array(10);
  length = 0;

  varint(value: number): this {
    if (value < 0 || !Number.isFinite(value)) throw new Error(`protobuf: bad varint ${value}`);
    let n = 0;
    let v = value;
    while (v >= 128) {
      this.scratch[n++] = (v % 128) | 0x80;
      v = Math.floor(v / 128);
    }
    this.scratch[n++] = v;
    this.raw(this.scratch.slice(0, n));
    return this;
  }

  tag(field: number, wire: number): this {
    return this.varint(field * 8 + wire);
  }

  varintField(field: number, value: number): this {
    return this.tag(field, WIRE_VARINT).varint(value);
  }

  bytesField(field: number, bytes: Uint8Array): this {
    this.tag(field, WIRE_LEN).varint(bytes.byteLength);
    return this.raw(bytes);
  }

  stringField(field: number, value: string): this {
    return this.bytesField(field, textEncoder.encode(value));
  }

  raw(bytes: Uint8Array): this {
    this.chunks.push(bytes);
    this.length += bytes.byteLength;
    return this;
  }

  finish(): Uint8Array {
    const out = new Uint8Array(this.length);
    let p = 0;
    for (const c of this.chunks) {
      out.set(c, p);
      p += c.byteLength;
    }
    this.chunks = [out];
    return out;
  }
}

/** TensorProto with raw_data storage. */
export function encodeTensorProto(name: string, dataType: number, dims: number[], raw: Uint8Array): Uint8Array {
  const w = new ProtoWriter();
  for (const d of dims) w.varintField(1, d);
  w.varintField(2, dataType);
  w.stringField(8, name);
  w.bytesField(9, raw);
  return w.finish();
}

export { f16BitsToF32, f32ToF16Bits };

type TensorInfo = {
  name: string;
  dataType: number;
  dims: number[];
  raw: Uint8Array | null;
  int32Data: Int32Array | null;
  floatData: number[] | null;
};

function parseTensor(buf: Uint8Array, f: Field): TensorInfo {
  const info: TensorInfo = { name: '', dataType: 0, dims: [], raw: null, int32Data: null, floatData: null };
  const int32: number[] = [];
  const floats: number[] = [];
  for (const t of parseProtoFields(buf, f.valueStart, f.valueEnd)) {
    switch (t.field) {
      case 1: // dims
        if (t.wire === WIRE_VARINT) info.dims.push(t.varint!);
        else
          for (let p = t.valueStart; p < t.valueEnd; ) {
            const [v, n] = readVarint(buf, p);
            info.dims.push(v);
            p = n;
          }
        break;
      case 2:
        info.dataType = t.varint ?? 0;
        break;
      case 4: // float_data
        if (t.wire === WIRE_FIXED32) floats.push(new DataView(buf.buffer, buf.byteOffset + t.valueStart, 4).getFloat32(0, true));
        else {
          const dv = new DataView(buf.buffer, buf.byteOffset + t.valueStart, t.valueEnd - t.valueStart);
          for (let p = 0; p + 4 <= dv.byteLength; p += 4) floats.push(dv.getFloat32(p, true));
        }
        break;
      case 5: // int32_data
        if (t.wire === WIRE_VARINT) int32.push(t.varint! | 0);
        else
          for (let p = t.valueStart; p < t.valueEnd; ) {
            const [v, n] = readVarint(buf, p);
            int32.push(v | 0);
            p = n;
          }
        break;
      case 8:
        info.name = fieldString(buf, t);
        break;
      case 9:
        info.raw = buf.subarray(t.valueStart, t.valueEnd);
        break;
      case 13: // external_data: not supported for folding
        info.raw = null;
        info.dataType = -1;
        break;
      default:
        break;
    }
  }
  if (int32.length) info.int32Data = Int32Array.from(int32);
  if (floats.length) info.floatData = floats;
  return info;
}

function elementCount(dims: number[]): number {
  let n = 1;
  for (const d of dims) n *= d;
  return n;
}

type TensorReader = { length: number; at(index: number): number };

/** Views/readers only: even packed int4 weights are never unpacked into a second tensor. */
function integerReader(t: TensorInfo): TensorReader | null {
  const length = elementCount(t.dims);
  if (!Number.isSafeInteger(length) || length < 0) return null;
  if (t.dataType === ONNX_INT8 || t.dataType === ONNX_UINT8) {
    if (t.raw) {
      if (t.raw.byteLength !== length) return null;
      const raw = t.raw;
      return { length, at: t.dataType === ONNX_INT8 ? (i) => (raw[i] << 24) >> 24 : (i) => raw[i] };
    }
    const data = t.int32Data;
    if (!data || data.length !== length) return null;
    return { length, at: t.dataType === ONNX_INT8 ? (i) => (data[i] << 24) >> 24 : (i) => data[i] & 255 };
  }
  if (t.dataType === ONNX_INT4 || t.dataType === ONNX_UINT4) {
    const raw = t.raw;
    if (!raw || raw.byteLength !== Math.ceil(length / 2)) return null;
    const nibble = (i: number) => (raw[i >> 1] >> ((i & 1) * 4)) & 15;
    return { length, at: t.dataType === ONNX_INT4 ? (i) => (nibble(i) << 28) >> 28 : nibble };
  }
  return null;
}

function floatReader(t: TensorInfo): TensorReader | null {
  const length = elementCount(t.dims);
  if (!Number.isSafeInteger(length) || length < 0) return null;
  if (t.dataType === ONNX_FLOAT && !t.raw) {
    const data = t.floatData;
    return data && data.length === length ? { length, at: (i) => data[i] } : null;
  }
  const width = t.dataType === ONNX_FLOAT ? 4 : t.dataType === ONNX_FLOAT16 ? 2 : 0;
  if (!width || !t.raw || t.raw.byteLength !== length * width) return null;
  const view = new DataView(t.raw.buffer, t.raw.byteOffset, t.raw.byteLength);
  const half = view as DataView & { getFloat16?: (offset: number, littleEndian?: boolean) => number };
  if (width === 2 && typeof half.getFloat16 === 'function') {
    return {
      length,
      at: (i) => {
        const value = half.getFloat16!(i * 2, true);
        return Number.isNaN(value) ? NaN : value; // Match f16BitsToF32's canonical NaN.
      },
    };
  }
  return { length, at: width === 4 ? (i) => view.getFloat32(i * 4, true) : (i) => f16BitsToF32(view.getUint16(i * 2, true)) };
}

type DequantPlan = { total: number; write(view: DataView, offset: number, dataType: number): void };

/** Validate/plan without materializing any dequantized weights. */
function planDequantize(x: TensorInfo, scale: TensorInfo, zeroPoint: TensorInfo | null, axisAttr: number, blockSize: number): DequantPlan | null {
  const q = integerReader(x);
  const s = floatReader(scale);
  if (!q || !s) return null;
  let zp: TensorReader | null = null;
  if (zeroPoint) {
    if (zeroPoint.dataType !== x.dataType) return null;
    zp = integerReader(zeroPoint);
    if (!zp || zp.length !== s.length) return null;
  }
  const dims = x.dims;
  const rank = dims.length;
  const total = q.length;
  const scalar = s.length === 1 && blockSize === 0;
  let outer = 1,
    inner = 1,
    axisDim = 0,
    blocks = 0;
  if (!scalar) {
    const axis = axisAttr < 0 ? axisAttr + rank : axisAttr;
    if (axis < 0 || axis >= rank || blockSize < 0) return null;
    for (let d = 0; d < axis; d++) outer *= dims[d];
    axisDim = dims[axis];
    for (let d = axis + 1; d < rank; d++) inner *= dims[d];
    if (blockSize === 0) {
      if (scale.dims.length !== 1 || scale.dims[0] !== axisDim) return null;
    } else {
      blocks = Math.ceil(axisDim / blockSize);
      if (scale.dims.length !== rank) return null;
      for (let d = 0; d < rank; d++) if (scale.dims[d] !== (d === axis ? blocks : dims[d])) return null;
    }
  }
  return {
    total,
    write(view, offset, dataType) {
      // Keep the old f64 -> f32 -> f16 rounding, not direct f64 -> f16.
      // Native half IO is exact for finite values; NaNs retain the legacy encoding.
      const half = view as DataView & { setFloat16?: (offset: number, value: number, littleEndian?: boolean) => void };
      const storeHalf =
        typeof half.setFloat16 === 'function'
          ? (i: number, value: number) => {
              if (Number.isNaN(value)) view.setUint16(offset + i * 2, f32ToF16Bits(value), true);
              else half.setFloat16!(offset + i * 2, Math.fround(value), true);
            }
          : (i: number, value: number) => view.setUint16(offset + i * 2, f32ToF16Bits(value), true);
      const store = dataType === ONNX_FLOAT16 ? storeHalf : (i: number, value: number) => view.setFloat32(offset + i * 4, value, true);
      if (scalar) {
        const sv = s.at(0),
          zv = zp ? zp.at(0) : 0;
        for (let i = 0; i < total; i++) store(i, (q.at(i) - zv) * sv);
        return;
      }
      let idx = 0;
      for (let o = 0; o < outer; o++) {
        for (let j = 0; j < axisDim; j++) {
          if (blockSize === 0) {
            const sv = s.at(j),
              zv = zp ? zp.at(j) : 0;
            for (let i = 0; i < inner; i++, idx++) store(idx, (q.at(idx) - zv) * sv);
          } else {
            const base = (o * blocks + Math.floor(j / blockSize)) * inner;
            if (zp) for (let i = 0; i < inner; i++, idx++) store(idx, (q.at(idx) - zp.at(base + i)) * s.at(base + i));
            else for (let i = 0; i < inner; i++, idx++) store(idx, q.at(idx) * s.at(base + i));
          }
        }
      }
    },
  };
}

function varintLength(value: number): number {
  if (!Number.isSafeInteger(value) || value < 0) throw new Error(`protobuf: bad size ${value}`);
  let length = 1;
  for (; value >= 128; value = Math.floor(value / 128)) length++;
  return length;
}

function bytesFieldLength(field: number, length: number): number {
  return varintLength(field * 8 + WIRE_LEN) + varintLength(length) + length;
}

/** Exact-size model output: no tensor, graph or final-model concatenation copies. */
class ModelWriter {
  readonly bytes: Uint8Array;
  readonly view: DataView;
  position = 0;
  constructor(length: number) {
    this.bytes = new Uint8Array(length);
    this.view = new DataView(this.bytes.buffer, this.bytes.byteOffset, this.bytes.byteLength);
  }
  varint(value: number): void {
    while (value >= 128) {
      this.bytes[this.position++] = (value % 128) | 0x80;
      value = Math.floor(value / 128);
    }
    this.bytes[this.position++] = value;
  }
  varintField(field: number, value: number): void {
    this.varint(field * 8);
    this.varint(value);
  }
  header(field: number, length: number): void {
    this.varint(field * 8 + WIRE_LEN);
    this.varint(length);
  }
  raw(bytes: Uint8Array): void {
    this.bytes.set(bytes, this.position);
    this.position += bytes.byteLength;
  }
}

type FoldedTensor = { name: Uint8Array; dataType: number; dims: number[]; plan: DequantPlan; rawBytes: number; byteLength: number };
function planFoldedTensor(name: string, dataType: number, dims: number[], plan: DequantPlan): FoldedTensor {
  const encodedName = textEncoder.encode(name);
  const rawBytes = plan.total * (dataType === ONNX_FLOAT16 ? 2 : 4);
  const byteLength =
    dims.reduce((n, dim) => n + 1 + varintLength(dim), 0) +
    1 +
    varintLength(dataType) +
    bytesFieldLength(8, encodedName.length) +
    bytesFieldLength(9, rawBytes);
  return { name: encodedName, dataType, dims, plan, rawBytes, byteLength };
}
function writeFoldedTensor(writer: ModelWriter, tensor: FoldedTensor): void {
  writer.header(5, tensor.byteLength);
  for (const dim of tensor.dims) writer.varintField(1, dim);
  writer.varintField(2, tensor.dataType);
  writer.header(8, tensor.name.length);
  writer.raw(tensor.name);
  writer.header(9, tensor.rawBytes);
  tensor.plan.write(writer.view, writer.position, tensor.dataType);
  writer.position += tensor.rawBytes;
}

type NodeInfo = {
  field: Field;
  opType: string;
  domain: string;
  inputs: string[];
  outputs: string[];
  axis: number;
  blockSize: number;
  /** DequantizeLinear output_dtype (opset 25+); 0 means "same as the scale". */
  outputDtype: number;
};

/**
 * Counts every tensor name a node reads, including reads from subgraphs nested in its
 * attributes (If/Loop/Scan bodies capture outer-scope initializers by name). Names that a
 * subgraph shadows with its own input/initializer are over-counted, which only keeps an
 * initializer alive that could have been dropped.
 */
function countNodeConsumers(buf: Uint8Array, node: Field, consumers: Map<string, number>): void {
  for (const t of parseProtoFields(buf, node.valueStart, node.valueEnd)) {
    if (t.field === 1 && t.wire === WIRE_LEN) {
      const name = fieldString(buf, t);
      consumers.set(name, (consumers.get(name) ?? 0) + 1);
    } else if (t.field === 5 && t.wire === WIRE_LEN) {
      for (const a of parseProtoFields(buf, t.valueStart, t.valueEnd)) {
        if ((a.field === 6 || a.field === 11) && a.wire === WIRE_LEN) countGraphConsumers(buf, a, consumers); // AttributeProto.g (6) / .graphs (11)
      }
    }
  }
}

function countGraphConsumers(buf: Uint8Array, graph: Field, consumers: Map<string, number>): void {
  for (const f of parseProtoFields(buf, graph.valueStart, graph.valueEnd)) {
    if (f.field === 1 && f.wire === WIRE_LEN) countNodeConsumers(buf, f, consumers);
  }
}

function parseNode(buf: Uint8Array, f: Field): NodeInfo {
  const node: NodeInfo = { field: f, opType: '', domain: '', inputs: [], outputs: [], axis: 1, blockSize: 0, outputDtype: 0 };
  for (const t of parseProtoFields(buf, f.valueStart, f.valueEnd)) {
    switch (t.field) {
      case 1:
        node.inputs.push(fieldString(buf, t));
        break;
      case 2:
        node.outputs.push(fieldString(buf, t));
        break;
      case 4:
        node.opType = fieldString(buf, t);
        break;
      case 7:
        node.domain = fieldString(buf, t);
        break;
      case 5: {
        let name = '';
        let ival: number | undefined;
        for (const a of parseProtoFields(buf, t.valueStart, t.valueEnd)) {
          if (a.field === 1) name = fieldString(buf, a);
          else if (a.field === 3) ival = a.varint;
        }
        if (ival === undefined) break;
        // AttributeProto.i is int64 two's complement; readVarint already returns it signed.
        if (name === 'axis') node.axis = ival;
        else if (name === 'block_size') node.blockSize = ival;
        else if (name === 'output_dtype') node.outputDtype = ival;
        break;
      }
      default:
        break;
    }
  }
  return node;
}

export type OnnxDequantFoldResult = {
  bytes: Uint8Array;
  /** DequantizeLinear nodes replaced by initializers. */
  foldedNodes: number;
  /** DequantizeLinear nodes left in the graph (non-constant inputs or unsupported layout/dtype). */
  skippedNodes: number;
  removedInitializers: number;
  bytesBefore: number;
  bytesAfter: number;
  elapsedMs: number;
};

function now(): number {
  return typeof performance === 'undefined' ? Date.now() : performance.now();
}

/** Count DequantizeLinear nodes without rewriting anything (cheap: only node headers are parsed). */
export function countOnnxDequantizeLinear(model: Uint8Array): number {
  const graph = parseProtoFields(model).find((f) => f.field === 7 && f.wire === WIRE_LEN);
  if (!graph) return 0;
  let n = 0;
  for (const f of parseProtoFields(model, graph.valueStart, graph.valueEnd)) {
    if (f.field !== 1 || f.wire !== WIRE_LEN) continue;
    const node = parseNode(model, f);
    if (node.opType === 'DequantizeLinear') n += 1;
  }
  return n;
}

export function foldOnnxDequantizeLinear(model: Uint8Array): OnnxDequantFoldResult {
  const t0 = now();
  const unchanged = (skipped: number): OnnxDequantFoldResult => ({
    bytes: model,
    foldedNodes: 0,
    skippedNodes: skipped,
    removedInitializers: 0,
    bytesBefore: model.byteLength,
    bytesAfter: model.byteLength,
    elapsedMs: now() - t0,
  });
  const modelFields = parseProtoFields(model);
  const graphField = modelFields.find((f) => f.field === 7 && f.wire === WIRE_LEN);
  if (!graphField) return unchanged(0);
  const graphFields = parseProtoFields(model, graphField.valueStart, graphField.valueEnd);

  const nodes: NodeInfo[] = [];
  const initializers = new Map<string, { field: Field; info: TensorInfo | null }>();
  const graphInputs = new Set<string>();
  const graphOutputs = new Set<string>();
  const consumers = new Map<string, number>();
  for (const f of graphFields) {
    if (f.wire !== WIRE_LEN) continue;
    if (f.field === 1) {
      nodes.push(parseNode(model, f));
      countNodeConsumers(model, f, consumers);
    } else if (f.field === 5) {
      let name = '';
      for (const t of parseProtoFields(model, f.valueStart, f.valueEnd))
        if (t.field === 8) {
          name = fieldString(model, t);
          break;
        }
      initializers.set(name, { field: f, info: null });
    } else if (f.field === 11 || f.field === 12) {
      for (const t of parseProtoFields(model, f.valueStart, f.valueEnd))
        if (t.field === 1) (f.field === 11 ? graphInputs : graphOutputs).add(fieldString(model, t));
    }
  }
  const dql = nodes.filter((n) => n.opType === 'DequantizeLinear' && (n.domain === '' || n.domain === 'ai.onnx'));
  if (dql.length === 0) return unchanged(0);

  const tensor = (name: string): TensorInfo | null => {
    const entry = initializers.get(name);
    if (!entry) return null;
    if (!entry.info) entry.info = parseTensor(model, entry.field);
    return entry.info;
  };

  const removedNodes = new Set<Field>();
  const folds = new Map<Field, FoldedTensor>();
  const droppedInitializers = new Set<Field>();
  let skipped = 0;
  for (const node of dql) {
    if (node.inputs.length < 2 || node.outputs.length !== 1) {
      skipped += 1;
      continue;
    }
    const x = tensor(node.inputs[0]);
    const scale = tensor(node.inputs[1]);
    const zpName = node.inputs.length > 2 && node.inputs[2] ? node.inputs[2] : null;
    // An initializer that is also a graph input is a default the caller may override at run
    // time, so it is not a constant we can bake in.
    if (graphInputs.has(node.inputs[0]) || graphInputs.has(node.inputs[1]) || (zpName && graphInputs.has(zpName))) {
      skipped += 1;
      continue;
    }
    const zp = zpName ? tensor(zpName) : null;
    if (!x || !scale || (zpName && !zp) || (scale.dataType !== ONNX_FLOAT && scale.dataType !== ONNX_FLOAT16)) {
      skipped += 1;
      continue;
    }
    if (initializers.has(node.outputs[0])) {
      skipped += 1;
      continue;
    }
    const outType = node.outputDtype || scale.dataType;
    if (outType !== ONNX_FLOAT && outType !== ONNX_FLOAT16) {
      skipped += 1;
      continue;
    }
    const plan = planDequantize(x, scale, zp, node.axis, node.blockSize);
    if (!plan) {
      skipped += 1;
      continue;
    }
    const xEntry = initializers.get(node.inputs[0])!;
    if (folds.has(xEntry.field)) {
      skipped += 1;
      continue;
    }
    removedNodes.add(node.field);
    folds.set(xEntry.field, planFoldedTensor(node.outputs[0], outType, x.dims, plan));
    for (const name of [node.inputs[0], node.inputs[1], zpName]) {
      if (!name) continue;
      const left = (consumers.get(name) ?? 1) - 1;
      consumers.set(name, left);
      if (left <= 0 && !graphInputs.has(name) && !graphOutputs.has(name)) droppedInitializers.add(initializers.get(name)!.field);
    }
  }
  if (removedNodes.size === 0) return unchanged(skipped);

  const graph = (writer?: ModelWriter): number => {
    let length = 0;
    for (const f of graphFields) {
      if (f.field === 1 && removedNodes.has(f)) continue;
      if (f.field === 5) {
        const folded = folds.get(f);
        if (folded) {
          length += bytesFieldLength(5, folded.byteLength);
          if (writer) writeFoldedTensor(writer, folded);
        }
        if (droppedInitializers.has(f)) continue;
      }
      length += f.end - f.start;
      writer?.raw(model.subarray(f.start, f.end));
    }
    return length;
  };
  const graphLength = graph();
  const modelLength = modelFields.reduce((length, f) => length + (f === graphField ? bytesFieldLength(7, graphLength) : f.end - f.start), 0);
  const writer = new ModelWriter(modelLength);
  for (const f of modelFields) {
    if (f === graphField) {
      writer.header(7, graphLength);
      graph(writer);
    } else writer.raw(model.subarray(f.start, f.end));
  }
  if (writer.position !== modelLength) throw new Error('onnx: folded model size mismatch');
  const bytes = writer.bytes;
  return {
    bytes,
    foldedNodes: removedNodes.size,
    skippedNodes: skipped,
    removedInitializers: droppedInitializers.size,
    bytesBefore: model.byteLength,
    bytesAfter: bytes.byteLength,
    elapsedMs: now() - t0,
  };
}
