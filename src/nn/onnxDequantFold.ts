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

// ---------------------------------------------------------------- writer

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

// ---------------------------------------------------------------- f16

export function f16BitsToF32(h: number): number {
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const mant = h & 0x3ff;
  if (exp === 0) return sign * mant * 2 ** -24;
  if (exp === 31) return mant ? NaN : sign * Infinity;
  return sign * (1 + mant / 1024) * 2 ** (exp - 15);
}

const f32Scratch = new Float32Array(1);
const u32Scratch = new Uint32Array(f32Scratch.buffer);

/** Round-to-nearest-even f32 -> f16 bits (IEEE 754 binary16). */
export function f32ToF16Bits(x: number): number {
  f32Scratch[0] = x;
  const b = u32Scratch[0];
  const sign = (b >>> 16) & 0x8000;
  const exp = (b >>> 23) & 0xff;
  let mant = b & 0x7fffff;
  if (exp === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0);
  const e = exp - 127 + 15;
  if (e >= 31) return sign | 0x7c00;
  if (e <= 0) {
    if (e < -10) return sign;
    mant |= 0x800000;
    const shift = 14 - e;
    let h = mant >>> shift;
    const rem = mant & ((1 << shift) - 1);
    const half = 1 << (shift - 1);
    if (rem > half || (rem === half && h & 1)) h += 1;
    return sign | h;
  }
  let h = (e << 10) | (mant >>> 13);
  const rem = mant & 0x1fff;
  if (rem > 0x1000 || (rem === 0x1000 && h & 1)) h += 1; // carry into the exponent is the correct rounding
  return sign | h;
}

// ---------------------------------------------------------------- tensors

type TensorInfo = {
  name: string;
  dataType: number;
  dims: number[];
  raw: Uint8Array | null;
  int32Data: Int32Array | null;
  floatData: Float32Array | null;
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
  if (floats.length) info.floatData = Float32Array.from(floats);
  return info;
}

function elementCount(dims: number[]): number {
  let n = 1;
  for (const d of dims) n *= d;
  return n;
}

/** Integer tensor as an int8-range array (int4/uint4 unpacked, low nibble first). */
function integerValues(t: TensorInfo): ArrayLike<number> | null {
  const n = elementCount(t.dims);
  switch (t.dataType) {
    case ONNX_INT8:
      if (t.raw) return t.raw.byteLength === n ? new Int8Array(t.raw.buffer, t.raw.byteOffset, n) : null;
      return t.int32Data && t.int32Data.length === n ? Int8Array.from(t.int32Data) : null;
    case ONNX_UINT8:
      if (t.raw) return t.raw.byteLength === n ? t.raw : null;
      return t.int32Data && t.int32Data.length === n ? Uint8Array.from(t.int32Data) : null;
    case ONNX_INT4: {
      if (!t.raw || t.raw.byteLength !== (n + 1) >> 1) return null;
      const out = new Int8Array(n);
      for (let i = 0; i < n; i++) {
        const nib = (t.raw[i >> 1] >> ((i & 1) * 4)) & 0xf;
        out[i] = nib >= 8 ? nib - 16 : nib;
      }
      return out;
    }
    case ONNX_UINT4: {
      if (!t.raw || t.raw.byteLength !== (n + 1) >> 1) return null;
      const out = new Uint8Array(n);
      for (let i = 0; i < n; i++) out[i] = (t.raw[i >> 1] >> ((i & 1) * 4)) & 0xf;
      return out;
    }
    default:
      return null;
  }
}

function floatValues(t: TensorInfo): Float32Array | null {
  const n = elementCount(t.dims);
  if (t.dataType === ONNX_FLOAT) {
    if (t.raw) {
      if (t.raw.byteLength !== n * 4) return null;
      const out = new Float32Array(n);
      const dv = new DataView(t.raw.buffer, t.raw.byteOffset, t.raw.byteLength);
      for (let i = 0; i < n; i++) out[i] = dv.getFloat32(i * 4, true);
      return out;
    }
    return t.floatData && t.floatData.length === n ? t.floatData : null;
  }
  if (t.dataType === ONNX_FLOAT16) {
    if (!t.raw || t.raw.byteLength !== n * 2) return null;
    const out = new Float32Array(n);
    const dv = new DataView(t.raw.buffer, t.raw.byteOffset, t.raw.byteLength);
    for (let i = 0; i < n; i++) out[i] = f16BitsToF32(dv.getUint16(i * 2, true));
    return out;
  }
  return null;
}

function dequantize(x: TensorInfo, scale: TensorInfo, zeroPoint: TensorInfo | null, axisAttr: number, blockSize: number): Float32Array | null {
  const q = integerValues(x);
  const s = floatValues(scale);
  if (!q || !s) return null;
  let zp: ArrayLike<number> | null = null;
  if (zeroPoint) {
    if (zeroPoint.dataType !== x.dataType) return null;
    zp = integerValues(zeroPoint);
    if (!zp || zp.length !== s.length) return null;
  }
  const dims = x.dims;
  const rank = dims.length;
  const total = elementCount(dims);
  const out = new Float32Array(total);
  const scaleCount = elementCount(scale.dims);
  if (scaleCount === 1 && blockSize === 0) {
    const sv = s[0];
    const zv = zp ? zp[0] : 0;
    for (let i = 0; i < total; i++) out[i] = (q[i] - zv) * sv;
    return out;
  }
  let axis = axisAttr < 0 ? axisAttr + rank : axisAttr;
  if (axis < 0 || axis >= rank) return null;
  let outer = 1;
  for (let d = 0; d < axis; d++) outer *= dims[d];
  const axisDim = dims[axis];
  let inner = 1;
  for (let d = axis + 1; d < rank; d++) inner *= dims[d];
  if (blockSize === 0) {
    if (scale.dims.length !== 1 || scale.dims[0] !== axisDim) return null;
    let idx = 0;
    for (let o = 0; o < outer; o++) {
      for (let j = 0; j < axisDim; j++) {
        const sv = s[j];
        const zv = zp ? zp[j] : 0;
        for (let i = 0; i < inner; i++, idx++) out[idx] = (q[idx] - zv) * sv;
      }
    }
    return out;
  }
  const blocks = Math.ceil(axisDim / blockSize);
  if (scale.dims.length !== rank) return null;
  for (let d = 0; d < rank; d++) if (scale.dims[d] !== (d === axis ? blocks : dims[d])) return null;
  let idx = 0;
  for (let o = 0; o < outer; o++) {
    for (let j = 0; j < axisDim; j++) {
      const base = (o * blocks + Math.floor(j / blockSize)) * inner;
      if (zp) for (let i = 0; i < inner; i++, idx++) out[idx] = (q[idx] - zp[base + i]) * s[base + i];
      else for (let i = 0; i < inner; i++, idx++) out[idx] = q[idx] * s[base + i];
    }
  }
  return out;
}

function encodeFloatTensor(name: string, dataType: number, dims: number[], values: Float32Array): Uint8Array {
  if (dataType === ONNX_FLOAT16) {
    const bits = new Uint16Array(values.length);
    for (let i = 0; i < values.length; i++) bits[i] = f32ToF16Bits(values[i]);
    return encodeTensorProto(name, ONNX_FLOAT16, dims, new Uint8Array(bits.buffer, bits.byteOffset, bits.byteLength));
  }
  const copy = Float32Array.from(values);
  return encodeTensorProto(name, ONNX_FLOAT, dims, new Uint8Array(copy.buffer, copy.byteOffset, copy.byteLength));
}

// ---------------------------------------------------------------- nodes

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
      // Only the name is needed up front; the payload is parsed for the tensors we fold.
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
  const replacement = new Map<Field, Uint8Array>(); // initializer field -> folded tensor bytes (emitted in its place)
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
    const values = dequantize(x, scale, zp, node.axis, node.blockSize);
    if (!values) {
      skipped += 1;
      continue;
    }
    const xEntry = initializers.get(node.inputs[0])!;
    if (replacement.has(xEntry.field)) {
      skipped += 1;
      continue;
    } // shared quantized input: fold only once
    const folded = encodeFloatTensor(node.outputs[0], outType, x.dims, values);
    removedNodes.add(node.field);
    replacement.set(xEntry.field, folded);
    for (const name of [node.inputs[0], node.inputs[1], zpName]) {
      if (!name) continue;
      const left = (consumers.get(name) ?? 1) - 1;
      consumers.set(name, left);
      if (left <= 0 && !graphInputs.has(name) && !graphOutputs.has(name)) droppedInitializers.add(initializers.get(name)!.field);
    }
  }
  if (removedNodes.size === 0) return unchanged(skipped);

  const graph = new ProtoWriter();
  for (const f of graphFields) {
    if (f.field === 1 && removedNodes.has(f)) continue;
    if (f.field === 5) {
      const folded = replacement.get(f);
      if (folded) {
        graph.bytesField(5, folded);
        if (droppedInitializers.has(f)) continue;
      } else if (droppedInitializers.has(f)) continue;
    }
    graph.raw(model.subarray(f.start, f.end));
  }
  const graphBytes = graph.finish();
  const out = new ProtoWriter();
  for (const f of modelFields) {
    if (f === graphField) out.bytesField(7, graphBytes);
    else out.raw(model.subarray(f.start, f.end));
  }
  const bytes = out.finish();
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
