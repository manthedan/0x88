// GPU-resident IO for an ORT WebGPU session (docs/ort_invoke_overhead_2026-11.md).
//
// The shipped path hands ORT a CPU input tensor and asks for CPU outputs. ORT
// then uploads the input through a fresh staging buffer and runs one
// MemcpyToHost per output, each its own submit + mapAsync + wait, one after
// the other. This binding instead:
//
//   - writes the input with queue.writeBuffer into one persistent GPU buffer;
//   - binds persistent, preallocated GPU output buffers (ORT IO binding);
//   - after session.run copies every output into ONE staging buffer and maps it
//     once (one submit, one mapAsync per invoke instead of one per output).
//
// The output bytes are the bytes ORT's kernels wrote: no math changes. Because
// the buffers never change, the session may also be created with
// `enableGraphCapture`, which lets ORT replay its recorded dispatches instead of
// re-running per-kernel CPU work.
//
// Captured sessions must use `outputs: 'session'` (ORT-owned output buffers),
// not preallocated fetches: see OrtGpuIoOutputMode for the native leak the
// fetch form causes under capture.
//
// Invokes are serialized: the buffers are shared, so a second run waits for the
// first one's readback.
import type * as ort from 'onnxruntime-web/webgpu';

export type OrtGpuIoTensorType = 'float16' | 'float32';

export type OrtGpuIoTensorSpec = {
  name: string;
  type: OrtGpuIoTensorType;
  dims: readonly number[];
};

export type OrtGpuIoOutputs = Record<string, Uint16Array | Float32Array>;

// Minimal WebGPU shapes (the repo has no @webgpu/types).
type GPUBuffer = {
  mapAsync(mode: number, offset?: number, size?: number): Promise<void>;
  getMappedRange(offset?: number, size?: number): ArrayBuffer;
  unmap(): void;
  destroy(): void;
};
type GPUCommandEncoder = {
  copyBufferToBuffer(source: GPUBuffer, sourceOffset: number, destination: GPUBuffer, destinationOffset: number, size: number): void;
  finish(): unknown;
};
export type OrtGpuIoDevice = {
  createBuffer(descriptor: { size: number; usage: number; label?: string }): GPUBuffer;
  createCommandEncoder(): GPUCommandEncoder;
  queue: { writeBuffer(buffer: GPUBuffer, offset: number, data: ArrayBufferLike, dataOffset?: number, size?: number): void; submit(buffers: unknown[]): void };
};
type GPUDevice = OrtGpuIoDevice;

export type OrtGpuIoTensorCtor = {
  fromGpuBuffer(buffer: never, options: { dataType: OrtGpuIoTensorType; dims: readonly number[] }): ort.Tensor;
};
type TensorCtor = OrtGpuIoTensorCtor;

type SessionLike = Pick<ort.InferenceSession, 'run'>;

/**
 * Where the outputs live.
 *
 * - 'preallocated': our own GPU buffers, passed to run() as fetches. ORT binds
 *   them on every run and releases what it binds. Use this without graph capture.
 * - 'session': ORT allocates the output buffers. Under graph capture they stay
 *   bound to the session and are reused by every replay. run() returns them as
 *   gpu-buffer tensors, and we dispose each one after the copy, which releases
 *   the per-run native OrtValue that ORT-web hands back.
 *
 * Captured sessions need 'session'. ORT-web 1.27 binds IO only on a captured
 * session's first run. On every later run, OrtRunWithBinding returns three new
 * native OrtValue copies of the bound outputs (GetBoundOutputValues). Because
 * those runs no longer list the fetches as pre-allocated, wasm-core-impl keeps
 * the copies alive for the returned tensors' dispose(). InferenceSession.run then
 * returns the caller's fetch tensors instead (`outputArray[i] ?? decode(...)`),
 * so nothing can ever dispose the copies. That leaks a few dozen bytes of wasm
 * heap per output per run, which the 'session' form avoids. Measured in
 * docs/ort_invoke_overhead_2026-11.md section 4.
 */
export type OrtGpuIoOutputMode = 'preallocated' | 'session';

type GpuBufferTensor = { location?: string; gpuBuffer?: unknown; type?: string; dims?: readonly number[]; dispose(): void };

const BYTES_PER_ELEMENT: Record<OrtGpuIoTensorType, number> = { float16: 2, float32: 4 };
// ORT's own gpu-buffer downloader rounds sizes up to 16 bytes; copyBufferToBuffer
// needs multiples of 4. Every buffer and staging slot uses 16.
const ALIGN = 16;

// WebGPU usage/map flags (spec constants), so this module imports cleanly where
// GPUBufferUsage is not a global (Node tests).
const USAGE_MAP_READ = 0x1;
const USAGE_COPY_SRC = 0x4;
const USAGE_COPY_DST = 0x8;
const USAGE_STORAGE = 0x80;
const MAP_MODE_READ = 0x1;

function elementCount(dims: readonly number[]): number {
  return dims.reduce((a, b) => a * b, 1);
}

function alignUp(n: number): number {
  return Math.ceil(n / ALIGN) * ALIGN;
}

/** Byte layout of the outputs inside the single staging buffer. Exported for tests. */
export function ortGpuIoLayout(outputs: readonly OrtGpuIoTensorSpec[]): {
  slots: Array<OrtGpuIoTensorSpec & { elements: number; bytes: number; paddedBytes: number; offset: number }>;
  stagingBytes: number;
} {
  let offset = 0;
  const slots = outputs.map((spec) => {
    const elements = elementCount(spec.dims);
    const bytes = elements * BYTES_PER_ELEMENT[spec.type];
    const paddedBytes = alignUp(bytes);
    const slot = { ...spec, elements, bytes, paddedBytes, offset };
    offset += paddedBytes;
    return slot;
  });
  return { slots, stagingBytes: offset };
}

/**
 * Session options that really turn on ORT WebGPU graph capture.
 *
 * ORT-web 1.27 forwards `enableGraphCapture` to the native WebGPU EP as the EP
 * option `enableGraphCapture=1` only when the WebGPU provider is given as an
 * object (`{ name: 'webgpu' }`, lib/wasm/session-options.ts); with the string
 * form `'webgpu'` the EP never sees it. The JS side still treats the session as
 * captured (IO bound once, gpu-buffer IO required) while ORT re-records every
 * kernel each run, so the string form gets the constraints without the benefit.
 */
export function ortGraphCaptureSessionOptions(): { enableGraphCapture: true; executionProviders: Array<{ name: 'webgpu' }> } {
  return { enableGraphCapture: true, executionProviders: [{ name: 'webgpu' }] };
}

/**
 * Keep ORT's import of a caller-owned GPUBuffer alive across runs.
 *
 * ORT-web 1.27's native WebGPU EP imports a gpu-buffer tensor's GPUBuffer into
 * its WebGPU handle table at the start of every run (webgpuRegisterBuffer: it
 * stores `[handle, refcount]` on the buffer under a private symbol) and releases
 * the handle when the run ends (webgpuUnregisterBuffer -> wgpuBufferRelease at
 * refcount 0). Without graph capture ORT rebinds IO every run, so a fresh handle
 * per run is fine, only wasted work. With graph capture ORT binds IO once, on the
 * first run, and keeps using that handle: after the release it points at a freed
 * handle (it works only while the allocator happens to hand back the same one).
 * ORT's own gpu-buffer outputs avoid this by registering with a NaN refcount,
 * which never reaches zero. This does the same for our buffers: a Proxy
 * prototype sees the `[handle, 1]` record being stored and makes the count NaN,
 * so one handle lives as long as the buffer. The buffer object, its methods and
 * its brand are unchanged. The handle-table entry is not freed when the buffer
 * is destroyed (a few bytes per binding).
 *
 * Returns a probe: true once ORT has registered (and we pinned) the buffer.
 */
export function pinOrtGpuBufferImport(buffer: object): () => boolean {
  let pinned = false;
  const proto = Object.getPrototypeOf(buffer) as object;
  Object.setPrototypeOf(
    buffer,
    new Proxy(proto, {
      set(_target, key, value, receiver) {
        if (receiver === buffer && typeof key === 'symbol' && Array.isArray(value) && value.length === 2 && value[1] === 1) {
          value[1] = Number.NaN;
          pinned = true;
        }
        Object.defineProperty(receiver, key, { value, writable: true, configurable: true, enumerable: true });
        return true;
      },
    }),
  );
  return () => pinned;
}

export class OrtGpuIoBinding {
  readonly input: OrtGpuIoTensorSpec;
  readonly outputs: readonly OrtGpuIoTensorSpec[];
  private readonly session: SessionLike;
  private readonly device: GPUDevice;
  private readonly inputBuffer: GPUBuffer;
  private readonly inputBytes: number;
  /** Our output buffers ('preallocated' only; empty for 'session'). */
  private readonly outputBuffers: GPUBuffer[];
  private readonly staging: GPUBuffer;
  private readonly layout: ReturnType<typeof ortGpuIoLayout>;
  private readonly feeds: Record<string, ort.Tensor>;
  private readonly fetches: Record<string, ort.Tensor> | null;
  readonly outputMode: OrtGpuIoOutputMode;
  private readonly pins: Array<() => boolean>;
  private pinsChecked = false;
  private tail: Promise<unknown> = Promise.resolve();
  private disposed = false;

  /** `pin: false` is for the lab bisection only (see pinOrtGpuBufferImport). */
  constructor(
    session: SessionLike,
    device: GPUDevice,
    Tensor: TensorCtor,
    input: OrtGpuIoTensorSpec,
    outputs: readonly OrtGpuIoTensorSpec[],
    { pin = true, outputMode = 'preallocated' }: { pin?: boolean; outputMode?: OrtGpuIoOutputMode } = {},
  ) {
    if (!outputs.length) throw new Error('OrtGpuIoBinding: no outputs');
    this.session = session;
    this.device = device;
    this.input = input;
    this.outputs = outputs;
    this.outputMode = outputMode;
    this.inputBytes = elementCount(input.dims) * BYTES_PER_ELEMENT[input.type];
    if (this.inputBytes % 4 !== 0) throw new Error(`OrtGpuIoBinding: input ${input.name} is ${this.inputBytes} bytes; writeBuffer needs a multiple of 4`);
    this.layout = ortGpuIoLayout(outputs);
    const created: GPUBuffer[] = [];
    const wrapped: ort.Tensor[] = [];
    const wrap = (buffer: GPUBuffer, spec: OrtGpuIoTensorSpec): ort.Tensor => {
      const tensor = Tensor.fromGpuBuffer(buffer as never, { dataType: spec.type, dims: spec.dims });
      wrapped.push(tensor);
      return tensor;
    };
    try {
      this.inputBuffer = device.createBuffer({ size: alignUp(this.inputBytes), usage: USAGE_STORAGE | USAGE_COPY_DST | USAGE_COPY_SRC });
      created.push(this.inputBuffer);
      this.outputBuffers = (outputMode === 'session' ? [] : this.layout.slots).map((slot) => {
        const buffer = device.createBuffer({ size: slot.paddedBytes, usage: USAGE_STORAGE | USAGE_COPY_SRC | USAGE_COPY_DST });
        created.push(buffer);
        return buffer;
      });
      this.staging = device.createBuffer({ size: this.layout.stagingBytes, usage: USAGE_MAP_READ | USAGE_COPY_DST });
      created.push(this.staging);
      this.pins = pin ? [this.inputBuffer, ...this.outputBuffers].map((buffer) => pinOrtGpuBufferImport(buffer)) : [];
      if (!pin) this.pinsChecked = true;
      this.feeds = { [input.name]: wrap(this.inputBuffer, input) };
      this.fetches = outputMode === 'session' ? null : Object.fromEntries(this.layout.slots.map((slot, i) => [slot.name, wrap(this.outputBuffers[i], slot)]));
    } catch (error) {
      // Keep the construction error while rolling back every caller-owned resource.
      for (const tensor of wrapped) {
        try { tensor.dispose(); } catch {}
      }
      for (const buffer of created) {
        try { buffer.destroy(); } catch {}
      }
      throw error;
    }
  }

  /** ORT-owned output buffers of one run, after checking they match the specs. */
  private sessionOutputBuffers(results: Record<string, GpuBufferTensor>): GPUBuffer[] {
    return this.layout.slots.map((slot) => {
      const tensor = results[slot.name];
      if (!tensor || tensor.location !== 'gpu-buffer' || !tensor.gpuBuffer)
        throw new Error(`OrtGpuIoBinding: output ${slot.name} is not a gpu-buffer tensor (create the session with preferredOutputLocation 'gpu-buffer')`);
      if (tensor.type !== slot.type || elementCount(tensor.dims ?? []) !== slot.elements)
        throw new Error(`OrtGpuIoBinding: output ${slot.name} is ${tensor.type} [${tensor.dims?.join(',')}], expected ${slot.type} [${slot.dims.join(',')}]`);
      return tensor.gpuBuffer as GPUBuffer;
    });
  }

  /** One invoke: upload `data`, run, read every output back with one map. Serialized per binding. */
  run(data: ArrayBufferView & { BYTES_PER_ELEMENT: number; byteLength: number }): Promise<OrtGpuIoOutputs> {
    const next = this.tail.then(() => this.runNow(data));
    this.tail = next.catch(() => undefined);
    return next;
  }

  private async runNow(data: ArrayBufferView & { BYTES_PER_ELEMENT: number; byteLength: number }): Promise<OrtGpuIoOutputs> {
    if (this.disposed) throw new Error('OrtGpuIoBinding has been disposed');
    if (data.byteLength !== this.inputBytes) throw new Error(`OrtGpuIoBinding: input is ${data.byteLength} bytes, expected ${this.inputBytes}`);
    // float16: the raw f16 bits (Uint16Array, or a Float16Array view of them).
    const bytesPerElement = BYTES_PER_ELEMENT[this.input.type];
    if (data.BYTES_PER_ELEMENT !== bytesPerElement || (this.input.type === 'float32' && !(data instanceof Float32Array)))
      throw new Error(
        `OrtGpuIoBinding: input must be ${this.input.type === 'float16' ? 'f16 bits (2-byte elements)' : 'a Float32Array'} for ${this.input.type}`,
      );
    const { queue } = this.device;
    queue.writeBuffer(this.inputBuffer, 0, data.buffer, data.byteOffset, data.byteLength);
    const results = (this.fetches ? await this.session.run(this.feeds, this.fetches) : await this.session.run(this.feeds)) as unknown as Record<
      string,
      GpuBufferTensor
    >;
    try {
      if (!this.pinsChecked) {
        // If ORT ever stops registering buffers the way pinOrtGpuBufferImport expects, refuse
        // rather than run on buffers ORT may have destroyed.
        if (!this.pins.every((pinned) => pinned())) throw new Error('OrtGpuIoBinding: ORT did not register the IO buffers as expected; cannot keep them alive');
        this.pinsChecked = true;
      }
      // ORT-owned outputs are 16-byte padded like ours (its own downloader copies the padded size).
      const sources = this.fetches ? this.outputBuffers : this.sessionOutputBuffers(results);
      const encoder = this.device.createCommandEncoder();
      this.layout.slots.forEach((slot, i) => {
        encoder.copyBufferToBuffer(sources[i], 0, this.staging, slot.offset, slot.paddedBytes);
      });
      queue.submit([encoder.finish()]);
    } finally {
      // 'session': release ORT's per-run OrtValue for each output. The buffers stay bound to the
      // session, and the copy above is already queued.
      if (!this.fetches) for (const tensor of Object.values(results)) tensor.dispose();
    }
    await this.staging.mapAsync(MAP_MODE_READ);
    let bytes: ArrayBuffer;
    try {
      bytes = this.staging.getMappedRange().slice(0);
    } finally {
      this.staging.unmap();
    }
    const out: OrtGpuIoOutputs = {};
    for (const slot of this.layout.slots) {
      out[slot.name] = slot.type === 'float16' ? new Uint16Array(bytes, slot.offset, slot.elements) : new Float32Array(bytes, slot.offset, slot.elements);
    }
    return out;
  }

  /** Destroys the buffers once in-flight invokes settle. Release the session first or after; ORT holds no use past run(). */
  async dispose(): Promise<void> {
    if (this.disposed) return;
    await this.tail;
    this.disposed = true;
    this.inputBuffer.destroy();
    for (const buffer of this.outputBuffers) buffer.destroy();
    this.staging.destroy();
  }
}
