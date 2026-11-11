// Runtime-agnostic WebGPU kernel profiler.
//
// Installs before any runtime creates its device (patches
// GPUAdapter.prototype.requestDevice), so it sees TVMJS devices, ORT's native
// WebGPU EP device (emdawnwebgpu calls the JS API), and our own lab kernels.
// While armed, every compute dispatch is split into its own compute pass with
// timestampWrites, so per-dispatch GPU time is attributed even for runtimes
// that put many dispatches in one pass (ORT keeps one pass open per flush).
// Disarmed, the runtime's own passes are used unchanged: timed headline runs
// must be taken disarmed; armed runs are for attribution only.
//
// Caveats recorded in every collect(): Chrome quantizes timestamps (observed
// 65.536 us on Apple) unless launched with --enable-webgpu-developer-features;
// splitting passes adds pass boundaries (measured nil end-to-end for TVMJS in
// June 2026), and any timestampWrites a runtime set itself are dropped.
import { fnv1a } from './common.mjs';

const QUANTUM_NS = 65536n;
const SLOTS_PER_SET = 2048; // 4096 timestamp queries, the WebGPU per-set maximum

function parseWorkgroupSizes(code) {
  const sizes = {};
  const re = /@compute\s*@workgroup_size\(([^)]*)\)\s*fn\s+(\w+)/g;
  for (const m of code.matchAll(re)) sizes[m[2]] = m[1].replace(/\s+/g, '');
  return sizes;
}

export function installGpuProfiler({ capacity = 16384, addTimestampFeature = true } = {}) {
  const original = GPUAdapter.prototype.requestDevice;
  const state = { devices: [], errors: [], capacity };
  GPUAdapter.prototype.requestDevice = async function patchedRequestDevice(descriptor = {}) {
    const wantsTs = addTimestampFeature && this.features.has('timestamp-query');
    const requiredFeatures = [...new Set([...(descriptor.requiredFeatures ?? []), ...(wantsTs ? ['timestamp-query'] : [])])];
    const device = await original.call(this, { ...descriptor, requiredFeatures });
    state.devices.push(instrumentDevice(device, state));
    return device;
  };
  return {
    state,
    restore() {
      GPUAdapter.prototype.requestDevice = original;
    },
    /** Instrument a device this page created before install (rare). */
    adopt(device) {
      const prof = instrumentDevice(device, state);
      state.devices.push(prof);
      return prof;
    },
    lastDevice() {
      return state.devices.at(-1) ?? null;
    },
  };
}

function instrumentDevice(device, state) {
  const prof = {
    device,
    shaders: [],
    armed: false,
    slots: [],
    dropped: 0,
    submits: 0,
    submitCpuMs: 0,
    mapAsyncCount: 0,
    dispatchesSeen: 0,
    passesSeen: 0,
    errors: state.errors,
    supported: device.features.has('timestamp-query'),
  };
  device.addEventListener?.('uncapturederror', (e) => state.errors.push(String(e.error?.message ?? e)));
  device.lost?.then((info) => {
    if (info.reason !== 'destroyed') state.errors.push(`device lost: ${info.message}`);
  });
  const moduleInfo = new WeakMap();
  const pipelineInfo = new WeakMap();
  const bindGroupInfo = new WeakMap();

  const createShaderModule = device.createShaderModule.bind(device);
  device.createShaderModule = (descriptor) => {
    const module = createShaderModule(descriptor);
    const code = String(descriptor?.code ?? '');
    const info = { hash: fnv1a(code), bytes: code.length, label: descriptor?.label ?? '', workgroupSizes: parseWorkgroupSizes(code), head: code.slice(0, 160) };
    moduleInfo.set(module, info);
    prof.shaders.push({ ...info, code });
    return module;
  };
  const describePipeline = (descriptor) => {
    const mod = moduleInfo.get(descriptor?.compute?.module) ?? null;
    const entry = descriptor?.compute?.entryPoint ?? '';
    const label = descriptor?.label ?? '';
    // TVM names kernels by entry point; ORT labels pipelines/modules by program.
    const name = (entry && entry !== 'main' ? entry : '') || label || mod?.label || `shader_${mod?.hash ?? 'unknown'}`;
    return {
      name,
      entry,
      label,
      moduleLabel: mod?.label ?? '',
      shaderHash: mod?.hash ?? '',
      workgroupSize: mod?.workgroupSizes?.[entry || 'main'] ?? Object.values(mod?.workgroupSizes ?? {})[0] ?? '',
    };
  };
  for (const method of ['createComputePipeline', 'createComputePipelineAsync']) {
    const create = device[method].bind(device);
    device[method] = (descriptor) => {
      const result = create(descriptor);
      const info = describePipeline(descriptor);
      if (method.endsWith('Async')) return result.then((p) => (pipelineInfo.set(p, info), p));
      pipelineInfo.set(result, info);
      return result;
    };
  }
  const createBindGroup = device.createBindGroup.bind(device);
  device.createBindGroup = (descriptor) => {
    const group = createBindGroup(descriptor);
    let bytes = 0;
    const sizes = [];
    for (const entry of descriptor?.entries ?? []) {
      const res = entry.resource;
      const buffer = res?.buffer ?? (res instanceof GPUBuffer ? res : null);
      if (!buffer) continue;
      const size = Number(res.size ?? buffer.size - (res.offset ?? 0));
      sizes.push(size);
      bytes += size;
    }
    bindGroupInfo.set(group, { bytes, sizes });
    return group;
  };

  // WebGPU caps a query set at 4096 queries, so slots are spread over lazily
  // created sets of SLOTS_PER_SET dispatches each.
  const querySets = [];
  const querySetFor = (slotIndex) => {
    const i = Math.floor(slotIndex / SLOTS_PER_SET);
    while (querySets.length <= i) querySets.push(device.createQuerySet({ type: 'timestamp', count: SLOTS_PER_SET * 2 }));
    return querySets[i];
  };
  const resolveBuffer = prof.supported
    ? device.createBuffer({ size: state.capacity * 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC })
    : null;
  const readBuffer = prof.supported ? device.createBuffer({ size: state.capacity * 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }) : null;
  let invokeIndex = 0;

  const createEncoder = device.createCommandEncoder.bind(device);
  device.createCommandEncoder = (...args) => {
    const encoder = createEncoder(...args);
    const begin = encoder.beginComputePass.bind(encoder);
    encoder.beginComputePass = (descriptor = {}) => {
      prof.passesSeen += 1;
      if (prof.armed && prof.supported && prof.mode === 'pass') return timedPass(begin, descriptor);
      if (!prof.armed || !prof.supported) {
        const pass = begin(descriptor);
        if (prof.armed) {
          const dispatch = pass.dispatchWorkgroups.bind(pass);
          pass.dispatchWorkgroups = (...d) => {
            prof.dispatchesSeen += 1;
            return dispatch(...d);
          };
        }
        return pass;
      }
      return splitPass(begin, descriptor);
    };
    return encoder;
  };

  // Pass mode: time each original pass as-is (no splitting), so busy/envelope
  // reflect the runtime's real pass structure with no profiler inflation.
  function timedPass(begin, descriptor) {
    const canTime = prof.slots.length < state.capacity;
    const slot = canTime ? { index: prof.slots.length, invoke: invokeIndex, name: 'pass', kind: 'pass', workgroups: null, bindBytes: 0, dispatches: 0 } : null;
    if (!slot) {
      prof.dropped += 1;
      return begin(descriptor);
    }
    prof.slots.push(slot);
    const real = begin({
      ...descriptor,
      timestampWrites: {
        querySet: querySetFor(slot.index),
        beginningOfPassWriteIndex: (slot.index % SLOTS_PER_SET) * 2,
        endOfPassWriteIndex: (slot.index % SLOTS_PER_SET) * 2 + 1,
      },
    });
    const dispatch = real.dispatchWorkgroups.bind(real);
    real.dispatchWorkgroups = (...d) => {
      slot.dispatches += 1;
      prof.dispatchesSeen += 1;
      return dispatch(...d);
    };
    return real;
  }

  // A recording proxy: each dispatch replays the current pipeline/bind groups
  // into a fresh, timestamped real pass.
  function splitPass(begin, descriptor) {
    let pipeline = null;
    const groups = new Map();
    const run = (kind, dispatchArgs) => {
      prof.dispatchesSeen += 1;
      const pInfo = pipelineInfo.get(pipeline) ?? { name: 'unknown' };
      let bindBytes = 0;
      for (const args of groups.values()) bindBytes += bindGroupInfo.get(args[1])?.bytes ?? 0;
      const canTime = prof.slots.length < state.capacity;
      const slot = canTime
        ? { index: prof.slots.length, invoke: invokeIndex, ...pInfo, kind, workgroups: kind === 'indirect' ? null : dispatchArgs.map((v) => v ?? 1), bindBytes }
        : null;
      if (slot) prof.slots.push(slot);
      else prof.dropped += 1;
      const real = begin(
        slot
          ? {
              label: descriptor.label,
              timestampWrites: {
                querySet: querySetFor(slot.index),
                beginningOfPassWriteIndex: (slot.index % SLOTS_PER_SET) * 2,
                endOfPassWriteIndex: (slot.index % SLOTS_PER_SET) * 2 + 1,
              },
            }
          : { label: descriptor.label },
      );
      if (pipeline) real.setPipeline(pipeline);
      for (const args of groups.values()) real.setBindGroup(...args);
      if (kind === 'indirect') real.dispatchWorkgroupsIndirect(...dispatchArgs);
      else real.dispatchWorkgroups(...dispatchArgs);
      real.end();
    };
    return {
      label: descriptor.label ?? '',
      setPipeline(p) {
        pipeline = p;
      },
      setBindGroup(index, group, ...rest) {
        if (group === null || group === undefined) groups.delete(index);
        else if (rest.length >= 3 && ArrayBuffer.isView(rest[0])) {
          // (data, start, length) form, e.g. emdawnwebgpu passing a HEAPU32
          // view: copy now, the heap may change before the replayed dispatch.
          const [data, start, length] = rest;
          groups.set(index, [index, group, Array.from(data.subarray(start, start + length))]);
        } else groups.set(index, [index, group, ...rest]);
      },
      dispatchWorkgroups(x, y, z) {
        run('direct', [x, y, z]);
      },
      dispatchWorkgroupsIndirect(buffer, offset) {
        run('indirect', [buffer, offset]);
      },
      setImmediates() {
        throw new Error('gpu-profiler: setImmediates not supported in split mode');
      },
      pushDebugGroup() {},
      popDebugGroup() {},
      insertDebugMarker() {},
      end() {},
    };
  }

  const queue = device.queue;
  const submit = queue.submit.bind(queue);
  queue.submit = (buffers) => {
    if (!prof.armed) return submit(buffers);
    const t0 = performance.now();
    prof.submits += 1;
    const r = submit(buffers);
    prof.submitCpuMs += performance.now() - t0;
    return r;
  };

  /** mode 'dispatch' (default): one timestamped pass per dispatch, for attribution; 'pass': time the runtime's own passes. */
  prof.arm = ({ mode = 'dispatch' } = {}) => {
    prof.mode = mode;
    prof.slots = [];
    prof.dropped = 0;
    prof.submits = 0;
    prof.submitCpuMs = 0;
    prof.dispatchesSeen = 0;
    prof.passesSeen = 0;
    invokeIndex = 0;
    prof.armed = true;
  };
  /** Mark the start of the next invoke (dispatches are tagged with it). */
  prof.nextInvoke = () => {
    invokeIndex += 1;
  };
  prof.disarm = () => {
    prof.armed = false;
  };
  prof.collect = async () => {
    prof.armed = false;
    const base = {
      supported: prof.supported,
      dispatches: prof.slots.length,
      dropped: prof.dropped,
      submits: prof.submits,
      submitCpuMs: prof.submitCpuMs,
      passesSeen: prof.passesSeen,
      errors: [...state.errors],
    };
    if (!prof.supported || !prof.slots.length) return { ...base, rows: [] };
    const n = prof.slots.length * 2;
    const encoder = createEncoder();
    for (let set = 0; set * SLOTS_PER_SET * 2 < n; set++) {
      const count = Math.min(SLOTS_PER_SET * 2, n - set * SLOTS_PER_SET * 2);
      encoder.resolveQuerySet(querySets[set], 0, count, resolveBuffer, set * SLOTS_PER_SET * 16);
    }
    encoder.copyBufferToBuffer(resolveBuffer, 0, readBuffer, 0, n * 8);
    submit([encoder.finish()]);
    await readBuffer.mapAsync(GPUMapMode.READ, 0, n * 8);
    const ticks = new BigUint64Array(readBuffer.getMappedRange(0, n * 8).slice(0));
    readBuffer.unmap();
    let quantized = true;
    let origin = null;
    const rows = prof.slots.map((slot) => {
      const b = ticks[slot.index * 2];
      const e = ticks[slot.index * 2 + 1];
      if (origin === null || b < origin) origin = b;
      const ns = e > b ? e - b : 0n;
      if (ns % QUANTUM_NS !== 0n || b % QUANTUM_NS !== 0n) quantized = false;
      return { ...slot, beginNs: b, endNs: e, ns: Number(ns) };
    });
    for (const row of rows) {
      row.beginNs = Number(row.beginNs - origin);
      row.endNs = Number(row.endNs - origin);
    }
    return { ...base, timestampsQuantized: quantized, quantumNs: quantized ? Number(QUANTUM_NS) : null, rows };
  };
  return prof;
}

/**
 * Aggregate collected dispatch rows: per-kernel totals and per-invoke busy /
 * envelope / gap figures. Rows must carry `invoke` tags (>=1 for profiled
 * invokes; 0 = before first nextInvoke()).
 */
export function aggregateProfile(collected, { invokes }) {
  const rows = collected.rows.filter((r) => r.invoke >= 1);
  const byInvoke = new Map();
  for (const r of rows) {
    const list = byInvoke.get(r.invoke) ?? [];
    list.push(r);
    byInvoke.set(r.invoke, list);
  }
  const perInvoke = [...byInvoke.entries()].map(([invoke, list]) => {
    const busyNs = list.reduce((s, r) => s + r.ns, 0);
    const begin = Math.min(...list.map((r) => r.beginNs));
    const end = Math.max(...list.map((r) => r.endNs));
    return {
      invoke,
      dispatches: list.length,
      busyMs: busyNs / 1e6,
      envelopeMs: (end - begin) / 1e6,
      gapMs: (end - begin - busyNs) / 1e6,
      zeroDispatches: list.filter((r) => r.ns === 0).length,
    };
  });
  const byKernel = new Map();
  for (const r of rows) {
    const key = r.name;
    const k = byKernel.get(key) ?? {
      name: key,
      shaderHash: r.shaderHash,
      workgroupSize: r.workgroupSize,
      dispatches: 0,
      totalNs: 0,
      zero: 0,
      workgroups: new Set(),
      bindBytes: new Set(),
      firstSeq: r.index,
    };
    k.dispatches += 1;
    k.totalNs += r.ns;
    if (r.ns === 0) k.zero += 1;
    if (r.workgroups) k.workgroups.add(r.workgroups.join('x'));
    k.bindBytes.add(r.bindBytes);
    byKernel.set(key, k);
  }
  const total = rows.reduce((s, r) => s + r.ns, 0);
  const kernels = [...byKernel.values()]
    .map((k) => ({
      name: k.name,
      shaderHash: k.shaderHash,
      workgroupSize: k.workgroupSize,
      dispatchesPerInvoke: k.dispatches / invokes,
      msPerInvoke: k.totalNs / 1e6 / invokes,
      meanUsPerDispatch: k.totalNs / 1e3 / k.dispatches,
      share: total ? k.totalNs / total : 0,
      zeroDispatches: k.zero,
      workgroups: [...k.workgroups].slice(0, 8),
      bindBytes: [...k.bindBytes].slice(0, 8),
      firstSeq: k.firstSeq,
    }))
    .sort((a, b) => b.msPerInvoke - a.msPerInvoke);
  // Dispatches past the slot capacity run untimed: per-kernel totals would be
  // silently low, so the pages fail the run when dropped > 0.
  const dropped = collected.dropped ?? 0;
  return { invokes, kernelMsPerInvoke: total / 1e6 / invokes, perInvoke, kernels, dropped, complete: dropped === 0 };
}

/**
 * Time `invokes` calls of `run` with the runtime's own passes timestamped (no
 * splitting). Returns median per-invoke GPU busy / envelope / gap and wall.
 */
export async function passModeProfile(prof, run, invokes) {
  const med = (xs) => {
    const s = [...xs].sort((a, b) => a - b);
    const m = s.length >> 1;
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
  };
  prof.arm({ mode: 'pass' });
  const wall = [];
  for (let i = 0; i < invokes; i++) {
    prof.nextInvoke();
    const t = performance.now();
    await run();
    wall.push(performance.now() - t);
  }
  const submits = prof.submits;
  const collected = await prof.collect();
  const agg = aggregateProfile(collected, { invokes });
  return {
    invokes,
    wallMs: med(wall),
    busyMs: med(agg.perInvoke.map((p) => p.busyMs)),
    envelopeMs: med(agg.perInvoke.map((p) => p.envelopeMs)),
    gapMs: med(agg.perInvoke.map((p) => p.gapMs)),
    passesPerInvoke: med(agg.perInvoke.map((p) => p.dispatches)),
    submitsPerInvoke: submits / invokes,
    timestampsQuantized: collected.timestampsQuantized,
    dropped: collected.dropped ?? 0,
  };
}
