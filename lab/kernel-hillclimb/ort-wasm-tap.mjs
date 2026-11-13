// Lab-only tap on ORT-web's Emscripten module (docs/ort_invoke_overhead_2026-11.md §4).
//
// ORT does not expose its wasm module. installOrtWasmTap() wraps env.wasm.wasmPaths so
// that, when ORT loads its glue, it imports a tiny blob module which imports the real
// glue, calls its factory, and hands the resolved Module to us. The wasm binary, the
// glue and every ORT code path stay the same.
//
// The tap counts native output OrtValues: a handle is "live" from the time ORT reads it
// with _OrtGetTensorData until something calls _OrtReleaseTensor on it. Handles still
// live after an invoke's outputs have been consumed (and disposed, where the caller
// disposes) are leaked. It also reports the wasm heap size, and the address of a 1 MiB
// probe allocation (malloc then free): a rising address means the allocator's top grew.

export function installOrtWasmTap(env) {
  const state = { module: null, live: new Set(), gotData: 0, released: 0 };
  const wasm = env.wasm;
  let value = wasm.wasmPaths;
  const blobs = new Map();
  globalThis.__ortWasmTap = (module) => {
    state.module = module;
    const getData = module._OrtGetTensorData;
    module._OrtGetTensorData = (tensor, ...rest) => {
      state.gotData++;
      state.live.add(tensor);
      return getData(tensor, ...rest);
    };
    const release = module._OrtReleaseTensor;
    module._OrtReleaseTensor = (tensor) => {
      state.released++;
      state.live.delete(tensor);
      return release(tensor);
    };
  };
  Object.defineProperty(wasm, 'wasmPaths', {
    configurable: true,
    enumerable: true,
    get() {
      if (!value || typeof value !== 'object' || !value.mjs) return value;
      const real = new URL(String(value.mjs), location.href).href;
      let tap = blobs.get(real);
      if (!tap) {
        const source = `import factory from ${JSON.stringify(real)};\nexport default async (config) => { const m = await factory(config); globalThis.__ortWasmTap(m); return m; };\n`;
        tap = URL.createObjectURL(new Blob([source], { type: 'text/javascript' }));
        blobs.set(real, tap);
      }
      return { ...value, mjs: tap };
    },
    set(next) {
      value = next;
    },
  });
  return {
    get attached() {
      return state.module !== null;
    },
    snapshot() {
      const m = state.module;
      if (!m) throw new Error('ort-wasm-tap: ORT module not captured (glue not loaded through env.wasm.wasmPaths?)');
      const probe = m._malloc(1 << 20);
      m._free(probe);
      return { liveOutputHandles: state.live.size, gotData: state.gotData, released: state.released, heapBytes: m.HEAPU8.length, probeAddress: probe };
    },
  };
}
