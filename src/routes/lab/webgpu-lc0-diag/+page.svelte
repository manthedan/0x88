<script lang="ts">
import { onMount } from 'svelte';

type JsonRecord = Record<string, unknown>;
type RunState = 'idle' | 'running' | 'done' | 'error';
type AnyGpuAdapter = {
  features: Set<string>;
  limits: JsonRecord;
  requestDevice: () => Promise<AnyGpuDevice>;
} & JsonRecord;
type AnyGpuDevice = JsonRecord & {
  createBuffer: (descriptor: JsonRecord) => AnyGpuBuffer;
  createShaderModule: (descriptor: JsonRecord) => unknown;
  createComputePipeline: (descriptor: JsonRecord) => AnyGpuPipeline;
  createBindGroup: (descriptor: JsonRecord) => unknown;
  createCommandEncoder: () => AnyGpuCommandEncoder;
  queue: { writeBuffer: (...args: unknown[]) => void; submit: (commands: unknown[]) => void };
  destroy: () => void;
};
type AnyGpuBuffer = JsonRecord & { mapAsync: (mode: number) => Promise<void>; getMappedRange: () => ArrayBuffer; unmap: () => void };
type AnyGpuPipeline = JsonRecord & { getBindGroupLayout: (index: number) => unknown };
type AnyGpuCommandEncoder = JsonRecord & { beginComputePass: () => AnyGpuComputePass; copyBufferToBuffer: (...args: unknown[]) => void; finish: () => unknown };
type AnyGpuComputePass = JsonRecord & {
  setPipeline: (pipeline: unknown) => void;
  setBindGroup: (index: number, bindGroup: unknown) => void;
  dispatchWorkgroups: (count: number) => void;
  end: () => void;
};

const GPU_BUFFER_USAGE = { MAP_READ: 0x0001, COPY_SRC: 0x0004, COPY_DST: 0x0008, STORAGE: 0x0080 } as const;
const GPU_MAP_MODE = { READ: 0x0001 } as const;

const title = 'LC0 WebGPU diagnostics';
const defaultAssetBase = 'https://assets.0x88.app';
const defaultModel = '/models/lc0/t1-256x10-distilled-swa-2432500.batch1.f16.qdq8.onnx';
const modelOverrideEnabled = import.meta.env.VITE_BROWSER_CHESS_DEPLOY_PROFILE !== 'v0';

let runState: RunState = 'idle';
let status = 'Ready. Click Run diagnostics, keep this tab foregrounded, then copy the JSON result.';
let warmup = 1;
let iterations = 3;
let assetBase = defaultAssetBase;
let model = defaultModel;
let iframeSrc = '';
let reportText = '';
let copyStatus = '';
let chromeGpuNotes = '';
let startedAt = '';
let finishedAt = '';
let benchFrame: HTMLIFrameElement;
let pollTimer: ReturnType<typeof setInterval> | undefined;
let runSeq = 0;
let mounted = false;

onMount(() => {
  mounted = true;
  return () => {
    mounted = false;
    if (pollTimer) clearInterval(pollTimer);
  };
});

function nowIso(): string {
  return new Date().toISOString();
}

function roundMs(value: number): number {
  return Math.round(value * 1000) / 1000;
}

function withTimeout<T>(promise: Promise<T>, timeoutMs: number, label: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`${label} timed out after ${timeoutMs}ms`)), timeoutMs);
    promise.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (error) => {
        clearTimeout(timer);
        reject(error);
      },
    );
  });
}

function jsonClone(value: unknown): unknown {
  try {
    return JSON.parse(JSON.stringify(value));
  } catch {
    return String(value);
  }
}

function adapterInfo(adapter: AnyGpuAdapter | null): JsonRecord | null {
  if (!adapter) return null;
  const anyAdapter = adapter as unknown as JsonRecord;
  const info = (anyAdapter.info ?? null) as JsonRecord | null;
  return {
    info: info
      ? {
          vendor: info.vendor ?? null,
          architecture: info.architecture ?? null,
          device: info.device ?? null,
          description: info.description ?? null,
          raw: jsonClone(info),
        }
      : null,
    features: Array.from(adapter.features).sort(),
    limits: {
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupSizeX: adapter.limits.maxComputeWorkgroupSizeX,
      maxComputeWorkgroupsPerDimension: adapter.limits.maxComputeWorkgroupsPerDimension,
    },
    isFallbackAdapter: anyAdapter.isFallbackAdapter ?? null,
  };
}

async function probeMainWebGpu(): Promise<JsonRecord> {
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter: (opts?: JsonRecord) => Promise<AnyGpuAdapter | null> } }).gpu;
  if (!gpu) return { hasNavigatorGpu: false, adapter: null };
  const started = performance.now();
  try {
    const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
    return {
      hasNavigatorGpu: true,
      requestAdapterMs: roundMs(performance.now() - started),
      adapter: !!adapter,
      shaderF16: !!adapter?.features.has('shader-f16'),
      adapterInfo: adapterInfo(adapter),
    };
  } catch (error) {
    return { hasNavigatorGpu: true, error: String(error) };
  }
}

async function probeWorkerWebGpu(): Promise<JsonRecord> {
  if (!('Worker' in window)) return { workerAvailable: false };
  const source = `
      self.onmessage = async () => {
        const started = performance.now();
        try {
          const gpu = self.navigator && self.navigator.gpu;
          if (!gpu) { self.postMessage({ hasNavigatorGpu: false }); return; }
          const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
          self.postMessage({
            hasNavigatorGpu: true,
            requestAdapterMs: Math.round((performance.now() - started) * 1000) / 1000,
            adapter: !!adapter,
            shaderF16: !!adapter && adapter.features.has('shader-f16'),
            features: adapter ? Array.from(adapter.features).sort() : [],
            limits: adapter ? {
              maxBufferSize: adapter.limits.maxBufferSize,
              maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
              maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
              maxComputeWorkgroupSizeX: adapter.limits.maxComputeWorkgroupSizeX,
            } : null,
            info: adapter && adapter.info ? {
              vendor: adapter.info.vendor || null,
              architecture: adapter.info.architecture || null,
              device: adapter.info.device || null,
              description: adapter.info.description || null,
            } : null,
          });
        } catch (error) {
          self.postMessage({ error: String(error), stack: error && error.stack ? String(error.stack) : null });
        }
      };
    `;
  const url = URL.createObjectURL(new Blob([source], { type: 'text/javascript' }));
  try {
    const worker = new Worker(url);
    const result = await new Promise<JsonRecord>((resolve) => {
      const timeout = setTimeout(() => resolve({ timeout: true }), 15000);
      worker.onmessage = (event) => {
        clearTimeout(timeout);
        resolve(event.data as JsonRecord);
      };
      worker.onerror = (event) => {
        clearTimeout(timeout);
        resolve({ error: event.message });
      };
      worker.postMessage({});
    });
    worker.terminate();
    return { workerAvailable: true, ...result };
  } finally {
    URL.revokeObjectURL(url);
  }
}

async function probeWasmSimd(): Promise<JsonRecord> {
  // Minimal SIMD module from v8 feature-detect examples. It validates support without executing app code.
  const simdModule = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11]);
  return {
    validate: typeof WebAssembly !== 'undefined' && WebAssembly.validate(simdModule),
  };
}

async function runWgslSmoke(): Promise<JsonRecord> {
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter: (opts?: JsonRecord) => Promise<AnyGpuAdapter | null> } }).gpu;
  if (!gpu) return { skipped: 'navigator.gpu unavailable' };
  const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) return { skipped: 'requestAdapter returned null' };
  const device = await withTimeout(adapter.requestDevice(), 20000, 'WebGPU requestDevice');
  try {
    const count = 262144;
    const bytes = count * 4;
    const input = device.createBuffer({ size: bytes, usage: GPU_BUFFER_USAGE.STORAGE | GPU_BUFFER_USAGE.COPY_DST });
    const output = device.createBuffer({ size: bytes, usage: GPU_BUFFER_USAGE.STORAGE | GPU_BUFFER_USAGE.COPY_SRC });
    const readback = device.createBuffer({ size: bytes, usage: GPU_BUFFER_USAGE.MAP_READ | GPU_BUFFER_USAGE.COPY_DST });
    const data = new Float32Array(count);
    for (let i = 0; i < count; i += 1) data[i] = (i % 97) / 97;
    device.queue.writeBuffer(input, 0, data);
    const shader = device.createShaderModule({
      code: `
        @group(0) @binding(0) var<storage, read> input: array<f32>;
        @group(0) @binding(1) var<storage, read_write> output: array<f32>;
        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
          let i = id.x;
          if (i >= ${count}u) { return; }
          var x = input[i];
          for (var k = 0u; k < 32u; k = k + 1u) {
            x = x * 1.000001 + 0.000001;
          }
          output[i] = x;
        }
      `,
    });
    const pipelineStarted = performance.now();
    const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module: shader, entryPoint: 'main' } });
    const pipelineCreateMs = roundMs(performance.now() - pipelineStarted);
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: input } },
        { binding: 1, resource: { buffer: output } },
      ],
    });
    const dispatchStarted = performance.now();
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(count / 256));
    pass.end();
    encoder.copyBufferToBuffer(output, 0, readback, 0, bytes);
    device.queue.submit([encoder.finish()]);
    await withTimeout(readback.mapAsync(GPU_MAP_MODE.READ), 30000, 'WebGPU readback');
    const dispatchAndReadbackMs = roundMs(performance.now() - dispatchStarted);
    const view = new Float32Array(readback.getMappedRange());
    const checksum = roundMs(view[0] + view[17] + view[count - 1]);
    readback.unmap();
    return {
      count,
      bytes,
      pipelineCreateMs,
      dispatchAndReadbackMs,
      checksum,
      shaderF16: adapter.features.has('shader-f16'),
    };
  } finally {
    device.destroy();
  }
}

function environmentReport(): JsonRecord {
  const nav = navigator as Navigator & { deviceMemory?: number; userAgentData?: unknown };
  return {
    pageUrl: location.href,
    userAgent: navigator.userAgent,
    platform: navigator.platform,
    language: navigator.language,
    hardwareConcurrency: navigator.hardwareConcurrency,
    deviceMemory: nav.deviceMemory ?? null,
    userAgentData: jsonClone(nav.userAgentData ?? null),
    crossOriginIsolated,
    secureContext: window.isSecureContext,
    sharedArrayBuffer: typeof SharedArrayBuffer !== 'undefined',
    screen: {
      width: screen.width,
      height: screen.height,
      colorDepth: screen.colorDepth,
      devicePixelRatio,
    },
  };
}

function buildBenchUrl(): string {
  const params = new URLSearchParams({
    bench: '1',
    ep: 'webgpu',
    benchWarmup: String(warmup),
    benchIters: String(iterations),
    cache: '0',
    assetBase,
    diagRun: String(runSeq),
  });
  if (modelOverrideEnabled && model.trim()) params.set('model', model.trim());
  return `/single-engine/?${params.toString()}`;
}

function readIframeBench(): { done: boolean; failed: boolean; frame: JsonRecord; bench: unknown } {
  const doc = benchFrame?.contentDocument;
  if (!doc) return { done: false, failed: false, frame: { error: 'iframe document unavailable' }, bench: null };
  const text = (id: string) => doc.querySelector(`#${id}`)?.textContent?.trim() ?? '';
  const benchText = text('benchResult');
  let bench: unknown = benchText;
  if (benchText.startsWith('{')) {
    try {
      bench = JSON.parse(benchText);
    } catch {}
  }
  const message = text('message');
  const objectStatus = typeof bench === 'object' && bench !== null ? String((bench as JsonRecord).status ?? '') : '';
  const failed = objectStatus === 'BENCH_FAILED' || /^BENCH_FAILED\b/.test(benchText) || /^(?:Model load failed|Page failed to initialize):/i.test(message);
  const done = objectStatus === 'BENCH_DONE' || failed;
  return {
    done,
    failed,
    bench,
    frame: {
      status: text('status'),
      backend: text('backend'),
      gpuStatus: text('gpuStatus'),
      message,
      modelPath: text('modelPath'),
      benchText,
    },
  };
}

async function runDiagnostics(): Promise<void> {
  if (pollTimer) clearInterval(pollTimer);
  runState = 'running';
  runSeq += 1;
  copyStatus = '';
  reportText = '';
  startedAt = nowIso();
  finishedAt = '';
  status = 'Running browser/WebGPU probes…';

  const env = environmentReport();
  const [mainWebGpu, workerWebGpu, wasmSimd, wgslSmoke] = await Promise.all([
    withTimeout(probeMainWebGpu(), 20000, 'main-thread WebGPU probe').catch((error) => ({ error: String(error) })),
    withTimeout(probeWorkerWebGpu(), 20000, 'worker WebGPU probe').catch((error) => ({ error: String(error) })),
    probeWasmSimd().catch((error) => ({ error: String(error) })),
    withTimeout(runWgslSmoke(), 60000, 'WGSL smoke').catch((error) => ({ error: String(error) })),
  ]);

  if (!mounted) return;

  const partial = {
    schema: 'lc0_browser.public_webgpu_lc0_diagnostics.v1',
    startedAt,
    environment: env,
    webgpu: { main: mainWebGpu, worker: workerWebGpu, wgslSmoke },
    wasm: { simd: wasmSimd },
    config: {
      warmup,
      iterations,
      assetBase,
      model: modelOverrideEnabled ? model : defaultModel,
      modelOverrideEnabled,
    },
    chromeGpuNotes,
    lc0Bench: { status: 'pending' },
  };
  reportText = JSON.stringify(partial, null, 2);

  status = 'Loading LC0 benchmark iframe. This may download the model and take a minute…';
  iframeSrc = buildBenchUrl();

  const deadline = Date.now() + 10 * 60 * 1000;
  pollTimer = setInterval(() => {
    try {
      const observed = readIframeBench();
      const elapsed = Math.round((Date.now() - Date.parse(startedAt)) / 1000);
      const frameMessage = String(observed.frame.message ?? '').slice(0, 160);
      status = `LC0 benchmark running (${elapsed}s): ${frameMessage || observed.frame.status || 'waiting for iframe'}`;
      const timedOut = Date.now() > deadline;
      if (observed.done || timedOut) {
        if (pollTimer) clearInterval(pollTimer);
        finishedAt = nowIso();
        const report = {
          ...partial,
          finishedAt,
          chromeGpuNotes,
          lc0Bench: {
            status: observed.failed ? 'failed' : observed.done ? 'complete' : 'timeout',
            iframeUrl: iframeSrc,
            frame: observed.frame,
            result: observed.bench,
          },
          classificationHint: observed.failed
            ? { warning: 'LC0 benchmark failed; inspect lc0Bench.frame and result.' }
            : classifyResult(observed.bench, wgslSmoke),
        };
        reportText = JSON.stringify(report, null, 2);
        runState = observed.done && !observed.failed ? 'done' : 'error';
        status = observed.failed
          ? 'LC0 benchmark failed. Copy the JSON below for diagnosis.'
          : observed.done
            ? 'Done. Copy the JSON below and send it back.'
            : 'Timed out. Copy the JSON below anyway.';
      }
    } catch (error) {
      if (pollTimer) clearInterval(pollTimer);
      finishedAt = nowIso();
      reportText = JSON.stringify(
        { ...partial, finishedAt, chromeGpuNotes, lc0Bench: { status: 'error', error: String(error), iframeUrl: iframeSrc } },
        null,
        2,
      );
      runState = 'error';
      status = `Error reading benchmark iframe: ${String(error)}`;
    }
  }, 1000);
}

function classifyResult(bench: unknown, wgslSmoke: unknown): JsonRecord {
  const avgMs = typeof bench === 'object' && bench !== null ? Number((bench as JsonRecord).avgMs) : NaN;
  const wgslMs = typeof wgslSmoke === 'object' && wgslSmoke !== null ? Number((wgslSmoke as JsonRecord).dispatchAndReadbackMs) : NaN;
  return {
    nativeGpuLikely: Number.isFinite(avgMs) ? avgMs < 500 : null,
    lc0AvgMs: Number.isFinite(avgMs) ? avgMs : null,
    wgslDispatchAndReadbackMs: Number.isFinite(wgslMs) ? wgslMs : null,
    warning:
      Number.isFinite(avgMs) && avgMs >= 500
        ? 'LC0 eval latency is software-like for this small model; verify Chrome is using the discrete GPU via chrome://gpu or Task Manager.'
        : null,
  };
}

async function copyReport(): Promise<void> {
  try {
    // Notes are commonly pasted after the benchmark finishes. Refresh the
    // serialized report at copy time so the copied JSON reflects the current
    // textarea rather than the completion-time snapshot.
    try {
      const report = JSON.parse(reportText) as JsonRecord;
      report.chromeGpuNotes = chromeGpuNotes;
      reportText = JSON.stringify(report, null, 2);
    } catch {
      // Preserve non-JSON failure text; clipboard errors are handled below.
    }
    await navigator.clipboard.writeText(reportText);
    copyStatus = 'Copied.';
  } catch (error) {
    copyStatus = `Copy failed: ${String(error)}`;
  }
}
</script>

<svelte:head>
  <title>{title}</title>
  <meta name="robots" content="noindex,nofollow,noarchive" />
  <meta name="description" content="Unlisted LC0/WebGPU benchmark and diagnostics page." />
</svelte:head>

<main>
  <section class="panel intro">
    <p class="eyebrow">Unlisted diagnostics page</p>
    <h1>{title}</h1>
    <p>
      This page gathers browser/WebGPU capability data, runs a small WGSL compute smoke, then runs the LC0
      f16/qDQ ONNX WebGPU benchmark through the normal single-engine worker path. It is intentionally not linked
      from the public site and is marked <code>noindex</code>.
    </p>
    <p class="warning">
      Please use Chrome or Edge on Windows, close other heavy GPU apps, keep this tab visible while the test runs,
      then send back the copied JSON. If possible, also paste the relevant <code>chrome://gpu</code> Graphics Feature
      Status / Problems text in the notes box before copying.
    </p>
  </section>

  <section class="panel controls" aria-label="Benchmark controls">
    <div class="field">
      <label for="assetBase">Asset base</label>
      <input id="assetBase" bind:value={assetBase} spellcheck="false" />
    </div>
    <div class="field">
      <label for="model">Model path or URL</label>
      <input id="model" bind:value={model} spellcheck="false" disabled={!modelOverrideEnabled} />
      {#if !modelOverrideEnabled}<span class="small">Production diagnostics are pinned to the deployed default model.</span>{/if}
    </div>
    <div class="row">
      <div class="field small-field">
        <label for="warmup">Warmup evals</label>
        <input id="warmup" type="number" min="0" max="20" bind:value={warmup} />
      </div>
      <div class="field small-field">
        <label for="iterations">Timed evals</label>
        <input id="iterations" type="number" min="1" max="100" bind:value={iterations} />
      </div>
      <button class="primary" type="button" on:click={runDiagnostics} disabled={runState === 'running'}>
        {runState === 'running' ? 'Running…' : 'Run diagnostics'}
      </button>
    </div>
    <div class="field">
      <label for="chromeGpuNotes">Optional chrome://gpu or Task Manager notes</label>
      <textarea id="chromeGpuNotes" rows="4" bind:value={chromeGpuNotes} placeholder="Paste Chrome GPU status, GPU name, or Task Manager observations here before copying JSON."></textarea>
    </div>
    <p class:busy={runState === 'running'} class="status">{status}</p>
  </section>

  {#if iframeSrc}
    <section class="panel">
      <h2>Benchmark frame</h2>
      <p class="small">The embedded frame below is the existing LC0 single-engine benchmark route.</p>
      <iframe bind:this={benchFrame} src={iframeSrc} title="LC0 benchmark frame"></iframe>
    </section>
  {/if}

  <section class="panel result" aria-label="Copyable JSON result">
    <div class="result-header">
      <h2>Result JSON</h2>
      <button type="button" on:click={copyReport} disabled={!reportText}>Copy JSON</button>
    </div>
    {#if copyStatus}<p class="small">{copyStatus}</p>{/if}
    <textarea readonly rows="28" value={reportText || 'Results will appear here after you click Run diagnostics.'}></textarea>
  </section>
</main>

<style>
  main {
    max-width: 1120px;
    margin: 0 auto;
    padding: 28px 16px 48px;
    display: grid;
    gap: 18px;
  }
  .panel {
    border: 1px solid var(--rule, #d8d6cd);
    border-radius: 14px;
    background: var(--panel, #fffdf7);
    padding: 18px;
    box-shadow: 0 8px 28px rgba(0,0,0,0.04);
  }
  .intro h1 { margin: 4px 0 10px; }
  .eyebrow {
    margin: 0;
    color: var(--muted, #6f6a5f);
    font-size: 12px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 700;
  }
  .warning {
    border-left: 4px solid var(--warn, #b7791f);
    background: var(--warn-soft, #fff7e6);
    padding: 10px 12px;
    border-radius: 8px;
  }
  .controls { display: grid; gap: 12px; }
  .row { display: flex; flex-wrap: wrap; gap: 12px; align-items: end; }
  .field { display: grid; gap: 5px; flex: 1 1 180px; }
  .small-field { max-width: 150px; }
  label { color: var(--muted, #6f6a5f); font-size: 13px; font-weight: 650; }
  input, textarea {
    font: inherit;
    font-family: var(--mono, ui-monospace, SFMono-Regular, Menlo, Consolas, monospace);
    border: 1px solid var(--rule, #d8d6cd);
    border-radius: 8px;
    padding: 9px 10px;
    background: var(--input-bg, white);
    color: var(--ink, #1f1d19);
  }
  textarea { width: 100%; box-sizing: border-box; resize: vertical; }
  button {
    border: 1px solid var(--rule, #d8d6cd);
    border-radius: 999px;
    padding: 10px 14px;
    background: var(--panel, white);
    color: var(--ink, #1f1d19);
    font-weight: 700;
    cursor: pointer;
  }
  button.primary { background: #1f1d19; color: white; border-color: #1f1d19; }
  button:disabled { opacity: 0.55; cursor: not-allowed; }
  .status {
    margin: 0;
    padding: 10px 12px;
    border-radius: 8px;
    background: var(--panel-inset, #f4f1e8);
    font-family: var(--mono, ui-monospace, SFMono-Regular, Menlo, Consolas, monospace);
    font-size: 13px;
  }
  .status.busy { background: var(--accent-soft, #eef6ff); }
  iframe {
    width: 100%;
    height: 620px;
    border: 1px solid var(--rule, #d8d6cd);
    border-radius: 10px;
    background: var(--panel, white);
  }
  .result-header { display:flex; justify-content:space-between; gap:12px; align-items:center; }
  .result h2 { margin: 0 0 10px; }
  .result textarea { min-height: 520px; font-size: 12px; line-height: 1.4; }
  .small { color: var(--muted, #6f6a5f); font-size: 13px; }
  code { font-family: var(--mono, ui-monospace, SFMono-Regular, Menlo, Consolas, monospace); }
</style>
