import '../nn/ortConsoleFilter.ts';
import { collectOrtRuntimeDiagnostics } from '../nn/ortRuntime.ts';
import { simulCancelGame, simulGeneration } from './arenaSimulGate.ts';
import { simulProbe } from './arenaSimulProbe.ts';
import { ARENA_SIMUL_PROTOCOL, runSimulTrial, SIMUL_OPENINGS, SIMUL_SEARCH_OPTIONS, validateSimulTrial } from './arenaSimulRunner.ts';
import { type Lc0Evaluation, type Lc0EvaluationProvider, type Lc0EvaluatorInput, Lc0OnnxEvaluator } from './onnxEvaluator.ts';
import { applyOrtExecutionProvider } from './ortWorkerExecutionProvider.ts';
import { Lc0PuctSearcher } from './search.ts';

interface Start {
  type: 'start';
  n: number;
  visits: number;
  maxPlies: number;
  pairs: number;
  runId: string;
  allow8?: boolean;
}
let run: AbortController | undefined;
let generation = '';
const controllers = new Map<number, AbortController>();
const post = (message: Record<string, unknown>) => self.postMessage(message);
type MemorySample = { bytes: number | null; note: string };
let memoryWaiter: ((value: MemorySample) => void) | undefined;

async function sampleMemory(signal: AbortSignal): Promise<MemorySample> {
  signal.throwIfAborted();
  return await new Promise<MemorySample>((resolve, reject) => {
    const finish = (sample: MemorySample) => {
      cleanup();
      resolve(sample);
    };
    const abort = () => {
      cleanup();
      reject(signal.reason);
    };
    const timer = setTimeout(() => {
      cleanup();
      reject(new Error('Memory sample timed out; reload before another gate'));
    }, 60000);
    const cleanup = () => {
      clearTimeout(timer);
      signal.removeEventListener('abort', abort);
      memoryWaiter = undefined;
    };
    memoryWaiter = finish;
    signal.addEventListener('abort', abort, { once: true });
    post({ type: 'memory', generation });
  });
}

function difference(a: Lc0Evaluation[], b: Lc0Evaluation[]) {
  let wdl = 0;
  let prior = 0;
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < 3; j++) wdl = Math.max(wdl, Math.abs(a[i].wdl[j] - b[i].wdl[j]));
    const byMove = new Map(b[i].legalPriors.map((entry) => [entry.uci, entry.prior]));
    for (const entry of a[i].legalPriors) prior = Math.max(prior, Math.abs(entry.prior - byMove.get(entry.uci)!));
  }
  return { maxWdl: wdl, maxPrior: prior };
}

// Cumulative createBuffer request bytes, NOT live/peak memory. Native releases and
// garbage collection need not call JS destroy(), so no residency claim is made.
let bufferAllocationBytes = 0;
const patched = new WeakSet<object>();
function installBufferProbe() {
  const gpu = (navigator as unknown as { gpu: { requestAdapter: (...args: unknown[]) => Promise<unknown> } }).gpu;
  if (!gpu) return;
  const requestAdapter = gpu.requestAdapter.bind(gpu);
  gpu.requestAdapter = async (...args) => {
    const adapter = (await requestAdapter(...args)) as { requestDevice?: (...args: unknown[]) => Promise<unknown> } | null;
    if (!adapter?.requestDevice) return adapter;
    const requestDevice = adapter.requestDevice.bind(adapter);
    adapter.requestDevice = async (...deviceArgs) => {
      const device = (await requestDevice(...deviceArgs)) as { createBuffer: (descriptor: { size: number }) => unknown };
      if (patched.has(device)) return device;
      patched.add(device);
      const createBuffer = device.createBuffer.bind(device);
      device.createBuffer = (descriptor) => {
        const buffer = createBuffer(descriptor);
        bufferAllocationBytes += Number(descriptor.size);
        return buffer;
      };
      return device;
    };
    return adapter;
  };
}
installBufferProbe();

async function benchmark(message: Start, signal: AbortSignal) {
  validateSimulTrial({ concurrency: message.n, visits: message.visits, maxPlies: message.maxPlies });
  if (![2, 4, 8].includes(message.n) || !Number.isInteger(message.pairs) || message.pairs < 1 || message.pairs > 3 || (message.n === 8 && !message.allow8))
    throw new Error('Run N=2/4 first; N=8 requires a helpful N=4 gate');
  applyOrtExecutionProvider('webgpu');
  const rows = [];
  let baselineProbe: Lc0Evaluation[] | undefined;
  const inputs: Lc0EvaluatorInput[] = SIMUL_OPENINGS.map((opening) => ({ positions: opening.positions! }));
  // ABAB... with fresh sessions; startup and equal warmup excluded. Last row isolates
  // scheduling from export numerics: serial, using the SAME batch-N artifact as simul.
  for (let index = 0; index <= message.pairs * 2; index++) {
    signal.throwIfAborted();
    const control = index === message.pairs * 2;
    const simul = !control && index % 2 === 1;
    const physical = simul || control ? (message.n <= 4 ? 4 : 8) : 1;
    const concurrency = simul ? message.n : 1;
    const modelUrl = `/models/lc0/t1-256x10-distilled-swa-2432500.batch${physical}.f16.onnx`;
    generation = simulGeneration(message.runId, index);
    bufferAllocationBytes = 0;
    post({ type: 'phase', phase: 'loading', generation, index, physical, concurrency, control });
    const response = await fetch(modelUrl, { signal });
    if (!response.ok) throw new Error(`Local model unavailable: ${response.status}`);
    const bytes = new Uint8Array(await response.arrayBuffer());
    const evaluator = await Lc0OnnxEvaluator.create(bytes);
    try {
      signal.throwIfAborted();
      const runtime = await collectOrtRuntimeDiagnostics();
      if (!runtime.describe.toLowerCase().includes('webgpu') || runtime.fallback) throw new Error('Gate requires strict WebGPU, no fallback');
      signal.throwIfAborted();
      const probe = await simulProbe(evaluator, inputs, physical, signal);
      if (!baselineProbe) baselineProbe = probe;
      const padded = [];
      for (const input of inputs) {
        signal.throwIfAborted();
        padded.push(await evaluator.evaluate(input));
      }
      const observed = probe.map((result) => (result.timing as { ortPhysicalBatchSize?: number }).ortPhysicalBatchSize);
      if (observed.some((size) => size !== physical)) throw new Error('Export did not execute the required physical batch');
      let physicalRuns = 0;
      const checked: Lc0EvaluationProvider = {
        evaluate: (input) => evaluator.evaluate(input),
        evaluateBatch: async (batch) => {
          const results = await evaluator.evaluateBatch(batch);
          if (
            results.filter((result) => (result.timing as { batchPosition?: number }).batchPosition === 0).length !== 1 ||
            results.some((result) => (result.timing as { ortPhysicalBatchSize?: number }).ortPhysicalBatchSize !== physical)
          ) {
            throw new Error('Broker batch was split into serial physical runs');
          }
          physicalRuns++;
          return results;
        },
      };
      signal.throwIfAborted();
      const warmup = new Lc0PuctSearcher(evaluator);
      await warmup.search(inputs[0], { ...SIMUL_SEARCH_OPTIONS, visits: message.visits, signal });
      let lastPost = 0;
      const trial = await runSimulTrial(checked, {
        concurrency,
        visits: message.visits,
        maxPlies: message.maxPlies,
        signal,
        controllers,
        onProgress: (progress) => {
          if (performance.now() - lastPost < 150 && progress.ply !== 0) return;
          lastPost = performance.now();
          post({ type: 'progress', generation, index, ...progress });
        },
      });
      // Never publish normal-looking performance rates for an incomplete trial.
      if (trial.completed !== 8) throw new Error('Trial cancelled; incomplete games are not scored or timed as completed games');
      // Memory is a SEPARATE pass after ALL timed rows, never between A/B trials:
      // UASM can trigger GC long after its caller's timeout. No subsequent timing
      // work may run while that promise is pending, including after Stop/restart.
      if (control) post({ type: 'phase', phase: 'memory', generation, index, physical, concurrency, control });
      const memory = control ? await sampleMemory(signal) : { bytes: null, note: 'Memory sampled after all timed trials on the same-export control' };
      const row = {
        ...trial,
        index,
        control,
        physical,
        physicalRuns,
        modelUrl,
        modelBytes: bytes.byteLength,
        runtime: runtime.describe,
        memory,
        bufferAllocationBytes,
        occupancy: trial.broker.evaluations / (physicalRuns * physical),
        exportDrift: difference(probe, baselineProbe),
        rowDrift: difference(probe, padded),
      };
      rows.push(row);
      post({ type: 'trial', row });
    } finally {
      await evaluator.dispose();
    }
  }
  post({
    type: 'done',
    result: {
      protocol: ARENA_SIMUL_PROTOCOL,
      n: message.n,
      visits: message.visits,
      maxPlies: message.maxPlies,
      pairs: message.pairs,
      options: SIMUL_SEARCH_OPTIONS,
      openings: SIMUL_OPENINGS.map(({ name, fen, moves }) => ({ name, fen, moves })),
      rows,
    },
  });
}

self.onmessage = (
  event: MessageEvent<
    Start | { type: 'stop' } | { type: 'cancelGame'; generation: string; game?: number } | { type: 'memory'; generation: string; sample: MemorySample }
  >,
) => {
  const message = event.data;
  if (message.type === 'stop') {
    run?.abort();
    return;
  }
  if (message.type === 'memory') {
    if (message.generation === generation) memoryWaiter?.(message.sample);
    return;
  }
  if (message.type === 'cancelGame') {
    if (message.generation === generation) simulCancelGame(controllers, message.game);
    return;
  }
  if (run) {
    post({ type: 'error', error: 'A gate is already active' });
    return;
  }
  run = new AbortController();
  void benchmark(message, run.signal)
    .catch((error) => {
      post({ type: 'error', error: error instanceof Error ? error.message : String(error) });
    })
    .finally(() => {
      run = undefined;
    });
};
