import '../nn/ortConsoleFilter.ts';
import { collectOrtRuntimeDiagnostics, type OrtExecutionProviderPreference } from '../nn/ortRuntime.ts';
import { loadLc0WebModelPack } from './modelPack.ts';
import type { Lc0EvaluatorInput } from './onnxEvaluator.ts';
import { applyOrtExecutionProvider } from './ortWorkerExecutionProvider.ts';
import {
  type Lc0WebAttentionOutProjKernelVariant,
  type Lc0WebAttentionQkvKernelVariant,
  type Lc0WebEncoderKernelVariant,
  type Lc0WebFfnKernelVariant,
  type Lc0WebHybridEncoderProfileMode,
  type Lc0WebHybridLegalPriorsBackend,
  type Lc0WebSmolgenKernelVariant,
  type Lc0WebWgslHeadsVsOrtFixtureInput,
  runLc0WebAttentionBlockBenchmark,
  runLc0WebAttentionOutputBenchmark,
  runLc0WebAttentionOutputOrtBenchmark,
  runLc0WebAttentionScoreBenchmark,
  runLc0WebAttentionScoreOrtBenchmark,
  runLc0WebAttentionValueBenchmark,
  runLc0WebAttentionValueOrtBenchmark,
  runLc0WebEncoder0BlockBenchmark,
  runLc0WebEncoder0BlockOrtBenchmark,
  runLc0WebEncoder0FfnBenchmark,
  runLc0WebEncoder0FfnOrtBenchmark,
  runLc0WebEncoderStackBenchmark,
  runLc0WebHybridEncoderProfile,
  runLc0WebHybridEvaluation,
  runLc0WebMappedPolicyProbe,
  runLc0WebMatmulAddKernelBenchmark,
  runLc0WebMatmulAddKernelProbe,
  runLc0WebMatmulAddOrtBenchmark,
  runLc0WebQkvProjectionBenchmark,
  runLc0WebQkvProjectionProbe,
  runLc0WebSmolgenBenchmark,
  runLc0WebSoftmaxBenchmark,
  runLc0WebWgslDeferredReadbackBenchmark,
  runLc0WebWgslHeadsProbe,
  runLc0WebWgslHeadsVsOrtFixtures,
} from './wgslMatmulAddProbe.ts';

type PackMessage = {
  id: number;
  packUrl: string;
  iterations?: number;
  warmup?: number;
  verifyShards?: boolean;
};

type WorkerRequest =
  | (PackMessage & { type: 'loadPack'; loadWeights?: boolean; tensorNames?: string[] })
  | (PackMessage & {
      type: 'kernelProbe' | 'kernelBenchmark';
      weightTensorName?: string;
      biasTensorName?: string;
      variant?: 'scalar' | 'tiled16' | 'scalar-transposed' | 'scalar-shader-f16-accum-f32';
    })
  | (PackMessage & { type: 'qkvProbe' | 'qkvBenchmark' | 'attentionScoreBenchmark' | 'softmaxBenchmark' | 'attentionValueBenchmark' })
  | (PackMessage & { type: 'attentionBlockBenchmark'; fusedScoreSoftmax?: boolean; attentionQkvKernelVariant?: Lc0WebAttentionQkvKernelVariant })
  | (PackMessage & { type: 'attentionOutputBenchmark'; encoderPrefix?: string; attentionOutProjKernelVariant?: Lc0WebAttentionOutProjKernelVariant })
  | (PackMessage & { type: 'encoder0FfnBenchmark'; encoderPrefix?: string; ffnKernelVariant?: Lc0WebFfnKernelVariant })
  | (PackMessage & { type: 'encoder0BlockBenchmark'; encoderPrefix?: string })
  | (PackMessage & { type: 'smolgenBenchmark'; encoderPrefix?: string; projectKernelVariant?: Lc0WebSmolgenKernelVariant })
  | (PackMessage & {
      type:
        | 'ortBenchmark'
        | 'attentionScoreOrtBenchmark'
        | 'attentionValueOrtBenchmark'
        | 'attentionOutputOrtBenchmark'
        | 'encoder0FfnOrtBenchmark'
        | 'encoder0BlockOrtBenchmark';
      ep: OrtExecutionProviderPreference;
      weightTensorName?: string;
      biasTensorName?: string;
      encoderPrefix?: string;
    })
  | (PackMessage & { type: 'encoderStackBenchmark'; ep: OrtExecutionProviderPreference; layers?: number; compareOrt?: boolean; compareHeads?: boolean })
  | (PackMessage & { type: 'wgslHeadsProbe'; ep: OrtExecutionProviderPreference })
  | (PackMessage & {
      type: 'wgslHeadsVsOrtFixtures';
      ep: OrtExecutionProviderPreference;
      fixtures: Lc0WebWgslHeadsVsOrtFixtureInput[];
      layers?: number;
      mappedPolicyTolerance?: number;
      wdlTolerance?: number;
      strictWebGpu?: boolean;
    })
  | { type: 'mappedPolicyProbe'; id: number }
  | (PackMessage & {
      type: 'hybridEvaluate';
      input: Lc0EvaluatorInput;
      layers?: number;
      headBackend?: 'ort' | 'wgsl';
      wgslBatchMode?: 'physical' | 'serial';
      inputBackend?: 'js' | 'wgsl' | 'wasm';
      legalPriorsBackend?: Lc0WebHybridLegalPriorsBackend;
    })
  | (PackMessage & {
      type: 'hybridEncoderProfile';
      input: Lc0EvaluatorInput;
      layers?: number;
      inputBackend?: 'js' | 'wgsl' | 'wasm';
      encoderKernelVariant?: Lc0WebEncoderKernelVariant;
      profileMode?: Lc0WebHybridEncoderProfileMode;
    })
  | (PackMessage & {
      type: 'wgslDeferredReadbackBenchmark';
      inputs: Lc0EvaluatorInput[];
      layers?: number;
      inputBackend?: 'js' | 'wgsl' | 'wasm';
      legalPriorsBackend?: Lc0WebHybridLegalPriorsBackend;
      batchSize?: number;
    });

type PackFootprint = {
  declaredTensorBytes: number;
  loadedTensorBytes: number;
  totalShardBytes: number;
  loadedShardBytes: number;
  tensorCount: number;
  loadedTensorCount: number;
  shardCount: number;
  loadedShardCount: number;
  dtypeHistogram: Record<string, number>;
};

function post(type: string, id: number, result: unknown): void {
  self.postMessage({ type, id, result });
}

async function assertStrictWebGpuOrt(message: string, options: { probeAdapter?: boolean; requireSession?: boolean; minSessionAttemptIndex?: number } = {}) {
  const diagnostics = await collectOrtRuntimeDiagnostics({ probeAdapter: options.probeAdapter });
  const adapterOk = diagnostics.adapter?.ok !== false;
  const sessionAttempts = diagnostics.sessionAttempts.slice(options.minSessionAttemptIndex ?? 0);
  const latestSuccessfulSession = [...sessionAttempts].reverse().find((attempt) => attempt.ok);
  const actualProviders = options.requireSession ? (latestSuccessfulSession?.providers ?? []) : diagnostics.resolvedExecutionProviders;
  const providerOk = actualProviders.includes('webgpu');
  if (!diagnostics.webgpuAvailable || !adapterOk || !providerOk || (options.requireSession && !latestSuccessfulSession)) {
    throw new Error(
      `${message}: strict ORT WebGPU required but actual providers were ${actualProviders.join(',') || 'none'} (webgpuAvailable=${diagnostics.webgpuAvailable}, adapterOk=${adapterOk}, sessionsSince=${sessionAttempts.length}, sessionsTotal=${diagnostics.sessionAttempts.length})`,
    );
  }
  return diagnostics;
}

async function handleLoadPack(message: Extract<WorkerRequest, { type: 'loadPack' }>): Promise<void> {
  const pack = await loadLc0WebModelPack(message.packUrl, {
    loadWeights: message.loadWeights,
    verifyShards: message.verifyShards,
    tensorNames: message.tensorNames,
  });
  let loadedTensorBytes = 0;
  const loadedShardFiles = new Set<string>();
  for (const tensor of pack.tensors.values()) {
    loadedTensorBytes += tensor.bytes.byteLength;
    loadedShardFiles.add(tensor.info.shard);
  }
  const totalShardBytes = pack.manifest.weights.shards.reduce((sum, shard) => sum + shard.bytes, 0);
  const loadedShardBytes = pack.manifest.weights.shards.filter((shard) => loadedShardFiles.has(shard.file)).reduce((sum, shard) => sum + shard.bytes, 0);
  const packFootprint: PackFootprint = {
    declaredTensorBytes: pack.manifest.weights.totalTensorBytes,
    loadedTensorBytes,
    totalShardBytes,
    loadedShardBytes,
    tensorCount: pack.manifest.weights.tensorCount,
    loadedTensorCount: pack.tensors.size,
    shardCount: pack.manifest.weights.shards.length,
    loadedShardCount: loadedShardFiles.size,
    dtypeHistogram:
      pack.manifest.weights.dtypeHistogram ??
      pack.manifest.weights.tensors.reduce<Record<string, number>>((histogram, tensor) => {
        histogram[tensor.dtype] = (histogram[tensor.dtype] ?? 0) + 1;
        return histogram;
      }, {}),
  };
  post('packLoadResult', message.id, {
    packUrl: pack.manifestUrl,
    modelName: pack.manifest.model.name,
    sourceSha256: pack.manifest.model.sourceSha256,
    layout: pack.manifest.model.layout,
    recommendedRuntime: pack.manifest.model.recommendedRuntime,
    tensorCount: pack.manifest.weights.tensorCount,
    loadedTensorCount: pack.tensors.size,
    loadedTensorBytes,
    shardCount: pack.manifest.weights.shards.length,
    verifiedShardCount: pack.verifiedShards.length,
    shardBytes: totalShardBytes,
    packFootprint,
    elapsedMs: pack.elapsedMs,
  });
}

async function handle(message: WorkerRequest): Promise<void> {
  if (message.type === 'loadPack') return handleLoadPack(message);
  if (message.type === 'mappedPolicyProbe') return post('mappedPolicyProbeResult', message.id, await runLc0WebMappedPolicyProbe());
  if (message.type === 'kernelProbe')
    return post(
      'kernelProbeResult',
      message.id,
      await runLc0WebMatmulAddKernelProbe({
        packUrl: message.packUrl,
        weightTensorName: message.weightTensorName,
        biasTensorName: message.biasTensorName,
        iterations: message.iterations,
        warmup: message.warmup,
        verifyShards: message.verifyShards,
        variant: message.variant,
      }),
    );
  if (message.type === 'kernelBenchmark')
    return post(
      'kernelBenchmarkResult',
      message.id,
      await runLc0WebMatmulAddKernelBenchmark({
        packUrl: message.packUrl,
        weightTensorName: message.weightTensorName,
        biasTensorName: message.biasTensorName,
        iterations: message.iterations,
        warmup: message.warmup,
        verifyShards: message.verifyShards,
        variant: message.variant,
      }),
    );
  if (message.type === 'qkvProbe') return post('qkvProbeResult', message.id, await runLc0WebQkvProjectionProbe(message));
  if (message.type === 'qkvBenchmark') return post('qkvBenchmarkResult', message.id, await runLc0WebQkvProjectionBenchmark(message));
  if (message.type === 'attentionScoreBenchmark') return post('attentionScoreBenchmarkResult', message.id, await runLc0WebAttentionScoreBenchmark(message));
  if (message.type === 'smolgenBenchmark') return post('smolgenBenchmarkResult', message.id, await runLc0WebSmolgenBenchmark(message));
  if (message.type === 'softmaxBenchmark') return post('softmaxBenchmarkResult', message.id, await runLc0WebSoftmaxBenchmark(message));
  if (message.type === 'attentionValueBenchmark') return post('attentionValueBenchmarkResult', message.id, await runLc0WebAttentionValueBenchmark(message));
  if (message.type === 'attentionBlockBenchmark') return post('attentionBlockBenchmarkResult', message.id, await runLc0WebAttentionBlockBenchmark(message));
  if (message.type === 'attentionOutputBenchmark') return post('attentionOutputBenchmarkResult', message.id, await runLc0WebAttentionOutputBenchmark(message));
  if (message.type === 'encoder0FfnBenchmark') return post('encoder0FfnBenchmarkResult', message.id, await runLc0WebEncoder0FfnBenchmark(message));
  if (message.type === 'encoder0BlockBenchmark') return post('encoder0BlockBenchmarkResult', message.id, await runLc0WebEncoder0BlockBenchmark(message));
  if (message.type === 'hybridEvaluate') return post('hybridEvaluationResult', message.id, await runLc0WebHybridEvaluation(message));
  if (message.type === 'hybridEncoderProfile') return post('hybridEncoderProfileResult', message.id, await runLc0WebHybridEncoderProfile(message));
  if (message.type === 'wgslDeferredReadbackBenchmark')
    return post('wgslDeferredReadbackBenchmarkResult', message.id, await runLc0WebWgslDeferredReadbackBenchmark(message));

  if (!('ep' in message)) throw new Error(`Unsupported LC0 lab request: ${message.type}`);
  applyOrtExecutionProvider(message.ep);
  if (message.type === 'ortBenchmark')
    return post(
      'ortBenchmarkResult',
      message.id,
      await runLc0WebMatmulAddOrtBenchmark({
        packUrl: message.packUrl,
        weightTensorName: message.weightTensorName,
        biasTensorName: message.biasTensorName,
        iterations: message.iterations,
        warmup: message.warmup,
        verifyShards: message.verifyShards,
      }),
    );
  if (message.type === 'attentionScoreOrtBenchmark')
    return post('attentionScoreOrtBenchmarkResult', message.id, await runLc0WebAttentionScoreOrtBenchmark(message));
  if (message.type === 'attentionValueOrtBenchmark')
    return post('attentionValueOrtBenchmarkResult', message.id, await runLc0WebAttentionValueOrtBenchmark(message));
  if (message.type === 'attentionOutputOrtBenchmark')
    return post('attentionOutputOrtBenchmarkResult', message.id, await runLc0WebAttentionOutputOrtBenchmark(message));
  if (message.type === 'encoder0FfnOrtBenchmark') return post('encoder0FfnOrtBenchmarkResult', message.id, await runLc0WebEncoder0FfnOrtBenchmark(message));
  if (message.type === 'encoder0BlockOrtBenchmark')
    return post('encoder0BlockOrtBenchmarkResult', message.id, await runLc0WebEncoder0BlockOrtBenchmark(message));
  if (message.type === 'encoderStackBenchmark') return post('encoderStackBenchmarkResult', message.id, await runLc0WebEncoderStackBenchmark(message));
  if (message.type === 'wgslHeadsProbe') return post('wgslHeadsProbeResult', message.id, await runLc0WebWgslHeadsProbe(message));
  if (message.type === 'wgslHeadsVsOrtFixtures') {
    const strictPreflight = message.strictWebGpu ? await assertStrictWebGpuOrt('WGSL heads vs ORT fixtures preflight', { probeAdapter: true }) : undefined;
    const result = await runLc0WebWgslHeadsVsOrtFixtures(message);
    if (message.strictWebGpu)
      await assertStrictWebGpuOrt('WGSL heads vs ORT fixtures postrun', {
        requireSession: true,
        minSessionAttemptIndex: strictPreflight?.sessionAttempts.length ?? 0,
      });
    return post('wgslHeadsVsOrtFixturesResult', message.id, result);
  }
}

let operationQueue: Promise<void> = Promise.resolve();
self.addEventListener('message', (event: MessageEvent<WorkerRequest>) => {
  const message = event.data;
  const run = operationQueue.then(
    () => handle(message),
    () => handle(message),
  );
  operationQueue = run.catch(() => undefined);
  void run.catch((error) => {
    self.postMessage({ type: 'error', id: message.id, error: error instanceof Error ? error.message : String(error) });
  });
});
