import { createOrtDeviceLossMonitor } from '../nn/ortDeviceLoss.ts';
import * as ort from '../nn/ortRuntime.ts';
import {
  collectOrtSessionDiagnostics,
  createOrtSession,
  describeOrtBackendConfig,
  type OrtExecutionProviderPreference,
  type OrtRuntimeDiagnosticOptions,
  ortRuntimeArtifactKindIsLocked,
  releaseOrtSession,
  setOrtDequantFoldForCurrentThread,
  setOrtRuntimeArtifactKindForCurrentThread,
  setOrtRuntimeDiagnosticOptionsForCurrentThread,
  setRequestedOrtExecutionProviderForCurrentThread,
  setRequestedOrtWasmThreadsForCurrentThread,
} from '../nn/ortRuntime.ts';
import { releaseOrtWorkerResources } from '../nn/ortWorkerDispose.ts';
import { ExclusiveQueue } from './exclusiveQueue.ts';
import { MAIA3_ORT_DEFAULTS } from './maia3RuntimeConfig.ts';
import { type Lc0ModelLoadInfo, loadLc0ModelForOrt } from './modelCache.ts';
import { isMetadataTrustedModel, isOnnxModelParseError } from './onnxModelRecovery.ts';

type InitMessage = {
  type: 'init';
  id: number;
  model: string | ArrayBuffer;
  /** Verified reload provenance; never an opaque/unverified model source. */
  modelUrl?: string;
  metadataTrustedModel?: boolean;
  /** ORT execution-provider preference; default 'auto' (WebGPU first, wasm fallback). */
  ep?: OrtExecutionProviderPreference;
  wasmThreads?: number;
  ortDiagnostics?: OrtRuntimeDiagnosticOptions;
  dequantFold?: boolean;
};

type EvaluateMessage = {
  type: 'evaluate';
  id: number;
  tokens: ArrayBuffer;
  eloSelf: number;
  eloOppo: number;
};

/** One position evaluated under several (eloSelf, eloOppo) conditions in one run. */
type EvaluateConditionsMessage = {
  type: 'evaluateConditions';
  id: number;
  tokens: ArrayBuffer;
  eloSelfs: number[];
  eloOppos: number[];
};

type DisposeMessage = {
  type: 'dispose';
  id: number;
};

type Maia3WorkerMessage = InitMessage | EvaluateMessage | EvaluateConditionsMessage | DisposeMessage;

let session: ort.InferenceSession | null = null;
const operations = new ExclusiveQueue();
let retiring = false;

function post(message: unknown, transfers: Transferable[] = []): void {
  (globalThis as unknown as { postMessage: (message: unknown, transfers?: Transferable[]) => void }).postMessage(message, transfers);
}

const deviceLossMonitor = createOrtDeviceLossMonitor((loss) => post({ type: 'deviceLost', loss }));

async function firstOutput(outputs: Awaited<ReturnType<ort.InferenceSession['run']>>, preferred: string, fallbackIndex: number): Promise<Float32Array> {
  const byName = outputs[preferred];
  const tensor = byName ?? Object.values(outputs)[fallbackIndex];
  if (!tensor) throw new Error(`Maia3 output ${preferred} missing`);
  const maybeGpuTensor = tensor as ort.Tensor & { location?: string; getData?: (releaseData?: boolean) => Promise<unknown> };
  const rawData = maybeGpuTensor.location === 'gpu-buffer' && typeof maybeGpuTensor.getData === 'function' ? await maybeGpuTensor.getData(true) : tensor.data;
  return new Float32Array(rawData as Float32Array | number[]);
}

(globalThis as unknown as { onmessage: ((event: MessageEvent<Maia3WorkerMessage>) => void) | null }).onmessage = (event) => {
  if ((event.data as { type: string }).type === 'cancel' || retiring) return;
  if (event.data.type === 'dispose') retiring = true;
  // ORT runs and session release must not overlap. The live panel coalesces
  // obsolete requests before posting; this also protects other worker callers.
  void operations
    .run(async () => {
      const message = event.data;
      let fallbackModelLoad: Lc0ModelLoadInfo | undefined;
      try {
        if (message.type === 'init') {
          deviceLossMonitor.stop();
          if (session) await releaseOrtSession(session);
          if (message.ep) setRequestedOrtExecutionProviderForCurrentThread(message.ep);
          setOrtRuntimeDiagnosticOptionsForCurrentThread(message.ortDiagnostics ?? null);
          setOrtDequantFoldForCurrentThread(message.dequantFold ?? MAIA3_ORT_DEFAULTS.dequantFold);
          setRequestedOrtWasmThreadsForCurrentThread(message.wasmThreads ?? null);
          // Unlike searchWorker this worker never switches provider per message,
          // so an ep=wasm request is a lifetime property and can pin the runtime
          // pair. Without this a WebGPU-capable browser running Maia on WASM
          // downloads the 24 MB asyncify build and stays single-threaded: the
          // selector deliberately ignores a programmatic provider pin, because
          // for a per-message worker it would not be safe.
          if (message.ep === 'wasm' && !ortRuntimeArtifactKindIsLocked()) setOrtRuntimeArtifactKindForCurrentThread('wasm');
          post({ type: 'progress', id: message.id, stage: 'creating-session' });
          const reloadModel = message.modelUrl
            ? async () => {
                const { model, ...info } = await loadLc0ModelForOrt(message.modelUrl!, {
                  cache: true,
                  cacheName: 'maia3-browser-models-v1',
                  manifestUrl: '/models/maia3/manifest.json',
                  allowUnverified: false,
                });
                fallbackModelLoad = info;
                return model;
              }
            : undefined;
          const creating = createOrtSession(message.model, {
            reloadModel,
            rejectModelParseErrors: () => (fallbackModelLoad ? isMetadataTrustedModel(fallbackModelLoad) : message.metadataTrustedModel === true),
          });
          message.model = ''; // The worker-owned init event must not pin the unfolded buffer.
          session = await creating;
          deviceLossMonitor.refresh();
          post({
            type: 'ready',
            id: message.id,
            inputNames: session.inputNames,
            outputNames: session.outputNames,
            backend: describeOrtBackendConfig(),
            ortRuntime: collectOrtSessionDiagnostics(),
          });
          return;
        }

        if (message.type === 'dispose') {
          deviceLossMonitor.stop();
          const previous = session;
          session = null;
          await releaseOrtWorkerResources(async () => {
            if (previous) await releaseOrtSession(previous);
          });
          post({ type: 'disposed', id: message.id });
          return;
        }

        if (!session) throw new Error('Maia3 worker is not initialized');

        if (message.type === 'evaluateConditions') {
          const single = new Float32Array(message.tokens);
          const n = message.eloSelfs.length;
          const batch = new Float32Array(n * single.length);
          for (let i = 0; i < n; i += 1) batch.set(single, i * single.length);
          const feeds: Record<string, ort.Tensor> = {
            tokens: new ort.Tensor('float32', batch, [n, 64, 12]),
            elo_self: new ort.Tensor('float32', Float32Array.from(message.eloSelfs), [n]),
            elo_oppo: new ort.Tensor('float32', Float32Array.from(message.eloOppos), [n]),
          };
          const outputs = await session.run(feeds);
          const logitsMove = await firstOutput(outputs, 'logits_move', 0);
          const logitsValue = await firstOutput(outputs, 'logits_value', 1);
          post(
            {
              type: 'result',
              id: message.id,
              logitsMove: logitsMove.buffer,
              logitsValue: logitsValue.buffer,
            },
            [logitsMove.buffer, logitsValue.buffer],
          );
          return;
        }

        const feeds: Record<string, ort.Tensor> = {
          tokens: new ort.Tensor('float32', new Float32Array(message.tokens), [1, 64, 12]),
          elo_self: new ort.Tensor('float32', new Float32Array([message.eloSelf]), [1]),
          elo_oppo: new ort.Tensor('float32', new Float32Array([message.eloOppo]), [1]),
        };
        const outputs = await session.run(feeds);
        const logitsMove = await firstOutput(outputs, 'logits_move', 0);
        const logitsValue = await firstOutput(outputs, 'logits_value', 1);
        post(
          {
            type: 'result',
            id: message.id,
            logitsMove: logitsMove.buffer,
            logitsValue: logitsValue.buffer,
          },
          [logitsMove.buffer, logitsValue.buffer],
        );
      } catch (error) {
        post({
          type: 'error',
          id: message.id,
          message: error instanceof Error ? error.message : String(error),
          ...(message.type === 'init' ? { modelParseError: isOnnxModelParseError(error) } : {}),
          ...(fallbackModelLoad ? { modelLoad: fallbackModelLoad } : {}),
        });
      }
    })
    .catch(() => undefined);
};
