import * as ort from '../nn/ortRuntime.ts';
import { createOrtSession, releaseOrtSession, describeOrtBackendConfig, setRequestedOrtExecutionProviderForCurrentThread, type OrtExecutionProviderPreference } from '../nn/ortRuntime.ts';

type InitMessage = {
  type: 'init';
  id: number;
  model: string | ArrayBuffer;
  /** ORT execution-provider preference; default 'auto' (WebGPU first, wasm fallback). */
  ep?: OrtExecutionProviderPreference;
};

type EvaluateMessage = {
  type: 'evaluate';
  id: number;
  tokens: ArrayBuffer;
  eloSelf: number;
  eloOppo: number;
};

/** One position evaluated under many (eloSelf, eloOppo) conditions in one run (rating-inference grids). */
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

function post(message: unknown, transfers: Transferable[] = []): void {
  (globalThis as unknown as { postMessage: (message: unknown, transfers?: Transferable[]) => void }).postMessage(message, transfers);
}

async function firstOutput(outputs: Awaited<ReturnType<ort.InferenceSession['run']>>, preferred: string, fallbackIndex: number): Promise<Float32Array> {
  const byName = outputs[preferred];
  const tensor = byName ?? Object.values(outputs)[fallbackIndex];
  if (!tensor) throw new Error(`Maia3 output ${preferred} missing`);
  const maybeGpuTensor = tensor as ort.Tensor & { location?: string; getData?: (releaseData?: boolean) => Promise<unknown> };
  const rawData = maybeGpuTensor.location === 'gpu-buffer' && typeof maybeGpuTensor.getData === 'function'
    ? await maybeGpuTensor.getData(true)
    : tensor.data;
  return new Float32Array(rawData as Float32Array | number[]);
}

(globalThis as unknown as { onmessage: ((event: MessageEvent<Maia3WorkerMessage>) => void) | null }).onmessage = (event) => {
  void (async () => {
    const message = event.data;
    try {
      if (message.type === 'init') {
        if (session) await releaseOrtSession(session);
        if (message.ep) setRequestedOrtExecutionProviderForCurrentThread(message.ep);
        post({ type: 'progress', id: message.id, stage: 'creating-session' });
        session = await createOrtSession(message.model);
        post({ type: 'ready', id: message.id, inputNames: session.inputNames, outputNames: session.outputNames, backend: describeOrtBackendConfig() });
        return;
      }

      if (message.type === 'dispose') {
        if (session) await releaseOrtSession(session);
        session = null;
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
        post({
          type: 'result',
          id: message.id,
          logitsMove: logitsMove.buffer,
          logitsValue: logitsValue.buffer,
        }, [logitsMove.buffer, logitsValue.buffer]);
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
      post({
        type: 'result',
        id: message.id,
        logitsMove: logitsMove.buffer,
        logitsValue: logitsValue.buffer,
      }, [logitsMove.buffer, logitsValue.buffer]);
    } catch (error) {
      post({ type: 'error', id: message.id, message: error instanceof Error ? error.message : String(error) });
    }
  })();
};
