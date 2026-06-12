import * as ort from '../nn/ortRuntime.ts';
import { createOrtSession, releaseOrtSession } from '../nn/ortRuntime.ts';

type InitMessage = {
  type: 'init';
  id: number;
  model: string | ArrayBuffer;
};

type EvaluateMessage = {
  type: 'evaluate';
  id: number;
  tokens: ArrayBuffer;
  eloSelf: number;
  eloOppo: number;
};

type DisposeMessage = {
  type: 'dispose';
  id: number;
};

type Maia3WorkerMessage = InitMessage | EvaluateMessage | DisposeMessage;

let session: ort.InferenceSession | null = null;

function post(message: unknown, transfers: Transferable[] = []): void {
  (globalThis as unknown as { postMessage: (message: unknown, transfers?: Transferable[]) => void }).postMessage(message, transfers);
}

function firstOutput(outputs: Awaited<ReturnType<ort.InferenceSession['run']>>, preferred: string, fallbackIndex: number): Float32Array {
  const byName = outputs[preferred];
  const tensor = byName ?? Object.values(outputs)[fallbackIndex];
  if (!tensor) throw new Error(`Maia3 output ${preferred} missing`);
  return new Float32Array(tensor.data as Float32Array | number[]);
}

(globalThis as unknown as { onmessage: ((event: MessageEvent<Maia3WorkerMessage>) => void) | null }).onmessage = (event) => {
  void (async () => {
    const message = event.data;
    try {
      if (message.type === 'init') {
        if (session) await releaseOrtSession(session);
        session = await createOrtSession(message.model);
        post({ type: 'ready', id: message.id, inputNames: session.inputNames, outputNames: session.outputNames });
        return;
      }

      if (message.type === 'dispose') {
        if (session) await releaseOrtSession(session);
        session = null;
        post({ type: 'disposed', id: message.id });
        return;
      }

      if (!session) throw new Error('Maia3 worker is not initialized');

      const feeds: Record<string, ort.Tensor> = {
        tokens: new ort.Tensor('float32', new Float32Array(message.tokens), [1, 64, 12]),
        elo_self: new ort.Tensor('float32', new Float32Array([message.eloSelf]), [1]),
        elo_oppo: new ort.Tensor('float32', new Float32Array([message.eloOppo]), [1]),
      };
      const outputs = await session.run(feeds);
      const logitsMove = firstOutput(outputs, 'logits_move', 0);
      const logitsValue = firstOutput(outputs, 'logits_value', 1);
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
