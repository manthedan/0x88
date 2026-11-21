import type { OrtDeviceLoss } from '../nn/ortDeviceLoss.ts';

export class WorkerDeviceLostError extends Error {
  readonly lossCount: number;

  constructor(loss: OrtDeviceLoss, lossCount: number) {
    super(loss.message ? `WebGPU device lost: ${loss.message}` : 'WebGPU device lost');
    this.lossCount = lossCount;
  }

  get retry(): boolean {
    return this.lossCount <= 2;
  }

  warning(engine: string): string {
    if (this.lossCount === 1) return `${engine}: GPU context was reset. Restarting the engine when needed.`;
    if (this.lossCount === 2) return `${engine}: GPU context was reset again. Switching to CPU (WASM).`;
    return `${engine}: GPU recovery did not succeed.`;
  }
}

export class WorkerDeviceRecovery {
  private losses = 0;

  provider<T extends string | undefined>(preference: T): T | 'wasm' {
    return this.losses >= 2 ? 'wasm' : preference;
  }

  lost(loss: OrtDeviceLoss): WorkerDeviceLostError {
    return new WorkerDeviceLostError(loss, ++this.losses);
  }
}

export async function retryWorkerDeviceLoss<T>(work: () => Promise<T>, signal?: AbortSignal): Promise<T> {
  for (;;) {
    signal?.throwIfAborted();
    try {
      return await work();
    } catch (error) {
      if (!(error instanceof WorkerDeviceLostError && error.retry)) throw error;
    }
  }
}

export function showWorkerDeviceWarning(target: HTMLElement, warning: string): void {
  target.textContent = warning;
  target.hidden = !warning;
}
