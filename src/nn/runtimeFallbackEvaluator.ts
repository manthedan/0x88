import type { BoardState } from '../chess/board.ts';
import type { Evaluation, EvaluationContext, Evaluator } from './evaluator.ts';
import { type BrowserRuntimeAuditDetail, publishBrowserRuntimeAudit } from './runtimeAudit.ts';

export async function disposeEvaluator(evaluator: Evaluator | undefined): Promise<void> {
  if (!evaluator) return;
  const owner = evaluator as Evaluator & { dispose?: () => void | Promise<void>; destroy?: () => void | Promise<void> };
  await (owner.dispose ? owner.dispose() : owner.destroy?.());
}

function evaluatorAbortError(message: string): Error {
  const error = new Error(message);
  error.name = 'AbortError';
  return error;
}

export class RuntimeFallbackEvaluator implements Evaluator {
  private fallbackPromise?: Promise<Evaluator>;
  private fallbackCreation?: Promise<Evaluator>;
  private primaryRetirement?: Promise<void>;
  private closePromise?: Promise<void>;
  private primaryFailed = false;
  private disposed = false;
  private readonly primary: Evaluator;
  private readonly createFallback: () => Promise<Evaluator>;
  private readonly auditBase: BrowserRuntimeAuditDetail;
  private readonly fallbackRuntime: 'tvmjs-webgpu-fallback-ort' | 'custom-webgpu-fallback-ort';

  constructor(
    primary: Evaluator,
    createFallback: () => Promise<Evaluator>,
    auditBase: BrowserRuntimeAuditDetail,
    fallbackRuntime: 'tvmjs-webgpu-fallback-ort' | 'custom-webgpu-fallback-ort' = 'tvmjs-webgpu-fallback-ort',
  ) {
    this.primary = primary;
    this.createFallback = createFallback;
    this.auditBase = auditBase;
    this.fallbackRuntime = fallbackRuntime;
  }

  private async fallback(error: unknown): Promise<Evaluator> {
    if (error instanceof Error && error.name === 'AbortError') throw error;
    if (this.disposed) throw evaluatorAbortError('Centipawn runtime evaluator is destroyed');
    if (!this.primaryFailed) {
      this.primaryFailed = true;
      const fallbackReason = error instanceof Error ? error.message : String(error);
      const primaryLabel = this.fallbackRuntime === 'custom-webgpu-fallback-ort' ? 'Custom WebGPU' : 'TVMJS WebGPU';
      console.warn(`${primaryLabel} runtime failed; switching Centipawn to ORT.`, { fallbackReason });
      this.primaryRetirement = disposeEvaluator(this.primary);
      publishBrowserRuntimeAudit({ ...this.auditBase, resolvedRuntime: this.fallbackRuntime, fallbackReason });
    }
    return this.fallbackEvaluator();
  }

  private async fallbackEvaluator(): Promise<Evaluator> {
    await this.primaryRetirement;
    if (this.disposed) throw evaluatorAbortError('Centipawn runtime evaluator is destroyed');
    if (!this.fallbackPromise) {
      this.fallbackCreation = this.createFallback();
      const creation = this.fallbackCreation.then((fallback) => {
        if (!this.disposed) return fallback;
        throw evaluatorAbortError('Centipawn runtime evaluator is destroyed');
      });
      this.fallbackPromise = creation;
      // The primary is already destroyed, so a failed creation must not be
      // cached: the next evaluation retries it instead of rethrowing forever.
      creation.catch(() => {
        if (this.fallbackPromise === creation) this.fallbackPromise = undefined;
      });
    }
    return this.fallbackPromise;
  }

  private async runWithFallback<T>(invoke: (evaluator: Evaluator) => Promise<T> | T): Promise<T> {
    if (this.disposed) throw evaluatorAbortError('Centipawn runtime evaluator is destroyed');
    if (this.primaryFailed) return invoke(await this.fallbackEvaluator());
    try {
      return await invoke(this.primary);
    } catch (error) {
      return invoke(await this.fallback(error));
    }
  }

  evaluate(board: BoardState, context: EvaluationContext = {}): Promise<Evaluation> {
    return this.runWithFallback((evaluator) => evaluator.evaluate(board, context));
  }

  evaluateBatch(boards: BoardState[], contexts: EvaluationContext[] = []): Promise<Evaluation[]> {
    return this.runWithFallback((evaluator) =>
      evaluator.evaluateBatch
        ? evaluator.evaluateBatch(boards, contexts)
        : Promise.all(boards.map((board, index) => evaluator.evaluate(board, contexts[index]))),
    );
  }

  dispose(): Promise<void> {
    if (this.closePromise) return this.closePromise;
    this.disposed = true;
    this.primaryRetirement ??= disposeEvaluator(this.primary);
    const fallbackRetirement = this.fallbackCreation?.then(disposeEvaluator, () => undefined);
    this.closePromise = Promise.allSettled([this.primaryRetirement, fallbackRetirement]).then(() => undefined);
    return this.closePromise;
  }

  destroy(): void {
    void this.dispose();
  }
}
