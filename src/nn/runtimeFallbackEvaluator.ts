import type { BoardState } from '../chess/board.ts';
import type { Evaluation, EvaluationContext, Evaluator } from './evaluator.ts';
import { type BrowserRuntimeAuditDetail, publishBrowserRuntimeAudit } from './runtimeAudit.ts';

function destroyEvaluator(evaluator: Evaluator | undefined): void {
  if (!evaluator) return;
  const destroy = (evaluator as Evaluator & { destroy?: () => void }).destroy;
  if (typeof destroy === 'function') destroy.call(evaluator);
}

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === 'AbortError';
}

function evaluatorAbortError(message: string): Error {
  const error = new Error(message);
  error.name = 'AbortError';
  return error;
}

export class RuntimeFallbackEvaluator implements Evaluator {
  private fallbackPromise?: Promise<Evaluator>;
  private primaryFailed = false;
  private disposed = false;
  private readonly primary: Evaluator;
  private readonly createFallback: () => Promise<Evaluator>;
  private readonly auditBase: BrowserRuntimeAuditDetail;

  constructor(primary: Evaluator, createFallback: () => Promise<Evaluator>, auditBase: BrowserRuntimeAuditDetail) {
    this.primary = primary;
    this.createFallback = createFallback;
    this.auditBase = auditBase;
  }

  private async fallback(error: unknown): Promise<Evaluator> {
    if (isAbortError(error)) throw error;
    if (this.disposed) throw evaluatorAbortError('Centipawn runtime evaluator is destroyed');
    if (!this.primaryFailed) {
      this.primaryFailed = true;
      const fallbackReason = error instanceof Error ? error.message : String(error);
      console.warn('TVMJS WebGPU runtime failed; switching Centipawn to ORT.', { fallbackReason });
      destroyEvaluator(this.primary);
      publishBrowserRuntimeAudit({
        ...this.auditBase,
        resolvedRuntime: 'tvmjs-webgpu-fallback-ort',
        fallbackReason,
      });
    }
    return this.fallbackEvaluator();
  }

  private fallbackEvaluator(): Promise<Evaluator> {
    if (!this.fallbackPromise) {
      const creation = this.createFallback().then((fallback) => {
        if (!this.disposed) return fallback;
        destroyEvaluator(fallback);
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

  async evaluate(board: BoardState, context: EvaluationContext = {}): Promise<Evaluation> {
    if (this.disposed) throw evaluatorAbortError('Centipawn runtime evaluator is destroyed');
    if (this.primaryFailed) return (await this.fallbackEvaluator()).evaluate(board, context);
    try {
      return await this.primary.evaluate(board, context);
    } catch (error) {
      return (await this.fallback(error)).evaluate(board, context);
    }
  }

  async evaluateBatch(boards: BoardState[], contexts: EvaluationContext[] = []): Promise<Evaluation[]> {
    if (this.disposed) throw evaluatorAbortError('Centipawn runtime evaluator is destroyed');
    if (this.primaryFailed) {
      const fallback = await this.fallbackEvaluator();
      return fallback.evaluateBatch
        ? await fallback.evaluateBatch(boards, contexts)
        : await Promise.all(boards.map((board, index) => fallback.evaluate(board, contexts[index])));
    }
    try {
      return this.primary.evaluateBatch
        ? await this.primary.evaluateBatch(boards, contexts)
        : await Promise.all(boards.map((board, index) => this.primary.evaluate(board, contexts[index])));
    } catch (error) {
      const fallback = await this.fallback(error);
      return fallback.evaluateBatch
        ? await fallback.evaluateBatch(boards, contexts)
        : await Promise.all(boards.map((board, index) => fallback.evaluate(board, contexts[index])));
    }
  }

  destroy(): void {
    if (this.disposed) return;
    this.disposed = true;
    destroyEvaluator(this.primary);
    void this.fallbackPromise?.then(destroyEvaluator).catch(() => undefined);
  }
}
