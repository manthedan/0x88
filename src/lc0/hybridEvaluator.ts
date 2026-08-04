import type { Lc0HistoryFill } from './encoder112.ts';
import { LC0_DEFAULT_POLICY_TEMPERATURE, type Lc0Evaluation, type Lc0EvaluatorInput } from './onnxEvaluator.ts';
import type {
  Lc0WebEncoderKernelVariant,
  Lc0WebExecutionFootprint,
  Lc0WebHybridEvaluationOptions,
  Lc0WebHybridLegalPriorsBackend,
  Lc0WebHybridRuntimeCreateOptions,
} from './wgslMatmulAddProbe.ts';

export type { Lc0WebEncoderKernelVariant, Lc0WebExecutionFootprint, Lc0WebHybridLegalPriorsBackend } from './wgslMatmulAddProbe.ts';

export type Lc0WebHybridHeadBackend = 'ort' | 'wgsl';
export type Lc0WebHybridWgslBatchMode = 'physical' | 'serial';
export type Lc0WebHybridInputBackend = 'js' | 'wgsl' | 'wasm';

type Lc0WebHybridEvaluatorRuntime = {
  evaluate(input: Lc0EvaluatorInput, options: { historyFill: Lc0HistoryFill; policyTemperature: number }): Promise<Lc0Evaluation>;
  evaluateBatch(inputs: Lc0EvaluatorInput[], options: { historyFill: Lc0HistoryFill; policyTemperature: number }): Promise<Lc0Evaluation[]>;
  evaluateWgslBatchesDeferredReadback(
    batches: Lc0EvaluatorInput[][],
    options: { historyFill: Lc0HistoryFill; policyTemperature: number },
  ): Promise<Lc0Evaluation[][]>;
  executionFootprint(): Lc0WebExecutionFootprint;
  dispose(): Promise<void>;
};

export interface Lc0WebHybridEvaluatorDependencies {
  createRuntime?: (options: Lc0WebHybridRuntimeCreateOptions) => Promise<Lc0WebHybridEvaluatorRuntime>;
}

function clampInteger(value: unknown, fallback: number, min: number, max: number): number {
  const numeric = Number(value ?? fallback);
  const finite = Number.isFinite(numeric) ? numeric : fallback;
  return Math.min(max, Math.max(min, Math.floor(finite)));
}

export class Lc0WebHybridEvaluator {
  readonly packUrl: string;
  readonly layers: number;
  readonly historyFill: Lc0HistoryFill;
  readonly policyTemperature: number;
  readonly verifyShards: boolean;
  readonly headBackend: Lc0WebHybridHeadBackend;
  readonly wgslBatchMode: Lc0WebHybridWgslBatchMode;
  readonly inputBackend: Lc0WebHybridInputBackend;
  readonly legalPriorsBackend: Lc0WebHybridLegalPriorsBackend;
  readonly encoderKernelVariant: Lc0WebEncoderKernelVariant;
  private readonly createRuntime: (options: Lc0WebHybridRuntimeCreateOptions) => Promise<Lc0WebHybridEvaluatorRuntime>;
  private runtimePromise?: Promise<Lc0WebHybridEvaluatorRuntime>;
  private currentRuntime?: Lc0WebHybridEvaluatorRuntime;
  private evaluationQueue: Promise<void> = Promise.resolve();
  private disposed = false;
  private disposePromise?: Promise<void>;

  constructor(options: Omit<Lc0WebHybridEvaluationOptions, 'input'>, dependencies: Lc0WebHybridEvaluatorDependencies = {}) {
    this.packUrl = options.packUrl;
    this.layers = clampInteger(options.layers, 10, 1, 32);
    this.historyFill = options.historyFill ?? 'fen_only';
    this.policyTemperature = options.policyTemperature ?? LC0_DEFAULT_POLICY_TEMPERATURE;
    this.verifyShards = options.verifyShards ?? true;
    this.headBackend = options.headBackend ?? 'ort';
    this.wgslBatchMode = options.wgslBatchMode ?? 'physical';
    this.inputBackend = options.inputBackend ?? 'js';
    this.legalPriorsBackend = options.legalPriorsBackend ?? 'js';
    if (this.legalPriorsBackend === 'gpu' && this.headBackend !== 'wgsl') throw new Error('GPU legal-prior backend requires WGSL heads');
    this.encoderKernelVariant = options.encoderKernelVariant ?? 'hand';
    // Preserve the production worker's code-split boundary: a static probe-module import would eagerly bundle every lab benchmark.
    this.createRuntime =
      dependencies.createRuntime ??
      (async (runtimeOptions) => {
        const { createLc0WebHybridRuntime } = await import('./wgslMatmulAddProbe.ts');
        return createLc0WebHybridRuntime(runtimeOptions);
      });
  }

  private runtime(): Promise<Lc0WebHybridEvaluatorRuntime> {
    if (!this.runtimePromise) {
      const runtimePromise = this.createRuntime({
        packUrl: this.packUrl,
        layers: this.layers,
        historyFill: this.historyFill,
        policyTemperature: this.policyTemperature,
        verifyShards: this.verifyShards,
        headBackend: this.headBackend,
        wgslBatchMode: this.wgslBatchMode,
        inputBackend: this.inputBackend,
        legalPriorsBackend: this.legalPriorsBackend,
        encoderKernelVariant: this.encoderKernelVariant,
      });
      runtimePromise
        .then((runtime) => {
          if (this.runtimePromise === runtimePromise) this.currentRuntime = runtime;
        })
        .catch(() => {
          if (this.runtimePromise === runtimePromise) this.runtimePromise = undefined;
        });
      this.runtimePromise = runtimePromise;
    }
    return this.runtimePromise;
  }

  executionFootprint(): Lc0WebExecutionFootprint | undefined {
    return this.currentRuntime?.executionFootprint();
  }

  private enqueueEvaluation<T>(work: () => Promise<T>): Promise<T> {
    if (this.disposed) return Promise.reject(new Error('LC0 WebGPU hybrid evaluator has been disposed'));
    const run = this.evaluationQueue.then(work, work);
    this.evaluationQueue = run.then(
      () => undefined,
      () => undefined,
    );
    return run;
  }

  async evaluate(input: Lc0EvaluatorInput): Promise<Lc0Evaluation> {
    return this.enqueueEvaluation(async () =>
      (await this.runtime()).evaluate(input, {
        historyFill: this.historyFill,
        policyTemperature: this.policyTemperature,
      }),
    );
  }

  async evaluateBatch(inputs: Lc0EvaluatorInput[]): Promise<Lc0Evaluation[]> {
    return this.enqueueEvaluation(async () => {
      const runtime = await this.runtime();
      return runtime.evaluateBatch(inputs, {
        historyFill: this.historyFill,
        policyTemperature: this.policyTemperature,
      });
    });
  }

  async evaluateBatchSequence(batches: Lc0EvaluatorInput[][]): Promise<Lc0Evaluation[][]> {
    return this.enqueueEvaluation(async () => {
      const runtime = await this.runtime();
      const options = { historyFill: this.historyFill, policyTemperature: this.policyTemperature };
      if (this.headBackend === 'wgsl' && this.wgslBatchMode === 'physical' && batches.length > 1)
        return runtime.evaluateWgslBatchesDeferredReadback(batches, options);
      const out: Lc0Evaluation[][] = [];
      for (const batch of batches) out.push(await runtime.evaluateBatch(batch, options));
      return out;
    });
  }

  async dispose(): Promise<void> {
    if (this.disposePromise) return this.disposePromise;
    this.disposed = true;
    this.disposePromise = (async () => {
      await this.evaluationQueue;
      const runtimePromise = this.runtimePromise;
      this.runtimePromise = undefined;
      this.currentRuntime = undefined;
      if (!runtimePromise) return;
      const runtime = await runtimePromise.catch(() => undefined);
      await runtime?.dispose();
    })();
    return this.disposePromise;
  }
}
