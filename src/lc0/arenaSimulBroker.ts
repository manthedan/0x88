import type { Lc0Evaluation, Lc0EvaluationProvider, Lc0EvaluatorInput } from './onnxEvaluator.ts';

// Research-only singleton-leaf broker. No product caller and no search changes.
interface Pending {
  game: string;
  input: Lc0EvaluatorInput;
  signal: AbortSignal;
  enqueued: number;
  resolve: (value: Lc0Evaluation) => void;
  reject: (error: unknown) => void;
  abort: () => void;
}

export interface SimulBrokerMetrics {
  batches: number;
  evaluations: number;
  batchSizes: Record<number, number>;
  maxResident: number;
  maxWaitMs: number;
  runMs: number;
  cancelled: number;
  perGame: Record<string, number>;
}

function aborted(): Error {
  const error = new Error('Simul game cancelled');
  error.name = 'AbortError';
  return error;
}

/** One outstanding leaf per game, FIFO admission, one awaited physical run at a time. */
export class ArenaSimulBroker {
  private queue: Pending[] = [];
  private active: Pending[] = [];
  private resident = new Set<string>();
  private timer: ReturnType<typeof setTimeout> | undefined;
  private running: Promise<void> | undefined;
  private closed = false;
  private state: SimulBrokerMetrics = {
    batches: 0,
    evaluations: 0,
    batchSizes: {},
    maxResident: 0,
    maxWaitMs: 0,
    runMs: 0,
    cancelled: 0,
    perGame: {},
  };

  private inner: Lc0EvaluationProvider;
  private batchSize: number;
  private capacity: number;
  private waitMs: number;

  constructor(inner: Lc0EvaluationProvider, batchSize: number, capacity = 8, waitMs = 1) {
    this.inner = inner;
    this.batchSize = batchSize;
    this.capacity = capacity;
    this.waitMs = waitMs;
    if (
      !Number.isInteger(batchSize) ||
      batchSize < 1 ||
      batchSize > 8 ||
      !Number.isInteger(capacity) ||
      capacity < batchSize ||
      capacity > 8 ||
      !Number.isFinite(waitMs) ||
      waitMs < 0 ||
      waitMs > 10
    )
      throw new Error('Invalid lab broker bounds');
  }

  metrics(): SimulBrokerMetrics {
    return { ...this.state, batchSizes: { ...this.state.batchSizes }, perGame: { ...this.state.perGame } };
  }

  client(game: string, signal: AbortSignal): Lc0EvaluationProvider {
    const evaluate = (input: Lc0EvaluatorInput) => this.enqueue(game, input, signal);
    return {
      evaluate,
      evaluateBatch: (inputs) => {
        // The gate preserves per-game search batch 1. Never silently split a logical search batch.
        if (inputs.length > 1) return Promise.reject(new Error('Simul requires per-game search batch 1'));
        return inputs.length ? evaluate(inputs[0]).then((value) => [value]) : Promise.resolve([]);
      },
    };
  }

  private enqueue(game: string, input: Lc0EvaluatorInput, signal: AbortSignal): Promise<Lc0Evaluation> {
    if (signal.aborted || this.closed) return Promise.reject(aborted());
    if (this.resident.has(game)) return Promise.reject(new Error('Only one outstanding leaf per game is allowed'));
    if (this.resident.size >= this.capacity) return Promise.reject(new Error('Simul queue capacity exceeded'));
    return new Promise((resolve, reject) => {
      const request: Pending = {
        game,
        input,
        signal,
        enqueued: performance.now(),
        resolve,
        reject,
        abort: () => {
          this.state.cancelled++;
          reject(aborted());
          // An in-flight row cannot be removed from session.run. Retain its slot until readback,
          // but reject immediately and discard its result. Never cancel another game's row.
          const index = this.queue.indexOf(request);
          if (index >= 0) {
            this.queue.splice(index, 1);
            this.release(request);
          }
        },
      };
      this.resident.add(game);
      this.state.maxResident = Math.max(this.state.maxResident, this.resident.size);
      this.queue.push(request);
      signal.addEventListener('abort', request.abort, { once: true });
      this.schedule();
    });
  }

  private release(request: Pending): void {
    request.signal.removeEventListener('abort', request.abort);
    this.resident.delete(request.game);
  }

  private schedule(): void {
    if (this.running || !this.queue.length || this.closed) return;
    if (this.timer !== undefined) clearTimeout(this.timer);
    // The deadline is measured from admission, including time behind the previous run.
    // Timers cannot preempt a GPU run or a busy worker; maxWait is a coalescing bound, not an SLA.
    const delay = this.queue.length >= this.batchSize ? 0 : Math.max(0, this.waitMs - (performance.now() - this.queue[0].enqueued));
    const launch = () => {
      this.timer = undefined;
      this.running = Promise.resolve()
        .then(() => this.flush())
        .finally(() => {
          this.running = undefined;
          this.schedule();
        });
    };
    if (delay === 0) launch();
    else this.timer = setTimeout(launch, delay);
  }

  private async flush(): Promise<void> {
    const batch = this.queue.splice(0, this.batchSize);
    if (!batch.length) return;
    this.active = batch;
    const start = performance.now();
    for (const request of batch) this.state.maxWaitMs = Math.max(this.state.maxWaitMs, start - request.enqueued);
    try {
      const inputs = batch.map((request) => request.input);
      const results = this.inner.evaluateBatch ? await this.inner.evaluateBatch(inputs) : await this.serialFallback(inputs);
      if (results.length !== batch.length) throw new Error('Simul evaluator result cardinality mismatch');
      this.state.batches++;
      this.state.evaluations += batch.length;
      this.state.batchSizes[batch.length] = (this.state.batchSizes[batch.length] ?? 0) + 1;
      for (let i = 0; i < batch.length; i++) {
        const request = batch[i];
        this.release(request);
        if (!request.signal.aborted && !this.closed) {
          this.state.perGame[request.game] = (this.state.perGame[request.game] ?? 0) + 1;
          request.resolve(results[i]);
        }
      }
    } catch (error) {
      for (const request of batch) {
        this.release(request);
        request.reject(error);
      }
    } finally {
      this.state.runMs += performance.now() - start;
      this.active = [];
    }
  }

  private async serialFallback(inputs: Lc0EvaluatorInput[]): Promise<Lc0Evaluation[]> {
    const results: Lc0Evaluation[] = [];
    for (const input of inputs) results.push(await this.inner.evaluate(input));
    return results;
  }

  async close(): Promise<void> {
    this.closed = true;
    if (this.timer !== undefined) clearTimeout(this.timer);
    for (const request of this.queue.splice(0)) {
      this.release(request);
      request.reject(aborted());
    }
    for (const request of this.active) request.reject(aborted());
    await this.running;
  }
}
