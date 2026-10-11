/** One running task plus one replaceable pending value. Invalidations cannot
 * preempt GPU work, but make its result stale immediately (before a debounce). */
export class LatestOnlyTask<T> {
  private generation = 0;
  private pending: { value: T; generation: number } | null = null;
  private running = false;
  private disposed = false;

  private readonly run: (value: T, isCurrent: () => boolean) => Promise<void>;
  private readonly onError: (error: unknown) => void;

  constructor(run: (value: T, isCurrent: () => boolean) => Promise<void>, onError: (error: unknown) => void) {
    this.run = run;
    this.onError = onError;
  }

  invalidate(): void {
    this.generation += 1;
    this.pending = null;
  }

  request(value: T): void {
    if (this.disposed) return;
    this.invalidate();
    this.pending = { value, generation: this.generation };
    this.pump();
  }

  dispose(): void {
    this.disposed = true;
    this.invalidate();
  }

  private pump(): void {
    if (this.running || this.disposed || !this.pending) return;
    const request = this.pending;
    this.pending = null;
    this.running = true;
    const isCurrent = () => !this.disposed && request.generation === this.generation;
    void (async () => {
      try {
        await this.run(request.value, isCurrent);
      } catch (error) {
        if (isCurrent()) this.onError(error);
      } finally {
        this.running = false;
        this.pump();
      }
    })().catch(() => undefined);
  }
}
