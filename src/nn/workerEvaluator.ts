import type { BoardState } from '../chess/board.ts';
import type { Move } from '../chess/moveCodec.ts';
import type { Evaluation, EvaluationContext, Evaluator } from './evaluator.ts';
import type { PrincipalVariationEntry, SearchBudgetMode, SearchEarlyStop, SearchPolicyEntry, SearchStats } from '../search/puct.ts';
import type { OnnxStudentMeta } from './onnxEvaluator.ts';
import type { SquareFormerMeta } from './squareformerEvaluator.ts';

type WorkerEvalMap = [number, number][];
type WorkerEvaluation = {
  policy: WorkerEvalMap;
  wdl: [number, number, number];
  auxiliaryWdls?: Record<string, [number, number, number]>;
  actionValues?: WorkerEvalMap;
  rankScores?: WorkerEvalMap;
  regrets?: WorkerEvalMap;
  risks?: WorkerEvalMap;
  uncertainties?: WorkerEvalMap;
};

export type WorkerSearchPolicyName = 'classic' | 'av' | 'aux' | 'monty' | 'widen';
export type WorkerChooseMoveOptions = {
  visits?: number;
  cpuct?: number;
  fpu?: number;
  temperature?: number;
  cpuctSchedule?: 'constant' | 'lc0-log';
  fpuStrategy?: 'constant' | 'lc0-reduction';
  fpuReduction?: number;
  historyFens?: string[];
  batchSize?: number;
  searchPolicyName?: WorkerSearchPolicyName;
  avWeight?: number;
  rankWeight?: number;
  regretWeight?: number;
  riskWeight?: number;
  uncertaintyWeight?: number;
  includePv?: boolean;
  pvDepth?: number;
  pvSelector?: 'visits' | 'q' | 'puct';
  rootMoves?: Move[];
  yieldEveryMs?: number;
  budgetMode?: SearchBudgetMode;
  maxVisitsMultiplier?: number;
  earlyStop?: SearchEarlyStop;
};
export type WorkerSearchResult = { move: Move | null; visits: number; value: number; policy: SearchPolicyEntry[]; principalVariation?: PrincipalVariationEntry[]; stats?: SearchStats };

type WorkerRequestPayload =
  | { type: 'init'; modelPath: string; meta: OnnxStudentMeta | SquareFormerMeta }
  | { type: 'evaluateBatch'; boards: BoardState[]; contexts: EvaluationContext[] }
  | { type: 'chooseMove'; board: BoardState; options: WorkerChooseMoveOptions };
type WorkerRequest = WorkerRequestPayload & { id: number };

type WorkerResponse =
  | { id: number; type: 'ready' }
  | { id: number; type: 'evaluated'; evaluations: WorkerEvaluation[] }
  | { id: number; type: 'searchResult'; result: WorkerSearchResult }
  | { id: number; type: 'error'; message: string; stack?: string };

function reviveMap(entries: WorkerEvalMap | undefined): Map<number, number> | undefined {
  return entries ? new Map(entries) : undefined;
}

function reviveEvaluation(value: WorkerEvaluation): Evaluation {
  const actionValues = reviveMap(value.actionValues);
  const rankScores = reviveMap(value.rankScores);
  const regrets = reviveMap(value.regrets);
  const risks = reviveMap(value.risks);
  const uncertainties = reviveMap(value.uncertainties);
  return {
    policy: new Map(value.policy),
    wdl: value.wdl,
    ...(value.auxiliaryWdls ? { auxiliaryWdls: value.auxiliaryWdls } : {}),
    ...(actionValues ? { actionValues } : {}),
    ...(rankScores ? { rankScores } : {}),
    ...(regrets ? { regrets } : {}),
    ...(risks ? { risks } : {}),
    ...(uncertainties ? { uncertainties } : {}),
  };
}

export function browserWorkerEvaluatorEnabled(): boolean {
  const params = new URLSearchParams(typeof location === 'undefined' ? '' : location.search);
  const raw = params.get('engineWorker') ?? params.get('evalWorker') ?? params.get('workerEval');
  return raw === '1' || raw === 'true' || raw === 'on' || raw === 'search';
}

function workerVisibleModelPath(modelPath: string): string {
  if (/^(?:[a-z][a-z0-9+.-]*:|\/)/i.test(modelPath)) return modelPath;
  return new URL(modelPath, typeof document === 'undefined' ? globalThis.location?.href : document.baseURI).toString();
}

export class WorkerEvaluator implements Evaluator {
  private worker: Worker;
  private nextId = 1;
  private pending = new Map<number, { resolve: (value: unknown) => void; reject: (reason?: unknown) => void }>();

  private constructor(worker: Worker) {
    this.worker = worker;
    this.worker.onmessage = (event: MessageEvent<WorkerResponse>) => this.handleMessage(event.data);
    this.worker.onerror = (event) => {
      const error = new Error(event.message || 'Worker evaluator failed');
      for (const pending of this.pending.values()) pending.reject(error);
      this.pending.clear();
    };
  }

  static async create(modelPath: string, meta: OnnxStudentMeta | SquareFormerMeta): Promise<WorkerEvaluator> {
    const worker = new Worker(new URL('./evaluatorWorker.ts', import.meta.url), { type: 'module', name: 'tiny-leela-evaluator' });
    const evaluator = new WorkerEvaluator(worker);
    await evaluator.call({ type: 'init', modelPath: workerVisibleModelPath(modelPath), meta });
    return evaluator;
  }

  dispose(): void {
    this.worker.terminate();
    for (const pending of this.pending.values()) pending.reject(new Error('Worker evaluator disposed'));
    this.pending.clear();
  }

  async evaluate(board: BoardState, context: EvaluationContext = {}): Promise<Evaluation> {
    return (await this.evaluateBatch([board], [context]))[0];
  }

  async evaluateBatch(boards: BoardState[], contexts: EvaluationContext[] = []): Promise<Evaluation[]> {
    const evaluations = await this.call({ type: 'evaluateBatch', boards, contexts }) as WorkerEvaluation[];
    return evaluations.map(reviveEvaluation);
  }

  async chooseMove(board: BoardState, options: WorkerChooseMoveOptions): Promise<WorkerSearchResult> {
    return await this.call({ type: 'chooseMove', board, options }) as WorkerSearchResult;
  }

  private call(message: WorkerRequestPayload): Promise<unknown> {
    const id = this.nextId++;
    const request = { id, ...message } as WorkerRequest;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage(request);
    });
  }

  private handleMessage(message: WorkerResponse): void {
    const pending = this.pending.get(message.id);
    if (!pending) return;
    this.pending.delete(message.id);
    if (message.type === 'error') {
      const error = new Error(message.message);
      if (message.stack) error.stack = message.stack;
      pending.reject(error);
    } else if (message.type === 'evaluated') {
      pending.resolve(message.evaluations);
    } else if (message.type === 'searchResult') {
      pending.resolve(message.result);
    } else {
      pending.resolve(undefined);
    }
  }
}
