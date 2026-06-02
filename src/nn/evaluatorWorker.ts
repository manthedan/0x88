import type { BoardState } from '../chess/board.ts';
import type { Move } from '../chess/moveCodec.ts';
import type { Evaluation, EvaluationContext, Evaluator } from './evaluator.ts';
import { OnnxEvaluator, type OnnxStudentMeta } from './onnxEvaluator.ts';
import { SquareFormerEvaluator, type SquareFormerMeta } from './squareformerEvaluator.ts';
import { actionValuePuctPolicy, auxPuctPolicy, chooseMove, classicPuctPolicy, montyLitePuctPolicy, progressiveWideningPuctPolicy, type PrincipalVariationEntry, type SearchBudgetMode, type SearchEarlyStop, type SearchPolicy, type SearchPolicyEntry, type SearchStats } from '../search/puct.ts';

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

type WorkerSearchPolicyName = 'classic' | 'av' | 'aux' | 'monty' | 'widen';
type WorkerChooseMoveOptions = {
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
type WorkerSearchResult = { move: Move | null; visits: number; value: number; policy: SearchPolicyEntry[]; principalVariation?: PrincipalVariationEntry[]; stats?: SearchStats };

type WorkerRequest =
  | { id: number; type: 'init'; modelPath: string; meta: OnnxStudentMeta | SquareFormerMeta }
  | { id: number; type: 'evaluateBatch'; boards: BoardState[]; contexts: EvaluationContext[] }
  | { id: number; type: 'chooseMove'; board: BoardState; options: WorkerChooseMoveOptions };

type WorkerResponse =
  | { id: number; type: 'ready' }
  | { id: number; type: 'evaluated'; evaluations: WorkerEvaluation[] }
  | { id: number; type: 'searchResult'; result: WorkerSearchResult }
  | { id: number; type: 'error'; message: string; stack?: string };

let evaluator: Evaluator | null = null;

function mapToEntries(map: Map<number, number> | undefined): WorkerEvalMap | undefined {
  return map ? Array.from(map.entries()) : undefined;
}

function serializeEvaluation(value: Evaluation): WorkerEvaluation {
  const actionValues = mapToEntries(value.actionValues);
  const rankScores = mapToEntries(value.rankScores);
  const regrets = mapToEntries(value.regrets);
  const risks = mapToEntries(value.risks);
  const uncertainties = mapToEntries(value.uncertainties);
  return {
    policy: Array.from(value.policy.entries()),
    wdl: value.wdl,
    ...(value.auxiliaryWdls ? { auxiliaryWdls: value.auxiliaryWdls } : {}),
    ...(actionValues ? { actionValues } : {}),
    ...(rankScores ? { rankScores } : {}),
    ...(regrets ? { regrets } : {}),
    ...(risks ? { risks } : {}),
    ...(uncertainties ? { uncertainties } : {}),
  };
}

function searchPolicyByName(name: WorkerSearchPolicyName | undefined): SearchPolicy {
  if (name === 'av') return actionValuePuctPolicy;
  if (name === 'aux') return auxPuctPolicy;
  if (name === 'monty') return montyLitePuctPolicy;
  if (name === 'widen') return progressiveWideningPuctPolicy;
  return classicPuctPolicy;
}

function post(message: WorkerResponse): void {
  (globalThis as unknown as { postMessage: (message: WorkerResponse) => void }).postMessage(message);
}

async function handleRequest(message: WorkerRequest): Promise<void> {
  try {
    if (message.type === 'init') {
      evaluator = (message.meta.kind === 'squareformer' || message.meta.kind === 'squareformer_v2')
        ? await SquareFormerEvaluator.create(message.modelPath, message.meta as SquareFormerMeta)
        : await OnnxEvaluator.create(message.modelPath, message.meta as OnnxStudentMeta);
      post({ id: message.id, type: 'ready' });
      return;
    }
    if (!evaluator) throw new Error('Worker evaluator is not initialized');
    if (message.type === 'evaluateBatch') {
      const raw = evaluator.evaluateBatch
        ? await evaluator.evaluateBatch(message.boards, message.contexts)
        : await Promise.all(message.boards.map((board, i) => evaluator!.evaluate(board, message.contexts[i] ?? {})));
      post({ id: message.id, type: 'evaluated', evaluations: raw.map(serializeEvaluation) });
      return;
    }
    const { searchPolicyName, ...options } = message.options;
    const result = await chooseMove(message.board, evaluator, { ...options, searchPolicy: searchPolicyByName(searchPolicyName) });
    post({ id: message.id, type: 'searchResult', result: { move: result.move, visits: result.visits, value: result.value, policy: result.policy, ...(result.principalVariation ? { principalVariation: result.principalVariation } : {}), ...(result.stats ? { stats: result.stats } : {}) } });
  } catch (err) {
    post({
      id: message.id,
      type: 'error',
      message: err instanceof Error ? err.message : String(err),
      ...(err instanceof Error && err.stack ? { stack: err.stack } : {}),
    });
  }
}

(globalThis as unknown as { onmessage: ((event: MessageEvent<WorkerRequest>) => void) | null }).onmessage = (event) => {
  void handleRequest(event.data);
};
