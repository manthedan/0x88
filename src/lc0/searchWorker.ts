import { collectOrtRuntimeDiagnostics, setRequestedOrtExecutionProviderForCurrentThread, type OrtExecutionProviderPreference } from '../nn/ortRuntime.ts';
import { runMatch, type BattleEngine, type GameResult, type MatchSummary } from './engineBattle.ts';
import { describeLc0ModelLoad, loadLc0ModelForOrt } from './modelCache.ts';
import { Lc0OnnxEvaluator, type Lc0Evaluation, type Lc0EvaluatorInput } from './onnxEvaluator.ts';
import { Lc0PuctSearcher, type Lc0SearchResult } from './search.ts';

type InitMessage = {
  type: 'init';
  id: number;
  modelUrl: string;
  ep: OrtExecutionProviderPreference;
  cacheModel: boolean;
};

type SearchMessage = {
  type: 'search';
  id: number;
  input: Lc0EvaluatorInput;
  visits: number;
  batchSize?: number;
  multiPv?: number;
};

type EvaluateMessage = {
  type: 'evaluate';
  id: number;
  input: Lc0EvaluatorInput;
};

type BattleMessage = {
  type: 'battle';
  id: number;
  games: number;
  visits: number;
  maxPlies?: number;
  startFen?: string;
};

type CancelMessage = {
  type: 'cancel';
  id: number;
  /** Optional target request id; when omitted, cancels any in-flight search/battle. */
  target?: number;
};

type WorkerRequest = InitMessage | SearchMessage | EvaluateMessage | BattleMessage | CancelMessage;

type SearchWorkerResult = Omit<Lc0SearchResult, 'search'> & {
  stats?: Lc0SearchResult['search']['stats'];
  elapsedMs: number;
  cancelled?: boolean;
};

type WorkerResponse =
  | { type: 'ready'; id: number; backend: string; modelCache: string }
  | { type: 'evaluationResult'; id: number; result: Lc0Evaluation }
  | { type: 'searchResult'; id: number; result: SearchWorkerResult }
  | { type: 'battleProgress'; id: number; game: number; total: number; result: GameResult }
  | { type: 'battleResult'; id: number; result: MatchSummary; elapsedMs: number }
  | { type: 'error'; id: number; error: string };

let evaluator: Lc0OnnxEvaluator | null = null;
let searcher: Lc0PuctSearcher | null = null;
let configuredModelUrl: string | null = null;
/** In-flight search/battle abort controllers keyed by request id, so cancel messages can stop them. */
const activeSearches = new Map<number, AbortController>();
const activeBattles = new Map<number, AbortController>();

function nowMs(): number {
  return typeof performance === 'undefined' ? Date.now() : performance.now();
}

function post(message: WorkerResponse): void {
  self.postMessage(message);
}

async function handleInit(message: InitMessage): Promise<void> {
  setRequestedOrtExecutionProviderForCurrentThread(message.ep);
  const modelLoad = await loadLc0ModelForOrt(message.modelUrl, { cache: message.cacheModel });
  evaluator = await Lc0OnnxEvaluator.create(modelLoad.model);
  searcher = new Lc0PuctSearcher(evaluator);
  configuredModelUrl = message.modelUrl;
  const diagnostics = await collectOrtRuntimeDiagnostics();
  post({ type: 'ready', id: message.id, backend: diagnostics.describe, modelCache: describeLc0ModelLoad(modelLoad) });
}

async function handleEvaluate(message: EvaluateMessage): Promise<void> {
  if (!evaluator) throw new Error('LC0 search worker evaluator is not initialized');
  post({ type: 'evaluationResult', id: message.id, result: await evaluator.evaluate(message.input) });
}

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === 'AbortError';
}

async function handleSearch(message: SearchMessage): Promise<void> {
  if (!searcher) throw new Error('LC0 search worker is not initialized');
  const started = nowMs();
  const controller = new AbortController();
  activeSearches.set(message.id, controller);
  try {
    const result = await searcher.search(message.input, {
      visits: message.visits,
      batchSize: message.batchSize ?? 1,
      multiPv: message.multiPv,
      signal: controller.signal,
      yieldEveryMs: 16,
    });
    post({
      type: 'searchResult',
      id: message.id,
      result: {
        fen: result.fen,
        move: result.move,
        visits: result.visits,
        value: result.value,
        children: result.children,
        pv: result.pv,
        multiPv: result.multiPv,
        stats: result.search.stats,
        elapsedMs: nowMs() - started,
        cancelled: controller.signal.aborted,
      },
    });
  } catch (error) {
    if (!isAbortError(error)) throw error;
    // Cancellation discards the partial tree; report an empty cancelled result.
    post({
      type: 'searchResult',
      id: message.id,
      result: { fen: '', visits: 0, value: 0, children: [], pv: [], cancelled: true, elapsedMs: nowMs() - started },
    });
  } finally {
    activeSearches.delete(message.id);
  }
}

async function handleBattle(message: BattleMessage): Promise<void> {
  if (!searcher || !evaluator) throw new Error('LC0 search worker is not initialized');
  const localSearcher = searcher;
  const localEvaluator = evaluator;
  const started = nowMs();
  const controller = new AbortController();
  activeBattles.set(message.id, controller);
  try {
    const searchEngine: BattleEngine = {
      name: `lc0-search-${message.visits}`,
      async chooseMove(positions) {
        const result = await localSearcher.search({ positions }, { visits: message.visits, signal: controller.signal, yieldEveryMs: 16 });
        return result.move ?? null;
      },
    };
    const policyEngine: BattleEngine = {
      name: 'lc0-policy',
      async chooseMove(positions) {
        return (await localEvaluator.evaluate({ positions })).bestMove ?? null;
      },
    };
    const summary = await runMatch(searchEngine, policyEngine, message.games, {
      maxPlies: message.maxPlies ?? 200,
      startFen: message.startFen,
      signal: controller.signal,
      onGame: (game, total, result) => post({ type: 'battleProgress', id: message.id, game, total, result }),
    });
    post({ type: 'battleResult', id: message.id, result: summary, elapsedMs: nowMs() - started });
  } finally {
    activeBattles.delete(message.id);
  }
}

function handleCancel(message: CancelMessage): void {
  if (message.target !== undefined) {
    activeSearches.get(message.target)?.abort();
    activeBattles.get(message.target)?.abort();
    return;
  }
  for (const controller of activeSearches.values()) controller.abort();
  for (const controller of activeBattles.values()) controller.abort();
}

self.addEventListener('message', (event: MessageEvent<WorkerRequest>) => {
  const message = event.data;
  // Cancellation is synchronous and must not be wrapped in the error reporter,
  // so an abort never posts a spurious error for the cancel request itself.
  if (message.type === 'cancel') {
    handleCancel(message);
    return;
  }
  void (async () => {
    try {
      if (message.type === 'init') await handleInit(message);
      else if (message.type === 'evaluate') {
        if (!configuredModelUrl) throw new Error('LC0 search worker missing model URL');
        await handleEvaluate(message);
      } else if (message.type === 'search') {
        if (!configuredModelUrl) throw new Error('LC0 search worker missing model URL');
        await handleSearch(message);
      } else if (message.type === 'battle') {
        if (!configuredModelUrl) throw new Error('LC0 search worker missing model URL');
        await handleBattle(message);
      }
    } catch (error) {
      post({ type: 'error', id: message.id, error: (error as Error).message });
    }
  })();
});
