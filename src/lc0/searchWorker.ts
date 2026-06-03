import { collectOrtRuntimeDiagnostics, setRequestedOrtExecutionProviderForCurrentThread, type OrtExecutionProviderPreference } from '../nn/ortRuntime.ts';
import { describeLc0ModelLoad, loadLc0ModelForOrt } from './modelCache.ts';
import type { Lc0EvaluatorInput } from './onnxEvaluator.ts';
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
};

type WorkerRequest = InitMessage | SearchMessage;

type SearchWorkerResult = Omit<Lc0SearchResult, 'search'> & {
  stats?: Lc0SearchResult['search']['stats'];
  elapsedMs: number;
};

type WorkerResponse =
  | { type: 'ready'; id: number; backend: string; modelCache: string }
  | { type: 'searchResult'; id: number; result: SearchWorkerResult }
  | { type: 'error'; id: number; error: string };

let searcher: Lc0PuctSearcher | null = null;
let configuredModelUrl: string | null = null;

function nowMs(): number {
  return typeof performance === 'undefined' ? Date.now() : performance.now();
}

function post(message: WorkerResponse): void {
  self.postMessage(message);
}

async function handleInit(message: InitMessage): Promise<void> {
  setRequestedOrtExecutionProviderForCurrentThread(message.ep);
  const modelLoad = await loadLc0ModelForOrt(message.modelUrl, { cache: message.cacheModel });
  searcher = await Lc0PuctSearcher.create(modelLoad.model);
  configuredModelUrl = message.modelUrl;
  const diagnostics = await collectOrtRuntimeDiagnostics();
  post({ type: 'ready', id: message.id, backend: diagnostics.describe, modelCache: describeLc0ModelLoad(modelLoad) });
}

async function handleSearch(message: SearchMessage): Promise<void> {
  if (!searcher) throw new Error('LC0 search worker is not initialized');
  const started = nowMs();
  const result = await searcher.search(message.input, { visits: message.visits, batchSize: message.batchSize ?? 1 });
  post({
    type: 'searchResult',
    id: message.id,
    result: {
      fen: result.fen,
      move: result.move,
      visits: result.visits,
      value: result.value,
      children: result.children,
      stats: result.search.stats,
      elapsedMs: nowMs() - started,
    },
  });
}

self.addEventListener('message', (event: MessageEvent<WorkerRequest>) => {
  const message = event.data;
  void (async () => {
    try {
      if (message.type === 'init') await handleInit(message);
      else if (message.type === 'search') {
        if (!configuredModelUrl) throw new Error('LC0 search worker missing model URL');
        await handleSearch(message);
      }
    } catch (error) {
      post({ type: 'error', id: message.id, error: (error as Error).message });
    }
  })();
});
