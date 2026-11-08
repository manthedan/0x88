import { boardToFen, parseFen } from '../chess/board.ts';
import { moveToUci } from '../chess/moveCodec.ts';
import { legalMoves, makeMove } from '../chess/movegen.ts';
import { lc0AnalysisLines, stockfishAnalysisLines, type Lc0SearchLike } from './analysisFormat.ts';
import { resolvePublicAssetUrl } from './assetUrls.ts';
import { type CriticalLane, type CriticalMoment, type CriticalProvenance, type CriticalResult } from './criticalMoments.ts';
import { cpuEngineHashMbForSurface } from './engineProvision.ts';
import { withEngineTimeout } from './engineTimeout.ts';
import { Maia3BrowserEvaluator, MAIA3_DEFAULT_MODEL_URL } from './maia3.ts';
import type { EngineResourceBroker } from './resourceBroker.ts';
import { StockfishEngine, stockfishFlavorUrl } from './stockfishEngine.ts';

export const CRITICAL_LC0_MODEL = resolvePublicAssetUrl('/models/lc0/t1-256x10-distilled-swa-2432500.batch1.f16.qdq8.onnx');

/** Own workers prevent a navigating interactive tab from resetting a report engine. Created only after consent. */
export function createCriticalEngines(broker: EngineResourceBroker, rating: number | undefined, signal: AbortSignal): { lanes: CriticalLane[]; dispose: () => void } {
  let sf: StockfishEngine | undefined;
  let worker: Worker | undefined;
  let maia: Maia3BrowserEvaluator | undefined;
  let lc0Ready: Promise<{ backend: string }> | undefined;
  let maiaReady: Promise<Maia3BrowserEvaluator> | undefined;
  let sequence = 0;
  const pending = new Map<number, { resolve: (value: any) => void; reject: (error: Error) => void; progress: (text: string) => void }>();
  const ids = ['critical-sf', 'critical-lc0', ...(rating === undefined ? [] : ['critical-maia'])];
  // Reserve CPU capacity even for neural providers: ORT may fall back to WASM.
  for (const id of ids) broker.register(id, { resourceClass: 'cpu', maxThreads: 1 });
  const stopWorker = () => {
    worker?.terminate();
    worker = undefined;
    for (const request of pending.values()) request.reject(new DOMException('Report worker stopped', 'AbortError'));
    pending.clear();
  };
  const dispose = () => {
    signal.removeEventListener('abort', dispose);
    sf?.dispose();
    stopWorker();
    maia?.terminate();
    for (const id of ids) broker.unregister(id);
  };
  signal.addEventListener('abort', dispose, { once: true });

  const post = <T>(message: Record<string, unknown>, progress: (text: string) => void): Promise<T> => {
    if (!worker) return Promise.reject(new Error('Lc0 worker unavailable.'));
    const id = ++sequence;
    return new Promise<T>((resolve, reject) => {
      pending.set(id, { resolve, reject, progress });
      worker!.postMessage({ ...message, id });
    });
  };
  const initLc0 = (progress: (text: string) => void) => {
    if (!lc0Ready) {
      worker = new Worker(new URL('./searchWorker.ts', import.meta.url), { type: 'module', name: 'critical-moments-lc0' });
      worker.addEventListener('message', (event) => {
        const data = event.data;
        const request = pending.get(data.id);
        if (!request) return;
        if (data.type === 'downloadProgress') request.progress(`Downloading ${(data.loadedBytes / 1e6).toFixed(1)} MB`);
        else if (data.type === 'searchProgress') request.progress(`${data.progress.visits ?? 0} / 400 visits`);
        else {
          pending.delete(data.id);
          if (data.type === 'error') request.reject(new Error(data.error ?? 'Lc0 worker failed.'));
          else request.resolve(data);
        }
      });
      worker.addEventListener('error', (event) => {
        for (const request of pending.values()) request.reject(new Error(event.message || 'Lc0 worker failed.'));
        pending.clear();
      });
      lc0Ready = post<{ backend: string }>({ type: 'init', modelUrl: CRITICAL_LC0_MODEL, ep: 'auto', runtime: 'onnx', cacheModel: true, evalCacheEntries: 2048, ortWasmThreads: 1, reportDownloadProgress: true }, progress);
    }
    return lc0Ready;
  };
  const protectedWork = async <T>(id: string, task: (operationSignal: AbortSignal) => Promise<T>): Promise<T> => {
    const lease = await broker.acquire({ engineId: id, signal });
    try {
      return await withEngineTimeout(id, signal, 180000, task);
    } finally { lease.release(); }
  };
  const sfBuild = stockfishFlavorUrl('lite-single');
  const sfProvenance: CriticalProvenance = { engine: 'Stockfish Lite', model: 'Stockfish 19 Lite NNUE (embedded)', build: sfBuild, runtime: 'WASM single-thread', budget: 'depth 12', options: `MultiPV 1; 1 thread; hash ${cpuEngineHashMbForSurface('analysis')} MB` };
  const lc0Provenance: CriticalProvenance = { engine: 'Lc0 Small', model: CRITICAL_LC0_MODEL, build: 't1-256x10-distilled-swa-2432500 · QDQ8', runtime: 'ORT auto (WebGPU / WASM)', budget: '400 visits', options: 'MultiPV 1; batch 1; reuseTree false; WASM threads 1' };
  const lanes: CriticalLane[] = [
    {
      id: 'sf', row: { family: 'sf', variant: 'lite', strength: 12 }, provenance: sfProvenance,
      evaluate: (moment, _signal, progress) => protectedWork('critical-sf', async (operationSignal) => {
        sf ??= new StockfishEngine({ threads: 1, hashMb: cpuEngineHashMbForSurface('analysis') }, sfBuild);
        const engine = sf;
        const abort = () => engine.dispose();
        operationSignal.addEventListener('abort', abort, { once: true });
        try {
          progress('Searching at depth 12');
          const infos = await engine.analyze(moment.history[0], { moves: historyMoves(moment), depth: 12, multipv: 1, signal: operationSignal });
          return { cacheKey: '', provenance: { ...sfProvenance }, line: stockfishAnalysisLines(infos, moment.fen, 'Stockfish Lite')[0] };
        } finally { operationSignal.removeEventListener('abort', abort); }
      }),
    },
    {
      id: 'lc0', row: { family: 'lc0', variant: 'small', strength: 400 }, provenance: lc0Provenance,
      evaluate: (moment, _signal, progress) => protectedWork('critical-lc0', async (operationSignal) => {
        const abort = () => stopWorker();
        operationSignal.addEventListener('abort', abort, { once: true });
        try {
          progress('Loading Lc0 Small');
          const ready = await initLc0(progress);
          const response = await post<{ result: Lc0SearchLike }>({ type: 'search', input: { positions: moment.history.map(parseFen) }, visits: 400, batchSize: 1, multiPv: 1, reuseTree: false, reportProgress: true }, progress);
          return { cacheKey: '', provenance: { ...lc0Provenance, runtime: ready.backend }, line: lc0AnalysisLines(response.result, moment.fen, 'Lc0 Small')[0] };
        } finally { operationSignal.removeEventListener('abort', abort); }
      }),
    },
  ];
  if (rating !== undefined) {
    const provenance: CriticalProvenance = { engine: 'Maia3', model: MAIA3_DEFAULT_MODEL_URL, build: 'maia3_simplified QDQ8 (fp16 fallback)', runtime: 'ORT auto', budget: '1 policy inference', options: `self ${rating}; opponent ${rating}; temperature 1` };
    lanes.push({
      id: 'maia', row: { family: 'lc0', variant: 'critical-maia3', strength: rating }, provenance,
      evaluate: (moment, _signal, progress) => protectedWork('critical-maia', async (operationSignal): Promise<CriticalResult> => {
        // Initialization is shared within this job, never between cancelled jobs.
        maiaReady ??= Maia3BrowserEvaluator.create({ signal: operationSignal, onProgress: (loaded) => progress(`Downloading ${(loaded / 1e6).toFixed(1)} MB`) }).then((evaluator) => (maia = evaluator));
        const abort = () => maia?.terminate();
        operationSignal.addEventListener('abort', abort, { once: true });
        try {
          const evaluator = await maiaReady;
          operationSignal.throwIfAborted();
          progress(`Predicting at rating ${rating}`);
          const [evaluation] = await evaluator.evaluateConditions(moment.fen, [{ selfElo: rating, oppoElo: rating }]);
          return { cacheKey: '', provenance: { ...provenance, model: evaluator.modelLoad.url, runtime: evaluator.backend }, rating, policy: Object.fromEntries(evaluation.legalPriors.map((entry) => [entry.uci, entry.prior])) };
        } finally { operationSignal.removeEventListener('abort', abort); }
      }),
    });
  }
  return { lanes, dispose };
}

// Reconstruct legal history for UCI rather than silently analyzing just the final FEN.
function historyMoves(moment: CriticalMoment): string[] {
  const moves: string[] = [];
  for (let i = 1; i < moment.history.length; i++) {
    const board = parseFen(moment.history[i - 1]);
    const move = legalMoves(board).find((candidate) => boardToFen(makeMove(board, candidate)) === moment.history[i]);
    if (!move) throw new Error('Invalid report history.');
    moves.push(moveToUci(move));
  }
  return moves;
}
