// Analysis's engine host: the resource broker, CPU engines, Centipawn
// evaluators and the Analysis view of the pooled big nets. No DOM: progress
// and runtime-info refreshes go through an AnalysisEnginePort. Lc0 Small and
// the Maia3 panel still live in the controller (analysisBrowser.ts).
import type { BoardState } from '../../chess/board.ts';
import type { createBrowserSquareformerRuntimeEvaluator } from '../../nn/browserRuntimeEvaluator.ts';
import type { Evaluator } from '../../nn/evaluator.ts';
import { centipawnModelIdForUrls } from '../../nn/runtimeRegistry.ts';
import { AnalysisCpuSessions, type analysisGamePosition } from '../analysisContinuity.ts';
import { type AnalysisLine, lc0AnalysisLines } from '../analysisFormat.ts';
import type { BerserkVariant } from '../berserkVariants.ts';
import { acquireBigNetSearcherAsync, type BigNetKey, disposeBigNetSearcherNow, peekBigNetSearcher, releaseBigNetSearcher } from '../bigNetSessionPool.ts';
import type { BrowserUciAnalysisOptions, BrowserUciEngine } from '../browserUciEngine.ts';
import { BIG_NETS, type BigNetConfig, type Bt4SearchProgress, type Bt4WorkerSearcher } from '../bt4Engine.ts';
import { createCentipawnEvaluatorCache } from '../centipawnEvaluatorCache.ts';
import { DisposableVariantPool } from '../disposableVariantPool.ts';
import { centipawnEngineLabel, type EngineFamily, type EngineRow, engineResourceProfile, isLc0BigNetVariant } from '../engineCatalog.ts';
import {
  berserkCacheKey,
  cpuEngineHashMbForSurface,
  createBerserkEngine,
  createPlentyChessEngine,
  createRecklessEngine,
  createStormphraxEngine,
  createViridithasEngine,
  plentyChessCacheKey,
  recklessCacheKey,
  stormphraxCacheKey,
  viridithasCacheKey,
} from '../engineProvision.ts';
import { withEngineTimeout } from '../engineTimeout.ts';
import type { GameTree } from '../gameTree.ts';
import type { LargeNetConsent } from '../largeNetConsent.ts';
import type { LoadingProgressItem } from '../loadingProgress.ts';
import { LC0_MODEL_DOWNLOAD_NOT_APPROVED } from '../modelCache.ts';
import { type PlentyChessVariant, plentyChessVariantUnsupportedReason } from '../plentychessVariants.ts';
import type { RecklessVariant } from '../recklessVariants.ts';
import { EngineResourceBroker, type PerformanceDial } from '../resourceBroker.ts';
import { StockfishEngine, stockfishFlavorUrl } from '../stockfishEngine.ts';
import { type StormphraxVariant, stormphraxVariantUnsupportedReason } from '../stormphraxVariants.ts';
import type { ViridithasVariant } from '../viridithasVariants.ts';

/** The page's side of the host. Ids are the loading-progress item ids ('model', `startup:${label}`). */
export interface AnalysisEnginePort {
  /** False once the owning mount is disposed. */
  live(): boolean;
  /** Show (`item`) or hide (`null`) one loading-progress item. */
  loading(id: string, item: LoadingProgressItem | null): void;
  /** A CPU engine's runtime status changed; the page re-renders its runtime info. */
  runtimeInfoChanged(): void;
}

export interface AnalysisCentipawnConfig {
  /** The legacy research override (?tinyModel…): BT4 anneal model with the custom/TVMJS runtimes. */
  legacyOverride: boolean;
  modelUrl: string;
  metaUrl: string;
  legacyModelUrl: string;
  legacyMetaUrl: string;
  manifestUrl: string;
}

/** Row variant keys to resolved variants; the controller owns URL and custom-variant selection. */
export interface AnalysisVariantResolvers {
  reckless(key: string): RecklessVariant;
  viridithas(key: string): ViridithasVariant;
  berserk(key: string): BerserkVariant;
  plentyChess(key: string): PlentyChessVariant;
  stormphrax(key: string): StormphraxVariant;
}

/** The shared big-net pool as Analysis uses it. */
export interface AnalysisBigNetPool {
  acquire(key: BigNetKey, signal?: AbortSignal): Promise<Bt4WorkerSearcher>;
  peek(key: BigNetKey): Bt4WorkerSearcher | null;
  release(key: BigNetKey, idleTtlMs: number): void;
  disposeNow(key: BigNetKey): void;
}

type CentipawnCache = Pick<ReturnType<typeof createCentipawnEvaluatorCache>, 'get' | 'has' | 'retain' | 'clear'>;

export interface AnalysisEngineDeps {
  port: AnalysisEnginePort;
  dial: PerformanceDial;
  centipawn: AnalysisCentipawnConfig;
  /** Injected so this module stays loadable without the WebGPU runtime graph. */
  loadCentipawn: typeof createBrowserSquareformerRuntimeEvaluator;
  variants: AnalysisVariantResolvers;
  /** The visible engine rows; their Centipawn evaluators stay alive. */
  activeRows(): EngineRow[];
  /** Download a CPU engine's assets before its first start. */
  preload(label: string, urls: (string | undefined)[], signal: AbortSignal): Promise<void>;
  createStockfish?: (options: ConstructorParameters<typeof StockfishEngine>[0], url: string) => StockfishEngine;
  bigNets?: AnalysisBigNetPool;
  centipawnCache?: CentipawnCache;
}

export type AnalysisBigNetKey = Extract<BigNetKey, 'bt4' | 't3'>;
export const ANALYSIS_BIG_NET_KEYS: readonly AnalysisBigNetKey[] = ['bt4', 't3'];

/**
 * The big-net session options of an Analysis search (worker eval cache off, as
 * before). The pool shares sessions with Play and Arena, which use 2048 or
 * arenaCacheEntries(); a different size makes init dispose and reload the
 * session. initBigNetsForRun and bigNetLines must both use this.
 */
export const ANALYSIS_BIG_NET_SESSION = { evalCacheEntries: 0 } as const;

/**
 * What a run's big-net init may ask the user. `ask` is the page's download prompt (the host has
 * no DOM) for a cache-only approval whose net was evicted (bigNetFor().cacheOnly); `current` is
 * false once the run is superseded, so a removed row is never asked about.
 */
export interface AnalysisBigNetPrompt {
  signal: AbortSignal;
  current(): boolean;
  ask(config: BigNetConfig, signal: AbortSignal): Promise<LargeNetConsent>;
}

const ANALYSIS_CPU_ENGINE_OPTIONS = { surface: 'analysis' } as const;

const sharedBigNetPool: AnalysisBigNetPool = {
  acquire: (key, signal) => acquireBigNetSearcherAsync(key, false, signal),
  peek: peekBigNetSearcher,
  release: releaseBigNetSearcher,
  disposeNow: disposeBigNetSearcherNow,
};

export function threadedStockfishAvailable(): boolean {
  return typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated === true && typeof SharedArrayBuffer !== 'undefined';
}

export function cpuAnalysisTimeoutMs(row: EngineRow): number {
  if (row.family === 'sf') return Math.max(60000, row.strength * 5000);
  return Math.max(45000, row.strength * 8000);
}

export function cpuColdStartTimeoutMs(row: EngineRow): number {
  if (row.family === 'sf') return 60000;
  return 300000;
}

export function createAnalysisEngines(deps: AnalysisEngineDeps) {
  const { port, variants } = deps;
  const createStockfish = deps.createStockfish ?? ((options, url) => new StockfishEngine(options, url));
  const bigNets = deps.bigNets ?? sharedBigNetPool;
  const config = deps.centipawn;

  // Multi-engine analysis runs all selected engines concurrently. CPU engines
  // are gated by a bounded broker (max 2 concurrent) to avoid saturating the
  // page; GPU engines (Lc0, Centipawn) run without a CPU lease.
  const broker = new EngineResourceBroker({ policy: 'bounded', maxConcurrentCpu: 2, dial: deps.dial });
  const registeredBrokerEngines = new Set<string>();

  function syncBrokerParticipants(rows: EngineRow[]): void {
    const desired = new Map<string, EngineFamily>();
    for (const row of rows) {
      if (engineResourceProfile(row.family).resourceClass !== 'cpu') continue;
      desired.set(row.family === 'sf' ? `sf-${row.variant === 'full' ? 'full' : 'lite'}` : `${row.family}:${row.variant}`, row.family);
    }
    // Register the exact participant set for this run; stale entries skew shares.
    for (const known of registeredBrokerEngines) {
      if (!desired.has(known)) {
        broker.unregister(known);
        registeredBrokerEngines.delete(known);
      }
    }
    for (const [engineId, family] of desired) {
      broker.register(engineId, { ...engineResourceProfile(family) });
      registeredBrokerEngines.add(engineId);
    }
  }

  async function withCpuLease<T>(engineId: string, signal: AbortSignal, task: (lease: { threads: number }) => Promise<T>): Promise<T> {
    const lease = await broker.acquire({ engineId, signal });
    try {
      return await task(lease);
    } finally {
      lease.release();
    }
  }

  // ---- Centipawn ----------------------------------------------------------

  const centipawnCache = deps.centipawnCache ?? createCentipawnEvaluatorCache(deps.loadCentipawn);
  // Row edits are deferred while a run is active, so the run's own row snapshot
  // must keep its Centipawn evaluators alive even if the visible rows changed.
  let centipawnRunKeys = new Set<string>();

  function centipawnRuntimeForVariant(variant: string): 'auto' | 'ort' | 'custom-webgpu' | 'tvmjs-webgpu' {
    if (variant.endsWith('-ort')) return 'ort';
    if (variant.endsWith('-custom')) return config.legacyOverride ? 'custom-webgpu' : 'tvmjs-webgpu';
    return 'auto';
  }

  function centipawnRuntimeFallbackForVariant(variant: string): boolean {
    return !variant.endsWith('-custom');
  }

  function centipawnModel(): { modelId: string; onnx: string; meta: string } {
    return config.legacyOverride
      ? { modelId: 'bt4-anneal-muon-best', onnx: config.legacyModelUrl, meta: config.legacyMetaUrl }
      : {
          modelId: centipawnModelIdForUrls(config.modelUrl, config.metaUrl) ?? config.modelUrl,
          onnx: config.modelUrl,
          meta: config.metaUrl,
        };
  }

  function centipawnEvaluatorCacheKey(variant: string): string {
    const runtime = centipawnRuntimeForVariant(variant);
    const fallback = centipawnRuntimeFallbackForVariant(variant);
    const model = centipawnModel();
    return `${runtime}:${fallback ? 'fallback' : 'strict'}:${model.onnx}:${model.meta}:${config.manifestUrl}`;
  }

  function centipawnEvaluatorKeysForRows(rows: EngineRow[]): Set<string> {
    return new Set(rows.filter((row) => row.family === 'centipawn').map((row) => centipawnEvaluatorCacheKey(row.variant)));
  }

  function activeCentipawnEvaluatorKeys(): Set<string> {
    return new Set([...centipawnEvaluatorKeysForRows(deps.activeRows()), ...centipawnRunKeys]);
  }

  function disposeUnusedCentipawnEvaluators(): void {
    centipawnCache.retain(activeCentipawnEvaluatorKeys());
  }

  function retireAnalysisRunCentipawnKeys(): void {
    if (!centipawnRunKeys.size) return;
    centipawnRunKeys = new Set();
    // Removals made while the snapshot was held could not dispose these evaluators.
    disposeUnusedCentipawnEvaluators();
  }

  /** `multiPv` only labels the runtime audit; each search passes its own. */
  function centipawnEvaluator(variant: string, multiPv: number): Promise<Evaluator> {
    const runtime = centipawnRuntimeForVariant(variant);
    const fallback = centipawnRuntimeFallbackForVariant(variant);
    const model = centipawnModel();
    const key = centipawnEvaluatorCacheKey(variant);
    return centipawnCache.get(
      key,
      {
        spec: {
          id: model.modelId,
          modelId: model.modelId,
          label: centipawnEngineLabel(variant),
          onnx: model.onnx,
          meta: model.meta,
          runtime,
          manifestUrl: config.manifestUrl,
        },
        options: {
          runtime,
          manifestUrl: config.manifestUrl,
          fallback,
          audit: { surface: 'analysis', searchBudget: `multipv=${multiPv}` },
        },
        maxEntries: 4096,
        label: `centipawn-analysis:${runtime}`,
      },
      () => activeCentipawnEvaluatorKeys().has(key),
    );
  }

  // ---- Big nets -----------------------------------------------------------

  /**
   * The served model URL of each big net as of the latest run's start, the way
   * lc0ServedModelUrl holds Lc0 Small's. Every cache and selection key reads this
   * snapshot (engineAnalysisCacheKey), never the live worker, whose
   * servedModelUrl a crash clears mid-run.
   */
  const bigNetServedModel = new Map<AnalysisBigNetKey, string>();
  /** The run whose big-net init is in flight, and the nets it is loading. */
  let bigNetInit: { owner: AbortController; keys: AnalysisBigNetKey[] } | null = null;
  /**
   * Big nets approved on a phone because they were cached ('cached' consent). That approves no
   * download: loading Lc0 Small (or anything else) can evict the net before the run loads it, so
   * initBigNetsForRun loads these cache-only and asks again before downloading.
   */
  const cacheOnlyBigNets = new Set<AnalysisBigNetKey>();
  /** Cache-only nets whose download the user declined: automatic runs fail them without asking again. */
  const declinedBigNetDownloads = new Set<AnalysisBigNetKey>();

  /** Record the page's large-net consent for a row choice; a cache-hit approval loads cache-only. */
  function recordBigNetConsent(key: AnalysisBigNetKey, consent: LargeNetConsent): void {
    if (consent === 'cached') cacheOnlyBigNets.add(key);
    else if (consent === 'download') cacheOnlyBigNets.delete(key);
    // Choosing the net again is the explicit retry of a declined download.
    if (consent) declinedBigNetDownloads.delete(key);
  }

  async function bigNetFor(variant: string, signal?: AbortSignal): Promise<{ config: BigNetConfig; searcher: Bt4WorkerSearcher; cacheOnly: boolean }> {
    const key: AnalysisBigNetKey = variant === 't3' ? 't3' : 'bt4'; // keep in step with engineAnalysisCacheKey
    return { config: BIG_NETS[key], searcher: await bigNets.acquire(key, signal), cacheOnly: cacheOnlyBigNets.has(key) };
  }

  async function bigNetLines(
    variant: string,
    fen: string,
    visits: number,
    positions: BoardState[],
    requestedMultiPv: number,
    servedModelUrl: string,
    signal: AbortSignal | undefined,
    onProgress: (label: string, progress: Bt4SearchProgress) => void,
  ): Promise<AnalysisLine[]> {
    const { config, searcher } = await bigNetFor(variant, signal);
    const label = `Lc0 ${config.name}`;
    // The run's keys name servedModelUrl; never search on another session under them.
    if (!searcher.isInitializedFor(ANALYSIS_BIG_NET_SESSION) || searcher.servedModelUrl !== servedModelUrl) {
      throw new Error(`${label} session changed during analysis; analyze again`);
    }
    // Like initBigNetsForRun: once this run is cancelled (Stop, a new run, or
    // unmount), cancelAnalysisOperation has already cleared the shared model
    // progress slot and the next owner may have shown its own. The cancelled
    // search still settles later, so neither its callbacks nor its finally may
    // touch the slot after that.
    const cancelled = () => signal?.aborted === true || !port.live();
    searcher.onDownloadProgress = (loaded, total) => {
      if (!cancelled()) port.loading('model', { label, loadedBytes: loaded, totalBytes: total, phase: 'Downloading' });
    };
    try {
      const result = await searcher.search(
        { positions },
        {
          visits,
          multiPv: requestedMultiPv,
          evalCacheEntries: ANALYSIS_BIG_NET_SESSION.evalCacheEntries,
          batchSize: config.recommendedBatchSize,
          batchPipelineDepth: config.recommendedPipelineDepth,
          signal,
          onProgress: (progress) => onProgress(label, progress),
        },
      );
      return result.cancelled ? [] : lc0AnalysisLines(result, fen, `Lc0 ${config.name}`);
    } finally {
      if (!cancelled()) port.loading('model', null);
    }
  }

  /**
   * A big net picks its served model (the shipped export or its split-K rewrite)
   * when its worker initializes, and that URL is part of every cache and
   * selection key (engineAnalysisCacheKey). So, like the Lc0 Small backend before
   * analyzeCurrent proceeds, the selected big nets are initialized, with the
   * options the search will pass, before a run takes its keys. Keys taken before
   * that init would not match the session the search runs on (none yet, or a
   * pooled one the search would reload), and the run's results would be discarded
   * or cached under the wrong model. initBigNetsForRun returns the init errors by
   * net, which the run reports as that engine's failure without retrying the
   * download.
   */
  function bigNetsAwaitingInit(rows: EngineRow[]): AnalysisBigNetKey[] {
    const keys = new Set<AnalysisBigNetKey>();
    for (const row of rows) {
      if (row.family !== 'lc0' || !isLc0BigNetVariant(row.variant)) continue;
      const key: AnalysisBigNetKey = row.variant === 't3' ? 't3' : 'bt4';
      if (!bigNets.peek(key)?.isInitializedFor(ANALYSIS_BIG_NET_SESSION)) keys.add(key);
    }
    return [...keys];
  }

  /** `prompt` owns the download prompt a cache-only approval may need; without it such a net fails as not approved. */
  async function initBigNetsForRun(
    keys: AnalysisBigNetKey[],
    owner: AbortController,
    cancelled: () => boolean,
    prompt?: AnalysisBigNetPrompt,
  ): Promise<Map<AnalysisBigNetKey, Error>> {
    bigNetInit = { owner, keys };
    const errors = new Map<AnalysisBigNetKey, Error>();
    try {
      await Promise.all(
        keys.map(async (key) => {
          try {
            const { config, searcher, cacheOnly } = await bigNetFor(key);
            // Checked before each attempt: a stopped run must not reload a pooled
            // session that Play or Arena may have taken back.
            if (cancelled()) return;
            const label = `Lc0 ${config.name}`;
            // A cancelled run no longer owns the shared model progress slot:
            // cancelAnalysisOperation already cleared it, and the next run or mount
            // may have shown its own. The pooled init keeps going, so its late
            // callbacks and this finally must not touch the slot.
            searcher.onDownloadProgress = (loaded, total) => {
              if (!cancelled()) port.loading('model', { label, loadedBytes: loaded, totalBytes: total, phase: 'Downloading' });
            };
            const load = async (options: { evalCacheEntries?: number; noDownload?: boolean }) => {
              await searcher.init(options);
              // An init already in flight for other options resolves with those; reload once.
              if (cancelled() || searcher.isInitializedFor(ANALYSIS_BIG_NET_SESSION)) return;
              await searcher.init(options);
            };
            try {
              if (!cacheOnly) {
                await load(ANALYSIS_BIG_NET_SESSION);
                return;
              }
              try {
                await load({ ...ANALYSIS_BIG_NET_SESSION, noDownload: true });
              } catch (error) {
                if (cancelled() || !(error instanceof Error && error.message.includes(LC0_MODEL_DOWNLOAD_NOT_APPROVED))) throw error;
                const notApproved = () => new Error(`Lc0 ${config.name} is no longer cached and its download was not approved.`);
                if (!prompt || declinedBigNetDownloads.has(key)) throw notApproved();
                if (!prompt.current()) return;
                const consent = await prompt.ask(config, prompt.signal);
                if (cancelled()) return;
                if (consent !== 'download') {
                  declinedBigNetDownloads.add(key);
                  throw notApproved();
                }
                cacheOnlyBigNets.delete(key);
                if (!prompt.current()) return;
                await load(ANALYSIS_BIG_NET_SESSION);
              }
            } finally {
              if (!cancelled()) port.loading('model', null);
            }
          } catch (error) {
            errors.set(key, error instanceof Error ? error : new Error(String(error)));
          }
        }),
      );
    } finally {
      if (bigNetInit?.owner === owner) bigNetInit = null;
    }
    return errors;
  }

  /** After initBigNetsForRun: record the URL of the session each selected big net's search will use. */
  function snapshotBigNetServedModels(rows: EngineRow[]): void {
    for (const row of rows) {
      if (row.family !== 'lc0' || !isLc0BigNetVariant(row.variant)) continue;
      const key: AnalysisBigNetKey = row.variant === 't3' ? 't3' : 'bt4';
      const searcher = bigNets.peek(key);
      if (searcher?.isInitializedFor(ANALYSIS_BIG_NET_SESSION)) bigNetServedModel.set(key, searcher.servedModelUrl);
      else bigNetServedModel.delete(key);
    }
  }

  // ---- CPU engines --------------------------------------------------------

  let stockfishLite: StockfishEngine | null = null;
  let stockfishFull: StockfishEngine | null = null;
  const preparedCpuEngineKeys = new Set<string>();
  const cpuSessions = new AnalysisCpuSessions();

  function getStockfish(kind: 'lite' | 'full'): StockfishEngine {
    // Constructor depth is just a default; each analyze() call passes the row
    // depth and a broker-leased thread count. Analysis defaults to the threaded
    // flavor whenever isolation allows it (flavor is fixed per page lifetime).
    const threaded = threadedStockfishAvailable();
    if (kind === 'lite') {
      if (!stockfishLite)
        stockfishLite = createStockfish(
          { depth: 14, hashMb: cpuEngineHashMbForSurface('analysis') },
          stockfishFlavorUrl(threaded ? 'lite-threaded' : 'lite-single'),
        );
      return stockfishLite;
    }
    if (!stockfishFull)
      stockfishFull = createStockfish({ depth: 14, hashMb: cpuEngineHashMbForSurface('analysis') }, stockfishFlavorUrl(threaded ? 'threaded' : 'single'));
    return stockfishFull;
  }

  function disposeStockfish(kind: 'lite' | 'full'): void {
    const engine = kind === 'lite' ? stockfishLite : stockfishFull;
    if (!engine) return;
    engine.dispose();
    if (kind === 'lite') stockfishLite = null;
    else stockfishFull = null;
    // A recreated worker downloads and starts again; show that, not a warm search.
    preparedCpuEngineKeys.delete(`sf:${kind}`);
  }

  const pools = {
    reckless: new DisposableVariantPool(recklessCacheKey, (variant: RecklessVariant) =>
      createRecklessEngine(variant, () => port.runtimeInfoChanged(), ANALYSIS_CPU_ENGINE_OPTIONS),
    ),
    viridithas: new DisposableVariantPool(viridithasCacheKey, (variant: ViridithasVariant) => createViridithasEngine(variant, {}, ANALYSIS_CPU_ENGINE_OPTIONS)),
    berserk: new DisposableVariantPool(berserkCacheKey, (variant: BerserkVariant) => createBerserkEngine(variant, ANALYSIS_CPU_ENGINE_OPTIONS)),
    plentyChess: new DisposableVariantPool(plentyChessCacheKey, (variant: PlentyChessVariant) => createPlentyChessEngine(variant, ANALYSIS_CPU_ENGINE_OPTIONS)),
    stormphrax: new DisposableVariantPool(stormphraxCacheKey, (variant: StormphraxVariant) => createStormphraxEngine(variant, ANALYSIS_CPU_ENGINE_OPTIONS)),
  };

  function disposeAnalysisCpuEngines(): void {
    preparedCpuEngineKeys.clear();
    disposeStockfish('lite');
    disposeStockfish('full');
    pools.reckless.disposeAll();
    pools.viridithas.disposeAll();
    pools.berserk.disposeAll();
    pools.plentyChess.disposeAll();
    pools.stormphrax.disposeAll();
  }

  async function prepareCpuEngine(
    row: EngineRow,
    engine: object,
    preparationKey: string,
    label: string,
    game: GameTree,
    parentSignal: AbortSignal,
    urls: (string | undefined)[],
    start: (signal: AbortSignal) => Promise<void>,
    beginSearch: () => void,
  ): Promise<void> {
    const prepared = preparedCpuEngineKeys.has(preparationKey);
    // A warm engine reset is part of the next search, not another model load.
    // Show one search activity state immediately instead of briefly flashing the
    // download/startup surface on every position.
    if (prepared) beginSearch();
    await withEngineTimeout(`${label} loading`, parentSignal, cpuColdStartTimeoutMs(row), async (signal) => {
      if (prepared) {
        if (cpuSessions.needsReset(engine, game)) await start(signal);
        if (!signal.aborted) cpuSessions.prepared(engine, game);
        return;
      }
      await deps.preload(label, urls, signal);
      if (signal.aborted) throw new DOMException('Analysis aborted', 'AbortError');
      port.loading(`startup:${label}`, { label, phase: 'Starting engine' });
      try {
        await start(signal);
        if (!signal.aborted) cpuSessions.prepared(engine, game);
        preparedCpuEngineKeys.add(preparationKey);
      } finally {
        port.loading(`startup:${label}`, null);
      }
    });
    if (!prepared) beginSearch();
  }

  async function analyzeCpuPosition(
    engine: Pick<BrowserUciEngine, 'analyze'>,
    row: EngineRow,
    fen: string,
    position: ReturnType<typeof analysisGamePosition>,
    options: BrowserUciAnalysisOptions,
  ) {
    // The experimental Reckless direct ABI only exposes set_fen, not UCI moves.
    // Keep that explicitly FEN-only rather than feeding it a malformed FEN or
    // silently switching the user's runtime. All UCI transports receive history.
    const fenOnly = row.family === 'reckless' && variants.reckless(row.variant).backend === 'browser-api';
    try {
      return await engine.analyze(fenOnly ? fen : position.fen, { ...options, moves: fenOnly ? undefined : position.moves });
    } catch (error) {
      cpuSessions.invalidate(engine);
      throw error;
    } finally {
      if (options.signal?.aborted) cpuSessions.invalidate(engine);
    }
  }

  return {
    broker,
    syncBrokerParticipants,
    withCpuLease,

    centipawnModel,
    centipawnRuntimeForVariant,
    centipawnEvaluatorCacheKey,
    centipawnEvaluator,
    hasCentipawnEvaluator: (key: string): boolean => centipawnCache.has(key),
    /** Pin the run's Centipawn evaluators until the run settles (retireAnalysisRunCentipawnKeys). */
    pinAnalysisRunCentipawnKeys(rows: EngineRow[]): void {
      centipawnRunKeys = centipawnEvaluatorKeysForRows(rows);
    },
    analysisRunCentipawnKeys: (): string[] => [...centipawnRunKeys],
    retireAnalysisRunCentipawnKeys,
    disposeUnusedCentipawnEvaluators,
    disposeCentipawnEvaluators(): void {
      centipawnCache.clear();
    },

    bigNetFor,
    recordBigNetConsent,
    bigNetLines,
    bigNetsAwaitingInit,
    initBigNetsForRun,
    snapshotBigNetServedModels,
    servedBigNetModels: (): Record<string, string> => Object.fromEntries(bigNetServedModel),
    /**
     * Stop ends a cold big-net load instead of leaving it to run on, so the run
     * settles now and Maia3 can reload once the worker retires. Navigation keeps
     * the load: the next run joins it rather than restarting the download.
     * True when `owner`'s load was in flight (the page then says it stopped).
     */
    stopBigNetInit(owner: AbortController | null): boolean {
      const init = bigNetInit;
      if (!init || init.owner !== owner) return false;
      for (const key of init.keys) if (!bigNets.peek(key)?.loaded) bigNets.disposeNow(key);
      return true;
    },
    /** Cancel and release the big nets no row in `rows` uses. */
    releaseUnusedBigNets(rows: EngineRow[]): void {
      for (const key of ANALYSIS_BIG_NET_KEYS) {
        if (rows.some((r) => r.family === 'lc0' && r.variant === key)) continue;
        bigNets.peek(key)?.cancel();
        // Match Play: an unused net must not overlap its replacement for three minutes.
        bigNets.release(key, 0);
      }
    },
    releaseBigNets(): void {
      for (const key of ANALYSIS_BIG_NET_KEYS) {
        bigNets.peek(key)?.cancel();
        bigNets.release(key, 0);
      }
    },

    cpuSessions,
    pools,
    getStockfish,
    getRecklessFor: (variantKey: string) => pools.reckless.getOrCreate(variants.reckless(variantKey)),
    getViridithasFor: (variantKey: string) => pools.viridithas.getOrCreate(variants.viridithas(variantKey)),
    getBerserkFor: (variantKey: string) => pools.berserk.getOrCreate(variants.berserk(variantKey)),
    getPlentyChessFor(variantKey: string) {
      const variant = variants.plentyChess(variantKey);
      const unsupportedReason = plentyChessVariantUnsupportedReason(variant);
      if (unsupportedReason) throw new Error(`${variant.label} ${unsupportedReason}.`);
      return pools.plentyChess.getOrCreate(variant);
    },
    getStormphraxFor(variantKey: string) {
      const variant = variants.stormphrax(variantKey);
      const unsupportedReason = stormphraxVariantUnsupportedReason(variant);
      if (unsupportedReason) throw new Error(`${variant.label} ${unsupportedReason}.`);
      return pools.stormphrax.getOrCreate(variant);
    },
    prepareCpuEngine,
    analyzeCpuPosition,
    disposeAnalysisCpuEngines,
    /** Dispose CPU engines no row uses; Stockfish also stays for `stockfishRows` (a running review). */
    disposeUnusedCpuEngines(activeRows: EngineRow[], stockfishRows: EngineRow[]): void {
      for (const kind of ['lite', 'full'] as const) {
        if (stockfishRows.some((row) => row.family === 'sf' && (row.variant === 'full' ? 'full' : 'lite') === kind)) continue;
        disposeStockfish(kind);
      }
      pools.reckless.retain(activeRows.filter((row) => row.family === 'reckless').map((row) => variants.reckless(row.variant)));
      pools.viridithas.retain(activeRows.filter((row) => row.family === 'viridithas').map((row) => variants.viridithas(row.variant)));
      pools.berserk.retain(activeRows.filter((row) => row.family === 'berserk').map((row) => variants.berserk(row.variant)));
      pools.plentyChess.retain(activeRows.filter((row) => row.family === 'plentychess').map((row) => variants.plentyChess(row.variant)));
      pools.stormphrax.retain(activeRows.filter((row) => row.family === 'stormphrax').map((row) => variants.stormphrax(row.variant)));
    },
    /** Runtime teardown: the run's Centipawn pins and warm-CPU marks lapse without disposing anything yet. */
    dropRunState(): void {
      centipawnRunKeys = new Set();
      preparedCpuEngineKeys.clear();
    },
  };
}

export type AnalysisEngines = ReturnType<typeof createAnalysisEngines>;
