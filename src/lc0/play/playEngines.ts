// Play's engine host: loads, caches and retires every engine the Play page
// uses. Engines outlive page mounts (verified model bytes and resident CPU
// engines are reused by the next mount), so this state is not mount-scoped.
// No DOM: callers pass an EngineUiPort for notes and download progress.
import type { createBrowserSquareformerRuntimeEvaluator } from '../../nn/browserRuntimeEvaluator.ts';
import type { Evaluator } from '../../nn/evaluator.ts';
import { resolvePublicAssetUrl } from '../assetUrls.ts';
import { acquireBigNetSearcherAsync, type BigNetKey, peekBigNetSearcher, preparePhoneBigNet, releaseUnusedBigNetSearchers } from '../bigNetSessionPool.ts';
import { createCentipawnEvaluatorCache } from '../centipawnEvaluatorCache.ts';
import { inactiveCpuEnginesToEvict, PendingCpuNewGames } from '../cpuEngineResidency.ts';
import type { EngineFamily } from '../engineCatalog.ts';
import { cpuEngineHashMbForSurface, createDefaultBrowserUciEngine, isDefaultBrowserUciFamily } from '../engineProvision.ts';
import { awaitEngineRetirements, trackEngineRetirement } from '../engineRetirement.ts';
import { CachedLc0Evaluator } from '../lc0EvaluatorBase.ts';
import type { createLc0OnnxEvaluatorFromUrl } from '../lc0SmallModelSelection.ts';
import { Maia3BrowserEvaluator, maia3ModelFallbackUrls } from '../maia3.ts';
import { Lc0PuctSearcher } from '../search.ts';
import { StockfishEngine, stockfishFlavorUrl } from '../stockfishEngine.ts';

const loadDefaultLc0Small: typeof createLc0OnnxEvaluatorFromUrl = async (...args) =>
  (await import('../lc0SmallModelSelection.ts')).createLc0OnnxEvaluatorFromUrl(...args);

export type PlayFamily = 'maia3' | EngineFamily;

export interface PlayEngineSpec {
  id: string;
  label: string;
  family: PlayFamily;
  /** Family-specific: Maia Elo, lc0 net key ('small' | 't3' | 'bt4' | 'lqo'), or stockfish kind. */
  variant: string;
}

export interface CpuEngine {
  setOptions(options: { depth?: number; movetimeMs?: number; threads?: number; skillLevel?: number }): void;
  bestMove(fen: string, signal?: AbortSignal, opts?: { moves?: readonly string[] }): Promise<string | null>;
  newGame?(signal?: AbortSignal): Promise<void>;
  dispose?(): void;
}

/** The page's view of an engine load. Late callbacks from a retired load must not reach a newer screen. */
export interface EngineUiPort {
  /** Snapshot the current engine preparation; the check fails once the page or that preparation is gone. */
  current(): () => boolean;
  /** False once the page mount that owns this port is disposed. */
  mounted(): boolean;
  note(text: string, warn?: boolean): void;
  /** Clear the note only while it still shows `text`. */
  clearNote(text: string): void;
  progress(label: string, loadedBytes?: number, totalBytes?: number): void;
  hideProgress(): void;
}

export interface Lc0SmallModel {
  modelUrl: string;
  deviceSelection: boolean;
}

export interface PlayEngineDeps {
  /** Injected so this module stays loadable without the WebGPU runtime graph. */
  loadCentipawn: typeof createBrowserSquareformerRuntimeEvaluator;
  loadLc0Small?: typeof createLc0OnnxEvaluatorFromUrl;
  createMaia3?: typeof Maia3BrowserEvaluator.create;
  createCpuEngine?: (option: PlayEngineSpec) => CpuEngine | Promise<CpuEngine>;
}

const BIG_NET_KEYS: readonly BigNetKey[] = ['bt4', 't3', 'lqo'];

export interface CentipawnAssets {
  onnx: string;
  meta: string;
}

/** Resolved against the page's asset base when called: a client-side navigation can change ?assetBase=. */
export function playCentipawnAssets(): CentipawnAssets {
  return { onnx: resolvePublicAssetUrl('/models/s4fix_v_ema.onnx'), meta: resolvePublicAssetUrl('/models/s4fix_v_ema.meta.json') };
}
const RECLAIM_NOTE = 'Freeing memory…';

function createPlayCpuEngine(option: PlayEngineSpec): CpuEngine | Promise<CpuEngine> {
  switch (option.family) {
    case 'sf':
      return new StockfishEngine(
        { depth: 4, threads: 1, hashMb: cpuEngineHashMbForSurface('play') },
        stockfishFlavorUrl(option.variant === 'lite' ? 'lite-single' : 'single'),
      );
    default:
      if (isDefaultBrowserUciFamily(option.family)) return createDefaultBrowserUciEngine(option.family);
      throw new Error(`unsupported engine family ${option.family}`);
  }
}

function bigNetKeep(option?: PlayEngineSpec): BigNetKey[] {
  return option?.family === 'lc0' && option.variant !== 'small' ? [option.variant as BigNetKey] : [];
}

export function createPlayEngines(deps: PlayEngineDeps) {
  const loadLc0Small = deps.loadLc0Small ?? loadDefaultLc0Small;
  const createMaia3 = deps.createMaia3 ?? ((options) => Maia3BrowserEvaluator.create(options));
  const createCpuEngine = deps.createCpuEngine ?? createPlayCpuEngine;
  // Neural sessions retire on switch/unmount; verified model bytes remain
  // cached independently of native session residency.
  let lc0Searcher: Lc0PuctSearcher | null = null;
  let lc0LoadPromise: Promise<Lc0PuctSearcher> | null = null;
  let lc0SmallOwner: { evaluator: CachedLc0Evaluator | null; abort: AbortController } = { evaluator: null, abort: new AbortController() };
  /** The request the cached Lc0 Small searcher was built from; a remount may ask for another (?model=). */
  let lc0SmallModel: Lc0SmallModel | null = null;
  let maia3Abort: AbortController | null = null;
  let maia3Promise: Promise<Maia3BrowserEvaluator> | null = null;
  /** The model URLs the cached Maia3 was loaded from; a remount may resolve them against another asset base. */
  let maia3Key: string | null = null;
  /** One-line model/cache status shown in the caption once Maia3 has loaded. */
  let maia3Status: string | null = null;
  const centipawn = createCentipawnEvaluatorCache(deps.loadCentipawn);
  const cpuEnginePromises = new Map<string, Promise<CpuEngine>>();
  const pendingCpuNewGames = new PendingCpuNewGames();

  /** Retire every neural engine except `keep`'s; resolves once their workers are gone. */
  function retireUnusedEngines(keep?: PlayEngineSpec): Promise<void> {
    if (keep?.family !== 'centipawn') centipawn.clear();
    if (keep?.family !== 'maia3') retireMaia3();
    if (!(keep?.family === 'lc0' && keep.variant === 'small')) retireLc0Small();
    releaseUnusedBigNetSearchers(bigNetKeep(keep), 0);
    return awaitEngineRetirements();
  }

  function retireMaia3(): Promise<void> | undefined {
    if (!maia3Promise) return undefined;
    const pending = maia3Promise;
    maia3Abort?.abort();
    maia3Abort = null;
    maia3Promise = null;
    maia3Status = null;
    maia3Key = null;
    // An aborted initialization publishes its worker cleanup through onInitCleanup.
    return trackEngineRetirement(
      pending.then(
        (evaluator) => evaluator.dispose(),
        () => undefined,
      ),
    );
  }

  /** Resolves once this session is gone; replacements wait on exactly this, not the global barrier. */
  function retireLc0Small(): Promise<void> | undefined {
    if (!lc0LoadPromise) return undefined;
    const pending = lc0LoadPromise;
    const owner = lc0SmallOwner;
    lc0SmallOwner = { evaluator: null, abort: new AbortController() };
    lc0LoadPromise = null;
    lc0Searcher = null;
    lc0SmallModel = null;
    owner.abort.abort();
    return trackEngineRetirement(
      pending.then(
        async (searcher) => {
          searcher.resetTree();
          await owner.evaluator?.dispose();
          owner.evaluator = null;
        },
        () => undefined,
      ),
    );
  }

  function disposeAllCpuEngines(): void {
    for (const id of cpuEnginePromises.keys()) disposeCpuEngine(id, true);
  }

  function acquireBigNet(key: BigNetKey, signal: AbortSignal, ui: EngineUiPort) {
    const reclaim = preparePhoneBigNet(key, 2048, signal, disposeAllCpuEngines, () => ui.note(RECLAIM_NOTE));
    if (!reclaim) return acquireBigNetSearcherAsync(key, false, signal);
    return reclaim.then(
      () => acquireBigNetSearcherAsync(key, false, signal),
      (error: unknown) => {
        // A cancelled wait (resign, new game, opponent change) must not leave the note on the next screen.
        ui.clearNote(RECLAIM_NOTE);
        throw error;
      },
    );
  }

  function ensureLc0Small(model: Lc0SmallModel, ui: EngineUiPort): Promise<Lc0PuctSearcher> {
    // Another model's session must be gone before this one loads: one Lc0 Small resident at a time.
    const replacing = !!lc0SmallModel && (lc0SmallModel.modelUrl !== model.modelUrl || lc0SmallModel.deviceSelection !== model.deviceSelection);
    // Not awaitEngineRetirements(): if this load is retired in turn, its own retirement
    // waits on this promise, and the global barrier would then wait on itself.
    const predecessor = replacing ? retireLc0Small() : undefined;
    if (lc0Searcher) return Promise.resolve(lc0Searcher);
    if (!lc0LoadPromise) {
      const current = ui.current();
      const owner = lc0SmallOwner;
      lc0SmallModel = model;
      if (current()) ui.note('Loading Lc0 small net…');
      lc0LoadPromise = (async () => {
        // Only a replacement yields here; a first load starts synchronously, before any retirement can abort it.
        if (predecessor) await predecessor;
        // Download progress hides only once the session exists: on Apple WebGPU a
        // failed session reloads the original model for its WASM fallback.
        const { evaluator } = await loadLc0Small(model.modelUrl, {
          deviceSelection: model.deviceSelection,
          load: {
            cache: true,
            requestPersistentStorage: true,
            signal: owner.abort.signal,
            onProgress: (loaded, total) => {
              if (current()) ui.progress('Lc0 small net', loaded, total);
            },
          },
        });
        if (current()) ui.hideProgress();
        owner.evaluator = new CachedLc0Evaluator(evaluator, { maxEntries: 2048 });
        const loaded = new Lc0PuctSearcher(owner.evaluator);
        if (owner === lc0SmallOwner) lc0Searcher = loaded;
        if (current()) ui.note('');
        return loaded;
      })().catch((error: Error) => {
        if (owner === lc0SmallOwner) {
          lc0LoadPromise = null;
          lc0SmallModel = null;
        }
        if (current()) {
          ui.hideProgress();
          ui.note(`Lc0 load failed: ${error.message}`, true);
        }
        throw error;
      });
    }
    return lc0LoadPromise;
  }

  function ensureCentipawn(ui: EngineUiPort, assets: CentipawnAssets = playCentipawnAssets()): Promise<Evaluator> {
    const key = `${assets.onnx}\n${assets.meta}`;
    // One Centipawn resident: a session from another asset base retires.
    centipawn.retain(new Set([key]));
    const starting = !centipawn.has(key);
    const current = ui.current();
    const loading = centipawn.get(
      key,
      {
        spec: { id: 's4fix-v-ema', modelId: 's4fix-v-ema', label: 'Centipawn', onnx: assets.onnx, meta: assets.meta, runtime: 'ort' },
        options: { runtime: 'ort', fallback: true, audit: { surface: 'play' } },
        maxEntries: 4096,
        label: 'centipawn-play:ort',
      },
      () => ui.mounted(),
    );
    if (!starting) return loading;
    if (current()) ui.note('Loading Centipawn…');
    return loading.then(
      (evaluator) => {
        if (current()) ui.note('');
        return evaluator;
      },
      (error: Error) => {
        if (current()) ui.note(`Centipawn load failed: ${error.message}`, true);
        throw error;
      },
    );
  }

  /** `elo` seeds a new evaluator; each move still passes its own Elo. */
  async function ensureMaia3(elo: number, ui: EngineUiPort, modelUrls: readonly string[] = maia3ModelFallbackUrls()): Promise<Maia3BrowserEvaluator> {
    const key = modelUrls.join('\n');
    // Another asset base's session must be gone first; wait on it only, as for Lc0 Small.
    const predecessor = maia3Key !== null && maia3Key !== key ? retireMaia3() : undefined;
    if (maia3Promise) {
      const cached = maia3Promise;
      const owner = maia3Abort;
      const evaluator = await cached;
      if (owner?.signal.aborted) throw new DOMException('Maia3 owner retired', 'AbortError');
      if (!evaluator.failure) return evaluator;
      if (maia3Promise === cached) {
        maia3Promise = null;
        maia3Key = null;
      }
      return ensureMaia3(elo, ui, modelUrls);
    }
    const current = ui.current();
    const owner = new AbortController();
    maia3Abort = owner;
    maia3Key = key;
    maia3Promise = (async () => {
      if (current()) ui.note('Loading Maia3 human model…');
      if (predecessor) await predecessor;
      const evaluator = await createMaia3({
        fallbackUrls: modelUrls,
        selfElo: elo,
        oppoElo: elo,
        signal: owner.signal,
        onInitCleanup: trackEngineRetirement,
        onProgress: (loaded, total) => {
          if (current()) ui.progress('Maia3', loaded, total);
        },
      });
      if (current()) ui.hideProgress();
      const load = evaluator.modelLoad;
      const origin = load.cacheStatus === 'hit' ? 'from cache' : load.cacheStatus === 'miss' ? 'downloaded' : 'loaded';
      const integrity = load.sha256Valid === true ? ', sha256 ✓' : load.sha256Valid === false ? ', sha256 MISMATCH' : '';
      if (owner === maia3Abort) maia3Status = `Model ${origin} (${((load.bytes ?? 0) / 1e6).toFixed(0)}MB${integrity})`;
      if (current()) ui.note('');
      return evaluator;
    })().catch((error: Error) => {
      if (owner === maia3Abort) {
        maia3Promise = null;
        maia3Key = null;
      }
      if (current()) {
        ui.hideProgress();
        ui.note(`Maia3 load failed: ${error.message}`, true);
      }
      throw error;
    });
    return maia3Promise;
  }

  function cpuEngineFor(option: PlayEngineSpec, ui: EngineUiPort): Promise<CpuEngine> {
    const existing = cpuEnginePromises.get(option.id);
    if (existing) {
      retainCpuEngine(option.id, existing);
      return existing;
    }
    const label = option.label;
    const current = ui.current();
    const slot: { current: Promise<CpuEngine> | null } = { current: null };
    const created = (async (): Promise<CpuEngine> => {
      if (current()) ui.note(`Loading ${label} (first use downloads the engine)…`);
      try {
        return createCpuEngine(option);
      } finally {
        if (current()) ui.note('');
      }
    })().catch((error: Error) => {
      if (slot.current && cpuEnginePromises.get(option.id) === slot.current) cpuEnginePromises.delete(option.id);
      if (current()) ui.note(`${label} load failed: ${error.message}`, true);
      throw error;
    });
    slot.current = created;
    retainCpuEngine(option.id, created);
    return created;
  }

  /** The engine for a move: a resident engine first clears the hash it kept from the previous game. */
  async function cpuEngineForMove(option: PlayEngineSpec, signal: AbortSignal, ui: EngineUiPort): Promise<CpuEngine> {
    const engine = await cpuEngineFor(option, ui);
    await pendingCpuNewGames.resetIfPending(option.id, engine, signal);
    return engine;
  }

  /** Keep `id` as the newest resident engine and dispose inactive engines past the cap. */
  function retainCpuEngine(id: string, pending: Promise<CpuEngine>): void {
    cpuEnginePromises.delete(id);
    cpuEnginePromises.set(id, pending);
    evictInactiveCpuEngines(id);
  }

  function evictInactiveCpuEngines(activeId: string): void {
    for (const evictId of inactiveCpuEnginesToEvict([...cpuEnginePromises.keys()], activeId)) disposeCpuEngine(evictId);
  }

  function disposeCpuEngine(id: string, track = false): void {
    const pending = cpuEnginePromises.get(id);
    if (!pending) return;
    cpuEnginePromises.delete(id);
    pendingCpuNewGames.forget(id);
    const disposal = pending.then((engine) => engine.dispose?.());
    if (track) trackEngineRetirement(disposal);
    else void disposal.catch(() => undefined);
  }

  return {
    retireUnusedEngines,
    acquireBigNet,
    ensureLc0Small,
    ensureCentipawn,
    ensureMaia3,
    cpuEngineFor,
    cpuEngineForMove,
    maia3Status: (): string | null => maia3Status,

    /** A game (re)starts from its first position: resident CPU engines reset before their next move. */
    markCpuNewGame(): void {
      pendingCpuNewGames.markAll(cpuEnginePromises.keys());
    },

    /** New game: drop reused search trees and inactive CPU engines past the cap. */
    startNewGame(activeEngineId: string): void {
      lc0Searcher?.resetTree();
      for (const key of BIG_NET_KEYS) {
        const searcher = peekBigNetSearcher(key);
        if (!searcher) continue;
        // Best effort: the same click may release this net (new opponent) and
        // dispose its worker before the reset replies; nothing waits on it.
        if (searcher.loaded) searcher.resetTree().catch(() => undefined);
      }
      evictInactiveCpuEngines(activeEngineId);
    },

    /** The committed opponent changed. A pending restart still belongs to the current game, so call this only on commit. */
    opponentCommitted(previousEngineId: string, nextEngineId: string): void {
      // Stormphrax's large WASM heap must not share residency with the next model.
      if (previousEngineId === 'stormphrax' && nextEngineId !== previousEngineId) disposeCpuEngine(previousEngineId);
    },

    /** The Play page unmounted. `retireAllCpu` is decided at mount time (the URL may already show the next route). */
    unmounted(retireAllCpu: boolean): void {
      void retireUnusedEngines().catch(() => undefined);
      // Stormphrax reserves a 512 MiB WASM heap, so unlike the smaller CPU
      // engines it must not remain cached after this Play route is gone.
      if (retireAllCpu) disposeAllCpuEngines();
      else disposeCpuEngine('stormphrax');
    },
  };
}

export type PlayEngines = ReturnType<typeof createPlayEngines>;
