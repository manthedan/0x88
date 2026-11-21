import { type BoardState, boardToFen, type Piece, type PieceRole, parseFen } from '../chess/board.ts';
import { moveToUci } from '../chess/moveCodec.ts';
import { legalMoves } from '../chess/movegen.ts';
import type { OrtDeviceLoss } from '../nn/ortDeviceLoss.ts';
import { type OrtSessionDiagnostics, ortDequantFoldEnabled, requestedOrtRuntimeDiagnosticOptions } from '../nn/ortConfig.ts';
import { resolvePublicAssetUrl } from './assetUrls.ts';
import { disposeOrtWorker } from './disposeOrtWorker.ts';
import { WorkerDeviceLostError, WorkerDeviceRecovery, retryWorkerDeviceLoss } from './workerDeviceRecovery.ts';
import { terminateEngineWorker } from './engineRetirement.ts';
import { MAIA3_ORT_DEFAULTS } from './maia3RuntimeConfig.ts';
import { clearLc0ModelCacheEntry, type Lc0ModelLoadInfo, type Lc0ModelLoadResult, loadLc0ModelForOrt } from './modelCache.ts';
import { isMetadataTrustedModel, isOnnxModelParseError } from './onnxModelRecovery.ts';

const MAIA3_MODEL_CACHE_NAME = 'maia3-browser-models-v1';

/**
 * Default browser model: the weight-only int8 QDQ variant after the
 * einsum,promo graph rewrite (scripts/maia3_graph_rewrite.py). Same weights
 * and I/O as the QDQ8 file and bit-identical on ORT WASM, with half the
 * WebGPU dispatches (docs/maia3_graph_rewrite_2026-11.md). Dynamic batch:
 * Play and critical moments run b1, the Analysis panel b6, the Elo grid b11.
 * The loader falls back to the plain QDQ8 file, then to the upstream fp16
 * file, when a derived artifact is not staged.
 */
const MAIA3_MODEL_PATHS = [
  '/models/maia3/maia3_simplified.qdq8.graph.onnx',
  // Pre-rewrite QDQ8 file (28.1MB vs 45.7MB fp16): the default before the graph rewrite, now the first fallback.
  '/models/maia3/maia3_simplified.qdq8.onnx',
  '/models/maia3/maia3_simplified.onnx',
] as const;
export const MAIA3_DEFAULT_MODEL_URL = resolvePublicAssetUrl(MAIA3_MODEL_PATHS[0]);
/**
 * Load order when no model is requested explicitly, against the page's asset base
 * now: a client-side navigation can change ?assetBase= without reloading this module.
 */
export function maia3ModelFallbackUrls(): string[] {
  return MAIA3_MODEL_PATHS.map(resolvePublicAssetUrl);
}
export const MAIA3_MODEL_MANIFEST_URL = '/models/maia3/manifest.json';
export const MAIA3_MIN_ELO = 600;
export const MAIA3_MAX_ELO = 2600;
export const MAIA3_DEFAULT_ELO = 1500;
export const MAIA3_POLICY_SIZE = 4352;

export type Maia3MoveStyle = 'argmax' | 'sample';

export interface Maia3MovePolicyEntry {
  uci: string;
  prior: number;
  logit: number;
  index: number;
}

export interface Maia3Evaluation {
  fen: string;
  selfElo: number;
  oppoElo: number;
  legalPriors: Maia3MovePolicyEntry[];
  /** Raw `logits_value`, decoded by decodeMaia3Value. Side to move. */
  valueLogits: Maia3Wdl;
  /** Softmax over the value logits. Side to move. */
  valueProbabilities: Maia3Wdl;
}

/**
 * Maia3 value head, side-to-move perspective, as named fields. The model emits
 * `logits_value` as [Loss, Draw, Win] (upstream maia-platform-frontend
 * processOutputsMaia3), the reverse of the [W, D, L] order the LC0 evaluators
 * here use. Named fields mean no caller can index the raw array as W,D,L;
 * decodeMaia3Value is the only place that knows the model order.
 */
export interface Maia3Wdl {
  win: number;
  draw: number;
  loss: number;
}

/** Decode one model-order [Loss, Draw, Win] `logits_value` row. A malformed row decodes to NaN fields. */
export function decodeMaia3Value(row: ArrayLike<number>): { logits: Maia3Wdl; probabilities: Maia3Wdl } {
  const values = Array.from(row);
  if (values.length !== 3 || !values.every(Number.isFinite)) {
    // NaN rather than softmax's uniform fallback, so a broken output is not read as an uncertain position.
    const missing = { win: Number.NaN, draw: Number.NaN, loss: Number.NaN };
    return { logits: missing, probabilities: { ...missing } };
  }
  const [loss, draw, win] = values;
  const [pLoss, pDraw, pWin] = softmax([loss, draw, win]);
  return { logits: { win, draw, loss }, probabilities: { win: pWin, draw: pDraw, loss: pLoss } };
}

/**
 * White-perspective expected score from a Maia3 evaluation — the prediction
 * of the HUMAN game outcome between players of the conditioned ratings, not
 * an engine eval. Mirrors upstream's processOutputsMaia3: P(win) + 0.5·P(draw)
 * for the side to move, flipped for black.
 */
export function maia3WinProbability(evaluation: Maia3Evaluation): number {
  const { win, draw, loss } = evaluation.valueProbabilities;
  if (!Number.isFinite(win) || !Number.isFinite(draw) || !Number.isFinite(loss)) return 0.5;
  const sideToMoveScore = win + 0.5 * draw;
  const blackToMove = evaluation.fen.split(' ')[1] === 'b';
  return blackToMove ? 1 - sideToMoveScore : sideToMoveScore;
}

export interface Maia3ChooseOptions {
  style?: Maia3MoveStyle;
  temperature?: number;
  topP?: number;
}

export interface Maia3Choice {
  move: string | null;
  evaluation: Maia3Evaluation;
}

export type Maia3EvaluateInput = BoardState | string | { board?: BoardState; fen?: string; positions?: BoardState[]; fens?: string[] };

type PendingRequest = {
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
};

function clampElo(elo: number): number {
  if (!Number.isFinite(elo)) return MAIA3_DEFAULT_ELO;
  return Math.max(MAIA3_MIN_ELO, Math.min(MAIA3_MAX_ELO, Math.round(elo)));
}

function normalizeTemperature(temperature: number | undefined): number {
  if (!Number.isFinite(temperature) || Number(temperature) <= 0) return 1;
  return Math.max(0.01, Math.min(5, Number(temperature)));
}

function boardFromInput(input: Maia3EvaluateInput): BoardState {
  if (typeof input === 'string') return parseFen(input);
  if ('squares' in input) return input;
  if (input.board) return input.board;
  if (input.fen) return parseFen(input.fen);
  if (input.positions?.length) return input.positions[input.positions.length - 1];
  if (input.fens?.length) return parseFen(input.fens[input.fens.length - 1]);
  throw new Error('Maia3 evaluation needs a board, FEN, or position history');
}

function verticalMirrorSquare(index: number): number {
  return index ^ 56;
}

function mirrorUciRanks(uci: string): string {
  return `${uci[0]}${9 - Number(uci[1])}${uci[2]}${9 - Number(uci[3])}${uci.slice(4)}`;
}

const PIECE_CHANNEL: Record<Piece, number> = { wp: 0, wn: 1, wb: 2, wr: 3, wq: 4, wk: 5, bp: 6, bn: 7, bb: 8, br: 9, bq: 10, bk: 11 };

/**
 * Maia3's browser ONNX takes side-to-move-normalized square tokens shaped
 * [batch, 64, 12]. For black to move we mirror ranks and swap colors so the
 * model always sees the player to move as white.
 */
export function boardToMaia3Tokens(board: BoardState): Float32Array {
  const tokens = new Float32Array(64 * 12);
  const black = board.turn === 'b';
  for (let square = 0; square < 64; square++) {
    const raw = board.squares[black ? verticalMirrorSquare(square) : square];
    if (!raw) continue;
    const piece: Piece = black ? `${raw[0] === 'w' ? 'b' : 'w'}${raw[1] as PieceRole}` : raw;
    tokens[square * 12 + PIECE_CHANNEL[piece]] = 1;
  }
  return tokens;
}

export function maia3MoveIndex(modelUci: string): number | undefined {
  if (!/^[a-h][1-8][a-h][1-8][qrbn]?$/.test(modelUci)) return undefined;
  const fromFile = modelUci.charCodeAt(0) - 97;
  const fromRank = Number(modelUci[1]) - 1;
  const toFile = modelUci.charCodeAt(2) - 97;
  const toRank = Number(modelUci[3]) - 1;
  if (modelUci.length === 4) return (fromFile + fromRank * 8) * 64 + (toFile + toRank * 8);
  if (fromRank !== 6 || toRank !== 7) return undefined;
  const promo = { q: 0, r: 1, b: 2, n: 3 }[modelUci[4] as 'q' | 'r' | 'b' | 'n'];
  return 4096 + (fromFile * 8 + toFile) * 4 + promo;
}

function legalMaia3Moves(board: BoardState): Array<{ uci: string; modelUci: string; index: number }> {
  const out: Array<{ uci: string; modelUci: string; index: number }> = [];
  for (const move of legalMoves(board)) {
    const uci = moveToUci(move);
    const modelUci = board.turn === 'b' ? mirrorUciRanks(uci) : uci;
    const index = maia3MoveIndex(modelUci);
    if (index !== undefined) out.push({ uci, modelUci, index });
  }
  return out;
}

function softmax(values: number[]): number[] {
  if (!values.length) return [];
  const max = Math.max(...values);
  const exp = values.map((value) => Math.exp(value - max));
  const sum = exp.reduce((acc, value) => acc + value, 0);
  return sum > 0 ? exp.map((value) => value / sum) : values.map(() => 1 / values.length);
}

function legalPolicyFromLogits(
  legal: Array<{ uci: string; modelUci: string; index: number }>,
  logits: Float32Array,
  temperature: number,
): Maia3MovePolicyEntry[] {
  if (!legal.length) return [];
  const probabilities = softmax(legal.map((entry) => Number(logits[entry.index] ?? -Infinity) / temperature));
  return legal
    .map((entry, i) => ({ uci: entry.uci, prior: probabilities[i], logit: Number(logits[entry.index] ?? -Infinity), index: entry.index }))
    .sort((a, b) => b.prior - a.prior);
}

export function chooseFromMaia3Policy(policy: Maia3MovePolicyEntry[], options: Maia3ChooseOptions = {}): string | null {
  if (!policy.length) return null;
  if ((options.style ?? 'sample') === 'argmax') return policy[0].uci;
  const topP = Number.isFinite(options.topP) ? Math.max(0.01, Math.min(1, Number(options.topP))) : 1;
  let pool = policy;
  if (topP < 1) {
    pool = [];
    let cumulative = 0;
    for (const entry of policy) {
      pool.push(entry);
      cumulative += entry.prior;
      if (cumulative >= topP) break;
    }
  }
  const total = pool.reduce((sum, entry) => sum + entry.prior, 0);
  if (!(total > 0)) return pool[0].uci;
  let r = Math.random() * total;
  for (const entry of pool) {
    r -= entry.prior;
    if (r <= 0) return entry.uci;
  }
  return pool[pool.length - 1].uci;
}

export interface Maia3BrowserEvaluatorOptions {
  /** Lab-only opt-out for arbitrary custom models. Unverified bytes are never cached. */
  allowUnverifiedModel?: boolean;
  /** Abort model loading and terminate this evaluator's worker, including pending inference. */
  signal?: AbortSignal;
  /** An owner can await retired initialization even when abort rejects create() immediately. */
  onInitCleanup?: (cleanup: Promise<void>) => void;
  /** Optional owner-supplied CPU reservation; ordinary Human/Play defaults are unchanged. */
  wasmThreads?: number;
  /** Fatal worker failure, including when no inference is in flight. */
  onFailure?: (error: Error) => void;
  deviceRecovery?: WorkerDeviceRecovery;
  onDeviceLossWarning?: (warning: string) => void;
  modelUrl?: string;
  /** Load order when `modelUrl` is not given; defaults to maia3ModelFallbackUrls() at create time. */
  fallbackUrls?: readonly string[];
  selfElo?: number;
  oppoElo?: number;
  onProgress?: (loadedBytes: number, totalBytes?: number) => void;
  /** ORT execution-provider preference; default 'auto' (WebGPU first, wasm fallback). */
  ep?: 'auto' | 'webgpu' | 'webgpu,wasm' | 'wasm';
}

export class Maia3BrowserEvaluator {
  private readonly worker: Worker;
  private nextId = 1;
  private readonly pending = new Map<number, PendingRequest>();
  readonly modelLoad: Lc0ModelLoadResult;
  readonly selfElo: number;
  readonly oppoElo: number;
  readonly inputNames: string[];
  readonly outputNames: string[];
  /** Resolved ORT backend, e.g. 'auto->webgpu' or 'wasm' (from the worker). */
  readonly backend: string;
  readonly ortRuntime: OrtSessionDiagnostics | null;
  deviceLost: OrtDeviceLoss | null = null;
  private abortCleanup?: () => void;
  private terminated = false;
  private readonly onFailure?: (error: Error) => void;
  private readonly loseDevice: (loss: OrtDeviceLoss) => WorkerDeviceLostError;
  private terminalError: Error | null = null;
  private readonly messageListener = (event: MessageEvent) => this.onMessage(event);
  private readonly errorListener = (event: ErrorEvent) => this.fail(new Error(event.message || 'Maia3 worker crashed'));
  private readonly messageErrorListener = () => this.fail(new Error('Maia3 worker message could not be decoded'));

  /** A fatal failure is distinct from owner cancellation and requires a fresh evaluator. */
  get failure(): Error | null {
    return this.terminalError?.name === 'AbortError' ? null : this.terminalError;
  }

  private constructor(
    worker: Worker,
    modelLoad: Lc0ModelLoadResult,
    selfElo: number,
    oppoElo: number,
    names: { inputNames: string[]; outputNames: string[]; backend: string; ortRuntime?: OrtSessionDiagnostics },
    onFailure: ((error: Error) => void) | undefined,
    loseDevice: (loss: OrtDeviceLoss) => WorkerDeviceLostError,
  ) {
    this.worker = worker;
    this.onFailure = onFailure;
    this.loseDevice = loseDevice;
    this.modelLoad = modelLoad;
    this.selfElo = selfElo;
    this.oppoElo = oppoElo;
    this.inputNames = names.inputNames;
    this.outputNames = names.outputNames;
    this.backend = names.backend;
    this.ortRuntime = names.ortRuntime ?? null;
    this.worker.addEventListener('message', this.messageListener);
    this.worker.addEventListener('error', this.errorListener);
    this.worker.addEventListener('messageerror', this.messageErrorListener);
  }

  static async create(
    options: Maia3BrowserEvaluatorOptions = {},
    modelDeps = { load: loadLc0ModelForOrt, clear: clearLc0ModelCacheEntry },
  ): Promise<Maia3BrowserEvaluator> {
    const deviceRecovery = options.deviceRecovery ?? new WorkerDeviceRecovery();
    const loseDevice = (loss: OrtDeviceLoss) => {
      const error = deviceRecovery.lost(loss);
      options.onDeviceLossWarning?.(error.warning('Maia3'));
      return error;
    };
    return retryWorkerDeviceLoss(() => this.createOnce(options, modelDeps, deviceRecovery, loseDevice), options.signal);
  }

  private static async createOnce(
    options: Maia3BrowserEvaluatorOptions,
    modelDeps: { load: typeof loadLc0ModelForOrt; clear: typeof clearLc0ModelCacheEntry },
    deviceRecovery: WorkerDeviceRecovery,
    loseDevice: (loss: OrtDeviceLoss) => WorkerDeviceLostError,
  ): Promise<Maia3BrowserEvaluator> {
    const selfElo = clampElo(options.selfElo ?? MAIA3_DEFAULT_ELO);
    const oppoElo = clampElo(options.oppoElo ?? selfElo);
    const load = (url: string, forceReload = false) =>
      modelDeps.load(url, {
        cache: true,
        forceReload,
        allowUnverified: forceReload ? false : options.allowUnverifiedModel,
        manifestUrl: MAIA3_MODEL_MANIFEST_URL,
        cacheName: MAIA3_MODEL_CACHE_NAME,
        onProgress: options.onProgress,
        signal: options.signal,
      });
    const urls = options.modelUrl ? [options.modelUrl] : (options.fallbackUrls ?? maia3ModelFallbackUrls());
    let modelLoad: Lc0ModelLoadResult;
    for (let i = 0; ; i++) {
      const url = urls[i];
      try {
        modelLoad = await load(url);
        break;
      } catch (error) {
        const next = urls[i + 1];
        if (options.signal?.aborted || !next) throw error;
        console.warn(`Maia3 model ${url} unavailable (${error instanceof Error ? error.message : String(error)}); falling back to ${next}`);
      }
    }
    const create = async (loaded: Lc0ModelLoadResult) => {
      options.signal?.throwIfAborted();
      const worker = new Worker(new URL('./maia3Worker.ts', import.meta.url), { type: 'module', name: 'maia3-evaluator' });
      // Publish the close once, synchronously when an abort starts it: an owner's
      // retirement snapshot taken right after abort() must already include it.
      let closing: Promise<void> | undefined;
      const close = () => {
        if (closing) return closing;
        closing = disposeOrtWorker(worker);
        options.onInitCleanup?.(closing);
        return closing;
      };
      try {
        const names = await Maia3BrowserEvaluator.postInit(
          worker,
          loaded.model,
          deviceRecovery.provider(options.ep),
          options.signal,
          options.wasmThreads,
          loaded.sha256Valid === true ? { url: loaded.logicalUrl ?? loaded.url, metadataTrusted: isMetadataTrustedModel(loaded) } : undefined,
          close,
          loseDevice,
        );
        options.signal?.throwIfAborted();
        const evaluator = new Maia3BrowserEvaluator(worker, loaded, selfElo, oppoElo, names, options.onFailure, loseDevice);
        const abort = () => evaluator.terminate();
        options.signal?.addEventListener('abort', abort, { once: true });
        evaluator.abortCleanup = () => options.signal?.removeEventListener('abort', abort);
        return evaluator;
      } catch (error) {
        if (error instanceof WorkerDeviceLostError) {
          terminateEngineWorker(worker);
          throw error;
        }
        const closed = close();
        if (options.signal?.aborted) void closed.catch(() => undefined);
        else await closed;
        throw error;
      }
    };
    try {
      return await create(modelLoad);
    } catch (error) {
      options.signal?.throwIfAborted();
      const workerFailure = error as { modelLoad?: Lc0ModelLoadInfo; modelParseError?: boolean } | null;
      const failed = workerFailure?.modelLoad ?? modelLoad;
      if (!(workerFailure?.modelParseError ?? isOnnxModelParseError(error)) || !isMetadataTrustedModel(failed)) throw error;
      await modelDeps.clear(failed.url, MAIA3_MODEL_CACHE_NAME);
      options.signal?.throwIfAborted();
      const reloaded = { ...(await load(failed.logicalUrl ?? modelLoad.logicalUrl ?? failed.url, true)), revalidated: true };
      if (reloaded.verification !== 'hashed' || reloaded.sha256Valid !== true) throw new Error('Maia3 model recovery requires freshly SHA-256-verified bytes');
      return create(reloaded); // One retry, in a fresh worker; no recursion or artifact fallback.
    }
  }

  private static postInit(
    worker: Worker,
    model: string | ArrayBuffer,
    ep?: Maia3BrowserEvaluatorOptions['ep'],
    signal?: AbortSignal,
    wasmThreads?: number,
    reloadModel?: { url: string; metadataTrusted: boolean },
    close: () => Promise<void> = () => disposeOrtWorker(worker),
    loseDevice: (loss: OrtDeviceLoss) => WorkerDeviceLostError = (loss) => new WorkerDeviceLostError(loss, 1),
  ): Promise<{ inputNames: string[]; outputNames: string[]; backend: string; ortRuntime?: OrtSessionDiagnostics }> {
    return new Promise((resolve, reject) => {
      const cleanup = () => {
        worker.removeEventListener('message', onMessage);
        worker.removeEventListener('error', onError);
        worker.removeEventListener('messageerror', onMessageError);
        signal?.removeEventListener('abort', onAbort);
      };
      const onAbort = () => {
        cleanup();
        void close().catch(() => undefined);
        reject(new DOMException('Maia3 cancelled', 'AbortError'));
      };
      const onError = (event: ErrorEvent) => {
        cleanup();
        reject(new Error(event.message || 'Maia3 worker failed to initialize'));
      };
      const onMessageError = () => {
        cleanup();
        reject(new Error('Maia3 worker initialization message could not be decoded'));
      };
      const onMessage = (event: MessageEvent) => {
        const data = event.data as {
          type?: string;
          id?: number;
          message?: string;
          inputNames?: string[];
          outputNames?: string[];
          backend?: string;
          ortRuntime?: OrtSessionDiagnostics;
          modelLoad?: Lc0ModelLoadInfo;
          modelParseError?: boolean;
          loss?: OrtDeviceLoss;
        };
        if (data.type === 'deviceLost') {
          cleanup();
          reject(loseDevice(data.loss!));
          return;
        }
        if (data.id !== 0) return;
        if (data.type === 'ready') {
          cleanup();
          resolve({ inputNames: data.inputNames ?? [], outputNames: data.outputNames ?? [], backend: data.backend ?? 'unknown', ortRuntime: data.ortRuntime });
        } else if (data.type === 'error') {
          cleanup();
          reject(
            Object.assign(new Error(data.message ?? 'Maia3 worker initialization failed'), {
              modelLoad: data.modelLoad,
              modelParseError: data.modelParseError,
            }),
          );
        }
      };
      worker.addEventListener('message', onMessage);
      worker.addEventListener('error', onError);
      worker.addEventListener('messageerror', onMessageError);
      signal?.addEventListener('abort', onAbort, { once: true });
      if (signal?.aborted) {
        onAbort();
        return;
      }
      const message = {
        type: 'init',
        id: 0,
        model,
        ep,
        wasmThreads,
        ortDiagnostics: requestedOrtRuntimeDiagnosticOptions(),
        dequantFold: ortDequantFoldEnabled(MAIA3_ORT_DEFAULTS.dequantFold),
        ...(reloadModel ? { modelUrl: reloadModel.url, metadataTrustedModel: reloadModel.metadataTrusted } : {}),
      };
      if (typeof model === 'string') worker.postMessage(message);
      else worker.postMessage(message, [model]);
    });
  }

  private onMessage(event: MessageEvent): void {
    const data = event.data as { type?: string; id?: number; message?: string; loss?: OrtDeviceLoss };
    if (data.type === 'deviceLost') {
      if (this.disposing) return;
      this.deviceLost = data.loss!;
      this.fail(this.loseDevice(this.deviceLost));
      return;
    }
    if (typeof data.id !== 'number') return;
    const pending = this.pending.get(data.id);
    if (!pending) return;
    this.pending.delete(data.id);
    if (data.type === 'error') pending.reject(new Error(data.message ?? 'Maia3 worker error'));
    else pending.resolve(data);
  }

  private post<T>(message: Record<string, unknown>, transfers: Transferable[] = []): Promise<T> {
    if (this.terminalError) return Promise.reject(this.terminalError);
    if (this.disposing) return Promise.reject(new DOMException('Maia3 disposed', 'AbortError'));
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as (value: unknown) => void, reject });
      this.worker.postMessage({ ...message, id }, transfers);
    });
  }

  async evaluate(input: Maia3EvaluateInput, options: { selfElo?: number; oppoElo?: number; temperature?: number } = {}): Promise<Maia3Evaluation> {
    const board = boardFromInput(input);
    const tokens = boardToMaia3Tokens(board);
    const selfElo = clampElo(options.selfElo ?? this.selfElo);
    const oppoElo = clampElo(options.oppoElo ?? this.oppoElo);
    const response = await this.post<{ logitsMove: ArrayBuffer; logitsValue: ArrayBuffer }>(
      { type: 'evaluate', tokens: tokens.buffer, eloSelf: selfElo, eloOppo: oppoElo },
      [tokens.buffer],
    );
    const value = decodeMaia3Value(new Float32Array(response.logitsValue));
    return {
      fen: boardToFen(board),
      selfElo,
      oppoElo,
      legalPriors: legalPolicyFromLogits(legalMaia3Moves(board), new Float32Array(response.logitsMove), normalizeTemperature(options.temperature)),
      valueLogits: value.logits,
      valueProbabilities: value.probabilities,
    };
  }

  async chooseMove(input: Maia3EvaluateInput, options: Maia3ChooseOptions & { selfElo?: number; oppoElo?: number } = {}): Promise<Maia3Choice> {
    const evaluation = await this.evaluate(input, options);
    return { move: chooseFromMaia3Policy(evaluation.legalPriors, options), evaluation };
  }

  /**
   * Evaluate ONE position under many (selfElo, oppoElo) conditions in a
   * single batched run — the rating-inference grid workload. Each condition
   * gets the full legal-policy + value treatment of evaluate().
   */
  async evaluateConditions(
    input: Maia3EvaluateInput,
    conditions: Array<{ selfElo: number; oppoElo: number }>,
    options: { temperature?: number } = {},
  ): Promise<Maia3Evaluation[]> {
    if (!conditions.length) return [];
    const board = boardFromInput(input);
    const fen = boardToFen(board);
    const tokens = boardToMaia3Tokens(board);
    const eloSelfs = conditions.map((c) => clampElo(c.selfElo));
    const eloOppos = conditions.map((c) => clampElo(c.oppoElo));
    const response = await this.post<{ logitsMove: ArrayBuffer; logitsValue: ArrayBuffer }>(
      { type: 'evaluateConditions', tokens: tokens.buffer, eloSelfs, eloOppos },
      [tokens.buffer],
    );
    const logitsMove = new Float32Array(response.logitsMove);
    const logitsValue = new Float32Array(response.logitsValue);
    const legal = legalMaia3Moves(board);
    const temperature = normalizeTemperature(options.temperature);
    return conditions.map((_, i) => {
      const value = decodeMaia3Value(logitsValue.subarray(i * 3, (i + 1) * 3));
      return {
        fen,
        selfElo: eloSelfs[i],
        oppoElo: eloOppos[i],
        legalPriors: legalPolicyFromLogits(legal, logitsMove.subarray(i * MAIA3_POLICY_SIZE, (i + 1) * MAIA3_POLICY_SIZE) as Float32Array, temperature),
        valueLogits: value.logits,
        valueProbabilities: value.probabilities,
      };
    });
  }

  private closing: Promise<void> = Promise.resolve();
  private disposing = false;

  private stop(error: Error, crashed = false, alreadyTerminated = false): void {
    if (this.terminated) return;
    this.terminated = true;
    this.terminalError = error;
    this.abortCleanup?.();
    this.worker.removeEventListener('message', this.messageListener);
    this.worker.removeEventListener('error', this.errorListener);
    this.worker.removeEventListener('messageerror', this.messageErrorListener);
    if (!alreadyTerminated && !this.disposing) {
      if (crashed) terminateEngineWorker(this.worker);
      else {
        this.closing = disposeOrtWorker(this.worker);
        void this.closing.catch(() => undefined);
      }
    }
    for (const pending of this.pending.values()) pending.reject(error);
    this.pending.clear();
  }

  private fail(error: Error): void {
    if (this.terminated) return;
    this.stop(error, true);
    this.onFailure?.(error);
  }

  /** Cancel the owner immediately; release the detached worker with a bounded grace period. */
  terminate(): void {
    this.stop(new DOMException('Maia3 cancelled', 'AbortError'));
  }

  dispose(): Promise<void> {
    if (!this.terminated && !this.disposing) {
      this.disposing = true;
      // Keep already admitted inference replies alive while the worker drains its queue.
      // Explicit terminate()/abort still rejects them immediately, independently of the ack.
      this.closing = disposeOrtWorker(this.worker).then(
        () => this.stop(new DOMException('Maia3 disposed', 'AbortError'), false, true),
        (error: Error) => {
          this.stop(error, false, true);
          throw error;
        },
      );
      void this.closing.catch(() => undefined);
    }
    return this.closing;
  }
}

export const maia3InternalsForTests = {
  clampElo,
  mirrorUciRanks,
  verticalMirrorSquare,
  legalMaia3Moves,
};
