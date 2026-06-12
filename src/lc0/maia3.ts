import { boardToFen, parseFen, type BoardState, type Piece, type PieceRole } from '../chess/board.ts';
import { legalMoves } from '../chess/movegen.ts';
import { moveToUci } from '../chess/moveCodec.ts';
import { loadLc0ModelForOrt, type Lc0ModelLoadResult } from './modelCache.ts';

export const MAIA3_DEFAULT_MODEL_URL = '/models/maia3/maia3_simplified.onnx';
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
  valueLogits: number[];
  /** Softmax over logits_value when the model exposes WDL-like logits. */
  valueProbabilities: number[];
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

function clampTopP(topP: number | undefined): number {
  if (!Number.isFinite(topP)) return 1;
  return Math.max(0.01, Math.min(1, Number(topP)));
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

function swapColor(piece: Piece): Piece {
  return `${piece[0] === 'w' ? 'b' : 'w'}${piece[1] as PieceRole}`;
}

function verticalMirrorSquare(index: number): number {
  const file = index & 7;
  const rank = index >> 3;
  return file + (7 - rank) * 8;
}

function mirrorUciRanks(uci: string): string {
  const mirrorSquareName = (square: string) => `${square[0]}${9 - Number(square[1])}`;
  return `${mirrorSquareName(uci.slice(0, 2))}${mirrorSquareName(uci.slice(2, 4))}${uci.slice(4)}`;
}

function perspectivePieceAt(board: BoardState, tokenSquare: number): Piece | null {
  if (board.turn === 'w') return board.squares[tokenSquare];
  const piece = board.squares[verticalMirrorSquare(tokenSquare)];
  return piece ? swapColor(piece) : null;
}

const PIECE_CHANNEL: Record<Piece, number> = {
  wp: 0,
  wn: 1,
  wb: 2,
  wr: 3,
  wq: 4,
  wk: 5,
  bp: 6,
  bn: 7,
  bb: 8,
  br: 9,
  bq: 10,
  bk: 11,
};

/**
 * Maia3's browser ONNX takes side-to-move-normalized square tokens shaped
 * [batch, 64, 12]. For black to move we mirror ranks and swap colors so the
 * model always sees the player to move as white.
 */
export function boardToMaia3Tokens(board: BoardState): Float32Array {
  const tokens = new Float32Array(64 * 12);
  for (let square = 0; square < 64; square++) {
    const piece = perspectivePieceAt(board, square);
    if (!piece) continue;
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
  const from = fromFile + fromRank * 8;
  const to = toFile + toRank * 8;
  if (modelUci.length === 4) return from * 64 + to;
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

function legalPolicyFromLogits(board: BoardState, logits: Float32Array, temperature: number): Maia3MovePolicyEntry[] {
  const legal = legalMaia3Moves(board);
  if (!legal.length) return [];
  const scaledLogits = legal.map((entry) => Number(logits[entry.index] ?? -Infinity) / temperature);
  const probabilities = softmax(scaledLogits);
  return legal.map((entry, i) => ({
    uci: entry.uci,
    prior: probabilities[i],
    logit: Number(logits[entry.index] ?? -Infinity),
    index: entry.index,
  })).sort((a, b) => b.prior - a.prior);
}

function topPPool(policy: Maia3MovePolicyEntry[], topP: number): Maia3MovePolicyEntry[] {
  if (topP >= 1) return policy;
  const pool: Maia3MovePolicyEntry[] = [];
  let cumulative = 0;
  for (const entry of policy) {
    pool.push(entry);
    cumulative += entry.prior;
    if (cumulative >= topP) break;
  }
  return pool.length ? pool : policy.slice(0, 1);
}

export function chooseFromMaia3Policy(policy: Maia3MovePolicyEntry[], options: Maia3ChooseOptions = {}): string | null {
  if (!policy.length) return null;
  if ((options.style ?? 'sample') === 'argmax') return policy[0].uci;
  const pool = topPPool(policy, clampTopP(options.topP));
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
  modelUrl?: string;
  selfElo?: number;
  oppoElo?: number;
  onProgress?: (loadedBytes: number, totalBytes?: number) => void;
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

  private constructor(worker: Worker, modelLoad: Lc0ModelLoadResult, selfElo: number, oppoElo: number, names: { inputNames: string[]; outputNames: string[] }) {
    this.worker = worker;
    this.modelLoad = modelLoad;
    this.selfElo = selfElo;
    this.oppoElo = oppoElo;
    this.inputNames = names.inputNames;
    this.outputNames = names.outputNames;
    this.worker.addEventListener('message', (event: MessageEvent) => this.onMessage(event));
  }

  static async create(options: Maia3BrowserEvaluatorOptions = {}): Promise<Maia3BrowserEvaluator> {
    const selfElo = clampElo(options.selfElo ?? MAIA3_DEFAULT_ELO);
    const oppoElo = clampElo(options.oppoElo ?? selfElo);
    const modelLoad = await loadLc0ModelForOrt(options.modelUrl ?? MAIA3_DEFAULT_MODEL_URL, {
      cache: true,
      manifestUrl: MAIA3_MODEL_MANIFEST_URL,
      cacheName: 'maia3-browser-models-v1',
      onProgress: options.onProgress,
    });
    const worker = new Worker(new URL('./maia3Worker.ts', import.meta.url), { type: 'module', name: 'maia3-evaluator' });
    const init = Maia3BrowserEvaluator.postInit(worker, modelLoad.model);
    return new Maia3BrowserEvaluator(worker, modelLoad, selfElo, oppoElo, await init);
  }

  private static postInit(worker: Worker, model: string | ArrayBuffer): Promise<{ inputNames: string[]; outputNames: string[] }> {
    return new Promise((resolve, reject) => {
      const id = 0;
      const cleanup = () => {
        worker.removeEventListener('message', onMessage);
        worker.removeEventListener('error', onError);
      };
      const onError = (event: ErrorEvent) => {
        cleanup();
        reject(new Error(event.message || 'Maia3 worker failed to initialize'));
      };
      const onMessage = (event: MessageEvent) => {
        const data = event.data as { type?: string; id?: number; message?: string; inputNames?: string[]; outputNames?: string[] };
        if (data.id !== id) return;
        if (data.type === 'ready') {
          cleanup();
          resolve({ inputNames: data.inputNames ?? [], outputNames: data.outputNames ?? [] });
        } else if (data.type === 'error') {
          cleanup();
          reject(new Error(data.message ?? 'Maia3 worker initialization failed'));
        }
      };
      worker.addEventListener('message', onMessage);
      worker.addEventListener('error', onError);
      if (typeof model === 'string') worker.postMessage({ type: 'init', id, model });
      else worker.postMessage({ type: 'init', id, model }, [model]);
    });
  }

  private onMessage(event: MessageEvent): void {
    const data = event.data as { type?: string; id?: number; message?: string };
    if (typeof data.id !== 'number') return;
    const pending = this.pending.get(data.id);
    if (!pending) return;
    this.pending.delete(data.id);
    if (data.type === 'error') pending.reject(new Error(data.message ?? 'Maia3 worker error'));
    else pending.resolve(data);
  }

  private post<T>(message: Record<string, unknown>, transfers: Transferable[] = []): Promise<T> {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as (value: unknown) => void, reject });
      this.worker.postMessage({ ...message, id }, transfers);
    });
  }

  async evaluate(input: Maia3EvaluateInput, options: { selfElo?: number; oppoElo?: number; temperature?: number } = {}): Promise<Maia3Evaluation> {
    const board = boardFromInput(input);
    const tokens = boardToMaia3Tokens(board);
    const response = await this.post<{ logitsMove: ArrayBuffer; logitsValue: ArrayBuffer }>(
      {
        type: 'evaluate',
        tokens: tokens.buffer,
        eloSelf: clampElo(options.selfElo ?? this.selfElo),
        eloOppo: clampElo(options.oppoElo ?? this.oppoElo),
      },
      [tokens.buffer],
    );
    const logitsMove = new Float32Array(response.logitsMove);
    const valueLogits = Array.from(new Float32Array(response.logitsValue));
    return {
      fen: boardToFen(board),
      selfElo: clampElo(options.selfElo ?? this.selfElo),
      oppoElo: clampElo(options.oppoElo ?? this.oppoElo),
      legalPriors: legalPolicyFromLogits(board, logitsMove, normalizeTemperature(options.temperature)),
      valueLogits,
      valueProbabilities: softmax(valueLogits),
    };
  }

  async chooseMove(input: Maia3EvaluateInput, options: Maia3ChooseOptions & { selfElo?: number; oppoElo?: number } = {}): Promise<Maia3Choice> {
    const evaluation = await this.evaluate(input, options);
    return { move: chooseFromMaia3Policy(evaluation.legalPriors, options), evaluation };
  }

  async dispose(): Promise<void> {
    try {
      await this.post({ type: 'dispose' });
    } finally {
      this.worker.terminate();
      this.pending.clear();
    }
  }
}

export const maia3InternalsForTests = {
  clampElo,
  mirrorUciRanks,
  verticalMirrorSquare,
  legalMaia3Moves,
};
