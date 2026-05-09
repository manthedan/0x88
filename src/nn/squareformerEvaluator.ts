import * as ort from 'onnxruntime-web';
import { boardToFen, type BoardState } from '../chess/board.ts';
import { legalMoves } from '../chess/movegen.ts';
import { moveToActionId, type Move } from '../chess/moveCodec.ts';
import type { Evaluation, EvaluationContext, Evaluator } from './evaluator.ts';

const PIECES = '.PNBRQKpnbrqk';
const PROMO_INDEX: Record<string, number> = { n: 0, b: 1, r: 2, q: 3 };

export interface SquareFormerMeta {
  kind: 'squareformer' | 'squareformer_v2';
  variant?: string;
  input_dim: number;
  token_features?: number;
  input_mode?: 'onehot' | 'embedding' | string;
  input_format?: 'float_onehot_rules' | 'compact_uint8_embeddings' | 'compact_uint8_tokens' | string;
  policy_size: number;
  history_plies: number;
  relation_bias?: boolean;
  av_head_exported?: boolean;
  action_value_move_encoding?: 'chessbench_compact_20480' | string;
  max_legal_moves?: number;
  onnx_fixed_legal_moves?: number;
  outputs?: string[];
}

function softmax(xs: ArrayLike<number>): number[] {
  let m = -Infinity;
  for (let i = 0; i < xs.length; i++) if (Number(xs[i]) > m) m = Number(xs[i]);
  const out = Array.from(xs, (x) => Math.exp(Number(x) - m));
  const total = out.reduce((a, b) => a + b, 0) || 1;
  return out.map((x) => x / total);
}

function pieceId(piece: string | null): number {
  if (!piece) return 0;
  const ch = piece[0] === 'w' ? piece[1].toUpperCase() : piece[1];
  return Math.max(0, PIECES.indexOf(ch));
}

function addBoardFeatures(data: Float32Array, board: BoardState, boardIndex: number, planesPerBoard: number, stride: number) {
  const base = boardIndex * planesPerBoard;
  for (let sq = 0; sq < 64; sq++) data[sq * stride + base + pieceId(board.squares[sq])] = 1;
}


function squareformerFloatInput(board: BoardState, meta: SquareFormerMeta, historyFens: string[] = []): Float32Array {
  const history = meta.history_plies;
  const planesPerBoard = 13;
  const inputDim = meta.input_dim;
  const data = new Float32Array(64 * inputDim);
  addBoardFeatures(data, board, 0, planesPerBoard, inputDim);
  for (let h = 0; h < history; h++) {
    if (!historyFens[h]) continue;
    try { addBoardFeatures(data, parseFenCached(historyFens[h]), h + 1, planesPerBoard, inputDim); } catch { /* ignore bad history */ }
  }
  const base = (history + 1) * planesPerBoard;
  for (let sq = 0; sq < 64; sq++) {
    data[sq * inputDim + base + 0] = board.turn === 'w' ? 1 : 0;
    data[sq * inputDim + base + 1] = board.turn === 'b' ? 1 : 0;
    data[sq * inputDim + base + 2] = board.castling.includes('K') ? 1 : 0;
    data[sq * inputDim + base + 3] = board.castling.includes('Q') ? 1 : 0;
    data[sq * inputDim + base + 4] = board.castling.includes('k') ? 1 : 0;
    data[sq * inputDim + base + 5] = board.castling.includes('q') ? 1 : 0;
    data[sq * inputDim + base + 7] = Math.min(255, board.halfmove) / 100;
  }
  if (board.epSquare !== null) data[board.epSquare * inputDim + base + 6] = 1;
  return data;
}

function castleMask(castling: string): number {
  return (castling.includes('K') ? 1 : 0) | (castling.includes('Q') ? 2 : 0) | (castling.includes('k') ? 4 : 0) | (castling.includes('q') ? 8 : 0);
}

function addCompactBoard(data: BigInt64Array, board: BoardState, boardIndex: number, stride: number) {
  for (let sq = 0; sq < 64; sq++) data[sq * stride + boardIndex] = BigInt(pieceId(board.squares[sq]));
}

function squareformerCompactInput(board: BoardState, meta: SquareFormerMeta, historyFens: string[] = []): BigInt64Array {
  const history = meta.history_plies;
  const stride = meta.token_features ?? history + 9;
  const data = new BigInt64Array(64 * stride);
  addCompactBoard(data, board, 0, stride);
  for (let h = 0; h < history; h++) {
    if (!historyFens[h]) continue;
    try { addCompactBoard(data, parseFenCached(historyFens[h]), h + 1, stride); } catch { /* ignore bad history */ }
  }
  const base = history + 1;
  const stm = board.turn === 'w' ? 1n : 2n;
  const flags = BigInt(castleMask(board.castling));
  const half = BigInt(Math.max(0, Math.min(255, Math.trunc(board.halfmove || 0))));
  for (let sq = 0; sq < 64; sq++) {
    data[sq * stride + base + 0] = stm;
    if (base + 1 < stride) data[sq * stride + base + 1] = flags;
    if (base + 2 < stride) data[sq * stride + base + 2] = board.epSquare === sq ? 1n : 0n;
    if (base + 3 < stride) data[sq * stride + base + 3] = half;
    // V2 compact token caches add static square topology tokens after rules.
    if (base + 4 < stride) data[sq * stride + base + 4] = BigInt(Math.floor(sq / 8));
    if (base + 5 < stride) data[sq * stride + base + 5] = BigInt(sq % 8);
    if (base + 6 < stride) data[sq * stride + base + 6] = BigInt((Math.floor(sq / 8) + (sq % 8)) & 1);
    if (base + 7 < stride) data[sq * stride + base + 7] = BigInt(sq);
  }
  return data;
}

function parseFenCached(fen: string): BoardState {
  // Local import cycle avoidance: parseFen is pure but kept separate from hot import line readability.
  const [placement, turn = 'w', castling = '-', ep = '-', half = '0', full = '1'] = fen.trim().split(/\s+/);
  const squares: BoardState['squares'] = Array(64).fill(null);
  const ranks = placement.split('/');
  for (let fenRank = 0; fenRank < 8; fenRank++) {
    let file = 0;
    for (const ch of ranks[fenRank] ?? '') {
      if (/\d/.test(ch)) file += Number(ch);
      else {
        const color = ch === ch.toUpperCase() ? 'w' : 'b';
        squares[file++ + (7 - fenRank) * 8] = `${color}${ch.toLowerCase()}` as never;
      }
    }
  }
  const epSquare = /^[a-h][1-8]$/.test(ep) ? ep.charCodeAt(0) - 97 + (Number(ep[1]) - 1) * 8 : null;
  return { squares, turn: turn as 'w' | 'b', castling, epSquare, halfmove: Number(half), fullmove: Number(full) };
}

function squareformerPolicyIndex(move: Move): number {
  const ft = move.from * 64 + move.to;
  if (move.promotion) return 4096 + ft * 4 + PROMO_INDEX[move.promotion];
  return ft;
}

function legalCandidateInputs(boards: BoardState[], width: number, contexts: EvaluationContext[] = []): { moves: Move[][]; classes: BigInt64Array; width: number } {
  const moves = boards.map((board, i) => contexts[i]?.legalMoves ?? legalMoves(board));
  const classes = new BigInt64Array(boards.length * width);
  for (let i = 0; i < moves.length; i++) {
    for (let j = 0; j < Math.min(width, moves[i].length); j++) classes[i * width + j] = BigInt(squareformerPolicyIndex(moves[i][j]));
  }
  return { moves, classes, width };
}

function isCompactMeta(meta: SquareFormerMeta): boolean {
  return meta.input_mode === 'embedding' || meta.input_format === 'compact_uint8_embeddings' || meta.input_format === 'compact_uint8_tokens';
}

function sessionOptions(): ort.InferenceSession.SessionOptions {
  const threads = Number(globalThis.process?.env?.ORT_INTRA_OP_NUM_THREADS ?? globalThis.process?.env?.ORT_NUM_THREADS ?? '0');
  const opts: ort.InferenceSession.SessionOptions = { graphOptimizationLevel: 'all' };
  if (Number.isFinite(threads) && threads > 0) {
    opts.intraOpNumThreads = Math.floor(threads);
    opts.interOpNumThreads = 1;
  }
  return opts;
}

export class SquareFormerEvaluator implements Evaluator {
  private session: ort.InferenceSession;
  private meta: SquareFormerMeta;
  constructor(session: ort.InferenceSession, meta: SquareFormerMeta) { this.session = session; this.meta = meta; }
  static async create(modelPath: string | Uint8Array | ArrayBuffer, meta: SquareFormerMeta): Promise<SquareFormerEvaluator> {
    return new SquareFormerEvaluator(await ort.InferenceSession.create(modelPath as never, sessionOptions()), meta);
  }
  async evaluate(board: BoardState, context: EvaluationContext = {}): Promise<Evaluation> {
    return (await this.evaluateBatch([board], [context]))[0];
  }

  async evaluateBatch(boards: BoardState[], contexts: EvaluationContext[] = []): Promise<Evaluation[]> {
    if (!boards.length) return [];
    const compact = isCompactMeta(this.meta);
    const stride = compact ? this.meta.token_features ?? this.meta.history_plies + 9 : this.meta.input_dim;
    const one = 64 * stride;
    const input = compact ? new BigInt64Array(boards.length * one) : new Float32Array(boards.length * one);
    for (let i = 0; i < boards.length; i++) {
      const row = compact ? squareformerCompactInput(boards[i], this.meta, contexts[i]?.historyFens ?? []) : squareformerFloatInput(boards[i], this.meta, contexts[i]?.historyFens ?? []);
      (input as BigInt64Array | Float32Array).set(row as never, i * one);
    }
    const shape: [number, number, number] = [boards.length, 64, stride];
    const feeds: Record<string, ort.Tensor> = { tokens: compact ? new ort.Tensor('int64', input as BigInt64Array, shape) : new ort.Tensor('float32', input as Float32Array, shape) };
    const avWidth = Math.max(1, Number(this.meta.onnx_fixed_legal_moves ?? this.meta.max_legal_moves ?? 0));
    const legalInfo = this.meta.av_head_exported && avWidth > 0 ? legalCandidateInputs(boards, avWidth, contexts) : null;
    if (legalInfo) feeds.legal_action_ids = new ort.Tensor('int64', legalInfo.classes, [boards.length, avWidth]);
    const outputs = await this.session.run(feeds);
    const policyRaw = (outputs.policy?.data ?? Object.values(outputs)[0].data) as Float32Array;
    const wdlRaw = (outputs.wdl?.data ?? Object.values(outputs)[1].data) as Float32Array;
    const avRaw = outputs.action_values?.data as Float32Array | undefined;
    const policySize = this.meta.policy_size;
    return boards.map((board, i) => {
      const legal = contexts[i]?.legalMoves ?? legalMoves(board);
      const policyRow = policyRaw.subarray(i * policySize, (i + 1) * policySize);
      const logits = legal.map((move) => Number(policyRow[squareformerPolicyIndex(move)] ?? -100));
      const probs = softmax(logits);
      const policy = new Map<number, number>();
      const actionValues = avRaw ? new Map<number, number>() : undefined;
      legal.forEach((move, j) => {
        const actionId = moveToActionId(move);
        policy.set(actionId, probs[j] ?? 0);
        if (avRaw && j < avWidth) actionValues?.set(actionId, Number(avRaw[i * avWidth + j] ?? 0));
      });
      const wdl = softmax(wdlRaw.subarray(i * 3, i * 3 + 3));
      return { policy, wdl: [wdl[0] ?? 0, wdl[1] ?? 0, wdl[2] ?? 0] as [number, number, number], actionValues };
    });
  }
}
