import * as ort from 'onnxruntime-web';
import { boardToFen, type BoardState } from '../chess/board.ts';
import { legalMoves } from '../chess/movegen.ts';
import { moveToActionId, type Move } from '../chess/moveCodec.ts';
import type { Evaluation, EvaluationContext, Evaluator } from './evaluator.ts';

const PIECES = '.PNBRQKpnbrqk';
const PROMO_INDEX: Record<string, number> = { n: 0, b: 1, r: 2, q: 3 };

export interface SquareFormerMeta {
  kind: 'squareformer';
  variant: string;
  input_dim: number;
  token_features?: number;
  input_format?: 'float_onehot_rules' | 'compact_uint8_embeddings' | string;
  policy_size: number;
  history_plies: number;
  relation_bias?: boolean;
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

function addBoardFeatures(data: Float32Array, board: BoardState, boardIndex: number, planesPerBoard: number) {
  const base = boardIndex * planesPerBoard;
  for (let sq = 0; sq < 64; sq++) data[sq * dataStride + base + pieceId(board.squares[sq])] = 1;
}

let dataStride = 0;

function squareformerFloatInput(board: BoardState, meta: SquareFormerMeta, historyFens: string[] = []): Float32Array {
  const history = meta.history_plies;
  const planesPerBoard = 13;
  const inputDim = meta.input_dim;
  dataStride = inputDim;
  const data = new Float32Array(64 * inputDim);
  addBoardFeatures(data, board, 0, planesPerBoard);
  for (let h = 0; h < history; h++) {
    if (!historyFens[h]) continue;
    try { addBoardFeatures(data, parseFenCached(historyFens[h]), h + 1, planesPerBoard); } catch { /* ignore bad history */ }
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
    data[sq * stride + base + 1] = flags;
    data[sq * stride + base + 2] = board.epSquare === sq ? 1n : 0n;
    data[sq * stride + base + 3] = half;
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

export class SquareFormerEvaluator implements Evaluator {
  private session: ort.InferenceSession;
  private meta: SquareFormerMeta;
  constructor(session: ort.InferenceSession, meta: SquareFormerMeta) { this.session = session; this.meta = meta; }
  static async create(modelPath: string | Uint8Array | ArrayBuffer, meta: SquareFormerMeta): Promise<SquareFormerEvaluator> {
    return new SquareFormerEvaluator(await ort.InferenceSession.create(modelPath as never), meta);
  }
  async evaluate(board: BoardState, context: EvaluationContext = {}): Promise<Evaluation> {
    const compact = this.meta.input_format === 'compact_uint8_embeddings';
    const input = compact ? squareformerCompactInput(board, this.meta, context.historyFens ?? []) : squareformerFloatInput(board, this.meta, context.historyFens ?? []);
    const shape: [number, number, number] = compact ? [1, 64, this.meta.token_features ?? this.meta.history_plies + 9] : [1, 64, this.meta.input_dim];
    const outputs = await this.session.run({ tokens: compact ? new ort.Tensor('int64', input as BigInt64Array, shape) : new ort.Tensor('float32', input as Float32Array, shape) });
    const policyRaw = (outputs.policy?.data ?? Object.values(outputs)[0].data) as Float32Array;
    const wdlRaw = (outputs.wdl?.data ?? Object.values(outputs)[1].data) as Float32Array;
    const legal = legalMoves(board);
    const logits = legal.map((move) => Number(policyRaw[squareformerPolicyIndex(move)] ?? -100));
    const probs = softmax(logits);
    const policy = new Map<number, number>();
    legal.forEach((move, i) => policy.set(moveToActionId(move), probs[i] ?? 0));
    const wdl = softmax(wdlRaw);
    return { policy, wdl: [wdl[0] ?? 0, wdl[1] ?? 0, wdl[2] ?? 0] };
  }
}
