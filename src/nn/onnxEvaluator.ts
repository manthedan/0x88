import * as ort from 'onnxruntime-web';
import { boardToFen, type BoardState } from '../chess/board.ts';
import { legalMoves } from '../chess/movegen.ts';
import { moveToActionId, type Move } from '../chess/moveCodec.ts';
import { POLICY_MAP, moveToChessBenchAvClass, moveToResidualPolicyIndex } from '../chess/moveEncodings.ts';
import type { Evaluation, EvaluationContext, Evaluator } from './evaluator.ts';

const PIECES = 'PNBRQKpnbrqk';

export interface OnnxStudentMeta {
  kind: string;
  architecture: 'residual_tower' | 'cnn_channel_transformer' | 'cnn_move_token_transformer' | 'cnn_square_move_transformer' | 'cnn_square_transformer';
  policy_map: string;
  moves: string[];
  channels: number;
  blocks: number;
  history_plies: number;
  input_planes: number;
  onnx?: string;
  channelformer_cpatch?: number;
  channelformer_dim?: number;
  channelformer_heads?: number;
  channelformer_layers?: number;
  channelformer_ff_dim?: number;
  trained_with_aux_av?: boolean;
  av_head_exported?: boolean;
  aux_heads_exported?: string[];
  action_value_move_encoding?: 'chessbench_compact_20480';
  max_legal_moves?: number;
  onnx_fixed_legal_moves?: number;
  num_move_features?: number;
  allow_legal_overflow_zero_prior?: boolean;
}

function softmax(xs: ArrayLike<number>): number[] {
  let m = -Infinity;
  for (let i = 0; i < xs.length; i++) if (xs[i] > m) m = xs[i];
  const out = Array.from(xs, (x) => Math.exp(Number(x) - m));
  const total = out.reduce((a, b) => a + b, 0) || 1;
  return out.map((x) => x / total);
}

function normalizePolicy(policyRaw: ArrayLike<number>, board: BoardState, legalOverride?: Move[]): Map<number, number> {
  const probs = softmax(policyRaw);
  const legal = legalOverride ?? legalMoves(board);
  const policyIndex = (move: Move) => moveToResidualPolicyIndex(move);
  let legalMass = 0;
  for (const move of legal) {
    const index = policyIndex(move);
    if (index !== undefined) legalMass += probs[index] ?? 0;
  }
  const policy = new Map<number, number>();
  if (legal.length && legalMass <= 0) for (const move of legal) policy.set(moveToActionId(move), 1 / legal.length);
  else for (const move of legal) { const index = policyIndex(move); policy.set(moveToActionId(move), index === undefined ? 0 : probs[index] / legalMass); }
  return policy;
}

function legalCandidateInputs(boards: BoardState[], contexts: EvaluationContext[] = []): { moves: Move[][]; classes: BigInt64Array; width: number } {
  const moves = boards.map((board, i) => contexts[i]?.legalMoves ?? legalMoves(board));
  const width = Math.max(1, ...moves.map((m) => m.length));
  const classes = new BigInt64Array(boards.length * width);
  for (let i = 0; i < moves.length; i++) {
    for (let j = 0; j < moves[i].length; j++) classes[i * width + j] = BigInt(moveToChessBenchAvClass(moves[i][j]));
  }
  return { moves, classes, width };
}

const ROLE_INDEX: Record<string, number> = { p: 1, n: 2, b: 3, r: 4, q: 5, k: 6 };
const PIECE_VALUE_BY_ROLE: Record<string, number> = { p: 1, n: 3, b: 3, r: 5, q: 9, k: 0 };
const PROMO_INDEX: Record<string, number> = { n: 1, b: 2, r: 3, q: 4 };

function kingSquare(board: BoardState, color: 'w' | 'b'): number {
  const target = `${color}k`;
  const i = board.squares.findIndex((p) => p === target);
  return i >= 0 ? i : (color === 'w' ? 4 : 60);
}

function chebyshev(a: number, b: number): number {
  return Math.max(Math.abs((a % 8) - (b % 8)), Math.abs(Math.floor(a / 8) - Math.floor(b / 8)));
}

export function moveformerLegalInputs(boards: BoardState[], width: number, featureCount: number, contexts: EvaluationContext[] = []): { moves: Move[][]; actionIds: BigInt64Array; features: Float32Array; mask: Float32Array; width: number } {
  const moves = boards.map((board, i) => contexts[i]?.legalMoves ?? legalMoves(board));
  const actionIds = new BigInt64Array(boards.length * width);
  const features = new Float32Array(boards.length * width * featureCount);
  const mask = new Float32Array(boards.length * width);
  for (let bi = 0; bi < boards.length; bi++) {
    const board = boards[bi];
    actionIds.fill(20480n, bi * width, (bi + 1) * width);
    const ownKing = kingSquare(board, board.turn);
    const enemyKing = kingSquare(board, board.turn === 'w' ? 'b' : 'w');
    for (let j = 0; j < Math.min(width, moves[bi].length); j++) {
      const move = moves[bi][j];
      const actionId = moveToActionId(move);
      actionIds[bi * width + j] = BigInt(actionId);
      mask[bi * width + j] = 1;
      const base = (bi * width + j) * featureCount;
      const moving = board.squares[move.from];
      const captured = board.squares[move.to];
      const movingRole = moving?.[1] ?? '';
      const capturedRole = captured?.[1] ?? '';
      const promo = move.promotion ? (PROMO_INDEX[move.promotion] ?? 0) : 0;
      const movingType = ROLE_INDEX[movingRole] ?? 0;
      const capturedType = ROLE_INDEX[capturedRole] ?? 0;
      const capturedValue = PIECE_VALUE_BY_ROLE[capturedRole] ?? 0;
      const promoValue = move.promotion ? ((PIECE_VALUE_BY_ROLE[move.promotion] ?? 0) - 1) : 0;
      if (featureCount > 0) features[base + 0] = movingType;
      if (featureCount > 1) features[base + 1] = capturedType;
      if (featureCount > 2) features[base + 2] = promo;
      if (featureCount > 3) features[base + 3] = captured ? 1 : 0;
      // Features 4/5/7..13/17 are expensive check/castle/attack/pin signals; keep zero here, matching the lightweight AV-cache training path.
      if (featureCount > 6) features[base + 6] = promo ? 1 : 0;
      if (featureCount > 14) features[base + 14] = PIECE_VALUE_BY_ROLE[movingRole] ?? 0;
      if (featureCount > 15) features[base + 15] = capturedValue;
      if (featureCount > 16) features[base + 16] = capturedValue + promoValue;
      if (featureCount > 18) features[base + 18] = chebyshev(move.to, enemyKing);
      if (featureCount > 19) features[base + 19] = chebyshev(move.to, ownKing);
    }
  }
  return { moves, actionIds, features, mask, width };
}

function addPiecePlanes(data: Float32Array, fen: string, offset: number, inputPlanes: number) {
  const placement = fen.split(/\s+/)[0];
  let rank = 0, file = 0;
  for (const ch of placement) {
    if (ch === '/') { rank++; file = 0; }
    else if (/\d/.test(ch)) file += Number(ch);
    else {
      const pi = PIECES.indexOf(ch);
      if (pi >= 0) data[((offset + pi) * 64) + rank * 8 + file] = 1;
      file++;
    }
  }
}

export function onnxInputPlanes(board: BoardState, meta: Pick<OnnxStudentMeta, 'input_planes' | 'history_plies'>, historyFens: string[] = []): Float32Array {
  const fen = boardToFen(board);
  const inputPlanes = meta.input_planes;
  const data = new Float32Array(inputPlanes * 64);
  addPiecePlanes(data, fen, 0, inputPlanes);
  for (let h = 0; h < meta.history_plies && h < historyFens.length; h++) addPiecePlanes(data, historyFens[h], 12 * (h + 1), inputPlanes);
  const parts = fen.split(/\s+/); const side = parts[1] ?? 'w'; const castling = parts[2] ?? '-'; const ep = parts[3] ?? '-';
  const state0 = 12 * (meta.history_plies + 1);
  const fillPlane = (p: number, v: number) => { if (p >= 0 && p < inputPlanes) data.fill(v, p * 64, (p + 1) * 64); };
  fillPlane(state0, side === 'w' ? 1 : -1);
  if (inputPlanes - state0 >= 10) {
    ['K', 'Q', 'k', 'q'].forEach((flag, i) => { if (castling.includes(flag)) fillPlane(state0 + 1 + i, 1); });
    if (ep !== '-' && ep.length >= 2) {
      const ef = ep.charCodeAt(0) - 97; const er = 8 - Number(ep[1]);
      if (er >= 0 && er < 8 && ef >= 0 && ef < 8) data[(state0 + 5) * 64 + er * 8 + ef] = 1;
    }
    fillPlane(state0 + 6, 1); fillPlane(state0 + 7, side === 'w' ? 1 : 0);
    // Check planes are intentionally zero for now; add runtime parity once history plumbing lands.
  } else fillPlane(state0 + 1, 1);
  return data;
}

function outputNames(outputs: Record<string, ort.Tensor>): string {
  return Object.keys(outputs).sort().join(', ') || '<none>';
}

function requiredFloatOutput(outputs: Record<string, ort.Tensor>, name: string): Float32Array {
  const tensor = outputs[name];
  if (!tensor) throw new Error(`ONNX output missing required tensor '${name}'. Available outputs: ${outputNames(outputs)}`);
  if (!(tensor.data instanceof Float32Array)) throw new Error(`ONNX output '${name}' expected float32 data, got ${Object.prototype.toString.call(tensor.data)}`);
  return tensor.data;
}

function optionalFloatOutput(outputs: Record<string, ort.Tensor>, ...names: string[]): Float32Array | undefined {
  for (const name of names) {
    const tensor = outputs[name];
    if (!tensor) continue;
    if (!(tensor.data instanceof Float32Array)) throw new Error(`ONNX output '${name}' expected float32 data, got ${Object.prototype.toString.call(tensor.data)}`);
    return tensor.data;
  }
  return undefined;
}

function sessionOptions(): ort.InferenceSession.SessionOptions {
  const threads = Number(globalThis.process?.env?.ORT_INTRA_OP_NUM_THREADS ?? globalThis.process?.env?.ORT_NUM_THREADS ?? '0');
  const opts: ort.InferenceSession.SessionOptions = { graphOptimizationLevel: 'all' };
  const epEnv = globalThis.process?.env?.TINY_LEELA_ORT_EP ?? globalThis.process?.env?.ORT_EXECUTION_PROVIDERS ?? '';
  const executionProviders = epEnv.split(',').map((s) => s.trim()).filter(Boolean);
  if (executionProviders.length) opts.executionProviders = executionProviders;
  if (Number.isFinite(threads) && threads > 0) {
    opts.intraOpNumThreads = Math.floor(threads);
    opts.interOpNumThreads = 1;
  }
  return opts;
}

export class OnnxEvaluator implements Evaluator {
  private session: ort.InferenceSession;
  private meta: OnnxStudentMeta;
  private historyFens: string[];
  constructor(session: ort.InferenceSession, meta: OnnxStudentMeta, historyFens: string[] = []) {
    this.session = session; this.meta = meta; this.historyFens = historyFens;
    if (meta.policy_map !== POLICY_MAP) throw new Error(`Unsupported policy map: ${meta.policy_map}`);
  }

  static async create(modelPath: string | Uint8Array | ArrayBuffer, meta: OnnxStudentMeta, historyFens: string[] = []): Promise<OnnxEvaluator> {
    const session = await ort.InferenceSession.create(modelPath as never, sessionOptions());
    return new OnnxEvaluator(session, meta, historyFens);
  }

  async evaluate(board: BoardState, context: EvaluationContext = {}): Promise<Evaluation> {
    return (await this.evaluateBatch([board], [context]))[0];
  }

  async evaluateBatch(boards: BoardState[], contexts: EvaluationContext[] = []): Promise<Evaluation[]> {
    if (!boards.length) return [];
    const one = this.meta.input_planes * 64;
    const input = new Float32Array(boards.length * one);
    for (let i = 0; i < boards.length; i++) input.set(onnxInputPlanes(boards[i], this.meta, contexts[i]?.historyFens ?? this.historyFens), i * one);
    const feeds: Record<string, ort.Tensor> = { planes: new ort.Tensor('float32', input, [boards.length, this.meta.input_planes, 8, 8]) };

    if (this.meta.architecture === 'cnn_move_token_transformer' || this.meta.architecture === 'cnn_square_move_transformer') {
      const width = Math.max(1, Number(this.meta.onnx_fixed_legal_moves ?? this.meta.max_legal_moves ?? 128));
      const featureCount = Math.max(1, Number(this.meta.num_move_features ?? 20));
      const legal = moveformerLegalInputs(boards, width, featureCount, contexts);
      const overflow = legal.moves.findIndex((moves) => moves.length > width);
      if (overflow >= 0 && this.meta.allow_legal_overflow_zero_prior !== true) {
        throw new Error(`Move-token ONNX legal move overflow: model accepts ${width} legal moves but position has ${legal.moves[overflow].length}. Use a larger legal bucket/export (for example k128), dynamic legal export, or set meta.allow_legal_overflow_zero_prior=true to keep legacy zero-prior truncation. fen=${boardToFen(boards[overflow])}`);
      }
      feeds.legal_action_ids = new ort.Tensor('int64', legal.actionIds, [boards.length, width]);
      feeds.legal_features = new ort.Tensor('float32', legal.features, [boards.length, width, featureCount]);
      feeds.legal_mask = new ort.Tensor('float32', legal.mask, [boards.length, width]);
      const outputs = await this.session.run(feeds);
      const policyRaw = requiredFloatOutput(outputs, 'policy_logits_legal');
      const wdlRaw = requiredFloatOutput(outputs, 'wdl_logits');
      const avRaw = optionalFloatOutput(outputs, 'action_values');
      const rankRaw = outputs.rank_scores?.data as Float32Array | undefined;
      const regretRaw = outputs.regrets?.data as Float32Array | undefined;
      const riskRaw = outputs.risks?.data as Float32Array | undefined;
      const uncertaintyRaw = outputs.uncertainties?.data as Float32Array | undefined;
      const out: Evaluation[] = [];
      for (let i = 0; i < boards.length; i++) {
        const n = Math.min(width, legal.moves[i].length);
        const probs = softmax(policyRaw.subarray(i * width, i * width + n));
        const policy = new Map<number, number>();
        const actionValues = new Map<number, number>();
        const rankScores = new Map<number, number>();
        const regrets = new Map<number, number>();
        const risks = new Map<number, number>();
        const uncertainties = new Map<number, number>();
        for (let j = 0; j < n; j++) {
          const actionId = moveToActionId(legal.moves[i][j]);
          policy.set(actionId, probs[j] ?? 0);
          if (avRaw) actionValues.set(actionId, Number(avRaw[i * width + j] ?? 0));
          if (rankRaw) rankScores.set(actionId, Number(rankRaw[i * width + j] ?? 0));
          if (regretRaw) regrets.set(actionId, Number(regretRaw[i * width + j] ?? 0));
          if (riskRaw) risks.set(actionId, Number(riskRaw[i * width + j] ?? 0));
          if (uncertaintyRaw) uncertainties.set(actionId, Number(uncertaintyRaw[i * width + j] ?? 0));
        }
        if (legal.moves[i].length > width) {
          // Legacy compatibility path, enabled only by meta.allow_legal_overflow_zero_prior.
          for (let j = width; j < legal.moves[i].length; j++) policy.set(moveToActionId(legal.moves[i][j]), 0);
        }
        const wdl = softmax(wdlRaw.subarray(i * 3, i * 3 + 3));
        out.push({ policy, wdl: [wdl[0] ?? 0, wdl[1] ?? 0, wdl[2] ?? 0], ...(actionValues.size ? { actionValues } : {}), ...(rankScores.size ? { rankScores } : {}), ...(regrets.size ? { regrets } : {}), ...(risks.size ? { risks } : {}), ...(uncertainties.size ? { uncertainties } : {}) });
      }
      return out;
    }

    let candidateMoves: Move[][] | null = null;
    let candidateWidth = 0;
    if (this.meta.av_head_exported) {
      const cand = legalCandidateInputs(boards, contexts);
      candidateMoves = cand.moves;
      candidateWidth = cand.width;
      feeds.candidate_moves = new ort.Tensor('int64', cand.classes, [boards.length, candidateWidth]);
    }
    const outputs = await this.session.run(feeds);
    const policyRaw = requiredFloatOutput(outputs, 'policy_logits');
    const wdlRaw = requiredFloatOutput(outputs, 'wdl_logits');
    const avRaw = optionalFloatOutput(outputs, 'action_values', 'action_value', 'action_value_logits');
    const rankRaw = outputs.rank_scores?.data as Float32Array | undefined;
    const regretRaw = outputs.regrets?.data as Float32Array | undefined;
    const riskRaw = outputs.risks?.data as Float32Array | undefined;
    const uncertaintyRaw = outputs.uncertainties?.data as Float32Array | undefined;
    const policySize = this.meta.moves?.length ?? Math.floor(policyRaw.length / boards.length);
    const out: Evaluation[] = [];
    for (let i = 0; i < boards.length; i++) {
      const policy = normalizePolicy(policyRaw.subarray(i * policySize, (i + 1) * policySize), boards[i], contexts[i]?.legalMoves);
      const wdl = softmax(wdlRaw.subarray(i * 3, i * 3 + 3));
      const actionValues = new Map<number, number>();
      const rankScores = new Map<number, number>();
      const regrets = new Map<number, number>();
      const risks = new Map<number, number>();
      const uncertainties = new Map<number, number>();
      if (candidateMoves) {
        for (let j = 0; j < candidateMoves[i].length; j++) {
          const actionId = moveToActionId(candidateMoves[i][j]);
          if (avRaw) actionValues.set(actionId, Number(avRaw[i * candidateWidth + j] ?? 0));
          if (rankRaw) rankScores.set(actionId, Number(rankRaw[i * candidateWidth + j] ?? 0));
          if (regretRaw) regrets.set(actionId, Number(regretRaw[i * candidateWidth + j] ?? 0));
          if (riskRaw) risks.set(actionId, Number(riskRaw[i * candidateWidth + j] ?? 0));
          if (uncertaintyRaw) uncertainties.set(actionId, Number(uncertaintyRaw[i * candidateWidth + j] ?? 0));
        }
      }
      out.push({ policy, wdl: [wdl[0] ?? 0, wdl[1] ?? 0, wdl[2] ?? 0], ...(actionValues.size ? { actionValues } : {}), ...(rankScores.size ? { rankScores } : {}), ...(regrets.size ? { regrets } : {}), ...(risks.size ? { risks } : {}), ...(uncertainties.size ? { uncertainties } : {}) });
    }
    return out;
  }
}
