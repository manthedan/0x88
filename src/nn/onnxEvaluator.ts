import * as ort from './ortRuntime.ts';
import { boardToFen, opposite, type BoardState, type Color, type Piece } from '../chess/board.ts';
import { isStmWhiteRankflip, normalizePositionForStmWhite, normalizedMoveToOriginal } from '../chess/boardNormalization.ts';
import { inCheck, kingSquare, legalMoves, makeMove } from '../chess/movegen.ts';
import { moveToActionId, type Move } from '../chess/moveCodec.ts';
import { POLICY_MAP, moveToChessBenchAvClass, moveToResidualPolicyIndex } from '../chess/moveEncodings.ts';
import type { Evaluation, EvaluationContext, Evaluator } from './evaluator.ts';
import { softmax } from './numerics.ts';

const PIECES = 'PNBRQKpnbrqk';
const INPUT_CACHE_ENTRIES = Math.max(0, Math.min(65536, Math.floor(Number(new URLSearchParams(typeof location === 'undefined' ? '' : location.search).get('evalInputCacheEntries') ?? '4096')) || 4096));

function lruGet<K, V>(map: Map<K, V>, key: K): V | undefined {
  const value = map.get(key);
  if (value === undefined) return undefined;
  map.delete(key);
  map.set(key, value);
  return value;
}

function lruPut<K, V>(map: Map<K, V>, key: K, value: V, maxEntries: number): void {
  if (maxEntries <= 0) return;
  if (map.has(key)) map.delete(key);
  map.set(key, value);
  while (map.size > maxEntries) {
    const oldest = map.keys().next().value as K | undefined;
    if (oldest === undefined) break;
    map.delete(oldest);
  }
}

function encodedInputKey(board: BoardState, historyFens: string[]): string {
  return `${boardToFen(board)}\nh:${historyFens.join('|')}`;
}

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
  board_normalization?: string;
}

function normalizePolicy(policyRaw: ArrayLike<number>, board: BoardState, legalOverride?: Move[], flipped = false): Map<number, number> {
  const probs = softmax(policyRaw);
  const legal = legalOverride ?? legalMoves(board);
  const policyIndex = (move: Move) => moveToResidualPolicyIndex(move);
  let legalMass = 0;
  for (const move of legal) {
    const index = policyIndex(move);
    if (index !== undefined) legalMass += probs[index] ?? 0;
  }
  const policy = new Map<number, number>();
  if (legal.length && legalMass <= 0) for (const move of legal) policy.set(moveToActionId(normalizedMoveToOriginal(move, flipped)), 1 / legal.length);
  else for (const move of legal) {
    const index = policyIndex(move);
    policy.set(moveToActionId(normalizedMoveToOriginal(move, flipped)), index === undefined ? 0 : probs[index] / legalMass);
  }
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
const KNIGHT_OFFSETS = [[1, 2], [2, 1], [-1, 2], [-2, 1], [1, -2], [2, -1], [-1, -2], [-2, -1]];
const KING_OFFSETS = [[1, 1], [1, 0], [1, -1], [0, 1], [0, -1], [-1, 1], [-1, 0], [-1, -1]];

const fileOf = (sq: number) => sq % 8;
const rankOf = (sq: number) => Math.floor(sq / 8);
const onBoard = (f: number, r: number) => f >= 0 && f < 8 && r >= 0 && r < 8;
const squareIndex = (f: number, r: number) => f + r * 8;

function chebyshev(a: number, b: number): number {
  return Math.max(Math.abs(fileOf(a) - fileOf(b)), Math.abs(rankOf(a) - rankOf(b)));
}

function capturedPieceForMove(board: BoardState, move: Move): Piece | null {
  const moving = board.squares[move.from];
  const direct = board.squares[move.to];
  const isEp = moving?.[1] === 'p' && move.to === board.epSquare && !direct && fileOf(move.from) !== fileOf(move.to);
  return isEp ? board.squares[squareIndex(fileOf(move.to), rankOf(move.from))] : direct;
}

function rayClear(board: BoardState, from: number, to: number, df: number, dr: number): boolean {
  let f = fileOf(from) + df;
  let r = rankOf(from) + dr;
  while (onBoard(f, r)) {
    const sq = squareIndex(f, r);
    if (sq === to) return true;
    if (board.squares[sq]) return false;
    f += df;
    r += dr;
  }
  return false;
}

function pieceAttacksSquare(board: BoardState, from: number, piece: Piece, to: number): boolean {
  const df = fileOf(to) - fileOf(from);
  const dr = rankOf(to) - rankOf(from);
  const role = piece[1];
  if (role === 'p') return dr === (piece[0] === 'w' ? 1 : -1) && Math.abs(df) === 1;
  if (role === 'n') return KNIGHT_OFFSETS.some(([f, r]) => f === df && r === dr);
  if (role === 'k') return KING_OFFSETS.some(([f, r]) => f === df && r === dr);
  if (role === 'b') return Math.abs(df) === Math.abs(dr) && df !== 0 && rayClear(board, from, to, Math.sign(df), Math.sign(dr));
  if (role === 'r') return ((df === 0) !== (dr === 0)) && (df !== 0 || dr !== 0) && rayClear(board, from, to, Math.sign(df), Math.sign(dr));
  if (role === 'q') {
    const bishopLike = Math.abs(df) === Math.abs(dr) && df !== 0;
    const rookLike = (df === 0) !== (dr === 0);
    return (bishopLike || rookLike) && rayClear(board, from, to, Math.sign(df), Math.sign(dr));
  }
  return false;
}

function attackerCount(board: BoardState, color: Color, sq: number): number {
  let count = 0;
  for (let from = 0; from < 64; from++) {
    const piece = board.squares[from];
    if (piece?.[0] === color && pieceAttacksSquare(board, from, piece, sq)) count++;
  }
  return count;
}

function isPinnedToKing(board: BoardState, color: Color, sq: number): boolean {
  const piece = board.squares[sq];
  if (!piece || piece[0] !== color || piece[1] === 'k') return false;
  const king = kingSquare(board, color);
  if (king === null) return false;
  const without = { ...board, squares: board.squares.slice() };
  without.squares[sq] = null;
  return attackerCount(without, opposite(color), king) > 0;
}

function terminalWdl(board: BoardState): [number, number, number] {
  if (!inCheck(board)) return [0, 1, 0];
  return [0, 0, 1];
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
      const captured = capturedPieceForMove(board, move);
      const movingRole = moving?.[1] ?? '';
      const movedColor: Color = (moving?.[0] ?? board.turn) as Color;
      const enemyColor = opposite(movedColor);
      const capturedRole = captured?.[1] ?? '';
      const promo = move.promotion ? (PROMO_INDEX[move.promotion] ?? 0) : 0;
      const movingType = ROLE_INDEX[movingRole] ?? 0;
      const capturedType = ROLE_INDEX[capturedRole] ?? 0;
      const capturedValue = PIECE_VALUE_BY_ROLE[capturedRole] ?? 0;
      const promoValue = move.promotion ? ((PIECE_VALUE_BY_ROLE[move.promotion] ?? 0) - 1) : 0;
      const isEp = movingRole === 'p' && move.to === board.epSquare && !!captured && !board.squares[move.to] && fileOf(move.from) !== fileOf(move.to);
      const isCastle = movingRole === 'k' && Math.abs(fileOf(move.to) - fileOf(move.from)) === 2;
      const fromEnemy = attackerCount(board, enemyColor, move.from);
      const fromOwn = attackerCount(board, movedColor, move.from);
      const after = makeMove(board, move);
      const givesCheck = inCheck(after, enemyColor);
      const toEnemy = attackerCount(after, enemyColor, move.to);
      const toOwn = attackerCount(after, movedColor, move.to);
      const ownKingAfter = kingSquare(after, movedColor) ?? (movedColor === 'w' ? 4 : 60);
      const enemyKingAfter = kingSquare(after, enemyColor) ?? (enemyColor === 'w' ? 4 : 60);
      const vals = [
        movingType,
        capturedType,
        promo,
        captured || isEp ? 1 : 0,
        givesCheck ? 1 : 0,
        isCastle ? 1 : 0,
        promo ? 1 : 0,
        isEp ? 1 : 0,
        fromEnemy > 0 ? 1 : 0,
        fromOwn > 0 ? 1 : 0,
        toEnemy > 0 ? 1 : 0,
        toOwn > 0 ? 1 : 0,
        Math.min(8, toEnemy),
        Math.min(8, toOwn),
        PIECE_VALUE_BY_ROLE[movingRole] ?? 0,
        capturedValue,
        capturedValue + promoValue,
        isPinnedToKing(board, movedColor, move.from) ? 1 : 0,
        chebyshev(move.to, enemyKingAfter),
        chebyshev(move.to, ownKingAfter),
      ];
      for (let k = 0; k < Math.min(featureCount, vals.length); k++) features[base + k] = vals[k];
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

function configureOrtWasmRuntime() {
  const wasm = ort.env.wasm as unknown as { numThreads?: number; proxy?: boolean };
  const isBrowserMainThread = typeof document !== 'undefined';
  const threads = Number(globalThis.process?.env?.ORT_INTRA_OP_NUM_THREADS ?? globalThis.process?.env?.ORT_NUM_THREADS ?? (isBrowserMainThread ? '1' : '0'));
  if (Number.isFinite(threads) && threads > 0) wasm.numThreads = Math.floor(threads);
  if (isBrowserMainThread) {
    // ORT's threaded WASM path starts Emscripten pthread workers from import.meta.url.
    // In a Vite single-bundle deploy that URL is the app bundle, which touches
    // `document` at top level and crashes inside workers on Netlify.
    wasm.numThreads = Number.isFinite(threads) && threads > 0 ? Math.floor(threads) : 1;
    wasm.proxy = false;
  }
}

function sessionOptions(): ort.InferenceSession.SessionOptions {
  configureOrtWasmRuntime();
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
  private planeCache = new Map<string, Float32Array>();
  constructor(session: ort.InferenceSession, meta: OnnxStudentMeta, historyFens: string[] = []) {
    this.session = session; this.meta = meta; this.historyFens = historyFens;
    if (meta.policy_map !== POLICY_MAP) throw new Error(`Unsupported policy map: ${meta.policy_map}`);
  }

  static async create(modelPath: string | Uint8Array | ArrayBuffer, meta: OnnxStudentMeta, historyFens: string[] = []): Promise<OnnxEvaluator> {
    const session = await ort.createOrtSession(modelPath);
    return new OnnxEvaluator(session, meta, historyFens);
  }

  async evaluate(board: BoardState, context: EvaluationContext = {}): Promise<Evaluation> {
    return (await this.evaluateBatch([board], [context]))[0];
  }

  private cachedInputPlanes(board: BoardState, historyFens: string[]): Float32Array {
    const key = encodedInputKey(board, historyFens);
    const cached = lruGet(this.planeCache, key);
    if (cached) return cached;
    const encoded = onnxInputPlanes(board, this.meta, historyFens);
    lruPut(this.planeCache, key, encoded, INPUT_CACHE_ENTRIES);
    return encoded;
  }

  async evaluateBatch(boards: BoardState[], contexts: EvaluationContext[] = []): Promise<Evaluation[]> {
    if (!boards.length) return [];
    const t0 = ort.tinyLeelaNowMs();
    const normalized = boards.map((board, i) => {
      const historyFens = contexts[i]?.historyFens ?? this.historyFens;
      return isStmWhiteRankflip(this.meta.board_normalization)
        ? normalizePositionForStmWhite(board, historyFens, contexts[i]?.legalMoves)
        : { board, historyFens, legalMoves: contexts[i]?.legalMoves, flipped: false };
    });
    const tNormalized = ort.tinyLeelaNowMs();
    const evalBoards = normalized.map((n) => n.board);
    const evalContexts = normalized.map((n) => ({ historyFens: n.historyFens, legalMoves: n.legalMoves }));
    const one = this.meta.input_planes * 64;
    const input = new Float32Array(boards.length * one);
    for (let i = 0; i < boards.length; i++) input.set(this.cachedInputPlanes(evalBoards[i], evalContexts[i]?.historyFens ?? []), i * one);
    const tEncoded = ort.tinyLeelaNowMs();
    const feeds: Record<string, ort.Tensor> = { planes: new ort.Tensor('float32', input, [boards.length, this.meta.input_planes, 8, 8]) };

    if (this.meta.architecture === 'cnn_move_token_transformer' || this.meta.architecture === 'cnn_square_move_transformer') {
      const width = Math.max(1, Number(this.meta.onnx_fixed_legal_moves ?? this.meta.max_legal_moves ?? 128));
      const featureCount = Math.max(1, Number(this.meta.num_move_features ?? 20));
      const legal = moveformerLegalInputs(evalBoards, width, featureCount, evalContexts);
      const overflow = legal.moves.findIndex((moves) => moves.length > width);
      if (overflow >= 0 && this.meta.allow_legal_overflow_zero_prior !== true) {
        throw new Error(`Move-token ONNX legal move overflow: model accepts ${width} legal moves but position has ${legal.moves[overflow].length}. Use a larger legal bucket/export (for example k128), dynamic legal export, or set meta.allow_legal_overflow_zero_prior=true to keep legacy zero-prior truncation. fen=${boardToFen(boards[overflow])}`);
      }
      feeds.legal_action_ids = new ort.Tensor('int64', legal.actionIds, [boards.length, width]);
      feeds.legal_features = new ort.Tensor('float32', legal.features, [boards.length, width, featureCount]);
      feeds.legal_mask = new ort.Tensor('float32', legal.mask, [boards.length, width]);
      const tLegal = ort.tinyLeelaNowMs();
      const tRun0 = ort.tinyLeelaNowMs();
      const outputs = await this.session.run(feeds);
      const tRun1 = ort.tinyLeelaNowMs();
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
        const policy = new Map<number, number>();
        if (n === 0) {
          out.push({ policy, wdl: terminalWdl(boards[i]) });
          continue;
        }
        const probs = softmax(policyRaw.subarray(i * width, i * width + n));
        const actionValues = new Map<number, number>();
        const rankScores = new Map<number, number>();
        const regrets = new Map<number, number>();
        const risks = new Map<number, number>();
        const uncertainties = new Map<number, number>();
        for (let j = 0; j < n; j++) {
          const originalMove = normalizedMoveToOriginal(legal.moves[i][j], normalized[i].flipped);
          const actionId = moveToActionId(originalMove);
          policy.set(actionId, probs[j] ?? 0);
          if (avRaw) actionValues.set(actionId, Number(avRaw[i * width + j] ?? 0));
          if (rankRaw) rankScores.set(actionId, Number(rankRaw[i * width + j] ?? 0));
          if (regretRaw) regrets.set(actionId, Number(regretRaw[i * width + j] ?? 0));
          if (riskRaw) risks.set(actionId, Number(riskRaw[i * width + j] ?? 0));
          if (uncertaintyRaw) uncertainties.set(actionId, Number(uncertaintyRaw[i * width + j] ?? 0));
        }
        if (legal.moves[i].length > width) {
          // Legacy compatibility path, enabled only by meta.allow_legal_overflow_zero_prior.
          for (let j = width; j < legal.moves[i].length; j++) policy.set(moveToActionId(normalizedMoveToOriginal(legal.moves[i][j], normalized[i].flipped)), 0);
        }
        const wdl = softmax(wdlRaw.subarray(i * 3, i * 3 + 3));
        out.push({ policy, wdl: [wdl[0] ?? 0, wdl[1] ?? 0, wdl[2] ?? 0], ...(actionValues.size ? { actionValues } : {}), ...(rankScores.size ? { rankScores } : {}), ...(regrets.size ? { regrets } : {}), ...(risks.size ? { risks } : {}), ...(uncertainties.size ? { uncertainties } : {}) });
      }
      const tDone = ort.tinyLeelaNowMs();
      ort.tinyLeelaLogLatency('onnx.evaluateBatch.moveToken', {
        batch: boards.length,
        legalWidth: width,
        featureCount,
        normalizeMs: tNormalized - t0,
        encodeMs: tEncoded - tNormalized,
        legalMs: tLegal - tEncoded,
        ortRunMs: tRun1 - tRun0,
        postprocessMs: tDone - tRun1,
        totalMs: tDone - t0,
      });
      return out;
    }

    let candidateMoves: Move[][] | null = null;
    let candidateWidth = 0;
    if (this.meta.av_head_exported) {
      const cand = legalCandidateInputs(evalBoards, evalContexts);
      candidateMoves = cand.moves;
      candidateWidth = cand.width;
      feeds.candidate_moves = new ort.Tensor('int64', cand.classes, [boards.length, candidateWidth]);
    }
    const tLegal = ort.tinyLeelaNowMs();
    const tRun0 = ort.tinyLeelaNowMs();
    const outputs = await this.session.run(feeds);
    const tRun1 = ort.tinyLeelaNowMs();
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
      const policy = normalizePolicy(policyRaw.subarray(i * policySize, (i + 1) * policySize), evalBoards[i], evalContexts[i]?.legalMoves, normalized[i].flipped);
      const wdl = softmax(wdlRaw.subarray(i * 3, i * 3 + 3));
      const actionValues = new Map<number, number>();
      const rankScores = new Map<number, number>();
      const regrets = new Map<number, number>();
      const risks = new Map<number, number>();
      const uncertainties = new Map<number, number>();
      if (candidateMoves) {
        for (let j = 0; j < candidateMoves[i].length; j++) {
          const actionId = moveToActionId(normalizedMoveToOriginal(candidateMoves[i][j], normalized[i].flipped));
          if (avRaw) actionValues.set(actionId, Number(avRaw[i * candidateWidth + j] ?? 0));
          if (rankRaw) rankScores.set(actionId, Number(rankRaw[i * candidateWidth + j] ?? 0));
          if (regretRaw) regrets.set(actionId, Number(regretRaw[i * candidateWidth + j] ?? 0));
          if (riskRaw) risks.set(actionId, Number(riskRaw[i * candidateWidth + j] ?? 0));
          if (uncertaintyRaw) uncertainties.set(actionId, Number(uncertaintyRaw[i * candidateWidth + j] ?? 0));
        }
      }
      out.push({ policy, wdl: [wdl[0] ?? 0, wdl[1] ?? 0, wdl[2] ?? 0], ...(actionValues.size ? { actionValues } : {}), ...(rankScores.size ? { rankScores } : {}), ...(regrets.size ? { regrets } : {}), ...(risks.size ? { risks } : {}), ...(uncertainties.size ? { uncertainties } : {}) });
    }
    const tDone = ort.tinyLeelaNowMs();
    ort.tinyLeelaLogLatency('onnx.evaluateBatch', {
      batch: boards.length,
      candidateWidth,
      normalizeMs: tNormalized - t0,
      encodeMs: tEncoded - tNormalized,
      legalMs: tLegal - tEncoded,
      ortRunMs: tRun1 - tRun0,
      postprocessMs: tDone - tRun1,
      totalMs: tDone - t0,
    });
    return out;
  }
}
