import { type BoardState, boardToFen, type Color, opposite } from '../chess/board.ts';
import { isStmWhiteRankflip, normalizedMoveToOriginal, normalizePositionForStmWhite } from '../chess/boardNormalization.ts';
import { type Move, moveToActionId } from '../chess/moveCodec.ts';
import { moveToChessBenchAvClass, moveToResidualPolicyIndex, POLICY_MAP } from '../chess/moveEncodings.ts';
import { inCheck, kingSquare, legalMoves, makeMove } from '../chess/movegen.ts';
import type { Evaluation, EvaluationContext, Evaluator } from './evaluator.ts';
import { softmax } from './numerics.ts';
import * as ort from './ortRuntime.ts';

const PIECES = 'PNBRQKpnbrqk';
const INPUT_CACHE_ENTRIES = Math.max(
  0,
  Math.min(
    65536,
    Math.floor(Number(new URLSearchParams(typeof location === 'undefined' ? '' : location.search).get('evalInputCacheEntries') ?? '4096')) || 4096,
  ),
);

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

const ROLE_INDEX: Record<string, number> = { p: 1, n: 2, b: 3, r: 4, q: 5, k: 6 };
const PIECE_VALUE_BY_ROLE: Record<string, number> = { p: 1, n: 3, b: 3, r: 5, q: 9, k: 0 };
const PROMO_INDEX: Record<string, number> = { n: 1, b: 2, r: 3, q: 4 };

const fileOf = (sq: number) => sq % 8;
const rankOf = (sq: number) => Math.floor(sq / 8);

function attackerCount(board: BoardState, color: Color, sq: number): number {
  const toF = fileOf(sq);
  const toR = rankOf(sq);
  let count = 0;
  for (let from = 0; from < 64; from++) {
    const piece = board.squares[from];
    if (piece?.[0] !== color) continue;
    const df = toF - fileOf(from);
    const dr = toR - rankOf(from);
    const role = piece[1];
    if (role === 'p') {
      if (dr === (color === 'w' ? 1 : -1) && Math.abs(df) === 1) count++;
    } else if (role === 'n') {
      if (Math.abs(df * dr) === 2) count++;
    } else if (role === 'k') {
      if (Math.max(Math.abs(df), Math.abs(dr)) === 1) count++;
    } else {
      const diag = Math.abs(df) === Math.abs(dr) && df !== 0;
      const ortho = (df === 0) !== (dr === 0);
      if ((role === 'b' && diag) || (role === 'r' && ortho) || (role === 'q' && (diag || ortho))) {
        const sf = Math.sign(df);
        const sr = Math.sign(dr);
        let f = fileOf(from) + sf;
        let r = rankOf(from) + sr;
        while (f >= 0 && f < 8 && r >= 0 && r < 8) {
          const cur = f + r * 8;
          if (cur === sq) {
            count++;
            break;
          }
          if (board.squares[cur]) break;
          f += sf;
          r += sr;
        }
      }
    }
  }
  return count;
}

export function moveformerLegalInputs(
  boards: BoardState[],
  width: number,
  featureCount: number,
  contexts: EvaluationContext[] = [],
): { moves: Move[][]; actionIds: BigInt64Array; features: Float32Array; mask: Float32Array; width: number } {
  const moves = boards.map((board, i) => contexts[i]?.legalMoves ?? legalMoves(board));
  const actionIds = new BigInt64Array(boards.length * width);
  const features = new Float32Array(boards.length * width * featureCount);
  const mask = new Float32Array(boards.length * width);
  for (let bi = 0; bi < boards.length; bi++) {
    const board = boards[bi];
    actionIds.fill(20480n, bi * width, (bi + 1) * width);
    for (let j = 0; j < Math.min(width, moves[bi].length); j++) {
      const move = moves[bi][j];
      actionIds[bi * width + j] = BigInt(moveToActionId(move));
      mask[bi * width + j] = 1;
      const base = (bi * width + j) * featureCount;
      const moving = board.squares[move.from];
      const movingRole = moving?.[1] ?? '';
      const directCapture = board.squares[move.to];
      const epCapture = movingRole === 'p' && move.to === board.epSquare && !directCapture && fileOf(move.from) !== fileOf(move.to);
      const captured = epCapture ? board.squares[fileOf(move.to) + rankOf(move.from) * 8] : directCapture;
      const isEp = epCapture && !!captured;
      const movedColor: Color = (moving?.[0] ?? board.turn) as Color;
      const enemyColor = opposite(movedColor);
      const capturedRole = captured?.[1] ?? '';
      const promo = move.promotion ? (PROMO_INDEX[move.promotion] ?? 0) : 0;
      const capturedValue = PIECE_VALUE_BY_ROLE[capturedRole] ?? 0;
      const promoValue = move.promotion ? (PIECE_VALUE_BY_ROLE[move.promotion] ?? 0) - 1 : 0;
      const isCastle = movingRole === 'k' && Math.abs(fileOf(move.to) - fileOf(move.from)) === 2;
      const fromEnemy = attackerCount(board, enemyColor, move.from);
      const fromOwn = attackerCount(board, movedColor, move.from);
      const after = makeMove(board, move);
      const toEnemy = attackerCount(after, enemyColor, move.to);
      const toOwn = attackerCount(after, movedColor, move.to);
      const ownKingAfter = kingSquare(after, movedColor) ?? (movedColor === 'w' ? 4 : 60);
      const enemyKingAfter = kingSquare(after, enemyColor) ?? (enemyColor === 'w' ? 4 : 60);
      let pinned = false;
      if (moving && moving[0] === movedColor && movingRole !== 'k') {
        const king = kingSquare(board, movedColor);
        if (king !== null) {
          const without = { ...board, squares: board.squares.slice() };
          without.squares[move.from] = null;
          pinned = attackerCount(without, enemyColor, king) > 0;
        }
      }
      const vals = [
        ROLE_INDEX[movingRole] ?? 0,
        ROLE_INDEX[capturedRole] ?? 0,
        promo,
        captured || isEp ? 1 : 0,
        inCheck(after, enemyColor) ? 1 : 0,
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
        pinned ? 1 : 0,
        Math.max(Math.abs(fileOf(move.to) - fileOf(enemyKingAfter)), Math.abs(rankOf(move.to) - rankOf(enemyKingAfter))),
        Math.max(Math.abs(fileOf(move.to) - fileOf(ownKingAfter)), Math.abs(rankOf(move.to) - rankOf(ownKingAfter))),
      ];
      for (let k = 0; k < Math.min(featureCount, vals.length); k++) features[base + k] = vals[k];
    }
  }
  return { moves, actionIds, features, mask, width };
}

function addPiecePlanes(data: Float32Array, fen: string, offset: number) {
  const placement = fen.split(/\s+/)[0];
  let rank = 0,
    file = 0;
  for (const ch of placement) {
    if (ch === '/') {
      rank++;
      file = 0;
    } else if (/\d/.test(ch)) file += Number(ch);
    else {
      const pi = PIECES.indexOf(ch);
      if (pi >= 0) data[(offset + pi) * 64 + rank * 8 + file] = 1;
      file++;
    }
  }
}

export function onnxInputPlanes(board: BoardState, meta: Pick<OnnxStudentMeta, 'input_planes' | 'history_plies'>, historyFens: string[] = []): Float32Array {
  const fen = boardToFen(board);
  const inputPlanes = meta.input_planes;
  const data = new Float32Array(inputPlanes * 64);
  addPiecePlanes(data, fen, 0);
  for (let h = 0; h < meta.history_plies && h < historyFens.length; h++) addPiecePlanes(data, historyFens[h], 12 * (h + 1));
  const parts = fen.split(/\s+/);
  const side = parts[1] ?? 'w';
  const castling = parts[2] ?? '-';
  const ep = parts[3] ?? '-';
  const state0 = 12 * (meta.history_plies + 1);
  const fillPlane = (p: number, v: number) => {
    if (p >= 0 && p < inputPlanes) data.fill(v, p * 64, (p + 1) * 64);
  };
  fillPlane(state0, side === 'w' ? 1 : -1);
  if (inputPlanes - state0 >= 10) {
    ['K', 'Q', 'k', 'q'].forEach((flag, i) => {
      if (castling.includes(flag)) fillPlane(state0 + 1 + i, 1);
    });
    if (ep !== '-' && ep.length >= 2) {
      const ef = ep.charCodeAt(0) - 97;
      const er = 8 - Number(ep[1]);
      if (er >= 0 && er < 8 && ef >= 0 && ef < 8) data[(state0 + 5) * 64 + er * 8 + ef] = 1;
    }
    fillPlane(state0 + 6, 1);
    fillPlane(state0 + 7, side === 'w' ? 1 : 0);
  } else fillPlane(state0 + 1, 1);
  return data;
}

function requiredFloatOutput(outputs: Record<string, ort.Tensor>, name: string): Float32Array {
  const tensor = outputs[name];
  if (!tensor) throw new Error(`ONNX output missing required tensor '${name}'. Available outputs: ${Object.keys(outputs).sort().join(', ') || '<none>'}`);
  if (!(tensor.data instanceof Float32Array))
    throw new Error(`ONNX output '${name}' expected float32 data, got ${Object.prototype.toString.call(tensor.data)}`);
  return tensor.data;
}

export class OnnxEvaluator implements Evaluator {
  private session: ort.InferenceSession;
  private meta: OnnxStudentMeta;
  private historyFens: string[];
  private planeCache = new Map<string, Float32Array>();

  constructor(session: ort.InferenceSession, meta: OnnxStudentMeta, historyFens: string[] = []) {
    this.session = session;
    this.meta = meta;
    this.historyFens = historyFens;
    if (meta.policy_map !== POLICY_MAP) throw new Error(`Unsupported policy map: ${meta.policy_map}`);
  }

  static async create(modelPath: string | Uint8Array | ArrayBuffer, meta: OnnxStudentMeta, historyFens: string[] = []): Promise<OnnxEvaluator> {
    return new OnnxEvaluator(await ort.createOrtSession(modelPath), meta, historyFens);
  }

  async evaluate(board: BoardState, context: EvaluationContext = {}): Promise<Evaluation> {
    return (await this.evaluateBatch([board], [context]))[0];
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
    const one = this.meta.input_planes * 64;
    const input = new Float32Array(boards.length * one);
    for (let i = 0; i < boards.length; i++) {
      const { board, historyFens } = normalized[i];
      const key = `${boardToFen(board)}\nh:${historyFens.join('|')}`;
      let planes = this.planeCache.get(key);
      if (planes !== undefined) {
        this.planeCache.delete(key);
        this.planeCache.set(key, planes);
      } else {
        planes = onnxInputPlanes(board, this.meta, historyFens);
        if (INPUT_CACHE_ENTRIES > 0) {
          this.planeCache.set(key, planes);
          while (this.planeCache.size > INPUT_CACHE_ENTRIES) {
            const oldest = this.planeCache.keys().next().value;
            if (oldest === undefined) break;
            this.planeCache.delete(oldest);
          }
        }
      }
      input.set(planes, i * one);
    }
    const tEncoded = ort.tinyLeelaNowMs();
    const feeds: Record<string, ort.Tensor> = { planes: new ort.Tensor('float32', input, [boards.length, this.meta.input_planes, 8, 8]) };

    const isMoveToken = this.meta.architecture === 'cnn_move_token_transformer' || this.meta.architecture === 'cnn_square_move_transformer';
    let moveTokenLegal: ReturnType<typeof moveformerLegalInputs> | null = null;
    let legalWidth = 0;
    let featureCount = 0;
    let candidateMoves: Move[][] | null = null;
    let candidateWidth = 0;
    if (isMoveToken) {
      legalWidth = Math.max(1, Number(this.meta.onnx_fixed_legal_moves ?? this.meta.max_legal_moves ?? 128));
      featureCount = Math.max(1, Number(this.meta.num_move_features ?? 20));
      moveTokenLegal = moveformerLegalInputs(
        normalized.map((n) => n.board),
        legalWidth,
        featureCount,
        normalized,
      );
      const overflow = moveTokenLegal.moves.findIndex((moves) => moves.length > legalWidth);
      if (overflow >= 0 && this.meta.allow_legal_overflow_zero_prior !== true) {
        throw new Error(
          `Move-token ONNX legal move overflow: model accepts ${legalWidth} legal moves but position has ${moveTokenLegal.moves[overflow].length}. Use a larger legal bucket/export (for example k128), dynamic legal export, or set meta.allow_legal_overflow_zero_prior=true to keep legacy zero-prior truncation. fen=${boardToFen(boards[overflow])}`,
        );
      }
      feeds.legal_action_ids = new ort.Tensor('int64', moveTokenLegal.actionIds, [boards.length, legalWidth]);
      feeds.legal_features = new ort.Tensor('float32', moveTokenLegal.features, [boards.length, legalWidth, featureCount]);
      feeds.legal_mask = new ort.Tensor('float32', moveTokenLegal.mask, [boards.length, legalWidth]);
    } else if (this.meta.av_head_exported) {
      candidateMoves = normalized.map((n) => n.legalMoves ?? legalMoves(n.board));
      candidateWidth = Math.max(1, ...candidateMoves.map((m) => m.length));
      const classes = new BigInt64Array(boards.length * candidateWidth);
      for (let i = 0; i < candidateMoves.length; i++) {
        for (let j = 0; j < candidateMoves[i].length; j++) classes[i * candidateWidth + j] = BigInt(moveToChessBenchAvClass(candidateMoves[i][j]));
      }
      feeds.candidate_moves = new ort.Tensor('int64', classes, [boards.length, candidateWidth]);
    }

    const tLegal = ort.tinyLeelaNowMs();
    const outputs = await this.session.run(feeds);
    await ort.downloadOrtOutputs(outputs);
    const tRun1 = ort.tinyLeelaNowMs();

    const policyRaw = requiredFloatOutput(outputs, isMoveToken ? 'policy_logits_legal' : 'policy_logits');
    const wdlRaw = requiredFloatOutput(outputs, 'wdl_logits');
    let avRaw: Float32Array | undefined;
    for (const name of isMoveToken ? ['action_values'] : ['action_values', 'action_value', 'action_value_logits']) {
      const tensor = outputs[name];
      if (!tensor) continue;
      if (!(tensor.data instanceof Float32Array))
        throw new Error(`ONNX output '${name}' expected float32 data, got ${Object.prototype.toString.call(tensor.data)}`);
      avRaw = tensor.data;
      break;
    }
    const auxHeads = (
      [
        ['actionValues', avRaw],
        ['rankScores', outputs.rank_scores?.data as Float32Array | undefined],
        ['regrets', outputs.regrets?.data as Float32Array | undefined],
        ['risks', outputs.risks?.data as Float32Array | undefined],
        ['uncertainties', outputs.uncertainties?.data as Float32Array | undefined],
      ] as const
    ).filter((entry): entry is readonly [(typeof entry)[0], Float32Array] => entry[1] !== undefined);
    const policySize = this.meta.moves?.length ?? Math.floor(policyRaw.length / boards.length);

    const out: Evaluation[] = [];
    for (let i = 0; i < boards.length; i++) {
      const { board, legalMoves: legalOverride, flipped } = normalized[i];
      const policy = new Map<number, number>();
      let auxMoves: Move[] | null = null;
      let auxCount = 0;
      let auxStride = 0;
      if (moveTokenLegal) {
        const moves = moveTokenLegal.moves[i];
        const n = Math.min(legalWidth, moves.length);
        if (n === 0) {
          out.push({ policy, wdl: inCheck(boards[i]) ? [0, 0, 1] : [0, 1, 0] });
          continue;
        }
        const probs = softmax(policyRaw.subarray(i * legalWidth, i * legalWidth + n));
        for (let j = 0; j < n; j++) policy.set(moveToActionId(normalizedMoveToOriginal(moves[j], flipped)), probs[j] ?? 0);
        // Legacy compatibility path, enabled only by meta.allow_legal_overflow_zero_prior.
        for (let j = legalWidth; j < moves.length; j++) policy.set(moveToActionId(normalizedMoveToOriginal(moves[j], flipped)), 0);
        auxMoves = moves;
        auxCount = n;
        auxStride = legalWidth;
      } else {
        const probs = softmax(policyRaw.subarray(i * policySize, (i + 1) * policySize));
        const legal = legalOverride ?? legalMoves(board);
        let legalMass = 0;
        for (const move of legal) {
          const index = moveToResidualPolicyIndex(move);
          if (index !== undefined) legalMass += probs[index] ?? 0;
        }
        for (const move of legal) {
          const actionId = moveToActionId(normalizedMoveToOriginal(move, flipped));
          if (legalMass <= 0) policy.set(actionId, 1 / legal.length);
          else {
            const index = moveToResidualPolicyIndex(move);
            policy.set(actionId, index === undefined ? 0 : probs[index] / legalMass);
          }
        }
        if (candidateMoves) {
          auxMoves = candidateMoves[i];
          auxCount = auxMoves.length;
          auxStride = candidateWidth;
        }
      }
      const wdl = softmax(wdlRaw.subarray(i * 3, i * 3 + 3));
      const evaluation: Evaluation = { policy, wdl: [wdl[0] ?? 0, wdl[1] ?? 0, wdl[2] ?? 0] };
      if (auxMoves && auxCount > 0 && auxHeads.length > 0) {
        const actionIds = auxMoves.slice(0, auxCount).map((m) => moveToActionId(normalizedMoveToOriginal(m, flipped)));
        for (const [field, raw] of auxHeads) {
          const map = new Map<number, number>();
          for (let j = 0; j < auxCount; j++) map.set(actionIds[j], Number(raw[i * auxStride + j] ?? 0));
          evaluation[field] = map;
        }
      }
      out.push(evaluation);
    }
    const tDone = ort.tinyLeelaNowMs();
    ort.tinyLeelaLogLatency(isMoveToken ? 'onnx.evaluateBatch.moveToken' : 'onnx.evaluateBatch', {
      batch: boards.length,
      ...(isMoveToken ? { legalWidth, featureCount } : { candidateWidth }),
      normalizeMs: tNormalized - t0,
      encodeMs: tEncoded - tNormalized,
      legalMs: tLegal - tEncoded,
      ortRunMs: tRun1 - tLegal,
      postprocessMs: tDone - tRun1,
      totalMs: tDone - t0,
    });
    return out;
  }
}
