import * as ort from './ortRuntime.ts';
import { boardToFen, opposite, type BoardState, type Color, type PieceRole } from '../chess/board.ts';
import { isStmWhiteRankflip, normalizePositionForStmWhite, normalizedMoveToOriginal, rankFlipSquare } from '../chess/boardNormalization.ts';
import { isSquareAttacked, legalMoves } from '../chess/movegen.ts';
import { moveToActionId, type Move } from '../chess/moveCodec.ts';
import { moveToSquareformerPolicyIndex } from '../chess/moveEncodings.ts';
import type { Evaluation, EvaluationContext, Evaluator } from './evaluator.ts';
import { softmax } from './numerics.ts';

const INPUT_CACHE_ENTRIES = Math.max(0, Math.min(65536, Math.floor(Number(new URLSearchParams(typeof location === 'undefined' ? '' : location.search).get('evalInputCacheEntries') ?? '4096')) || 4096));
type SquareFormerEncodedInput = Float32Array | BigInt64Array | Int32Array;
type CompactIndexDtype = 'int64' | 'int32';
type CompactIndexArray = BigInt64Array | Int32Array;
const PIECES = '.PNBRQKpnbrqk';

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

function encodedInputKey(board: BoardState, historyFens: string[], compact: boolean): string {
  return `${compact ? 'c' : 'f'}:${boardToFen(board)}\nh:${historyFens.join('|')}`;
}

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
  onnx_dynamic_batch?: boolean;
  board_normalization?: string;
  input_index_dtype?: CompactIndexDtype | string;
  attack_summary_feature_count?: number;
  attack_summary_schema?: 'threatgraph_square_summary_v1' | 'threatgraph_square_summary_v2' | string | null;
  attack_summary_scale?: number;
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


export function squareformerFloatInput(board: BoardState, meta: SquareFormerMeta, historyFens: string[] = []): Float32Array {
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

function compactIndexDtype(meta: SquareFormerMeta): CompactIndexDtype {
  return String(meta.input_index_dtype ?? '').toLowerCase() === 'int32' ? 'int32' : 'int64';
}

function compactTensorType(dtype: CompactIndexDtype): 'int64' | 'int32' {
  return dtype === 'int32' ? 'int32' : 'int64';
}

function makeCompactArray(length: number, dtype: CompactIndexDtype): CompactIndexArray {
  return dtype === 'int32' ? new Int32Array(length) : new BigInt64Array(length);
}

function setCompact(data: CompactIndexArray, index: number, value: number) {
  if (data instanceof BigInt64Array) data[index] = BigInt(value);
  else data[index] = value | 0;
}

function addCompactBoard(data: CompactIndexArray, board: BoardState, boardIndex: number, stride: number) {
  for (let sq = 0; sq < 64; sq++) setCompact(data, sq * stride + boardIndex, pieceId(board.squares[sq]));
}

function repetitionKeyCached(board: BoardState): string {
  return boardToFen(board).trim().split(/\s+/).slice(0, 4).join(' ');
}

function compactRepetitionFlags(board: BoardState, historyBoards: Array<BoardState | null>, history: number): number[] {
  const keys: Array<string | null> = [repetitionKeyCached(board)];
  for (let h = 0; h < history; h++) keys.push(historyBoards[h] ? repetitionKeyCached(historyBoards[h]!) : null);
  return keys.map((key, i) => key == null ? 0 : keys.slice(i + 1).some((older) => older === key) ? 1 : 0);
}

export function squareformerCompactInput(board: BoardState, meta: SquareFormerMeta, historyFens: string[] = [], dtype: CompactIndexDtype = compactIndexDtype(meta)): CompactIndexArray {
  const history = meta.history_plies;
  const stride = meta.token_features ?? history + 9;
  const data = makeCompactArray(64 * stride, dtype);
  const historyBoards: Array<BoardState | null> = Array(history).fill(null);
  addCompactBoard(data, board, 0, stride);
  for (let h = 0; h < history; h++) {
    if (!historyFens[h]) continue;
    try {
      historyBoards[h] = parseFenCached(historyFens[h]);
      addCompactBoard(data, historyBoards[h]!, h + 1, stride);
    } catch { /* ignore bad history */ }
  }
  const base = history + 1;
  const stm = board.turn === 'w' ? 1 : 2;
  const flags = castleMask(board.castling);
  const half = Math.max(0, Math.min(255, Math.trunc(board.halfmove || 0)));
  for (let sq = 0; sq < 64; sq++) {
    setCompact(data, sq * stride + base + 0, stm);
    if (base + 1 < stride) setCompact(data, sq * stride + base + 1, flags);
    if (base + 2 < stride) setCompact(data, sq * stride + base + 2, board.epSquare === sq ? 1 : 0);
    if (base + 3 < stride) setCompact(data, sq * stride + base + 3, half);
    // V2 compact token caches add static square topology tokens after rules.
    if (base + 4 < stride) setCompact(data, sq * stride + base + 4, Math.floor(sq / 8));
    if (base + 5 < stride) setCompact(data, sq * stride + base + 5, sq % 8);
    if (base + 6 < stride) setCompact(data, sq * stride + base + 6, (Math.floor(sq / 8) + (sq % 8)) & 1);
    if (base + 7 < stride) setCompact(data, sq * stride + base + 7, sq);
  }
  if (stride >= history + 9 + history + 1) {
    const repFlags = compactRepetitionFlags(board, historyBoards, history);
    const repBase = history + 9;
    for (let sq = 0; sq < 64; sq++) {
      for (let i = 0; i < Math.min(history + 1, repFlags.length); i++) setCompact(data, sq * stride + repBase + i, repFlags[i]);
    }
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

export function squareformerLegalCandidateInputs(boards: BoardState[], width: number, contexts: EvaluationContext[] = [], dtype: CompactIndexDtype = 'int64'): { moves: Move[][]; classes: CompactIndexArray; width: number } {
  const moves = boards.map((board, i) => contexts[i]?.legalMoves ?? legalMoves(board));
  const classes = makeCompactArray(boards.length * width, dtype);
  for (let i = 0; i < moves.length; i++) {
    for (let j = 0; j < Math.min(width, moves[i].length); j++) setCompact(classes, i * width + j, moveToSquareformerPolicyIndex(moves[i][j]));
  }
  return { moves, classes, width };
}

export const THREATGRAPH_SQUARE_SUMMARY_V1_FEATURES = 28;
export const THREATGRAPH_SQUARE_SUMMARY_V2_FEATURES = 64;
export const THREATGRAPH_SQUARE_SUMMARY_V1_FEATURE_NAMES = [
  'stm_attackers_pawn_count',
  'stm_attackers_knight_count',
  'stm_attackers_bishop_count',
  'stm_attackers_rook_count',
  'stm_attackers_queen_count',
  'stm_attackers_king_count',
  'opp_attackers_pawn_count',
  'opp_attackers_knight_count',
  'opp_attackers_bishop_count',
  'opp_attackers_rook_count',
  'opp_attackers_queen_count',
  'opp_attackers_king_count',
  'stm_attackers_total_count',
  'opp_attackers_total_count',
  'stm_min_attacker_value_bucket',
  'opp_min_attacker_value_bucket',
  'occupant_role_type',
  'occupant_color_stm_or_opp',
  'occupant_value_bucket',
  'occupant_attacked_by_enemy',
  'occupant_defended_by_own',
  'occupant_hanging',
  'occupant_pinned_to_king',
  'occupant_value_at_risk_bucket',
  'square_adjacent_to_stm_king',
  'square_adjacent_to_opp_king',
  'chebyshev_distance_to_stm_king_clip8',
  'chebyshev_distance_to_opp_king_clip8',
] as const;
export const THREATGRAPH_SQUARE_SUMMARY_V2_EXTRA_FEATURE_NAMES = [
  'source_forks_ge2_enemy_minor_or_better_targets',
  'source_forks_two_enemy_rook_or_better_targets',
  'source_attacks_undefended_enemy_queen_or_rook',
  'source_attacks_enemy_king_ring_and_material_target',
  'source_sole_defender_of_ge2_own_attacked_targets',
  'source_sole_defender_of_own_queen_or_rook_attacked',
  'source_blocks_friendly_rooklike_attack_on_enemy_queen_or_king',
  'source_blocks_friendly_bishoplike_attack_on_enemy_queen_or_king',
  'occupant_is_fork_target',
  'occupant_is_high_value_fork_target',
  'occupant_safe_mobility_bucket_clip3',
  'occupant_guarded_only_by_attacked_or_overloaded_defender',
  'outgoing_attacks_enemy_pawn_count_clip3',
  'outgoing_attacks_enemy_minor_count_clip3',
  'outgoing_attacks_enemy_rook_count_clip3',
  'outgoing_attacks_enemy_queen_count_clip3',
  'outgoing_attacks_enemy_king_ring_count_clip3',
  'outgoing_attacks_enemy_escape_candidate_count_clip3',
  'outgoing_defends_own_pawn_count_clip3',
  'outgoing_defends_own_minor_count_clip3',
  'outgoing_defends_own_rook_count_clip3',
  'outgoing_defends_own_queen_count_clip3',
  'outgoing_defends_own_king_ring_count_clip3',
  'outgoing_defends_own_escape_candidate_count_clip3',
  'piece_pinned_to_own_king_by_rooklike',
  'piece_pinned_to_own_king_by_bishoplike',
  'piece_relatively_pinned_to_own_queen_or_rook',
  'piece_pins_enemy_to_enemy_king',
  'piece_pins_enemy_to_enemy_queen_or_rook',
  'piece_has_xray_or_skewer_pressure_on_enemy_queen_or_king',
  'square_is_own_king_escape_candidate',
  'own_king_escape_candidate_attacked_by_opp',
  'own_king_escape_candidate_defended_by_stm_nonking',
  'square_is_enemy_king_escape_candidate',
  'enemy_king_escape_candidate_attacked_by_stm',
  'enemy_king_escape_candidate_defended_by_opp_nonking',
] as const;
export const THREATGRAPH_SQUARE_SUMMARY_V2_FEATURE_NAMES = [...THREATGRAPH_SQUARE_SUMMARY_V1_FEATURE_NAMES, ...THREATGRAPH_SQUARE_SUMMARY_V2_EXTRA_FEATURE_NAMES] as const;
export const THREATGRAPH_SQUARE_SUMMARY_V1_SWAP_MOBILITY_OUTGOING_FEATURES = 28;
export const THREATGRAPH_SQUARE_SUMMARY_V1_SWAP_MOBILITY_OUTGOING_SCHEMA = 'threatgraph_square_summary_v1_swap_mobility_outgoing_28';
export const THREATGRAPH_SQUARE_SUMMARY_V1_SWAP_MOBILITY_OUTGOING_V2_INDICES = [
  0, 1, 2, 3, 4, 38,
  6, 7, 8, 9, 10, 40,
  12, 13, 14, 15,
  16, 17, 18, 19, 20, 41, 22, 23,
  42, 43, 26, 27,
] as const;
export const THREATGRAPH_SQUARE_SUMMARY_V1_SWAP_MOBILITY_OUTGOING_FEATURE_NAMES = THREATGRAPH_SQUARE_SUMMARY_V1_SWAP_MOBILITY_OUTGOING_V2_INDICES.map((i) => THREATGRAPH_SQUARE_SUMMARY_V2_FEATURE_NAMES[i]);
const ROLE_INDEX: Record<PieceRole, number> = { p: 0, n: 1, b: 2, r: 3, q: 4, k: 5 };
const ROLE_TYPE: Record<PieceRole, number> = { p: 1, n: 2, b: 3, r: 4, q: 5, k: 6 };
const ROLE_VALUE_BUCKET: Record<PieceRole, number> = { p: 1, n: 3, b: 3, r: 5, q: 9, k: 20 };
const KNIGHT_DELTAS = [[1, 2], [2, 1], [-1, 2], [-2, 1], [1, -2], [2, -1], [-1, -2], [-2, -1]] as const;
const KING_DELTAS = [[1, 1], [1, 0], [1, -1], [0, 1], [0, -1], [-1, 1], [-1, 0], [-1, -1]] as const;
const BISHOP_DELTAS = [[1, 1], [1, -1], [-1, 1], [-1, -1]] as const;
const ROOK_DELTAS = [[1, 0], [-1, 0], [0, 1], [0, -1]] as const;

type AttackAccumulator = {
  byRole: Uint8Array;
  total: Uint8Array;
  minValue: Uint8Array;
};

const fileOf = (sq: number) => sq & 7;
const rankOf = (sq: number) => sq >> 3;
const onBoard = (file: number, rank: number) => file >= 0 && file < 8 && rank >= 0 && rank < 8;
const sqIndex = (file: number, rank: number) => file + rank * 8;
const colorIndex = (color: Color) => color === 'w' ? 0 : 1;
const roleOf = (piece: string): PieceRole => piece[1] as PieceRole;
const chebyshev = (a: number, b: number) => Math.max(Math.abs(fileOf(a) - fileOf(b)), Math.abs(rankOf(a) - rankOf(b)));

function kingSquareLocal(board: BoardState, color: Color): number | null {
  const target = `${color}k`;
  const sq = board.squares.findIndex((piece) => piece === target);
  return sq >= 0 ? sq : null;
}

function markAttack(acc: AttackAccumulator, color: Color, role: PieceRole, sq: number): void {
  const c = colorIndex(color);
  const r = ROLE_INDEX[role];
  const roleIndex = (c * 64 + sq) * 6 + r;
  acc.byRole[roleIndex] = Math.min(8, acc.byRole[roleIndex] + 1);
  const idx = c * 64 + sq;
  acc.total[idx] = Math.min(8, acc.total[idx] + 1);
  const v = ROLE_VALUE_BUCKET[role];
  if (acc.minValue[idx] === 0 || v < acc.minValue[idx]) acc.minValue[idx] = v;
}

function addSliderAttacks(board: BoardState, acc: AttackAccumulator, from: number, color: Color, role: PieceRole, dirs: readonly (readonly [number, number])[]): void {
  for (const [df, dr] of dirs) {
    let f = fileOf(from) + df;
    let r = rankOf(from) + dr;
    while (onBoard(f, r)) {
      const sq = sqIndex(f, r);
      markAttack(acc, color, role, sq);
      if (board.squares[sq]) break;
      f += df;
      r += dr;
    }
  }
}

function threatgraphAttackAccumulator(board: BoardState): AttackAccumulator {
  const acc = {
    byRole: new Uint8Array(2 * 64 * 6),
    total: new Uint8Array(2 * 64),
    minValue: new Uint8Array(2 * 64),
  };
  for (let from = 0; from < 64; from++) {
    const piece = board.squares[from];
    if (!piece) continue;
    const color = piece[0] as Color;
    const role = roleOf(piece);
    const f0 = fileOf(from);
    const r0 = rankOf(from);
    if (role === 'p') {
      const dr = color === 'w' ? 1 : -1;
      for (const df of [-1, 1]) {
        const f = f0 + df;
        const r = r0 + dr;
        if (onBoard(f, r)) markAttack(acc, color, role, sqIndex(f, r));
      }
    } else if (role === 'n') {
      for (const [df, dr] of KNIGHT_DELTAS) {
        const f = f0 + df;
        const r = r0 + dr;
        if (onBoard(f, r)) markAttack(acc, color, role, sqIndex(f, r));
      }
    } else if (role === 'k') {
      for (const [df, dr] of KING_DELTAS) {
        const f = f0 + df;
        const r = r0 + dr;
        if (onBoard(f, r)) markAttack(acc, color, role, sqIndex(f, r));
      }
    } else if (role === 'b') addSliderAttacks(board, acc, from, color, role, BISHOP_DELTAS);
    else if (role === 'r') addSliderAttacks(board, acc, from, color, role, ROOK_DELTAS);
    else if (role === 'q') {
      addSliderAttacks(board, acc, from, color, role, BISHOP_DELTAS);
      addSliderAttacks(board, acc, from, color, role, ROOK_DELTAS);
    }
  }
  return acc;
}

function isPinnedToKingLocal(board: BoardState, color: Color, sq: number): boolean {
  const piece = board.squares[sq];
  if (!piece || piece[0] !== color || piece[1] === 'k') return false;
  const king = kingSquareLocal(board, color);
  if (king === null) return false;
  const without: BoardState = { ...board, squares: [...board.squares] };
  without.squares[sq] = null;
  return isSquareAttacked(without, king, opposite(color));
}

export function threatgraphSquareSummaryV1(board: BoardState, rankflipColorSwap = false): Float32Array {
  const fdim = THREATGRAPH_SQUARE_SUMMARY_V1_FEATURES;
  const acc = threatgraphAttackAccumulator(board);
  const out = new Float32Array(64 * fdim);
  const stm = board.turn;
  const opp = opposite(stm);
  const stmIndex = colorIndex(stm);
  const oppIndex = colorIndex(opp);
  const stmKing = kingSquareLocal(board, stm);
  const oppKing = kingSquareLocal(board, opp);
  for (let outSq = 0; outSq < 64; outSq++) {
    const sq = rankflipColorSwap ? rankFlipSquare(outSq) : outSq;
    const base = outSq * fdim;
    for (let i = 0; i < 6; i++) {
      out[base + i] = acc.byRole[(stmIndex * 64 + sq) * 6 + i];
      out[base + 6 + i] = acc.byRole[(oppIndex * 64 + sq) * 6 + i];
    }
    const stmAttackers = acc.total[stmIndex * 64 + sq];
    const oppAttackers = acc.total[oppIndex * 64 + sq];
    out[base + 12] = stmAttackers;
    out[base + 13] = oppAttackers;
    out[base + 14] = acc.minValue[stmIndex * 64 + sq];
    out[base + 15] = acc.minValue[oppIndex * 64 + sq];
    const piece = board.squares[sq];
    if (piece) {
      const color = piece[0] as Color;
      const role = roleOf(piece);
      const enemy = opposite(color);
      const ownIndex = colorIndex(color);
      const enemyIndex = colorIndex(enemy);
      const ownAttackers = acc.total[ownIndex * 64 + sq];
      const enemyAttackers = acc.total[enemyIndex * 64 + sq];
      const minEnemy = acc.minValue[enemyIndex * 64 + sq];
      const pieceValue = ROLE_VALUE_BUCKET[role];
      out[base + 16] = ROLE_TYPE[role];
      out[base + 17] = color === stm ? 1 : 2;
      out[base + 18] = pieceValue;
      out[base + 19] = enemyAttackers > 0 ? 1 : 0;
      out[base + 20] = ownAttackers > 0 ? 1 : 0;
      out[base + 21] = enemyAttackers > 0 && ownAttackers === 0 ? 1 : 0;
      out[base + 22] = isPinnedToKingLocal(board, color, sq) ? 1 : 0;
      out[base + 23] = role !== 'k' && enemyAttackers > 0 && (ownAttackers === 0 || (minEnemy > 0 && minEnemy <= pieceValue)) ? pieceValue : 0;
    }
    out[base + 24] = stmKing !== null && chebyshev(stmKing, sq) <= 1 ? 1 : 0;
    out[base + 25] = oppKing !== null && chebyshev(oppKing, sq) <= 1 ? 1 : 0;
    out[base + 26] = stmKing === null ? 8 : Math.min(8, chebyshev(stmKing, sq));
    out[base + 27] = oppKing === null ? 8 : Math.min(8, chebyshev(oppKing, sq));
  }
  return out;
}

function lineDirection(a: number, b: number): readonly [number, number] | null {
  const df = fileOf(b) - fileOf(a);
  const dr = rankOf(b) - rankOf(a);
  if (df === 0 && dr === 0) return null;
  if (df === 0) return [0, dr > 0 ? 1 : -1];
  if (dr === 0) return [df > 0 ? 1 : -1, 0];
  if (Math.abs(df) === Math.abs(dr)) return [df > 0 ? 1 : -1, dr > 0 ? 1 : -1];
  return null;
}

function raySquaresFrom(sq: number, direction: readonly [number, number]): number[] {
  const out: number[] = [];
  let f = fileOf(sq) + direction[0];
  let r = rankOf(sq) + direction[1];
  while (onBoard(f, r)) {
    out.push(sqIndex(f, r));
    f += direction[0];
    r += direction[1];
  }
  return out;
}

function directAttackTargets(board: BoardState, from: number): number[] {
  const piece = board.squares[from];
  if (!piece) return [];
  const color = piece[0] as Color;
  const role = roleOf(piece);
  const f0 = fileOf(from);
  const r0 = rankOf(from);
  const out: number[] = [];
  if (role === 'p') {
    const dr = color === 'w' ? 1 : -1;
    for (const df of [-1, 1]) {
      const f = f0 + df;
      const r = r0 + dr;
      if (onBoard(f, r)) out.push(sqIndex(f, r));
    }
  } else if (role === 'n') {
    for (const [df, dr] of KNIGHT_DELTAS) {
      const f = f0 + df;
      const r = r0 + dr;
      if (onBoard(f, r)) out.push(sqIndex(f, r));
    }
  } else if (role === 'k') {
    for (const [df, dr] of KING_DELTAS) {
      const f = f0 + df;
      const r = r0 + dr;
      if (onBoard(f, r)) out.push(sqIndex(f, r));
    }
  } else if (role === 'b' || role === 'r' || role === 'q') {
    const dirs: readonly (readonly [number, number])[] = role === 'b' ? BISHOP_DELTAS : role === 'r' ? ROOK_DELTAS : [...BISHOP_DELTAS, ...ROOK_DELTAS];
    for (const dir of dirs) {
      for (const sq of raySquaresFrom(from, dir)) {
        out.push(sq);
        if (board.squares[sq]) break;
      }
    }
  }
  return out;
}

function attackLinks(board: BoardState): { attacksFrom: number[][]; attackersTo: number[][]; defendersTo: number[][] } {
  const attacksFrom = Array.from({ length: 64 }, () => [] as number[]);
  const attackersTo = Array.from({ length: 64 }, () => [] as number[]);
  const defendersTo = Array.from({ length: 64 }, () => [] as number[]);
  for (let src = 0; src < 64; src++) {
    const sp = board.squares[src];
    if (!sp) continue;
    for (const dst of directAttackTargets(board, src)) {
      attacksFrom[src].push(dst);
      attackersTo[dst].push(src);
      const tp = board.squares[dst];
      if (tp && tp[0] === sp[0]) defendersTo[dst].push(src);
    }
  }
  return { attacksFrom, attackersTo, defendersTo };
}

function occupiedTargetsAttackedBySource(board: BoardState, src: number, color?: Color): number[] {
  return directAttackTargets(board, src).filter((dst) => {
    const piece = board.squares[dst];
    return !!piece && (color === undefined || piece[0] === color);
  });
}

function kingRing(board: BoardState, color: Color): Set<number> {
  const k = kingSquareLocal(board, color);
  if (k === null) return new Set();
  return new Set(Array.from({ length: 64 }, (_, sq) => sq).filter((sq) => chebyshev(k, sq) <= 1));
}

function kingEscapeCandidates(board: BoardState, color: Color): Set<number> {
  const k = kingSquareLocal(board, color);
  const out = new Set<number>();
  if (k === null) return out;
  for (const [df, dr] of KING_DELTAS) {
    const f = fileOf(k) + df;
    const r = rankOf(k) + dr;
    if (!onBoard(f, r)) continue;
    const sq = sqIndex(f, r);
    const piece = board.squares[sq];
    if (!piece || piece[0] !== color) out.add(sq);
  }
  return out;
}

function sliderRolesForDirection(direction: readonly [number, number]): readonly PieceRole[] {
  return (direction[0] === 0 || direction[1] === 0) ? ['r', 'q'] : ['b', 'q'];
}

function pinToOwnKingKind(board: BoardState, color: Color, sq: number): 'rooklike' | 'bishoplike' | null {
  const piece = board.squares[sq];
  const king = kingSquareLocal(board, color);
  if (!piece || piece[0] !== color || piece[1] === 'k' || king === null) return null;
  const dir = lineDirection(king, sq);
  if (!dir || !raySquaresFrom(king, dir).includes(sq)) return null;
  const allowed = sliderRolesForDirection(dir);
  let seenSq = false;
  for (const cur of raySquaresFrom(king, dir)) {
    if (cur === sq) { seenSq = true; continue; }
    const blocker = board.squares[cur];
    if (!blocker) continue;
    if (seenSq && blocker[0] !== color && allowed.includes(roleOf(blocker))) return allowed[0] === 'r' ? 'rooklike' : 'bishoplike';
    return null;
  }
  return null;
}

function relativelyPinnedToOwnMajor(board: BoardState, color: Color, sq: number): boolean {
  const piece = board.squares[sq];
  if (!piece || piece[0] !== color || piece[1] === 'k') return false;
  for (const dir of [...BISHOP_DELTAS, ...ROOK_DELTAS]) {
    const pairs = [[dir, [-dir[0], -dir[1]] as const], [[-dir[0], -dir[1]] as const, dir]] as const;
    for (const [ownDir, enemyDir] of pairs) {
      let ownMajor = false;
      for (const cur of raySquaresFrom(sq, ownDir)) {
        const blocker = board.squares[cur];
        if (!blocker) continue;
        ownMajor = blocker[0] === color && (blocker[1] === 'q' || blocker[1] === 'r');
        break;
      }
      if (!ownMajor) continue;
      const allowed = sliderRolesForDirection(enemyDir);
      for (const cur of raySquaresFrom(sq, enemyDir)) {
        const blocker = board.squares[cur];
        if (!blocker) continue;
        if (blocker[0] !== color && allowed.includes(roleOf(blocker))) return true;
        break;
      }
    }
  }
  return false;
}

function sourcePinsEnemyToRole(board: BoardState, src: number, targetRoles: Set<PieceRole>): boolean {
  const source = board.squares[src];
  if (!source || !['b', 'r', 'q'].includes(source[1])) return false;
  const color = source[0] as Color;
  const role = roleOf(source);
  const dirs: readonly (readonly [number, number])[] = role === 'b' ? BISHOP_DELTAS : role === 'r' ? ROOK_DELTAS : [...BISHOP_DELTAS, ...ROOK_DELTAS];
  for (const dir of dirs) {
    let firstEnemy = false;
    for (const cur of raySquaresFrom(src, dir)) {
      const piece = board.squares[cur];
      if (!piece) continue;
      if (!firstEnemy) {
        if (piece[0] !== color && piece[1] !== 'k') { firstEnemy = true; continue; }
        break;
      }
      if (piece[0] !== color && targetRoles.has(roleOf(piece))) return true;
      break;
    }
  }
  return false;
}

function sourceHasXrayOrSkewerOnEnemyMajorOrKing(board: BoardState, src: number): boolean {
  const source = board.squares[src];
  if (!source || !['b', 'r', 'q'].includes(source[1])) return false;
  const color = source[0] as Color;
  const role = roleOf(source);
  const dirs: readonly (readonly [number, number])[] = role === 'b' ? BISHOP_DELTAS : role === 'r' ? ROOK_DELTAS : [...BISHOP_DELTAS, ...ROOK_DELTAS];
  for (const dir of dirs) {
    let sawBlocker = false;
    for (const cur of raySquaresFrom(src, dir)) {
      const piece = board.squares[cur];
      if (!piece) continue;
      if (!sawBlocker) { sawBlocker = true; continue; }
      if (piece[0] !== color && (piece[1] === 'q' || piece[1] === 'k')) return true;
      break;
    }
  }
  return false;
}

function sourceBlocksFriendlySliderAttack(board: BoardState, src: number, dirs: readonly (readonly [number, number])[], friendlyRoles: Set<PieceRole>): boolean {
  const piece = board.squares[src];
  if (!piece) return false;
  const color = piece[0];
  for (const dir of dirs) {
    const pairs = [[dir, [-dir[0], -dir[1]] as const], [[-dir[0], -dir[1]] as const, dir]] as const;
    for (const [friendlyDir, enemyDir] of pairs) {
      let friendly = false;
      for (const cur of raySquaresFrom(src, friendlyDir)) {
        const blocker = board.squares[cur];
        if (!blocker) continue;
        friendly = blocker[0] === color && friendlyRoles.has(roleOf(blocker));
        break;
      }
      if (!friendly) continue;
      for (const cur of raySquaresFrom(src, enemyDir)) {
        const blocker = board.squares[cur];
        if (!blocker) continue;
        if (blocker[0] !== color && (blocker[1] === 'q' || blocker[1] === 'k')) return true;
        break;
      }
    }
  }
  return false;
}

function movedBoard(board: BoardState, src: number, dst: number): BoardState {
  const squares = [...board.squares];
  squares[dst] = squares[src];
  squares[src] = null;
  return { ...board, squares };
}

function isLegalUnderAbsolutePinLowerBound(board: BoardState, src: number, dst: number, color: Color): boolean {
  const kind = pinToOwnKingKind(board, color, src);
  if (kind === null) return true;
  const king = kingSquareLocal(board, color);
  if (king === null) return true;
  const dir = lineDirection(king, src);
  return !!dir && raySquaresFrom(king, dir).includes(dst);
}

function safeMobilityClip3(board: BoardState, sq: number): number {
  const piece = board.squares[sq];
  if (!piece || !['n', 'b', 'r', 'q'].includes(piece[1])) return 0;
  const color = piece[0] as Color;
  const enemy = opposite(color);
  let count = 0;
  for (const dst of directAttackTargets(board, sq)) {
    const target = board.squares[dst];
    if (target && target[0] === color) continue;
    if (!isLegalUnderAbsolutePinLowerBound(board, sq, dst, color)) continue;
    const nb = movedBoard(board, sq, dst);
    const ownKing = kingSquareLocal(nb, color);
    if (ownKing !== null && isSquareAttacked(nb, ownKing, enemy)) continue;
    if (isSquareAttacked(nb, dst, enemy)) continue;
    count += 1;
    if (count >= 3) return 3;
  }
  return count;
}

function soleDefendedAttackedTargetsForSource(board: BoardState, src: number, exclude?: number): number[] {
  const source = board.squares[src];
  if (!source) return [];
  const color = source[0] as Color;
  const enemy = opposite(color);
  const acc = threatgraphAttackAccumulator(board);
  const colorI = colorIndex(color);
  const enemyI = colorIndex(enemy);
  const out: number[] = [];
  for (const dst of directAttackTargets(board, src)) {
    if (exclude !== undefined && dst === exclude) continue;
    const target = board.squares[dst];
    if (!target || target[0] !== color || target[1] === 'k') continue;
    if (ROLE_VALUE_BUCKET[roleOf(target)] < 3) continue;
    if (acc.total[enemyI * 64 + dst] > 0 && acc.total[colorI * 64 + dst] === 1) out.push(dst);
  }
  return out;
}

function defenderIsAttackedOrOverloaded(board: BoardState, defenderSq: number, defendedVictimSq: number): boolean {
  const defender = board.squares[defenderSq];
  if (!defender) return false;
  const enemy = opposite(defender[0] as Color);
  return isSquareAttacked(board, defenderSq, enemy) || soleDefendedAttackedTargetsForSource(board, defenderSq, defendedVictimSq).length > 0;
}

const clip3 = (n: number) => Math.min(3, Math.max(0, Math.trunc(n)));

export function threatgraphSquareSummaryV1SwapMobilityOutgoing(board: BoardState, rankflipColorSwap = false): Float32Array {
  const source = threatgraphSquareSummaryV2(board, rankflipColorSwap);
  const fdim = THREATGRAPH_SQUARE_SUMMARY_V1_SWAP_MOBILITY_OUTGOING_FEATURES;
  const out = new Float32Array(64 * fdim);
  for (let sq = 0; sq < 64; sq++) {
    const srcBase = sq * THREATGRAPH_SQUARE_SUMMARY_V2_FEATURES;
    const dstBase = sq * fdim;
    for (let i = 0; i < fdim; i++) out[dstBase + i] = source[srcBase + THREATGRAPH_SQUARE_SUMMARY_V1_SWAP_MOBILITY_OUTGOING_V2_INDICES[i]];
  }
  return out;
}

export function threatgraphSquareSummaryV2(board: BoardState, rankflipColorSwap = false): Float32Array {
  const fdim = THREATGRAPH_SQUARE_SUMMARY_V2_FEATURES;
  const prefix = threatgraphSquareSummaryV1(board, rankflipColorSwap);
  const acc = threatgraphAttackAccumulator(board);
  const { attacksFrom, attackersTo, defendersTo } = attackLinks(board);
  const stm = board.turn;
  const opp = opposite(stm);
  const stmI = colorIndex(stm);
  const oppI = colorIndex(opp);
  const ownEscape = kingEscapeCandidates(board, stm);
  const enemyEscape = kingEscapeCandidates(board, opp);
  const out = new Float32Array(64 * fdim);
  for (let outSq = 0; outSq < 64; outSq++) {
    const sq = rankflipColorSwap ? rankFlipSquare(outSq) : outSq;
    const base = outSq * fdim;
    const oldBase = outSq * THREATGRAPH_SQUARE_SUMMARY_V1_FEATURES;
    out.set(prefix.subarray(oldBase, oldBase + THREATGRAPH_SQUARE_SUMMARY_V1_FEATURES), base);
    const piece = board.squares[sq];
    if (piece) {
      const color = piece[0] as Color;
      const enemy = opposite(color);
      const colorI = colorIndex(color);
      const enemyI = colorIndex(enemy);
      const enemyRing = kingRing(board, enemy);
      const enemyEsc = kingEscapeCandidates(board, enemy);
      const ownEsc = kingEscapeCandidates(board, color);
      const enemyTargets = occupiedTargetsAttackedBySource(board, sq, enemy);
      const enemyMinorPlus = enemyTargets.filter((t) => ROLE_VALUE_BUCKET[roleOf(board.squares[t]!)] >= 3);
      const enemyRookPlus = enemyTargets.filter((t) => ['r', 'q', 'k'].includes(board.squares[t]![1]));
      if (enemyMinorPlus.length >= 2) out[base + 28] = 1;
      if (enemyRookPlus.length >= 2) out[base + 29] = 1;
      if (enemyTargets.some((t) => ['q', 'r'].includes(board.squares[t]![1]) && acc.total[enemyI * 64 + t] === 0)) out[base + 30] = 1;
      if (attacksFrom[sq].some((t) => enemyRing.has(t)) && enemyMinorPlus.length > 0) out[base + 31] = 1;
      const sole = soleDefendedAttackedTargetsForSource(board, sq);
      if (sole.length >= 2) out[base + 32] = 1;
      if (sole.some((t) => ['q', 'r'].includes(board.squares[t]![1]))) out[base + 33] = 1;
      if (sourceBlocksFriendlySliderAttack(board, sq, ROOK_DELTAS, new Set<PieceRole>(['r', 'q']))) out[base + 34] = 1;
      if (sourceBlocksFriendlySliderAttack(board, sq, BISHOP_DELTAS, new Set<PieceRole>(['b', 'q']))) out[base + 35] = 1;

      const ownTargetsByEnemySource: number[][] = [];
      for (const src of attackersTo[sq]) {
        const sp = board.squares[src];
        if (!sp || sp[0] !== enemy) continue;
        const targets = occupiedTargetsAttackedBySource(board, src, color).filter((t) => ROLE_VALUE_BUCKET[roleOf(board.squares[t]!)] >= 3);
        if (targets.includes(sq) && targets.length >= 2) ownTargetsByEnemySource.push(targets);
      }
      if (ownTargetsByEnemySource.length > 0) out[base + 36] = 1;
      if (ownTargetsByEnemySource.some((targets) => targets.some((t) => ['r', 'q', 'k'].includes(board.squares[t]![1])))) out[base + 37] = 1;
      out[base + 38] = safeMobilityClip3(board, sq);
      const defenders = defendersTo[sq].filter((src) => board.squares[src]?.[0] === color);
      if (defenders.length && defenders.every((d) => defenderIsAttackedOrOverloaded(board, d, sq))) out[base + 39] = 1;

      const attackCounts = { p: 0, minor: 0, r: 0, q: 0, kingRing: 0, escape: 0 };
      const defendCounts = { p: 0, minor: 0, r: 0, q: 0, kingRing: 0, escape: 0 };
      for (const dst of attacksFrom[sq]) {
        const target = board.squares[dst];
        if (target && target[0] === enemy) {
          if (target[1] === 'p') attackCounts.p += 1;
          else if (target[1] === 'n' || target[1] === 'b') attackCounts.minor += 1;
          else if (target[1] === 'r') attackCounts.r += 1;
          else if (target[1] === 'q') attackCounts.q += 1;
        }
        if (enemyRing.has(dst)) attackCounts.kingRing += 1;
        if (enemyEsc.has(dst)) attackCounts.escape += 1;
        if (target && target[0] === color) {
          if (target[1] === 'p') defendCounts.p += 1;
          else if (target[1] === 'n' || target[1] === 'b') defendCounts.minor += 1;
          else if (target[1] === 'r') defendCounts.r += 1;
          else if (target[1] === 'q') defendCounts.q += 1;
        }
        if (ownEsc.has(dst)) defendCounts.escape += 1;
        const ownKingSq = kingSquareLocal(board, color);
        if (ownKingSq !== null && chebyshev(ownKingSq, dst) <= 1) defendCounts.kingRing += 1;
      }
      out[base + 40] = clip3(attackCounts.p);
      out[base + 41] = clip3(attackCounts.minor);
      out[base + 42] = clip3(attackCounts.r);
      out[base + 43] = clip3(attackCounts.q);
      out[base + 44] = clip3(attackCounts.kingRing);
      out[base + 45] = clip3(attackCounts.escape);
      out[base + 46] = clip3(defendCounts.p);
      out[base + 47] = clip3(defendCounts.minor);
      out[base + 48] = clip3(defendCounts.r);
      out[base + 49] = clip3(defendCounts.q);
      out[base + 50] = clip3(defendCounts.kingRing);
      out[base + 51] = clip3(defendCounts.escape);

      const pinKind = pinToOwnKingKind(board, color, sq);
      out[base + 52] = pinKind === 'rooklike' ? 1 : 0;
      out[base + 53] = pinKind === 'bishoplike' ? 1 : 0;
      out[base + 54] = relativelyPinnedToOwnMajor(board, color, sq) ? 1 : 0;
      out[base + 55] = sourcePinsEnemyToRole(board, sq, new Set<PieceRole>(['k'])) ? 1 : 0;
      out[base + 56] = sourcePinsEnemyToRole(board, sq, new Set<PieceRole>(['q', 'r'])) ? 1 : 0;
      out[base + 57] = sourceHasXrayOrSkewerOnEnemyMajorOrKing(board, sq) ? 1 : 0;
    }
    out[base + 58] = ownEscape.has(sq) ? 1 : 0;
    out[base + 59] = ownEscape.has(sq) && acc.total[oppI * 64 + sq] > 0 ? 1 : 0;
    out[base + 60] = ownEscape.has(sq) && (acc.total[stmI * 64 + sq] - acc.byRole[(stmI * 64 + sq) * 6 + ROLE_INDEX.k]) > 0 ? 1 : 0;
    out[base + 61] = enemyEscape.has(sq) ? 1 : 0;
    out[base + 62] = enemyEscape.has(sq) && acc.total[stmI * 64 + sq] > 0 ? 1 : 0;
    out[base + 63] = enemyEscape.has(sq) && (acc.total[oppI * 64 + sq] - acc.byRole[(oppI * 64 + sq) * 6 + ROLE_INDEX.k]) > 0 ? 1 : 0;
  }
  return out;
}

export function isCompactMeta(meta: SquareFormerMeta): boolean {
  return meta.input_mode === 'embedding' || meta.input_format === 'compact_uint8_embeddings' || meta.input_format === 'compact_uint8_tokens';
}

function outputNames(outputs: Record<string, ort.Tensor>): string {
  return Object.keys(outputs).sort().join(', ') || '<none>';
}

function requiredFloatOutput(outputs: Record<string, ort.Tensor>, name: string): Float32Array {
  const tensor = outputs[name];
  if (!tensor) throw new Error(`SquareFormer ONNX output missing required tensor '${name}'. Available outputs: ${outputNames(outputs)}`);
  if (!(tensor.data instanceof Float32Array)) throw new Error(`SquareFormer ONNX output '${name}' expected float32 data, got ${Object.prototype.toString.call(tensor.data)}`);
  return tensor.data;
}

function optionalFloatOutput(outputs: Record<string, ort.Tensor>, name: string): Float32Array | undefined {
  const tensor = outputs[name];
  if (!tensor) return undefined;
  if (!(tensor.data instanceof Float32Array)) throw new Error(`SquareFormer ONNX output '${name}' expected float32 data, got ${Object.prototype.toString.call(tensor.data)}`);
  return tensor.data;
}

const SQUAREFORMER_USED_OUTPUTS = ['policy', 'wdl', 'wdl_sf18', 'action_values', 'rank_scores', 'regrets'] as const;

function selectedSquareformerOutputs(session: ort.InferenceSession): string[] | undefined {
  const names = (session as unknown as { outputNames?: string[] }).outputNames;
  if (!Array.isArray(names)) return undefined;
  const fetches = SQUAREFORMER_USED_OUTPUTS.filter((name) => names.includes(name));
  return fetches.length ? fetches : undefined;
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

export class SquareFormerEvaluator implements Evaluator {
  private session: ort.InferenceSession;
  private meta: SquareFormerMeta;
  private inputCache = new Map<string, SquareFormerEncodedInput>();
  constructor(session: ort.InferenceSession, meta: SquareFormerMeta) { this.session = session; this.meta = meta; }
  static async create(modelPath: string | Uint8Array | ArrayBuffer, meta: SquareFormerMeta): Promise<SquareFormerEvaluator> {
    return new SquareFormerEvaluator(await ort.createOrtSession(modelPath), meta);
  }
  async evaluate(board: BoardState, context: EvaluationContext = {}): Promise<Evaluation> {
    return (await this.evaluateBatch([board], [context]))[0];
  }

  private cachedEncodedInput(board: BoardState, historyFens: string[], compact: boolean): SquareFormerEncodedInput {
    const key = encodedInputKey(board, historyFens, compact);
    const cached = lruGet(this.inputCache, key);
    if (cached) return cached;
    const encoded = compact ? squareformerCompactInput(board, this.meta, historyFens) : squareformerFloatInput(board, this.meta, historyFens);
    lruPut(this.inputCache, key, encoded, INPUT_CACHE_ENTRIES);
    return encoded;
  }

  async evaluateBatch(boards: BoardState[], contexts: EvaluationContext[] = []): Promise<Evaluation[]> {
    if (!boards.length) return [];
    const t0 = ort.tinyLeelaNowMs();
    if (boards.length > 1 && this.meta.onnx_dynamic_batch !== true) {
      const out: Evaluation[] = [];
      for (let i = 0; i < boards.length; i++) out.push((await this.evaluateBatch([boards[i]], [contexts[i] ?? {}]))[0]);
      ort.tinyLeelaLogLatency('squareformer.evaluateBatch.static-loop', { batch: boards.length, totalMs: ort.tinyLeelaNowMs() - t0 });
      return out;
    }
    const normalized = boards.map((board, i) => {
      const context = contexts[i] ?? {};
      const n = isStmWhiteRankflip(this.meta.board_normalization)
        ? normalizePositionForStmWhite(board, context.historyFens ?? [], context.legalMoves)
        : { board, historyFens: context.historyFens ?? [], legalMoves: context.legalMoves, flipped: false };
      return { ...n, attackSummaryChannelMask: context.attackSummaryChannelMask };
    });
    const tNormalized = ort.tinyLeelaNowMs();
    const evalBoards = normalized.map((n) => n.board);
    const evalContexts = normalized.map((n) => ({ historyFens: n.historyFens, legalMoves: n.legalMoves }));
    const compact = isCompactMeta(this.meta);
    const indexDtype = compactIndexDtype(this.meta);
    const stride = compact ? this.meta.token_features ?? this.meta.history_plies + 9 : this.meta.input_dim;
    const one = 64 * stride;
    const input = compact ? makeCompactArray(boards.length * one, indexDtype) : new Float32Array(boards.length * one);
    for (let i = 0; i < boards.length; i++) {
      const row = this.cachedEncodedInput(evalBoards[i], evalContexts[i]?.historyFens ?? [], compact);
      (input as BigInt64Array | Int32Array | Float32Array).set(row as never, i * one);
    }
    const tEncoded = ort.tinyLeelaNowMs();
    const shape: [number, number, number] = [boards.length, 64, stride];
    const feeds: Record<string, ort.Tensor> = { tokens: compact ? new ort.Tensor(compactTensorType(indexDtype), input as CompactIndexArray, shape) : new ort.Tensor('float32', input as Float32Array, shape) };
    const attackFeatures = Number(this.meta.attack_summary_feature_count ?? 0);
    if (attackFeatures > 0) {
      const encodeAttack = this.meta.attack_summary_schema === 'threatgraph_square_summary_v1' && attackFeatures === THREATGRAPH_SQUARE_SUMMARY_V1_FEATURES
        ? threatgraphSquareSummaryV1
        : this.meta.attack_summary_schema === THREATGRAPH_SQUARE_SUMMARY_V1_SWAP_MOBILITY_OUTGOING_SCHEMA && attackFeatures === THREATGRAPH_SQUARE_SUMMARY_V1_SWAP_MOBILITY_OUTGOING_FEATURES
          ? threatgraphSquareSummaryV1SwapMobilityOutgoing
          : this.meta.attack_summary_schema === 'threatgraph_square_summary_v2' && attackFeatures === THREATGRAPH_SQUARE_SUMMARY_V2_FEATURES
            ? threatgraphSquareSummaryV2
            : null;
      if (!encodeAttack) throw new Error(`Unsupported SquareFormer attack_summary schema/features: schema=${this.meta.attack_summary_schema} features=${attackFeatures}`);
      const attack = new Float32Array(boards.length * 64 * attackFeatures);
      for (let i = 0; i < boards.length; i++) {
        const row = encodeAttack(evalBoards[i]);
        const mask = normalized[i].attackSummaryChannelMask;
        if (mask) {
          if (mask.length !== attackFeatures) throw new Error(`attackSummaryChannelMask length ${mask.length} does not match attack features ${attackFeatures}`);
          for (let sq = 0; sq < 64; sq++) {
            const base = sq * attackFeatures;
            for (let ch = 0; ch < attackFeatures; ch++) row[base + ch] *= Number(mask[ch] ?? 0);
          }
        }
        attack.set(row, i * 64 * attackFeatures);
      }
      feeds.attack_summary = new ort.Tensor('float32', attack, [boards.length, 64, attackFeatures]);
    }
    const tAttack = ort.tinyLeelaNowMs();
    const avWidth = Math.max(1, Number(this.meta.onnx_fixed_legal_moves ?? this.meta.max_legal_moves ?? 0));
    const legalInfo = this.meta.av_head_exported && avWidth > 0 ? squareformerLegalCandidateInputs(evalBoards, avWidth, evalContexts, indexDtype) : null;
    if (legalInfo) feeds.legal_action_ids = new ort.Tensor(compactTensorType(indexDtype), legalInfo.classes, [boards.length, avWidth]);
    const tLegal = ort.tinyLeelaNowMs();
    const fetches = selectedSquareformerOutputs(this.session);
    const tRun0 = ort.tinyLeelaNowMs();
    const outputs = fetches ? await this.session.run(feeds, fetches) : await this.session.run(feeds);
    const tRun1 = ort.tinyLeelaNowMs();
    const policyRaw = requiredFloatOutput(outputs, 'policy');
    const wdlRaw = requiredFloatOutput(outputs, 'wdl');
    const wdlSf18Raw = optionalFloatOutput(outputs, 'wdl_sf18');
    const avRaw = optionalFloatOutput(outputs, 'action_values');
    const rankRaw = optionalFloatOutput(outputs, 'rank_scores');
    const regretRaw = optionalFloatOutput(outputs, 'regrets');
    const policySize = this.meta.policy_size;
    const out = boards.map((board, i) => {
      const evalLegal = evalContexts[i]?.legalMoves ?? legalMoves(evalBoards[i]);
      const policyRow = policyRaw.subarray(i * policySize, (i + 1) * policySize);
      const logits = evalLegal.map((move) => Number(policyRow[moveToSquareformerPolicyIndex(move)] ?? -100));
      const probs = softmax(logits);
      const policy = new Map<number, number>();
      const actionValues = avRaw ? new Map<number, number>() : undefined;
      const rankScores = rankRaw ? new Map<number, number>() : undefined;
      const regrets = regretRaw ? new Map<number, number>() : undefined;
      evalLegal.forEach((move, j) => {
        const originalMove = normalizedMoveToOriginal(move, normalized[i].flipped);
        const actionId = moveToActionId(originalMove);
        policy.set(actionId, probs[j] ?? 0);
        if (avRaw && j < avWidth) actionValues?.set(actionId, Number(avRaw[i * avWidth + j] ?? 0));
        if (rankRaw && j < avWidth) rankScores?.set(actionId, Number(rankRaw[i * avWidth + j] ?? 0));
        if (regretRaw && j < avWidth) regrets?.set(actionId, Number(regretRaw[i * avWidth + j] ?? 0));
      });
      const wdl = softmax(wdlRaw.subarray(i * 3, i * 3 + 3));
      const wdlSf18 = wdlSf18Raw ? softmax(wdlSf18Raw.subarray(i * 3, i * 3 + 3)) : null;
      return {
        policy,
        wdl: [wdl[0] ?? 0, wdl[1] ?? 0, wdl[2] ?? 0] as [number, number, number],
        ...(wdlSf18 ? { auxiliaryWdls: { wdl_sf18: [wdlSf18[0] ?? 0, wdlSf18[1] ?? 0, wdlSf18[2] ?? 0] as [number, number, number] } } : {}),
        actionValues,
        rankScores,
        regrets,
      };
    });
    const tDone = ort.tinyLeelaNowMs();
    ort.tinyLeelaLogLatency('squareformer.evaluateBatch', {
      batch: boards.length,
      compact,
      indexDtype,
      attackFeatures,
      legalWidth: legalInfo ? avWidth : 0,
      outputs: Object.keys(outputs).join(','),
      normalizeMs: tNormalized - t0,
      encodeMs: tEncoded - tNormalized,
      attackMs: tAttack - tEncoded,
      legalMs: tLegal - tAttack,
      ortRunMs: tRun1 - tRun0,
      postprocessMs: tDone - tRun1,
      totalMs: tDone - t0,
    });
    return out;
  }
}
