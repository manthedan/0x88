import { boardToFen, parseFen, type BoardState } from '../chess/board.ts';
import { inCheck, legalMoves } from '../chess/movegen.ts';
import { moveToActionId, moveToUci, type Move } from '../chess/moveCodec.ts';
import { POLICY_INDEX, POLICY_MAP, moveToPolicyIndex } from '../chess/policyMap.ts';
import type { Evaluation, Evaluator } from './evaluator.ts';

const PIECES = 'PNBRQKpnbrqk';
const PIECE_INDEX = new Map([...PIECES].map((piece, index) => [piece, index]));

export interface StudentArtifact {
  kind: 'linear_fen_student' | 'frozen_conv_fen_student' | 'frozen_conv_feature_mlp_student' | 'tiny_board_cnn_student' | 'tiny_board_residual_student';
  moves: string[];
  policy_weights?: number[][];
  wdl_weights?: number[][];
  policy_feature_dim?: number;
  wdl_feature_dim?: number;
  weight_average_count?: number;
  conv_channels?: number;
  conv_layers?: number;
  feature_dim?: number;
  hidden?: number;
  w1?: number[][];
  b1?: number[];
  policy_w?: number[][];
  policy_b?: number[];
  wdl_w?: number[][];
  wdl_b?: number[];
  channels?: number;
  c1_weight?: number[][][][];
  c1_bias?: number[];
  c2_weight?: number[][][][];
  c2_bias?: number[];
  c3_weight?: number[][][][];
  c3_bias?: number[];
  policy_weight?: number[][];
  policy_bias?: number[];
  wdl_weight?: number[][];
  wdl_bias?: number[];
  policy_head?: 'pooled' | 'spatial';
  policy_map?: string | null;
  history_plies?: number;
  input_planes?: number;
  architecture?: 'legacy3' | 'residual_tower';
  blocks?: number;
}

function softmax(xs: number[]): number[] {
  const m = Math.max(...xs);
  const exps = xs.map((x) => Math.exp(x - m));
  const total = exps.reduce((a, b) => a + b, 0);
  return exps.map((x) => x / total);
}

function dot(weights: number[], values: number[]): number {
  let total = 0;
  for (let i = 0; i < Math.min(weights.length, values.length); i++) total += weights[i] * values[i];
  return total;
}

function fenFeatures(fen: string): number[] {
  const [placement, side] = fen.split(/\s+/);
  const counts = Object.fromEntries([...PIECES].map((p) => [p, 0])) as Record<string, number>;
  for (const ch of placement) if (ch in counts) counts[ch]++;
  const vals: Record<string, number> = { P: 1, N: 3, B: 3, R: 5, Q: 9, K: 0 };
  const whiteMat = [...'PNBRQK'].reduce((sum, p) => sum + counts[p] * vals[p], 0);
  const blackMat = [...'pnbrqk'].reduce((sum, p) => sum + counts[p] * vals[p.toUpperCase()], 0);
  return [
    1,
    side === 'w' ? 1 : -1,
    ...[...PIECES].map((p) => (counts[p] - 2) / 8),
    (whiteMat - blackMat) / 39,
  ];
}

function wdlFeatures(fen: string): number[] {
  const base = fenFeatures(fen);
  const sideSign = base[1];
  return [...base, ...base.slice(2).map((v) => sideSign * v)];
}

function stableWeight(...values: number[]): number {
  let seed = 0x9E3779B97F4A7C15n;
  const mask = 0xFFFFFFFFFFFFFFFFn;
  for (const value of values) {
    const v = BigInt(value);
    seed ^= (v + 0x9E3779B9n + ((seed << 6n) & mask) + (seed >> 2n)) & mask;
    seed &= mask;
  }
  return ((Number(seed % 2001n) / 1000) - 1) / Math.sqrt(values.length + 1);
}

const CONV_PARAM_CACHE = new Map<string, { biases: number[][]; kernels: number[][][][][] }>();

function convParams(channels: number, layers: number) {
  const key = `${channels}x${layers}`;
  const cached = CONV_PARAM_CACHE.get(key);
  if (cached) return cached;
  const biases: number[][] = [];
  const kernels: number[][][][][] = [];
  for (let layer = 0; layer < layers; layer++) {
    const prevChannels = layer === 0 ? 13 : channels;
    biases[layer] = [];
    kernels[layer] = [];
    for (let c = 0; c < channels; c++) {
      biases[layer][c] = stableWeight(layer, c, 99);
      kernels[layer][c] = [];
      for (let pc = 0; pc < prevChannels; pc++) {
        kernels[layer][c][pc] = [];
        for (let dri = 0; dri < 3; dri++) {
          kernels[layer][c][pc][dri] = [];
          for (let dfi = 0; dfi < 3; dfi++) kernels[layer][c][pc][dri][dfi] = stableWeight(layer, c, pc, dri, dfi);
        }
      }
    }
  }
  const params = { biases, kernels };
  CONV_PARAM_CACHE.set(key, params);
  return params;
}

function boardPlanes(fen: string, inputPlanes = 14, historyPlies = 0): number[][][] {
  const [placement, side = 'w', castling = '-', ep = '-'] = fen.split(/\s+/);
  const maps = Array.from({ length: inputPlanes }, () => Array.from({ length: 8 }, () => Array(8).fill(0)));
  let rank = 0;
  let file = 0;
  for (const ch of placement) {
    if (ch === '/') { rank++; file = 0; }
    else if (/\d/.test(ch)) file += Number(ch);
    else if (PIECE_INDEX.has(ch)) maps[PIECE_INDEX.get(ch)!][rank][file++] = 1;
  }
  const state0 = 12 * (historyPlies + 1);
  const sideValue = side === 'w' ? 1 : -1;
  for (let r = 0; r < 8; r++) for (let f = 0; f < 8; f++) maps[state0][r][f] = sideValue;
  if (inputPlanes - state0 >= 10) {
    ['K', 'Q', 'k', 'q'].forEach((flag, i) => { if (castling.includes(flag)) for (let r = 0; r < 8; r++) for (let f = 0; f < 8; f++) maps[state0 + 1 + i][r][f] = 1; });
    if (ep !== '-' && ep.length >= 2) {
      const ef = ep.charCodeAt(0) - 97;
      const er = 8 - Number(ep[1]);
      if (er >= 0 && er < 8 && ef >= 0 && ef < 8) maps[state0 + 5][er][ef] = 1;
    }
    let stmCheck = 0;
    let oppCheck = 0;
    if (inputPlanes - state0 >= 10) {
      const board = parseFen(fen);
      stmCheck = inCheck(board, board.turn) ? 1 : 0;
      oppCheck = inCheck(board, board.turn === 'w' ? 'b' : 'w') ? 1 : 0;
    }
    for (let r = 0; r < 8; r++) for (let f = 0; f < 8; f++) { maps[state0 + 6][r][f] = 1; maps[state0 + 7][r][f] = side === 'w' ? 1 : 0; maps[state0 + 8][r][f] = stmCheck; maps[state0 + 9][r][f] = oppCheck; }
  } else {
    for (let r = 0; r < 8; r++) for (let f = 0; f < 8; f++) maps[state0 + 1][r][f] = 1;
  }
  return maps;
}

function convReluResidual(input: number[][][], weights: number[][][][], biases: number[], residual: boolean): number[][][] {
  const out = Array.from({ length: biases.length }, () => Array.from({ length: 8 }, () => Array(8).fill(0)));
  for (let oc = 0; oc < biases.length; oc++) {
    for (let r = 0; r < 8; r++) for (let f = 0; f < 8; f++) {
      let acc = biases[oc];
      for (let ic = 0; ic < input.length; ic++) for (let kr = 0; kr < 3; kr++) for (let kf = 0; kf < 3; kf++) {
        const rr = r + kr - 1;
        const ff = f + kf - 1;
        if (rr >= 0 && rr < 8 && ff >= 0 && ff < 8) acc += input[ic][rr][ff] * (weights[oc]?.[ic]?.[kr]?.[kf] ?? 0);
      }
      const v = Math.max(0, acc);
      out[oc][r][f] = residual && oc < input.length ? v + input[oc][r][f] : v;
    }
  }
  return out;
}

function boardCnnLogits(fen: string, artifact: StudentArtifact): { policy: number[]; wdl: number[] } {
  const h1 = convReluResidual(boardPlanes(fen, artifact.input_planes ?? artifact.c1_weight?.[0]?.length ?? 14, artifact.history_plies ?? 0), artifact.c1_weight ?? [], artifact.c1_bias ?? [], false);
  const h2 = convReluResidual(h1, artifact.c2_weight ?? [], artifact.c2_bias ?? [], true);
  const h3 = convReluResidual(h2, artifact.c3_weight ?? [], artifact.c3_bias ?? [], true);
  const pooled = h3.map((channel) => channel.flat().reduce((a, b) => a + b, 0) / 64);
  const spatial = h3.flatMap((channel) => channel.flat());
  const policyFeatures = artifact.policy_head === 'spatial' ? spatial : pooled;
  return {
    policy: (artifact.policy_bias ?? []).map((bias, i) => bias + dot(artifact.policy_weight?.[i] ?? [], policyFeatures)),
    wdl: (artifact.wdl_bias ?? []).map((bias, i) => bias + dot(artifact.wdl_weight?.[i] ?? [], pooled)),
  };
}

function convStudentFeatures(fen: string, channels: number, layers: number): number[] {
  const [placement, side] = fen.split(/\s+/);
  let maps = Array.from({ length: 13 }, () => Array.from({ length: 8 }, () => Array(8).fill(0)));
  let rank = 0;
  let file = 0;
  for (const ch of placement) {
    if (ch === '/') {
      rank++;
      file = 0;
    } else if (/\d/.test(ch)) {
      file += Number(ch);
    } else if (PIECE_INDEX.has(ch)) {
      maps[PIECE_INDEX.get(ch)!][rank][file] = 1;
      file++;
    }
  }
  const sideValue = side === 'w' ? 1 : -1;
  maps[12] = Array.from({ length: 8 }, () => Array(8).fill(sideValue));

  const params = convParams(channels, layers);
  let prev = maps;
  for (let layer = 0; layer < layers; layer++) {
    const prevChannels = prev.length;
    const out = Array.from({ length: channels }, () => Array.from({ length: 8 }, () => Array(8).fill(0)));
    for (let c = 0; c < channels; c++) {
      for (let r = 0; r < 8; r++) {
        for (let f = 0; f < 8; f++) {
          let acc = params.biases[layer][c];
          for (let pc = 0; pc < prevChannels; pc++) {
            for (const dr of [-1, 0, 1]) {
              const rr = r + dr;
              if (rr < 0 || rr >= 8) continue;
              for (const df of [-1, 0, 1]) {
                const ff = f + df;
                if (ff >= 0 && ff < 8) acc += prev[pc][rr][ff] * params.kernels[layer][c][pc][dr + 1][df + 1];
              }
            }
          }
          out[c][r][f] = Math.tanh(acc / Math.sqrt(prevChannels * 4));
        }
      }
    }
    prev = out;
  }

  const feats = [1, sideValue];
  for (const channel of prev) {
    const flat = channel.flat();
    feats.push(flat.reduce((a, b) => a + b, 0) / 64, Math.max(...flat), Math.min(...flat));
  }
  return feats;
}

export class StudentEvaluator implements Evaluator {
  private artifact: StudentArtifact;
  private moveIndex: Map<string, number>;
  private cache = new Map<string, number[]>();

  constructor(artifact: StudentArtifact) {
    this.artifact = artifact;
    this.moveIndex = artifact.policy_map === POLICY_MAP ? POLICY_INDEX : new Map(artifact.moves.map((move, index) => [move, index]));
  }

  static fromJson(json: string): StudentEvaluator {
    return new StudentEvaluator(JSON.parse(json) as StudentArtifact);
  }

  private features(board: BoardState, valueHead = false): number[] {
    const fen = boardToFen(board);
    const key = `${valueHead ? 'v' : 'p'}:${fen}`;
    const cached = this.cache.get(key);
    if (cached) return cached;
    let features: number[];
    if (this.artifact.kind === 'frozen_conv_fen_student' || this.artifact.kind === 'frozen_conv_feature_mlp_student') {
      features = convStudentFeatures(fen, this.artifact.conv_channels ?? 64, this.artifact.conv_layers ?? 6);
    } else {
      features = valueHead ? wdlFeatures(fen) : fenFeatures(fen);
    }
    this.cache.set(key, features);
    return features;
  }

  evaluate(board: BoardState): Evaluation {
    const policyFeatures = this.features(board, false);
    const valueFeatures = this.features(board, true);
    let logits: number[];
    let wdlLogits: number[];
    if (this.artifact.kind === 'tiny_board_residual_student' || this.artifact.architecture === 'residual_tower') {
      throw new Error('Residual tower artifacts require the upcoming ONNX/runtime path');
    } else if (this.artifact.kind === 'tiny_board_cnn_student') {
      const cnn = boardCnnLogits(boardToFen(board), this.artifact);
      logits = cnn.policy;
      wdlLogits = cnn.wdl;
    } else if (this.artifact.kind === 'frozen_conv_feature_mlp_student') {
      const hidden = (this.artifact.b1 ?? []).map((bias, h) => Math.max(0, bias + dot((this.artifact.w1 ?? []).map((row) => row[h] ?? 0), policyFeatures)));
      logits = (this.artifact.policy_b ?? []).map((bias, move) => bias + dot((this.artifact.policy_w ?? []).map((row) => row[move] ?? 0), hidden));
      wdlLogits = (this.artifact.wdl_b ?? []).map((bias, k) => bias + dot((this.artifact.wdl_w ?? []).map((row) => row[k] ?? 0), hidden));
    } else {
      logits = (this.artifact.policy_weights ?? []).map((weights) => dot(weights, policyFeatures));
      wdlLogits = (this.artifact.wdl_weights ?? []).map((weights) => dot(weights, valueFeatures));
    }
    const probs = softmax(logits);
    const legal = legalMoves(board);
    const policyIndex = (move: Move) => this.artifact.policy_map === POLICY_MAP ? moveToPolicyIndex(move) : this.moveIndex.get(moveToUci(move));
    let legalMass = 0;
    for (const move of legal) {
      const index = policyIndex(move);
      if (index !== undefined) legalMass += probs[index] ?? 0;
    }
    const policy = new Map<number, number>();
    if (legal.length && legalMass <= 0) {
      for (const move of legal) policy.set(moveToActionId(move), 1 / legal.length);
    } else {
      for (const move of legal) {
        const index = policyIndex(move);
        policy.set(moveToActionId(move), index === undefined ? 0 : probs[index] / legalMass);
      }
    }
    const wdl = softmax(wdlLogits) as [number, number, number];
    return { policy, wdl };
  }
}
