import { boardToFen, type BoardState } from '../chess/board.ts';
import { pseudoLegalMoves } from '../chess/movegen.ts';
import { moveToActionId, moveToUci } from '../chess/moveCodec.ts';
import type { Evaluation, Evaluator } from './evaluator.ts';

const PIECES = 'PNBRQKpnbrqk';
const PIECE_INDEX = new Map([...PIECES].map((piece, index) => [piece, index]));

export interface StudentArtifact {
  kind: 'linear_fen_student' | 'frozen_conv_fen_student';
  moves: string[];
  policy_weights: number[][];
  wdl_weights: number[][];
  policy_feature_dim: number;
  wdl_feature_dim: number;
  weight_average_count?: number;
  conv_channels?: number;
  conv_layers?: number;
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
    this.moveIndex = new Map(artifact.moves.map((move, index) => [move, index]));
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
    if (this.artifact.kind === 'frozen_conv_fen_student') {
      features = convStudentFeatures(fen, this.artifact.conv_channels ?? 0, this.artifact.conv_layers ?? 0);
    } else {
      features = valueHead ? wdlFeatures(fen) : fenFeatures(fen);
    }
    this.cache.set(key, features);
    return features;
  }

  evaluate(board: BoardState): Evaluation {
    const policyFeatures = this.features(board, false);
    const valueFeatures = this.features(board, true);
    const logits = this.artifact.policy_weights.map((weights) => dot(weights, policyFeatures));
    const probs = softmax(logits);
    const legal = pseudoLegalMoves(board);
    const legalMass = legal.reduce((sum, move) => sum + (probs[this.moveIndex.get(moveToUci(move)) ?? -1] ?? 0), 0);
    const policy = new Map<number, number>();
    if (legal.length && legalMass <= 0) {
      for (const move of legal) policy.set(moveToActionId(move), 1 / legal.length);
    } else {
      for (const move of legal) {
        const index = this.moveIndex.get(moveToUci(move));
        policy.set(moveToActionId(move), index === undefined ? 0 : probs[index] / legalMass);
      }
    }
    const wdl = softmax(this.artifact.wdl_weights.map((weights) => dot(weights, valueFeatures))) as [number, number, number];
    return { policy, wdl };
  }
}
