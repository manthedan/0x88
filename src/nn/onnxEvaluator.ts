import * as ort from 'onnxruntime-web';
import { boardToFen, type BoardState } from '../chess/board.ts';
import { legalMoves } from '../chess/movegen.ts';
import { moveToActionId, type Move } from '../chess/moveCodec.ts';
import { POLICY_MAP, moveToPolicyIndex } from '../chess/policyMap.ts';
import type { Evaluation, EvaluationContext, Evaluator } from './evaluator.ts';

const PIECES = 'PNBRQKpnbrqk';

export interface OnnxStudentMeta {
  kind: string;
  architecture: 'residual_tower';
  policy_map: string;
  moves: string[];
  channels: number;
  blocks: number;
  history_plies: number;
  input_planes: number;
  onnx?: string;
}

function softmax(xs: ArrayLike<number>): number[] {
  let m = -Infinity;
  for (let i = 0; i < xs.length; i++) if (xs[i] > m) m = xs[i];
  const out = Array.from(xs, (x) => Math.exp(Number(x) - m));
  const total = out.reduce((a, b) => a + b, 0) || 1;
  return out.map((x) => x / total);
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

export class OnnxEvaluator implements Evaluator {
  private session: ort.InferenceSession;
  private meta: OnnxStudentMeta;
  private historyFens: string[];
  constructor(session: ort.InferenceSession, meta: OnnxStudentMeta, historyFens: string[] = []) {
    this.session = session; this.meta = meta; this.historyFens = historyFens;
    if (meta.policy_map !== POLICY_MAP) throw new Error(`Unsupported policy map: ${meta.policy_map}`);
  }

  static async create(modelPath: string | Uint8Array | ArrayBuffer, meta: OnnxStudentMeta, historyFens: string[] = []): Promise<OnnxEvaluator> {
    const session = await ort.InferenceSession.create(modelPath as never);
    return new OnnxEvaluator(session, meta, historyFens);
  }

  async evaluate(board: BoardState, context: EvaluationContext = {}): Promise<Evaluation> {
    const input = onnxInputPlanes(board, this.meta, context.historyFens ?? this.historyFens);
    const tensor = new ort.Tensor('float32', input, [1, this.meta.input_planes, 8, 8]);
    const outputs = await this.session.run({ planes: tensor });
    const policyRaw = outputs.policy_logits?.data ?? Object.values(outputs)[0].data;
    const wdlRaw = outputs.wdl_logits?.data ?? Object.values(outputs)[1].data;
    const probs = softmax(policyRaw as Float32Array);
    const legal = legalMoves(board);
    const policyIndex = (move: Move) => moveToPolicyIndex(move);
    const legalMass = legal.reduce((sum, move) => sum + (probs[policyIndex(move) ?? -1] ?? 0), 0);
    const policy = new Map<number, number>();
    if (legal.length && legalMass <= 0) for (const move of legal) policy.set(moveToActionId(move), 1 / legal.length);
    else for (const move of legal) { const index = policyIndex(move); policy.set(moveToActionId(move), index === undefined ? 0 : probs[index] / legalMass); }
    const wdl = softmax(wdlRaw as Float32Array);
    return { policy, wdl: [wdl[0] ?? 0, wdl[1] ?? 0, wdl[2] ?? 0] };
  }
}
