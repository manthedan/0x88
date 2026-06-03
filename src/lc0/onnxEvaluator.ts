import * as ort from '../nn/ortRuntime.ts';
import { parseFen, type BoardState } from '../chess/board.ts';
import { legalMoves } from '../chess/movegen.ts';
import { moveToUci, type Move } from '../chess/moveCodec.ts';
import { encodeLc0Classical112, type Lc0HistoryFill } from './encoder112.ts';
import { LC0_MIRROR_TRANSFORM, uciToLc0PolicyIndex } from './policyMap.ts';

export const LC0_ONNX_INPUT_PLANES = '/input/planes';
export const LC0_ONNX_OUTPUT_POLICY = '/output/policy';
export const LC0_ONNX_OUTPUT_WDL = '/output/wdl';
export const LC0_ONNX_OUTPUT_MLH = '/output/mlh';
export const LC0_DEFAULT_POLICY_TEMPERATURE = 1.359;

export interface Lc0LegalPrior {
  uci: string;
  index: number;
  logit: number;
  prior: number;
}

export interface Lc0Evaluation {
  fen: string;
  wdl: [number, number, number];
  q: number;
  mlh: number;
  legalPriors: Lc0LegalPrior[];
  bestMove?: string;
}

export interface Lc0OnnxEvaluatorOptions {
  policyTemperature?: number;
  historyFill?: Lc0HistoryFill;
}

function fileOf(square: number): number {
  return square % 8;
}

function isStandardCastlingMove(board: BoardState, move: Move): boolean {
  const piece = board.squares[move.from];
  return piece?.[1] === 'k' && Math.abs(fileOf(move.to) - fileOf(move.from)) === 2;
}

function tensorData(outputs: Awaited<ReturnType<ort.InferenceSession['run']>>, name: string): Float32Array | Float16Array | number[] {
  const tensor = outputs[name];
  if (!tensor) throw new Error(`LC0 ONNX output ${name} missing`);
  return tensor.data as Float32Array | Float16Array | number[];
}

function legalPolicyPriors(board: BoardState, logits: ArrayLike<number>, policyTemperature: number): Lc0LegalPrior[] {
  const moveTransform = board.turn === 'b' ? LC0_MIRROR_TRANSFORM : 0;
  const legal = legalMoves(board).map((move) => {
    const uci = moveToUci(move);
    const index = uciToLc0PolicyIndex(uci, moveTransform, { standardCastling: isStandardCastlingMove(board, move) });
    if (index === undefined) throw new Error(`No LC0 policy index for legal move ${uci}`);
    return { uci, index, logit: Number(logits[index]) / policyTemperature };
  });
  if (legal.length === 0) return [];
  const max = Math.max(...legal.map((entry) => entry.logit));
  const sum = legal.reduce((acc, entry) => acc + Math.exp(entry.logit - max), 0);
  return legal
    .map((entry) => ({ ...entry, prior: Math.exp(entry.logit - max) / sum }))
    .sort((a, b) => b.prior - a.prior);
}

export class Lc0OnnxEvaluator {
  readonly policyTemperature: number;
  readonly historyFill: Lc0HistoryFill;
  private readonly session: ort.InferenceSession;

  constructor(session: ort.InferenceSession, options: Lc0OnnxEvaluatorOptions = {}) {
    this.session = session;
    this.policyTemperature = options.policyTemperature ?? LC0_DEFAULT_POLICY_TEMPERATURE;
    this.historyFill = options.historyFill ?? 'fen_only';
  }

  static async create(modelPath: string | Uint8Array | ArrayBuffer, options: Lc0OnnxEvaluatorOptions = {}): Promise<Lc0OnnxEvaluator> {
    return new Lc0OnnxEvaluator(await ort.createOrtSession(modelPath), options);
  }

  async evaluate(boardOrFen: BoardState | string): Promise<Lc0Evaluation> {
    const board = typeof boardOrFen === 'string' ? parseFen(boardOrFen) : boardOrFen;
    const fen = typeof boardOrFen === 'string' ? boardOrFen : '';
    const encoded = encodeLc0Classical112(board, { historyFill: this.historyFill });
    const outputs = await this.session.run({
      [LC0_ONNX_INPUT_PLANES]: new ort.Tensor('float32', encoded.planes, [1, 112, 8, 8]),
    });
    const policy = tensorData(outputs, LC0_ONNX_OUTPUT_POLICY);
    const wdlRaw = tensorData(outputs, LC0_ONNX_OUTPUT_WDL);
    const mlhRaw = tensorData(outputs, LC0_ONNX_OUTPUT_MLH);
    const wdl: [number, number, number] = [Number(wdlRaw[0]), Number(wdlRaw[1]), Number(wdlRaw[2])];
    const legalPriors = legalPolicyPriors(board, policy, this.policyTemperature);
    return {
      fen,
      wdl,
      q: wdl[0] - wdl[2],
      mlh: Number(mlhRaw[0]),
      legalPriors,
      bestMove: legalPriors[0]?.uci,
    };
  }
}
