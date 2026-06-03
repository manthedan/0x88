import { Lc0OnnxEvaluator, type Lc0Evaluation, type Lc0EvaluatorInput, type Lc0OnnxEvaluatorOptions } from './onnxEvaluator.ts';

export interface Lc0PolicyOnlyChoice {
  fen: string;
  move?: string;
  evaluation: Lc0Evaluation;
}

export interface Lc0PolicyOnlyEvaluator {
  evaluate(boardOrFen: Lc0EvaluatorInput): Promise<Lc0Evaluation>;
}

export class Lc0PolicyOnlyPlayer {
  private readonly evaluator: Lc0PolicyOnlyEvaluator;

  constructor(evaluator: Lc0PolicyOnlyEvaluator) {
    this.evaluator = evaluator;
  }

  static async create(modelPath: string | Uint8Array | ArrayBuffer, options: Lc0OnnxEvaluatorOptions = {}): Promise<Lc0PolicyOnlyPlayer> {
    return new Lc0PolicyOnlyPlayer(await Lc0OnnxEvaluator.create(modelPath, options));
  }

  async chooseMove(boardOrFen: Lc0EvaluatorInput): Promise<Lc0PolicyOnlyChoice> {
    const evaluation = await this.evaluator.evaluate(boardOrFen);
    return {
      fen: evaluation.fen,
      move: evaluation.bestMove,
      evaluation,
    };
  }
}
