// Node-side Maia3 evaluator for calibration scripts: one ORT session, batched
// over (selfElo, oppoElo) pairs for a single position. Reuses the exported
// browser encoding (boardToMaia3Tokens / maia3MoveIndex) so script results
// describe exactly what the browser runtime computes.
import { readFileSync } from 'node:fs';
import * as ort from '../src/nn/ortRuntime.ts';
import { legalMoves } from '../src/chess/movegen.ts';
import { moveToUci } from '../src/chess/moveCodec.ts';
import { boardToMaia3Tokens, maia3MoveIndex, MAIA3_POLICY_SIZE } from '../src/lc0/maia3.ts';

const DEFAULT_MODEL = 'public/models/maia3/maia3_simplified.onnx';

function mirrorUciRanks(uci) {
  const flip = (sq) => `${sq[0]}${9 - Number(sq[1])}`;
  return `${flip(uci.slice(0, 2))}${flip(uci.slice(2, 4))}${uci.slice(4)}`;
}

export async function createMaia3NodeEvaluator(modelPath = DEFAULT_MODEL) {
  const session = await ort.createOrtSession(readFileSync(modelPath).buffer);

  /**
   * Evaluate one position under a batch of (selfElo, oppoElo) conditions.
   * Returns per-condition { legalPriors: Map<uci, prior>, valueProbabilities }
   * with valueProbabilities in model order [Loss, Draw, Win] for side to move.
   */
  async function evaluateConditions(board, conditions) {
    const tokens = boardToMaia3Tokens(board);
    const n = conditions.length;
    const batchTokens = new Float32Array(n * tokens.length);
    for (let i = 0; i < n; i += 1) batchTokens.set(tokens, i * tokens.length);
    const feeds = {
      tokens: new ort.Tensor('float32', batchTokens, [n, 64, 12]),
      elo_self: new ort.Tensor('float32', Float32Array.from(conditions.map((c) => c.selfElo)), [n]),
      elo_oppo: new ort.Tensor('float32', Float32Array.from(conditions.map((c) => c.oppoElo)), [n]),
    };
    const outputs = await session.run(feeds);
    const moveLogits = outputs.logits_move.data;
    const valueLogits = outputs.logits_value.data;

    const legal = [];
    for (const move of legalMoves(board)) {
      const uci = moveToUci(move);
      const modelUci = board.turn === 'b' ? mirrorUciRanks(uci) : uci;
      const index = maia3MoveIndex(modelUci);
      if (index !== undefined) legal.push({ uci, index });
    }

    const results = [];
    for (let i = 0; i < n; i += 1) {
      const base = i * MAIA3_POLICY_SIZE;
      const logits = legal.map((entry) => Number(moveLogits[base + entry.index]));
      const max = Math.max(...logits);
      const exp = logits.map((l) => Math.exp(l - max));
      const sum = exp.reduce((a, b) => a + b, 0) || 1;
      const legalPriors = new Map(legal.map((entry, j) => [entry.uci, exp[j] / sum]));
      const v = [Number(valueLogits[i * 3]), Number(valueLogits[i * 3 + 1]), Number(valueLogits[i * 3 + 2])];
      const vMax = Math.max(...v);
      const vExp = v.map((x) => Math.exp(x - vMax));
      const vSum = vExp.reduce((a, b) => a + b, 0) || 1;
      results.push({ legalPriors, valueProbabilities: vExp.map((x) => x / vSum) });
    }
    return results;
  }

  return { evaluateConditions, session };
}
