import { parseFen, START_FEN } from '../../src/chess/board.ts';
import { isStmWhiteRankflip, normalizePositionForStmWhite } from '../../src/chess/boardNormalization.ts';
import { squareformerCompactInput, threatgraphSquareSummaryV1 } from '../../src/nn/squareformerEvaluator.ts';

export const encodingFens = [
  START_FEN,
  'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1',
  'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2',
  'r3k2r/ppp2ppp/2n1bn2/3qp3/3P4/2N1BN2/PPP2PPP/R2QK2R b KQkq - 17 10',
  '4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 20',
  '4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 20',
  '4k3/P7/8/8/8/8/7p/4K3 w - - 255 50',
  '4k3/8/8/8/8/8/8/4K3 b - - 99 80',
];

export function encodingBatch(size, offset = 0, masked = false) {
  const fens = Array.from({ length: size }, (_, index) => encodingFens[(index + offset) % encodingFens.length]);
  const boards = fens.map(parseFen);
  const contexts = fens.map((fen, index) => ({
    // Includes missing history, full history and repetition features. Each row differs.
    historyFens: (index + offset) % 3 === 0 ? [] : Array.from({ length: (index + offset) % 3 === 1 ? 7 : 3 }, (_, h) => (h % 2 ? START_FEN : fen)),
    ...(masked ? { attackSummaryChannelMask: Array.from({ length: 28 }, (_, channel) => [0, 1, 0.3, -0.5, -0][(channel + index) % 5]) } : {}),
  }));
  return { boards, contexts };
}

// Independent reference: the original evaluator normalized real rows, padded
// those rows, then ran both encoders (and the mask) for every physical slot.
export function legacyPaddedInputs(boards, contexts, meta, physicalBatchSize = 16) {
  const rows = boards.map((board, index) => {
    const context = contexts[index] ?? {};
    const normalized = isStmWhiteRankflip(meta.board_normalization)
      ? normalizePositionForStmWhite(board, context.historyFens ?? [], context.legalMoves)
      : { board, historyFens: context.historyFens ?? [] };
    return { ...normalized, attackSummaryChannelMask: context.attackSummaryChannelMask };
  });
  const padded = [...rows];
  while (padded.length < physicalBatchSize) padded.push(rows.at(-1));
  const stride = meta.token_features ?? meta.history_plies + 9;
  const attackFeatures = meta.attack_summary_feature_count;
  const tokens = new Int32Array(physicalBatchSize * 64 * stride);
  const attack = new Float32Array(physicalBatchSize * 64 * attackFeatures);
  for (let index = 0; index < padded.length; index++) {
    const row = padded[index];
    tokens.set(squareformerCompactInput(row.board, meta, row.historyFens, 'int32'), index * 64 * stride);
    const attackRow = threatgraphSquareSummaryV1(row.board);
    if (row.attackSummaryChannelMask) {
      for (let square = 0; square < 64; square++) {
        const base = square * attackFeatures;
        for (let channel = 0; channel < attackFeatures; channel++) {
          attackRow[base + channel] *= Number(row.attackSummaryChannelMask[channel] ?? 0);
        }
      }
    }
    attack.set(attackRow, index * 64 * attackFeatures);
  }
  return { tokens: new Uint8Array(tokens.buffer), attack: new Uint8Array(attack.buffer) };
}
