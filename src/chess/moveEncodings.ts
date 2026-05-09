import { moveFromUci, moveToActionId, moveToUci, type Move } from './moveCodec.ts';
import { POLICY_INDEX, POLICY_MOVES, moveToPolicyIndex } from './policyMap.ts';

export { ACTION_SPACE, moveFromUci, moveToActionId, moveToUci, type Move } from './moveCodec.ts';
export { POLICY_INDEX, POLICY_MAP, POLICY_MOVES, POLICY_SIZE, moveToPolicyIndex, policyIndexToMove } from './policyMap.ts';

const CHESSBENCH_PROMO: Record<NonNullable<Move['promotion']>, number> = { n: 0, b: 1, r: 2, q: 3 };

export function moveToChessBenchAvClass(move: Move): number {
  const ft = move.from * 64 + move.to;
  if (!move.promotion) return ft;
  return 4096 + ft * 4 + CHESSBENCH_PROMO[move.promotion];
}

export const moveToSquareformerPolicyIndex = moveToChessBenchAvClass;

export function moveToResidualPolicyIndex(move: Move): number | undefined {
  return moveToPolicyIndex(move);
}

export function residualPolicyIndexToUci(index: number): string {
  const uci = POLICY_MOVES[index];
  if (uci === undefined) throw new Error(`Residual policy index out of range: ${index}`);
  return uci;
}

export function residualPolicyIndexToMove(index: number): Move {
  return moveFromUci(residualPolicyIndexToUci(index));
}

export function actionIdToMove(actionId: number): Move {
  if (!Number.isInteger(actionId) || actionId < 0 || actionId >= 64 * 64 * 5) throw new Error(`Action id out of range: ${actionId}`);
  const promoIndex = actionId % 5;
  const ft = Math.floor(actionId / 5);
  const from = Math.floor(ft / 64);
  const to = ft % 64;
  const promotion = ([undefined, 'n', 'b', 'r', 'q'] as const)[promoIndex];
  return promotion ? { from, to, promotion } : { from, to };
}

export function chessBenchAvClassToMove(klass: number): Move {
  if (!Number.isInteger(klass) || klass < 0 || klass >= 4096 + 4096 * 4) throw new Error(`ChessBench AV class out of range: ${klass}`);
  if (klass < 4096) return { from: Math.floor(klass / 64), to: klass % 64 };
  const x = klass - 4096;
  const ft = Math.floor(x / 4);
  const promoIndex = x % 4;
  const promotion = (['n', 'b', 'r', 'q'] as const)[promoIndex];
  return { from: Math.floor(ft / 64), to: ft % 64, promotion };
}

export function assertCanonicalMoveEncoding(move: Move): void {
  const actionRoundtrip = actionIdToMove(moveToActionId(move));
  if (moveToUci(actionRoundtrip) !== moveToUci(move)) throw new Error(`action-id roundtrip mismatch for ${moveToUci(move)}`);
  const avRoundtrip = chessBenchAvClassToMove(moveToChessBenchAvClass(move));
  if (moveToUci(avRoundtrip) !== moveToUci(move)) throw new Error(`AV-class roundtrip mismatch for ${moveToUci(move)}`);
  const policyIndex = moveToPolicyIndex(move);
  if (policyIndex !== undefined && POLICY_INDEX.get(moveToUci(move)) !== policyIndex) throw new Error(`policy-index mismatch for ${moveToUci(move)}`);
}
