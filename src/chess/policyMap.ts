import { moveFromUci, moveToUci, type Move } from './moveCodec.ts';

const FILES = 'abcdefgh';
const PROMOS = ['q', 'r', 'b', 'n'] as const;
const DIRS = [[1,0],[-1,0],[0,1],[0,-1],[1,1],[1,-1],[-1,1],[-1,-1]] as const;
const KNIGHTS = [[1,2],[2,1],[-1,2],[-2,1],[1,-2],[2,-1],[-1,-2],[-2,-1]] as const;
const on = (f: number, r: number) => f >= 0 && f < 8 && r >= 0 && r < 8;
const sq = (f: number, r: number) => `${FILES[f]}${r + 1}`;

export const POLICY_MAP = 'uci_queen_knight_promo_v1';

function buildPolicyMoves(): string[] {
  const out = new Set<string>();
  for (let r = 0; r < 8; r++) for (let f = 0; f < 8; f++) {
    const from = sq(f, r);
    for (const [df, dr] of DIRS) for (let n = 1; n < 8; n++) {
      const tf = f + df * n, tr = r + dr * n;
      if (!on(tf, tr)) break;
      out.add(`${from}${sq(tf, tr)}`);
    }
    for (const [df, dr] of KNIGHTS) if (on(f + df, r + dr)) out.add(`${from}${sq(f + df, r + dr)}`);
  }
  for (const r of [1, 6]) {
    const tr = r === 6 ? 7 : 0;
    for (let f = 0; f < 8; f++) for (const df of [-1, 0, 1]) if (on(f + df, tr)) {
      for (const p of PROMOS) out.add(`${sq(f, r)}${sq(f + df, tr)}${p}`);
    }
  }
  return [...out].sort();
}

export const POLICY_MOVES = buildPolicyMoves();
export const POLICY_SIZE = POLICY_MOVES.length;
export const POLICY_INDEX = new Map(POLICY_MOVES.map((uci, i) => [uci, i]));

export function moveToPolicyIndex(move: Move): number | undefined { return POLICY_INDEX.get(moveToUci(move)); }
export function policyIndexToMove(index: number): Move { return moveFromUci(POLICY_MOVES[index]); }
