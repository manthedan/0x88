export type RootPolicyVariant = 'argmax' | 'prior_proportional' | 'sqrt_prior';

export type MovePolicy = Record<string, number>;

function normalize(policy: MovePolicy): MovePolicy {
  const total = Object.values(policy).reduce((a, b) => a + b, 0);
  if (total <= 0) throw new Error('Cannot normalize empty or zero policy');
  return Object.fromEntries(Object.entries(policy).map(([move, p]) => [move, p / total]));
}

export function rootVisitDistribution(policy: MovePolicy, variant: RootPolicyVariant): MovePolicy {
  const moves = Object.keys(policy);
  if (moves.length === 0) return {};
  if (variant === 'argmax') {
    const best = moves.reduce((a, b) => policy[a] >= policy[b] ? a : b);
    return Object.fromEntries(moves.map((move) => [move, move === best ? 1 : 0]));
  }
  if (variant === 'prior_proportional') return normalize(policy);
  if (variant === 'sqrt_prior') {
    return normalize(Object.fromEntries(moves.map((move) => [move, Math.sqrt(Math.max(0, policy[move]))])));
  }
  const _exhaustive: never = variant;
  return _exhaustive;
}
