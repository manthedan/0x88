export function softmax(xs: ArrayLike<number>): number[] {
  let max = -Infinity;
  for (let i = 0; i < xs.length; i++) if (Number(xs[i]) > max) max = Number(xs[i]);
  const out = Array.from(xs, (x) => Math.exp(Number(x) - max));
  const total = out.reduce((sum, x) => sum + x, 0) || 1;
  return out.map((x) => x / total);
}
