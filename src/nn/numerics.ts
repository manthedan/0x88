export function softmax(xs: ArrayLike<number>): number[] {
  if (xs.length === 0) return [];
  let max = -Infinity;
  for (let i = 0; i < xs.length; i++) {
    const value = Number(xs[i]);
    if (Number.isNaN(value) || value === Infinity) throw new Error(`Softmax input ${i} is not finite: ${value}`);
    if (value > max) max = value;
  }
  if (max === -Infinity) throw new Error('Softmax requires at least one finite input');
  const out = new Array<number>(xs.length);
  let total = 0;
  for (let i = 0; i < xs.length; i++) {
    const value = Math.exp(Number(xs[i]) - max);
    out[i] = value;
    total += value;
  }
  if (!Number.isFinite(total) || total <= 0) throw new Error(`Softmax normalization is invalid: ${total}`);
  for (let i = 0; i < out.length; i++) out[i] /= total;
  return out;
}
