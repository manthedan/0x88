/** Median of a non-empty numeric sample. Even lengths use the mean of the two central values. */
export function median(values) {
  if (!values.length) throw new Error('median requires at least one sample');
  const sorted = [...values].map(Number).sort((a, b) => a - b);
  if (sorted.some((value) => !Number.isFinite(value))) throw new Error('median requires finite samples');
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/** Median absolute deviation around the sample median. */
export function mad(values) {
  const center = median(values);
  return median(values.map((value) => Math.abs(value - center)));
}

/**
 * Compare one canary metric with the previous run.
 * A swing counts only when it clears both a relative floor (default 10%) and
 * `madMultiple` times this run's MAD/median. A run whose own MAD exceeds the
 * floor is `noisy-run` and is not called a regression.
 */
export function compareMetric({ name, previous, samples, higherIsBetter, floor = 0.1, madMultiple = 3 }) {
  if (!Number.isFinite(previous)) throw new Error(`${name} previous value must be finite`);
  const current = median(samples);
  const spread = mad(samples);
  const madRatio = spread / Math.max(Math.abs(current), 1e-9);
  const threshold = Math.max(floor, madMultiple * madRatio);
  const deltaRatio = previous === 0 ? null : (current - previous) / Math.abs(previous);
  let status = 'within-noise';
  if (samples.length < 3) status = 'insufficient-repeats';
  else if (madRatio > floor) status = 'noisy-run';
  else if (deltaRatio !== null && Math.abs(deltaRatio) > threshold) status = (higherIsBetter ? deltaRatio > 0 : deltaRatio < 0) ? 'improved' : 'regressed';
  return { name, previous, current, mad: spread, madRatio, deltaRatio, threshold, higherIsBetter, status };
}
