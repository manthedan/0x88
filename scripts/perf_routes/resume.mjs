// Resume bookkeeping: a run directory may only continue the exact cohort it started.

export const sessionId = (cell, i) => `${String(i).padStart(3, '0')}-${cell.browser}-${cell.journey}-${cell.arm}-r${cell.repeat}`;

// Everything that changes which sessions exist or what they measure. Lock/driver/output paths do not.
export function runManifest(options, arms, { host, harness }) {
  return {
    version: 1,
    options: { browser: options.browser, repeats: options.repeats, journeys: options.journeys },
    arms,
    host,
    harness,
  };
}

const flatten = (value, prefix = '', out = {}) => {
  if (value && typeof value === 'object' && !Array.isArray(value))
    for (const [key, v] of Object.entries(value)) flatten(v, prefix ? `${prefix}.${key}` : key, out);
  else out[prefix] = JSON.stringify(value);
  return out;
};

export function manifestMismatches(saved, current) {
  const a = flatten(saved),
    b = flatten(current);
  return [...new Set([...Object.keys(a), ...Object.keys(b)])].filter((key) => a[key] !== b[key]).sort();
}

// Operator stops and Safari-busy attempts are not outcomes (a fresh run would have retried them).
const notAnOutcome = (result) => result.status !== 'ok' && /^(?:Error: )?Interrupted\b|already paired/.test(result.error ?? '');

// Reuse every recorded outcome, failures included: retrying only failures until clean would be
// selective outlier removal. Missing, unreadable, cancelled or interrupted sessions run again.
export function selectSessions(cells, inspect) {
  return cells.map((cell, i) => {
    const id = sessionId(cell, i);
    const { result, cleanupPending, cancelled } = inspect(id);
    if (cleanupPending)
      throw Error(`Browser cleanup unconfirmed for ${id}; confirm the browser is closed and remove its .cleanup-pending marker before resuming`);
    if (!result || cancelled || notAnOutcome(result)) return { cell, id, reuse: null };
    for (const key of ['browser', 'journey', 'arm', 'repeat'])
      if (result[key] !== cell[key]) throw Error(`Session file ${id} records a different ${key}: ${result[key]}`);
    return { cell, id, reuse: result };
  });
}
