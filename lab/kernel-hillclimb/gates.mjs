// Pure correctness gates for the profile pages. No browser APIs, so the unit
// tests can feed them deliberately broken candidates.
//
// Every gate compares a CANDIDATE against an independent REFERENCE: for ORT
// pages the reference model is the preset's shipped model (not the candidate
// URL), for TVMJS pages it is the ORT evaluator on the batch-1 ONNX export.

/** Per-output max |candidate - reference|; non-finite diffs and length mismatches are counted, not skipped. */
export function maxAbsDiff(candidate, reference) {
  const rows = {};
  for (const name of new Set([...Object.keys(candidate), ...Object.keys(reference)])) {
    const a = candidate[name];
    const b = reference[name];
    if (!a || !b) {
      rows[name] = { maxAbs: null, nonFinite: 0, length: a?.length ?? 0, referenceLength: b?.length ?? 0, missing: !a ? 'candidate' : 'reference' };
      continue;
    }
    let max = 0;
    let nonFinite = 0;
    const n = Math.min(a.length, b.length);
    for (let i = 0; i < n; i++) {
      const d = Math.abs(a[i] - b[i]);
      if (!Number.isFinite(d)) nonFinite++;
      else if (d > max) max = d;
    }
    rows[name] = { maxAbs: max, nonFinite, length: a.length, referenceLength: b.length };
  }
  return rows;
}

/**
 * ORT parity: every limited output must exist in both, have equal length, no
 * non-finite diffs, and max-abs within its limit. No limits = no pass.
 */
export function ortParity(candidate, reference, limits) {
  const outputs = maxAbsDiff(candidate, reference);
  const failures = limits
    ? Object.entries(limits)
        .filter(([name, max]) => {
          const o = outputs[name];
          return !(o && o.maxAbs !== null && o.length === o.referenceLength && o.nonFinite === 0 && o.maxAbs <= max);
        })
        .map(([name]) => name)
    : [];
  return { outputs, maxAbsLimits: limits ?? null, failures, pass: !!limits && failures.length === 0 };
}

const argmax = (priors) => {
  let best = null;
  for (const p of priors) if (best === null || p.prior > best.prior) best = p;
  return best?.uci ?? null;
};

/**
 * Lc0 evaluator parity (candidate rows `a`, reference rows `b`, each
 * { legalPriors: [{uci, prior}], wdl: [w, d, l] }).
 *
 * Pass requires, on every row: the same legal-move set, finite priors and
 * WDL, max |prior diff| <= maxPriorAbs, max |WDL diff| <= maxWdlAbs, and the
 * same best move unless the flip is a tie IN THE REFERENCE distribution
 * (refPrior(refBest) - refPrior(candBest) <= tieEpsilon). Measuring the gap in
 * the candidate's own priors would let a flattened candidate excuse itself.
 */
export function compareLc0(a, b, { tieEpsilon = null, maxPriorAbs, maxWdlAbs }) {
  if (!(maxPriorAbs >= 0) || !(maxWdlAbs >= 0)) throw new Error('compareLc0: maxPriorAbs and maxWdlAbs are required');
  let maxPrior = 0;
  let maxWdl = 0;
  let bestMatches = 0;
  const tolerated = [];
  const flips = [];
  const problems = [];
  if (a.length !== b.length) problems.push(`row count ${a.length} vs reference ${b.length}`);
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    const pa = new Map(a[i].legalPriors.map((p) => [p.uci, p.prior]));
    const pb = new Map(b[i].legalPriors.map((p) => [p.uci, p.prior]));
    if (pa.size !== pb.size || [...pb.keys()].some((k) => !pa.has(k))) problems.push(`row ${i}: legal-move set differs`);
    for (const [uci, prior] of pb) {
      const d = Math.abs((pa.get(uci) ?? Number.NaN) - prior);
      maxPrior = Number.isFinite(d) ? Math.max(maxPrior, d) : Number.NaN;
      if (Number.isNaN(maxPrior)) break;
    }
    if (Number.isNaN(maxPrior)) problems.push(`row ${i}: non-finite or missing prior`);
    for (let j = 0; j < 3; j++) {
      const d = Math.abs(a[i].wdl?.[j] - b[i].wdl?.[j]);
      if (!Number.isFinite(d)) {
        problems.push(`row ${i}: non-finite WDL`);
        maxWdl = Number.NaN;
        break;
      }
      maxWdl = Math.max(maxWdl, d);
    }
    const ba = argmax(a[i].legalPriors);
    const bb = argmax(b[i].legalPriors);
    if (ba === bb) bestMatches++;
    else {
      const gap = (pb.get(bb) ?? Number.NaN) - (pb.get(ba) ?? Number.NaN);
      const flip = { row: i, candidate: ba, reference: bb, referencePriorGap: gap };
      if (tieEpsilon !== null && Number.isFinite(gap) && gap <= tieEpsilon) tolerated.push(flip);
      else flips.push(flip);
    }
    if (Number.isNaN(maxPrior) || Number.isNaN(maxWdl)) break;
  }
  if (Number.isFinite(maxPrior) && maxPrior > maxPriorAbs) problems.push(`max prior diff ${maxPrior} > ${maxPriorAbs}`);
  if (Number.isFinite(maxWdl) && maxWdl > maxWdlAbs) problems.push(`max WDL diff ${maxWdl} > ${maxWdlAbs}`);
  if (flips.length) problems.push(`${flips.length} best-move flip(s) outside the reference tie band`);
  return {
    rows: a.length,
    bestMoveMatches: bestMatches,
    tieTolerated: tolerated,
    flips,
    maxLegalPriorAbs: maxPrior,
    maxWdlAbs: maxWdl,
    limits: { maxPriorAbs, maxWdlAbs, tieEpsilon },
    problems,
    pass: problems.length === 0,
  };
}

/** Centipawn parity: rows { policy: Map<move, p>, wdl: [3] }; same move set, finite, max-abs <= tolerance. */
export function compareCentipawn(a, b, tolerance = 0.001) {
  let policy = 0;
  let wdl = 0;
  const problems = [];
  if (a.length !== b.length) problems.push(`row count ${a.length} vs reference ${b.length}`);
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    if (a[i].policy.size !== b[i].policy.size || [...b[i].policy.keys()].some((k) => !a[i].policy.has(k))) problems.push(`row ${i}: move set differs`);
    for (const [k, p] of b[i].policy) {
      const d = Math.abs((a[i].policy.get(k) ?? Number.NaN) - p);
      if (!Number.isFinite(d)) {
        problems.push(`row ${i}: non-finite or missing policy`);
        policy = Number.NaN;
        break;
      }
      policy = Math.max(policy, d);
    }
    for (let j = 0; j < 3; j++) {
      const d = Math.abs(a[i].wdl?.[j] - b[i].wdl?.[j]);
      if (!Number.isFinite(d)) {
        problems.push(`row ${i}: non-finite WDL`);
        wdl = Number.NaN;
        break;
      }
      wdl = Math.max(wdl, d);
    }
    if (Number.isNaN(policy) || Number.isNaN(wdl)) break;
  }
  if (Number.isFinite(policy) && policy > tolerance) problems.push(`max policy diff ${policy} > ${tolerance}`);
  if (Number.isFinite(wdl) && wdl > tolerance) problems.push(`max WDL diff ${wdl} > ${tolerance}`);
  return { rows: a.length, maxLegalPolicyAbs: policy, maxWdlAbs: wdl, tolerance, problems, pass: problems.length === 0 };
}

/** A collected profile is only trustworthy when no dispatch went untimed. */
export function profileComplete(profile) {
  if (!profile || profile.supported === false) return true; // nothing claimed
  return !(profile.dropped > 0) && !(profile.passMode?.dropped > 0);
}
