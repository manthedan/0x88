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

/**
 * Profiled invokes that fit the profiler's timestamp capacity. In dispatch
 * mode every dispatch takes one slot, so `requested` invokes of
 * `dispatchesPerInvoke` overflow past `capacity` and the tail runs untimed.
 * Unknown dispatch counts leave the request alone (completeness still gates).
 */
export function profileInvokeCap(requested, dispatchesPerInvoke, capacity) {
  if (!(dispatchesPerInvoke > 0) || !(capacity > 0)) return { invokes: requested, capped: false };
  const fit = Math.floor(capacity / dispatchesPerInvoke);
  return fit >= requested ? { invokes: requested, capped: false } : { invokes: fit, capped: true };
}

/**
 * Auxiliary ab profiles that are not trustworthy: an arm's abProfile
 * dispatch-mode profile that dropped timestamps, did not run (capacity below
 * one invoke) or reported complete:false, and per-round pass-mode profiles
 * that dropped timestamps. Any entry fails the page's ok.
 */
export function abIncompleteProfiles(ab) {
  if (!ab) return [];
  const arms = [
    { key: 'a', model: null, pass: ab.a?.pass, profile: null },
    { key: 'b', model: ab.b?.model ?? null, pass: ab.b?.pass, profile: ab.b?.profile },
    ...(ab.extra ?? []).map((arm) => ({ key: arm.key, model: arm.model, pass: arm.pass, profile: arm.profile })),
  ];
  const out = [];
  for (const arm of arms) {
    const p = arm.profile;
    if (p && (p.complete !== true || p.dropped > 0 || !(p.invokes > 0)))
      out.push({
        arm: arm.key,
        model: arm.model,
        profile: 'dispatch',
        invokes: p.invokes ?? null,
        requestedInvokes: p.requestedInvokes ?? null,
        dropped: p.dropped ?? null,
      });
    if (arm.pass?.dropped > 0) out.push({ arm: arm.key, model: arm.model, profile: 'pass', dropped: arm.pass.dropped });
  }
  return out;
}

/**
 * Split the full fixture corpus into physical-batch chunks for parity. The
 * last chunk is padded by repeating its final position (as the production
 * evaluators do); `real` is how many leading rows are fixtures. Parity runs
 * over these chunks, not the timing feeds: a b1 timing feed holds only the
 * start position and a b8 feed drops the tail fixtures, including the only
 * nonzero rule50 position.
 */
export function fixtureChunks(fens, batch) {
  if (!(batch >= 1)) throw new Error(`fixtureChunks: bad batch ${batch}`);
  if (!fens.length) throw new Error('fixtureChunks: empty corpus');
  const chunks = [];
  for (let i = 0; i < fens.length; i += batch) {
    const real = fens.slice(i, i + batch);
    chunks.push({ fens: [...real, ...Array(batch - real.length).fill(real[real.length - 1])], real: real.length });
  }
  return chunks;
}

/**
 * Run `run(chunkFens) -> {output: Float32Array}` over fixtureChunks and
 * concatenate the real rows of each output, so the result has one row per
 * fixture. Every output's leading dimension must be the batch.
 */
export async function runChunked(fens, batch, run) {
  const parts = {};
  for (const chunk of fixtureChunks(fens, batch)) {
    const out = await run(chunk.fens);
    for (const [name, data] of Object.entries(out)) {
      if (data.length % batch) throw new Error(`runChunked: ${name} length ${data.length} is not a multiple of batch ${batch}`);
      const row = data.length / batch;
      parts[name] ??= [];
      parts[name].push(data.subarray(0, chunk.real * row));
    }
  }
  const out = {};
  for (const [name, list] of Object.entries(parts)) {
    const all = new Float32Array(list.reduce((n, a) => n + a.length, 0));
    let off = 0;
    for (const a of list) {
      all.set(a, off);
      off += a.length;
    }
    out[name] = all;
  }
  return out;
}

/**
 * Compare the requested manifest with the one the evaluator actually loaded
 * (both resolved against `base`). A mismatch means a silent runtime fallback,
 * and the run must not be recorded as the requested candidate.
 */
export function manifestCheck(requested, effective, base) {
  const req = new URL(requested, base).href;
  const eff = new URL(effective, base).href;
  return { ok: req === eff, requested: new URL(req).pathname, effective: new URL(eff).pathname };
}

/**
 * Maia3 parity rows: every fixture under several (self, opponent) rating
 * conditions, all unequal except one, so a candidate that swaps or ignores
 * one Elo input changes its outputs. (Equal ratings at every row made
 * elo_self and elo_oppo indistinguishable.)
 */
export const MAIA3_PARITY_CONDITIONS = [
  [1100, 1900],
  [1900, 1100],
  [1500, 1500],
  [2300, 800],
];
export function maia3ParityRows(fens) {
  return fens.flatMap((fen) => MAIA3_PARITY_CONDITIONS.map(([eloSelf, eloOppo]) => ({ fen, eloSelf, eloOppo })));
}

function legalSoftmax(logits, base, indices) {
  const v = indices.map((i) => logits[base + i]);
  const max = Math.max(...v);
  const e = v.map((x) => Math.exp(x - max));
  const sum = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / sum);
}

/**
 * Legal-move policy parity on raw policy logits (`cand`, `ref`: flat
 * [rows * rowSize]). `rows[r].indices` are the policy indices of row r's legal
 * moves (`ucis` optional, for reports; `fen` and `ucis` needed for
 * `allowedFlips`: [{ fen, moves: [a, b], maxGap }], measured drift pairs that
 * may flip up to maxGap in that position only). Per row, on the softmax over legal
 * moves only: max |prob diff| <= maxProbAbs; the candidate's best move is the
 * reference's unless the REFERENCE prior gap between them is <= tieEpsilon;
 * and among the reference's top-k moves, every pair whose reference gap
 * exceeds tieEpsilon keeps its order in the candidate. Raw-logit max-abs
 * alone lets a swap of the two leading logits pass ([1, 0] -> [0, 1] is a
 * max-abs of 1 but reverses a 73/27 split).
 */
export function legalPolicyParity(cand, ref, rows, { maxProbAbs, tieEpsilon, topK = 3, allowedFlips = [] }) {
  if (!(maxProbAbs >= 0) || !(tieEpsilon >= 0)) throw new Error('legalPolicyParity: maxProbAbs and tieEpsilon are required');
  const problems = [];
  const rowSize = ref.length / rows.length;
  const shapeOk = Number.isInteger(rowSize) && cand.length === ref.length;
  if (!shapeOk) problems.push(`shape: candidate ${cand.length}, reference ${ref.length}, rows ${rows.length}`);
  let maxProb = 0;
  let maxKl = 0;
  let bestMatches = 0;
  const tieTolerated = [];
  for (let r = 0; shapeOk && r < rows.length; r++) {
    const { indices, ucis } = rows[r];
    const name = (k) => ucis?.[k] ?? indices[k];
    const pc = legalSoftmax(cand, r * rowSize, indices);
    const pr = legalSoftmax(ref, r * rowSize, indices);
    if (pc.some((x) => !Number.isFinite(x))) {
      problems.push(`row ${r}: non-finite candidate policy`);
      continue;
    }
    let kl = 0;
    for (let k = 0; k < indices.length; k++) {
      maxProb = Math.max(maxProb, Math.abs(pc[k] - pr[k]));
      if (pr[k] > 0) kl += pr[k] * Math.log(pr[k] / Math.max(pc[k], 1e-30));
    }
    maxKl = Math.max(maxKl, kl);
    // A pair may flip if the reference gap is a tie, or if this exact
    // position/move pair is allowlisted (measured drift) within its maxGap.
    const flipOk = (a, b) => {
      const gap = Math.abs(pr[a] - pr[b]);
      if (gap <= tieEpsilon) return 'tie';
      const ua = ucis?.[a];
      const ub = ucis?.[b];
      const entry = allowedFlips.find((f) => f.fen === rows[r].fen && ua !== ub && f.moves.includes(ua) && f.moves.includes(ub));
      return entry && gap <= entry.maxGap ? 'allowlisted' : null;
    };
    const argmax = (p) => p.reduce((b, x, k) => (x > p[b] ? k : b), 0);
    const bc = argmax(pc);
    const br = argmax(pr);
    const why = bc === br ? null : flipOk(br, bc);
    if (bc === br) bestMatches++;
    else if (why) tieTolerated.push({ row: r, candidate: name(bc), reference: name(br), referenceGap: pr[br] - pr[bc], why });
    else problems.push(`row ${r}: best move ${name(bc)} vs reference ${name(br)} (reference gap ${(pr[br] - pr[bc]).toFixed(4)})`);
    const top = pr
      .map((p, k) => [p, k])
      .sort((a, b) => b[0] - a[0])
      .slice(0, topK)
      .map(([, k]) => k);
    for (let i = 0; i < top.length; i++)
      for (let j = i + 1; j < top.length; j++) {
        const [a, b] = [top[i], top[j]];
        if (!(pc[a] > pc[b]) && !flipOk(a, b)) problems.push(`row ${r}: top-${topK} order ${name(a)} > ${name(b)} reversed`);
      }
  }
  if (maxProb > maxProbAbs) problems.push(`max legal prob diff ${maxProb.toFixed(4)} > ${maxProbAbs}`);
  return {
    rows: rows.length,
    maxLegalProbAbs: maxProb,
    maxKl,
    bestMoveMatches: bestMatches,
    tieTolerated,
    limits: { maxProbAbs, tieEpsilon, topK, allowedFlips },
    problems,
    pass: problems.length === 0,
  };
}
