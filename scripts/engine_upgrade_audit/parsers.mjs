export function parseCommitBench(message) {
  const match = message.match(/^\s*Bench:\s*(\d+)\s*$/im);
  return match ? Number(match[1]) : null;
}

export function parseBenchOutput(text, { expectedRows, requireConsistentSummaries = false } = {}) {
  // Recognize records before validating them, and validate ALL summaries.
  // Callers may combine separately captured stdout/stderr without emission
  // ordering; neither stream may hide a malformed record in the other.
  // UCI info and table position rows are not summaries.
  const summaryRecord = /^(?:(?:Bench|Results)\s*:|Nodes\s+searched\b|Total\s+nodes\b|TOTAL\b|\S+\s+nodes\b|\S+\s+[\d.+-]\S*s(?:\s|$)|\S+\s+\S+\s+\S+\s+N\/s\b)/i;
  const summaries = text.split(/\r?\n/).map((line) => line.trim()).filter((line) => summaryRecord.test(line));
  const summary = summaries.at(-1);
  const patterns = [
    /^(\d+)\s+nodes in (\d+(?:\.\d+)?)s(?:\s+\((\d+) nps\))?$/, // Viridithas
    /^(?:Bench:\s*)?(\d+)\s+nodes\s+(\d+) nps$/, // Reckless, Stormphrax
    /^(\d+)\s+(\d+(?:\.\d+)?)s\s+(\d+) N\/s$/, // Reckless table footer
    /^Nodes searched\s*:\s*(\d+)$/, // Stockfish
    /^Results:\s*(\d+)\s+nodes(?:\s+(\d+) nps)?$/, // Berserk
    /^(?:Total nodes(?: searched)?\s*[:=]?|TOTAL)\s+(\d+)$/i,
  ];
  if (!summaries.length) throw new Error('Missing/invalid full bench node summary');
  let nodes;
  let firstNodes;
  for (const record of summaries) {
    const match = patterns.map((pattern) => record.match(pattern)).find(Boolean);
    nodes = match ? Number(match[1]) : null;
    if (!match || match.slice(1).some((token) => token !== undefined && !Number.isFinite(Number(token)))
      || !Number.isSafeInteger(nodes) || nodes <= 0) throw new Error(`Missing/invalid full bench node summary: ${record}`);
    if (requireConsistentSummaries && firstNodes !== undefined && nodes !== firstNodes) {
      throw new Error(`Conflicting full bench summaries: ${firstNodes} != ${nodes}; output order cannot establish a reference`);
    }
    firstNodes ??= nodes;
  }
  const rows = [];
  for (const line of text.split(/\r?\n/).map((record) => record.trim())) {
    if (/^Bench\s*\[/.test(line)) {
      const row = line.match(/^Bench \[#\s*(\d+)\]: bestmove\s+([a-h][1-8][a-h][1-8][qrbn]?|0000|\(none\))\s+score\s+(-?\d+)\s+(\d+) nodes\s+(\d+) nps\s+\|\s+.+$/);
      if (!row || [row[1], row[3], row[4], row[5]].some((token) => !Number.isSafeInteger(Number(token)))) {
        throw new Error(`Malformed bench position record: ${line}`);
      }
      rows.push({ position: Number(row[1]), bestmove: row[2], nodes: Number(row[4]) });
    } else if (/^\S+\s+\S+\s+\S+\s+\S+\s+N\/s\b|^\d+\s+\S+\s+[\d.+-]\S*s(?:\s|$)/.test(line)) {
      const row = line.match(/^(\d+)\s+(\d+)\s+(\d+(?:\.\d+)?)s\s+(\d+) N\/s$/);
      if (!row || [row[1], row[2], row[4]].some((token) => !Number.isSafeInteger(Number(token))) || !Number.isFinite(Number(row[3]))) {
        throw new Error(`Malformed bench position record: ${line}`);
      }
      rows.push({ position: Number(row[1]), nodes: Number(row[2]) });
    }
  }
  if ((expectedRows !== undefined && rows.length !== expectedRows) || (patterns[2].test(summary) && !rows.length)) {
    throw new Error(`Missing bench position coverage: expected ${expectedRows ?? 'table rows'}, got ${rows.length}`);
  }
  if (rows.length && (rows.some((row, i) => row.position !== rows[0].position + i)
    || ![0, 1].includes(rows[0].position) || rows.reduce((sum, row) => sum + row.nodes, 0) !== nodes)) {
    throw new Error('Invalid bench position coverage/node total');
  }
  return { nodes, rows };
}

export function parseTags(text, upstream) {
  const tags = new Map();
  for (const line of text.trim().split('\n')) {
    const match = line.match(/^([a-f0-9]{40})\s+refs\/tags\/(.+?)(\^\{\})?$/);
    if (!match) continue;
    const [, commit, name, peeled] = match;
    if (peeled || !tags.has(name)) tags.set(name, commit);
  }
  const version = upstream.tag ?? upstream.version;
  // Preserve the literal version check; a v-prefixed alternative is evidence,
  // not permission to claim that e.g. "v60" is an engine release tag.
  const literal = tags.get(version) ?? null;
  const alternatives = [...tags].filter(([name]) => name === `v${version}` || name === `b-v${version}`);
  const newestTags = [...tags].sort(([a], [b]) => b.localeCompare(a, 'en', { numeric: true })).slice(0, 10)
    .map(([tag, commit]) => ({ tag, commit }));
  return { version, tagCommit: literal, tagMatchesRef: literal === upstream.ref, alternatives, newestTags,
    ordering: 'version/name descending (ls-remote has no tag dates)' };
}

export function compareBench(candidate, reference, baselineDiverged = false) {
  if (!reference) return { status: 'SKIP', detail: `nodes=${candidate.nodes}; native unavailable`, data: { candidate, reference: null } };
  const delta = candidate.nodes - reference.nodes;
  const mismatches = [];
  if (reference.rows?.length) {
    for (let i = 0; i < Math.max(reference.rows.length, candidate.rows?.length ?? 0); i += 1) {
      const a = reference.rows[i];
      const b = candidate.rows?.[i];
      if (!a || !b || a.position !== b.position || a.nodes !== b.nodes || a.bestmove !== b.bestmove) mismatches.push({ index: i, native: a ?? null, wasm: b ?? null });
    }
  }
  const differs = delta !== 0 || mismatches.length > 0;
  return {
    status: differs ? (baselineDiverged ? 'WARN' : 'FAIL') : 'PASS',
    detail: `${differs ? `DIFF(${delta >= 0 ? '+' : ''}${delta})` : 'EXACT'} nodes=${candidate.nodes} native=${reference.nodes}${mismatches.length ? `; ${mismatches.length} position mismatch(es)` : ''}${differs && baselineDiverged ? '; baseline divergence explicitly waived' : ''}`,
    data: { candidate, reference, delta, perPositionCompared: Boolean(reference.rows?.length), mismatches, baselineDiverged },
  };
}

export function simdStatus(tier, result) {
  if (/(simd|sse41|relaxed)/i.test(tier) && result.total === 0) return 'FAIL';
  if (/relaxed/i.test(tier) && (result.relaxedF32.madd || result.relaxedF32.nmadd)) return 'WARN';
  return 'PASS';
}

export const hasFailures = (gates) => gates.some((gate) => gate.status === 'FAIL');
