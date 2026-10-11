export function parseCommitBench(message) {
  const match = message.match(/^\s*Bench:\s*(\d+)\s*$/im);
  return match ? Number(match[1]) : null;
}

export function parseBenchOutput(text) {
  // Identify the final summary record BEFORE validating its fields. Filtering
  // for valid matches would silently fall back past a malformed final record.
  // Unlabelled formats retain their nodes/table markers; UCI info and table
  // position rows are not summaries.
  const summaryRecord = /^(?:(?:Bench|Results)\s*:|Nodes\s+searched\b|Total\s+nodes\b|TOTAL\b|\S+\s+nodes\b|\S+\s+[\d.+-]\S*s(?:\s|$)|\S+\s+\S+\s+\S+\s+N\/s\b)/i;
  const summary = text.split(/\r?\n/).map((line) => line.trim()).filter((line) => summaryRecord.test(line)).at(-1);
  const patterns = [
    /^(\d+)\s+nodes in (\d+(?:\.\d+)?)s(?:\s+\((\d+) nps\))?$/, // Viridithas
    /^(?:Bench:\s*)?(\d+)\s+nodes\s+(\d+) nps$/, // Reckless, Stormphrax
    /^(\d+)\s+(\d+(?:\.\d+)?)s\s+(\d+) N\/s$/, // Reckless table footer
    /^Nodes searched\s*:\s*(\d+)$/, // Stockfish
    /^Results:\s*(\d+)\s+nodes(?:\s+(\d+) nps)?$/, // Berserk
    /^(?:Total nodes(?: searched)?\s*[:=]?|TOTAL)\s+(\d+)$/i,
  ];
  const match = summary && patterns.map((pattern) => summary.match(pattern)).find(Boolean);
  const nodes = match ? Number(match[1]) : null;
  if (!match || match.slice(1).some((token) => token !== undefined && !Number.isFinite(Number(token)))) {
    throw new Error(`Missing/invalid full bench node summary${summary ? `: ${summary}` : ''}`);
  }
  const rows = [...text.matchAll(/Bench \[#\s*(\d+)\]: bestmove\s+(\S+)\s+score\s+-?\d+\s+(\d+) nodes/g)]
    .map((m) => ({ position: Number(m[1]), bestmove: m[2], nodes: Number(m[3]) }));
  if (!rows.length) {
    rows.push(...[...text.matchAll(/^\s*(\d+)\s+(\d+)\s+[\d.]+s\s+[\d.]+ N\/s\s*$/gm)]
      .map((m) => ({ position: Number(m[1]), nodes: Number(m[2]) })));
  }
  if (nodes === null || !Number.isSafeInteger(nodes) || nodes <= 0) throw new Error(`Missing/invalid full bench node summary: ${summary}`);
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
  if (reference.rows?.length && candidate.rows?.length) {
    for (let i = 0; i < Math.max(reference.rows.length, candidate.rows.length); i += 1) {
      const a = reference.rows[i];
      const b = candidate.rows[i];
      if (!a || !b || a.nodes !== b.nodes || a.bestmove !== b.bestmove) mismatches.push({ index: i, native: a ?? null, wasm: b ?? null });
    }
  }
  const differs = delta !== 0 || mismatches.length > 0;
  return {
    status: differs ? (baselineDiverged ? 'WARN' : 'FAIL') : 'PASS',
    detail: `${differs ? `DIFF(${delta >= 0 ? '+' : ''}${delta})` : 'EXACT'} nodes=${candidate.nodes} native=${reference.nodes}${mismatches.length ? `; ${mismatches.length} position mismatch(es)` : ''}${differs && baselineDiverged ? '; baseline divergence explicitly waived' : ''}`,
    data: { candidate, reference, delta, perPositionCompared: Boolean(reference.rows?.length && candidate.rows?.length), mismatches, baselineDiverged },
  };
}

export function simdStatus(tier, result) {
  if (/(simd|sse41|relaxed)/i.test(tier) && result.total === 0) return 'FAIL';
  if (/relaxed/i.test(tier) && (result.relaxedF32.madd || result.relaxedF32.nmadd)) return 'WARN';
  return 'PASS';
}

export const hasFailures = (gates) => gates.some((gate) => gate.status === 'FAIL');
