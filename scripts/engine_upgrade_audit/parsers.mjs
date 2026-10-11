export function parseCommitBench(message) {
  const match = message.match(/^\s*Bench:\s*(\d+)\s*$/im);
  return match ? Number(match[1]) : null;
}

export function parseBenchOutput(text) {
  // Summaries only: never accidentally accept an intermediate UCI nodes line.
  const patterns = [
    /^\s*(\d+) nodes in [\d.]+s\b/gm, // Viridithas
    /^\s*(?:Bench:\s*)?(\d+) nodes\s+[\d.]+ nps\b/gm, // Reckless, Stormphrax
    /^\s*(\d+)\s+[\d.]+s\s+[\d.]+ N\/s\s*$/gm, // Reckless table footer
    /^\s*Nodes searched\s*:\s*(\d+)/gm, // Stockfish
    /^\s*Results:\s*(\d+) nodes\b/gm, // Berserk
    /^\s*(?:Total nodes(?: searched)?\s*[:=]?|TOTAL)\s+(\d+)\s*$/gim,
  ];
  const matches = patterns.flatMap((pattern) => [...text.matchAll(pattern)]).sort((a, b) => a.index - b.index);
  const nodes = matches.length ? Number(matches.at(-1)[1]) : null;
  const rows = [...text.matchAll(/Bench \[#\s*(\d+)\]: bestmove\s+(\S+)\s+score\s+-?\d+\s+(\d+) nodes/g)]
    .map((m) => ({ position: Number(m[1]), bestmove: m[2], nodes: Number(m[3]) }));
  if (!rows.length) {
    rows.push(...[...text.matchAll(/^\s*(\d+)\s+(\d+)\s+[\d.]+s\s+[\d.]+ N\/s\s*$/gm)]
      .map((m) => ({ position: Number(m[1]), nodes: Number(m[2]) })));
  }
  if (nodes === null || !Number.isSafeInteger(nodes) || nodes <= 0) throw new Error('Missing/invalid full bench node summary');
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
