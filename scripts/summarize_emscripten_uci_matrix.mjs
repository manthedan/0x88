#!/usr/bin/env node
// Usage: node scripts/summarize_emscripten_uci_matrix.mjs results.json
import fs from 'node:fs';
import { pathToFileURL } from 'node:url';
import { exactSearch, median } from './emscripten_uci_protocol.mjs';

export function summarizeMatrix(report) {
  const spread = (values) => ({ median: median(values), min: Math.min(...values), max: Math.max(...values) });
  const out = {};
  for (const engine of [...new Set(report.cells.map((c) => c.engine))]) {
    out[engine] = {};
    const cells = report.cells.filter((c) => c.engine === engine);
    for (const label of [...new Set(cells.map((c) => c.label))]) {
      const group = cells.filter((c) => c.label === label);
      const comparisons = group.map((cell) => {
        const ref = cells.find((c) => c.label === 'current' && c.round === cell.round);
        if (!ref) throw Error(`Missing current reference for ${engine} round ${cell.round}`);
        const mismatches = [];
        let exact = 0;
        for (const row of cell.rows) {
          const base = ref.rows.find((r) => r.position === row.position && r.depth === row.depth);
          if (base && exactSearch(base, row)) exact += 1;
          else mismatches.push({ round: cell.round, position: row.position, baseline: base ?? null, candidate: row });
        }
        if (cell.rows.length !== report.positions || ref.rows.length !== report.positions) throw Error('Incomplete suite');
        return { exact, pairs: cell.rows.length, mismatches,
          wallGainPct: (cell.summary.wallNps / ref.summary.wallNps - 1) * 100,
          engineGainPct: (cell.summary.engineNps / ref.summary.engineNps - 1) * 100 };
      });
      out[engine][label] = { runs: group.length,
        exact: comparisons.reduce((s, c) => s + c.exact, 0), pairs: comparisons.reduce((s, c) => s + c.pairs, 0),
        mismatches: comparisons.flatMap((c) => c.mismatches),
        wallNps: spread(group.map((c) => c.summary.wallNps)),
        engineNps: spread(group.map((c) => c.summary.engineNps)),
        wallMs: spread(group.map((c) => c.summary.wallMs)),
        wallGainPct: spread(comparisons.map((c) => c.wallGainPct)),
        engineGainPct: spread(comparisons.map((c) => c.engineGainPct)),
        compileMs: spread(group.map((c) => c.compileMs)),
        startupMs: spread(group.map((c) => c.startupMs)) };
    }
  }
  return out;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  console.log(JSON.stringify(summarizeMatrix(JSON.parse(fs.readFileSync(process.argv[2], 'utf8'))), null, 2));
}
