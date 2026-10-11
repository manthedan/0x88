import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { exactSearch } from '../emscripten_uci_protocol.mjs';
import { runReplay, stageEmscripten } from './bench.mjs';
import { runProcess } from './process.mjs';

export function validateParityRows(rows, artifacts, { depth = 9, positions = 20, requireExitCode = false } = {}) {
  if (!Array.isArray(rows)) return ['harness did not emit search rows'];
  const issues = [];
  const seen = new Set();
  if (rows.length !== artifacts.length * positions) issues.push(`expected ${artifacts.length * positions} rows, got ${rows.length}`);
  for (const row of rows) {
    const key = `${row.artifact}:${row.position}:${row.depth}`;
    if (seen.has(key)) issues.push(`duplicate search ${key}`);
    seen.add(key);
    if (!artifacts.includes(row.artifact) || !row.position || row.depth !== depth ||
        !row.bestmove || !row.score || !row.pv || !Number.isSafeInteger(row.nodes) || row.nodes <= 0 ||
        (requireExitCode && row.exitCode !== 0)) issues.push(`failed/incomplete search ${key}`);
  }
  for (const artifact of artifacts) if (rows.filter((row) => row.artifact === artifact).length !== positions) issues.push(`missing coverage: ${artifact}`);
  return issues;
}

export async function parityGate(context, adapter, tiers) {
  if (!adapter.parity) return { name: 'parity', status: 'SKIP', detail: 'no existing rotated-FEN Viridithas harness in v1', data: {} };
  if (adapter.parity === 'stockfish') {
    const references = new Map();
    const comparisons = [];
    for (const tier of tiers) {
      const candidate = await runReplay(context, stageEmscripten(context, tier, true), context.defaults, 10, false, true);
      const issues = validateParityRows(candidate.parityRows.map((row) => ({ ...row, artifact: tier.wasm })), [tier.wasm]);
      if (issues.length) throw new Error(`Stockfish parity: ${issues.join('; ')}`);
      const ref = references.get(tier.flavor);
      if (!ref) { references.set(tier.flavor, candidate); continue; }
      const mismatches = candidate.parityRows.filter((row, i) => !exactSearch(ref.parityRows[i], row));
      comparisons.push({ tier: tier.name, pairs: candidate.parityRows.length, exact: candidate.parityRows.length - mismatches.length, mismatches });
    }
    return { name: 'parity', status: comparisons.length ? comparisons.some((row) => row.exact !== row.pairs) ? 'FAIL' : 'PASS' : 'SKIP',
      detail: 'existing Stockfish rotated 20-FEN replay; separate full/lite cross-tier references', data: { comparisons } };
  }
  const reckless = adapter.parity === 'reckless';
  const supported = reckless ? tiers.filter((tier) => !tier.api && !tier.external) : tiers;
  if (supported.length < 2) return { name: 'parity', status: 'SKIP', detail: 'fewer than two tiers supported by existing parity harness', data: {} };
  const out = join(context.tmp, 'parity.json');
  const script = fileURLToPath(new URL(reckless ? '../reckless_wasi_node_bench.mjs' : '../emscripten_uci_bench.mjs', import.meta.url));
  const output = await runProcess(process.execPath, [script, reckless ? '--wasm' : '--js', supported.map((tier) => reckless ? tier.wasm : tier.js).join(','),
    '--depths', '9', '--positions', '20', '--json', out], { timeout: context.options.timeout, allowFailure: true, env: { ...process.env, TMPDIR: context.tmp } });
  if (output.timedOut) throw new Error('existing parity harness timed out');
  if (!existsSync(out)) throw new Error(`existing parity harness exit ${output.code} without a report: ${output.stderr.slice(-1500)}`);
  const report = JSON.parse(readFileSync(out, 'utf8'));
  const comparisons = Object.values(report.parity);
  const issues = validateParityRows(report.rows, supported.map((tier) => reckless ? tier.wasm : tier.js), { requireExitCode: reckless });
  const failed = output.code !== 0 || issues.length > 0 || !comparisons.length || comparisons.some((row) => row.pairs !== 20 || row.exact !== row.pairs);
  return { name: 'parity', status: failed ? 'FAIL' : 'PASS', detail: `existing ${reckless ? 'Reckless WASI' : 'Emscripten UCI'} harness; 20 rotated FENs, depth 9, wasm cross-tier reference${issues.length ? `; ${issues.length} row validation failure(s)` : ''}`,
    data: { report, issues, exitCode: output.code, excluded: tiers.filter((tier) => !supported.includes(tier)).map((tier) => ({ tier: tier.name, reason: 'API/external facade not supported by wrapper' })) } };
}
