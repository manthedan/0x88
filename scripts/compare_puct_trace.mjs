import { execFileSync } from 'node:child_process';
import { writeFileSync } from 'node:fs';
import { moveFromUci, moveToActionId } from '../src/chess/moveCodec.ts';

const artifact = process.argv[2] ?? 'artifacts/student_distill_benchmark.json';
const fen = process.argv[3] ?? 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
const sims = process.argv[4] ?? '8';

const ts = execFileSync('node', ['--experimental-strip-types', 'scripts/trace_ts_puct.mjs', artifact, fen, sims], { encoding: 'utf8' });
const rust = execFileSync('cargo', ['run', '--release', '--quiet', '--manifest-path', 'rust/tiny_leela_core/Cargo.toml', '--bin', 'tiny-leela-rust-trace', '--', artifact, fen, sims], { encoding: 'utf8' });
const trace = (s) => [...s.matchAll(/^TRACE sim=(\d+) move=([^ ]+)/gm)].map((m) => m[2]);
const traceSteps = (s) => [...s.matchAll(/^TRACE sim=(\d+) move=([^ ]+) prior=([0-9.eE+-]+) q_before=([0-9.eE+-]+) score=([0-9.eE+-]+) visits_before=(\d+) child_value=([0-9.eE+-]+) q_after=([0-9.eE+-]+)/gm)].map((m) => ({
  sim: Number(m[1]),
  selected_uci: m[2],
  selected_action_id: moveToActionId(moveFromUci(m[2])),
  prior: Number(m[3]),
  q_before: Number(m[4]),
  score: Number(m[5]),
  visits_before: Number(m[6]),
  child_value: Number(m[7]),
  q_after: Number(m[8]),
}));
const priors = (s) => [...s.matchAll(/^PRIOR \d+ ([^ ]+) ([0-9.eE+-]+)/gm)].map((m) => [m[1], Number(m[2])]);
const rootValue = (s) => Number(s.match(/^root_value=([0-9.eE+-]+)$/m)?.[1] ?? NaN);
const tsTrace = trace(ts), rustTrace = trace(rust);
let matches = 0;
for (let i = 0; i < Math.min(tsTrace.length, rustTrace.length); i++) if (tsTrace[i] === rustTrace[i]) matches++;
let priorMaxAbsError = 0;
const tsP = priors(ts), rustP = priors(rust);
for (let i = 0; i < Math.min(tsP.length, rustP.length); i++) if (tsP[i][0] === rustP[i][0]) priorMaxAbsError = Math.max(priorMaxAbsError, Math.abs(tsP[i][1] - rustP[i][1]));
console.log('TS_TRACE'); console.log(ts.trim());
console.log('RUST_TRACE'); console.log(rust.trim());
console.log(`METRIC puct_trace_steps=${tsTrace.length}`);
console.log(`METRIC puct_trace_move_match_rate=${(matches / Math.max(1, tsTrace.length)).toFixed(6)}`);
console.log(`METRIC puct_trace_prior_max_abs_error=${priorMaxAbsError.toExponential(6)}`);

const outPath = process.env.PUCT_TRACE_JSON_OUT;
if (outPath) {
  const toContract = (implementation, text) => ({
    schema: 'puct_trace_v1',
    implementation,
    root_fen: fen,
    model: { artifact },
    options: { visits: Number(sims), cpuct: 1.5, temperature: 0, batch_size: 1, search_policy: 'classic' },
    root: {
      value: rootValue(text),
      legal_count: undefined,
      priors: priors(text).map(([uci, prior]) => ({ uci, action_id: moveToActionId(moveFromUci(uci)), prior })),
    },
    steps: traceSteps(text),
  });
  writeFileSync(outPath, JSON.stringify({ schema: 'puct_trace_comparison_v1', ts: toContract('typescript', ts), rust: toContract('rust', rust), metrics: { steps: tsTrace.length, move_match_rate: matches / Math.max(1, tsTrace.length), prior_max_abs_error: priorMaxAbsError } }, null, 2));
  console.log(`WROTE ${outPath}`);
}
