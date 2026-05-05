import { execFileSync } from 'node:child_process';

const artifact = process.argv[2] ?? 'artifacts/student_distill_benchmark.json';
const fen = process.argv[3] ?? 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
const sims = process.argv[4] ?? '8';

const ts = execFileSync('node', ['--experimental-strip-types', 'scripts/trace_ts_puct.mjs', artifact, fen, sims], { encoding: 'utf8' });
const rust = execFileSync('cargo', ['run', '--release', '--quiet', '--manifest-path', 'rust/tiny_leela_core/Cargo.toml', '--bin', 'tiny-leela-rust-trace', '--', artifact, fen, sims], { encoding: 'utf8' });
const trace = (s) => [...s.matchAll(/^TRACE sim=(\d+) move=([^ ]+)/gm)].map((m) => m[2]);
const priors = (s) => [...s.matchAll(/^PRIOR \d+ ([^ ]+) ([0-9.eE+-]+)/gm)].map((m) => [m[1], Number(m[2])]);
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
