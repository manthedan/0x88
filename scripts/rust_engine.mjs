import { execFileSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { boardToFen } from '../src/chess/board.ts';
import { legalMoves } from '../src/chess/movegen.ts';
import { moveFromUci, moveToUci } from '../src/chess/moveCodec.ts';

const manifest = 'rust/tiny_leela_core/Cargo.toml';
const bin = 'rust/tiny_leela_core/target/release/tiny-leela-rust-eval';
const builtBins = new Set();

export function ensureRustBin(name) {
  const path = `rust/tiny_leela_core/target/release/${name}`;
  if (builtBins.has(name) && existsSync(path)) return path;
  execFileSync('cargo', ['build', '--release', '--quiet', '--manifest-path', manifest, '--bin', name], { stdio: 'inherit' });
  builtBins.add(name);
  return path;
}

export function ensureRustEngine() {
  return ensureRustBin('tiny-leela-rust-eval');
}

function parseOutput(out) {
  const bestMove = out.match(/^best_move=(.*)$/m)?.[1]?.trim() ?? 'none';
  const policyJson = out.match(/^root_policy_json=(.*)$/m)?.[1]?.trim() ?? '[]';
  const wdlText = out.match(/^wdl=([0-9.eE+,-]+)$/m)?.[1]?.trim() ?? '0,1,0';
  const metrics = Object.fromEntries([...out.matchAll(/^METRIC ([^=]+)=(-?[0-9.eE+]+)$/gm)].map((m) => [m[1], Number(m[2])]));
  let policy = [];
  try { policy = JSON.parse(policyJson); } catch { policy = []; }
  const wdl = wdlText.split(',').map(Number);
  return { bestMove, policy, wdl, metrics, raw: out };
}

export function rustEvaluateFen(fen, { model = 'artifacts/student_distill_benchmark.json', visits = 64, temperature = 0 } = {}) {
  ensureRustEngine();
  const out = execFileSync(bin, [model, fen, String(visits), String(temperature)], { encoding: 'utf8', maxBuffer: 1024 * 1024 * 32 });
  return parseOutput(out);
}

export function rustChooseMove(board, options = {}) {
  const legalUci = new Set(legalMoves(board).map(moveToUci));
  const result = rustEvaluateFen(boardToFen(board), options);
  if (!result.bestMove || result.bestMove === 'none' || !legalUci.has(result.bestMove)) return { ...result, move: null };
  return { ...result, move: moveFromUci(result.bestMove) };
}

export function rustPolicyForBoard(board, options = {}) {
  const legalByUci = new Map(legalMoves(board).map((move) => [moveToUci(move), move]));
  const result = rustEvaluateFen(boardToFen(board), options);
  const policy = result.policy
    .map((entry) => ({ ...entry, move: legalByUci.get(entry.move) ?? null }))
    .filter((entry) => entry.move && entry.probability > 0);
  const move = legalByUci.get(result.bestMove) ?? policy[0]?.move ?? null;
  return { ...result, move, policy };
}
