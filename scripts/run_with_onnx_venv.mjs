#!/usr/bin/env node
import { spawnSync } from 'node:child_process';
import { requireOnnxVenv } from './lib/prerequisites.mjs';

const [script, ...args] = process.argv.slice(2);
if (!script) {
  console.error('Usage: node scripts/run_with_onnx_venv.mjs <script.py> [args...]');
  process.exit(1);
}

let python;
try {
  python = requireOnnxVenv();
} catch (error) {
  console.error(error.message);
  process.exit(1);
}

const result = spawnSync(python, [script, ...args], { stdio: 'inherit' });
if (result.error) {
  console.error(result.error.message);
  process.exit(1);
}
process.exit(result.status ?? 1);
