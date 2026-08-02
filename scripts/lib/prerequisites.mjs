import { accessSync, constants } from 'node:fs';
import { resolve } from 'node:path';

/**
 * Prerequisite checks for scripts that depend on machine-local state (python
 * venvs, sibling checkouts, downloaded models). Fail with an actionable
 * message instead of a bare ENOENT from deep inside the script.
 */

export function requireReadableFile(path, hint) {
  try {
    accessSync(path, constants.R_OK);
    return path;
  } catch {
    throw new Error(`Required file not found or unreadable: ${path}${hint ? `\n${hint}` : ''}`);
  }
}

export function requireDirectory(path, hint) {
  try {
    accessSync(path, constants.R_OK);
    return path;
  } catch {
    throw new Error(`Required directory not found: ${path}${hint ? `\n${hint}` : ''}`);
  }
}

/** Resolve the repo's ONNX python venv interpreter, or throw setup instructions. */
export function requireOnnxVenv(repoRoot = process.cwd()) {
  const python = resolve(repoRoot, '.venv-onnx/bin/python');
  try {
    accessSync(python, constants.X_OK);
    return python;
  } catch {
    throw new Error(
      `This script needs the ONNX python environment at .venv-onnx/. Set it up with:\n` +
        '  python3 -m venv .venv-onnx\n' +
        '  .venv-onnx/bin/pip install -r requirements-onnx.txt',
    );
  }
}
