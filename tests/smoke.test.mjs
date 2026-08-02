import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

test('research scaffold documents LC0 browser target', () => {
  const source = readFileSync(new URL('../src/index.ts', import.meta.url), 'utf8');
  assert.match(source, /PUCT/);
  assert.match(source, /ONNX Runtime Web/);
});
