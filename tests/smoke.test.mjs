import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

test('research scaffold documents tiny leela target', () => {
  const source = readFileSync(new URL('../src/index.ts', import.meta.url), 'utf8');
  assert.match(source, /PUCT/);
  assert.match(source, /ONNX Runtime Web/);
});
