import assert from 'node:assert/strict';
import test from 'node:test';
import { setImmediate as turn } from 'node:timers/promises';

class BenchElement extends EventTarget {
  textContent = '';
  checked = false;
  value = '';
  querySelector() {
    return this;
  }
  toggleAttribute() {}
}

test('benchmark validates the selected Emscripten variants before the sweep', async (t) => {
  const elements = new Map();
  function el(id) {
    if (!elements.has(id)) elements.set(id, new BenchElement());
    return elements.get(id);
  }
  const globals = { document: { getElementById: el }, location: new URL('http://localhost:5181/lab/reckless-benchmark.html') };
  for (const [name, value] of Object.entries(globals)) {
    const previous = Object.getOwnPropertyDescriptor(globalThis, name);
    Object.defineProperty(globalThis, name, { value, configurable: true });
    t.after(() => {
      if (previous) Object.defineProperty(globalThis, name, previous);
      else delete globalThis[name];
    });
  }
  const fetch = t.mock.method(globalThis, 'fetch', async () => new Response(null));
  el('hashInput').value = '512';
  el('depthsInput').value = '2';
  el('fenInput').value = 'startpos';
  for (const id of ['benchFull', 'benchPersistent', 'benchBerserk', 'benchPlentyChess']) el(id).checked = true;
  await import('../src/lc0/recklessBenchmark.ts');
  await turn();
  fetch.mock.resetCalls();

  el('run').dispatchEvent(new Event('click'));
  assert.match(el('status').textContent, /^Use at most 256 MiB of Hash or deselect .*Berserk.*PlentyChess/);
  assert.equal(fetch.mock.callCount(), 0);

  el('benchPersistent').checked = false;
  el('benchFull').checked = false;
  el('benchOneShot').checked = true;
  el('run').dispatchEvent(new Event('click'));
  await turn();
  assert.match(el('status').textContent, /^Done\./);
});
