// Classic Worker, matching the product's importScripts + modularized factory path.
// Stage beside the protocol module on an isolated production static server.
self.onmessage = async ({ data: { cell, depth, hash, positions, nonce } }) => {
  try {
    const started = performance.now();
    const { runSuite, summarizeCell } = await import('./emscripten_uci_protocol.mjs');
    importScripts(`${cell.js}?cpuMatrix=${encodeURIComponent(nonce)}`);
    const fetchStarted = performance.now();
    const getBytes = async (url) => {
      const response = await fetch(url, { cache: 'no-store' });
      if (!response.ok) throw Error(`${url}: HTTP ${response.status}`);
      return new Uint8Array(await response.arrayBuffer());
    };
    const [wasm, data] = await Promise.all([getBytes(cell.js.replace(/\.js$/, '.wasm')), getBytes(cell.data)]);
    const fetchMs = performance.now() - fetchStarted;
    // A valid ignored custom section with a unique nonce defeats same-byte V8
    // module cache reuse between workers. It changes no executable instruction.
    const payload = new TextEncoder().encode(`\u0000${nonce}`); // zero-length section name
    if (payload.length >= 128) throw Error('Nonce too long');
    const bytes = new Uint8Array(wasm.length + 2 + payload.length);
    bytes.set(wasm); bytes.set([0, payload.length], wasm.length); bytes.set(payload, wasm.length + 2);
    const compileStarted = performance.now();
    const compiled = await WebAssembly.compile(bytes);
    const compileMs = performance.now() - compileStarted;
    const stdout = [];
    let instantiateMs;
    const module = await self[cell.factory]({
      print: (line) => {
        stdout.push(String(line));
        self.postMessage({ type: 'line', line: String(line), stream: 'stdout' });
      },
      printErr: (line) => self.postMessage({ type: 'line', line: String(line), stream: 'stderr' }),
      locateFile: (file) => file.endsWith('.data') ? cell.data : cell.js.replace(/\.js$/, '.wasm'),
      getPreloadedPackage: () => data.buffer,
      instantiateWasm(imports, receive) {
        const t = performance.now();
        const instance = new WebAssembly.Instance(compiled, imports);
        instantiateMs = performance.now() - t;
        receive(instance, compiled);
        return instance.exports;
      },
    });
    const startupMs = performance.now() - started;
    const rows = runSuite(module, stdout, { depth, hash, positions });
    self.postMessage({ ...cell, depth, hash, threads: 1, fetchMs, compileMs, instantiateMs, startupMs,
      rows, summary: summarizeCell(rows) });
  } catch (error) { self.postMessage({ error: String(error.stack ?? error) }); }
};
