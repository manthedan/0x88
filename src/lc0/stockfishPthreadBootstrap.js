const [encodedWasmUrl] = self.location.hash.slice(1).split(',');
const wasmUrl = decodeURIComponent(encodedWasmUrl ?? '');

if (!wasmUrl) {
  throw new Error('Stockfish pthread bootstrap requires an encoded WASM URL');
}

const scriptUrl = wasmUrl.replace(/\.wasm(?:[?#].*)?$/, '.js');
if (scriptUrl === wasmUrl) {
  throw new Error(`Stockfish pthread bootstrap expected a .wasm URL, received ${wasmUrl}`);
}

importScripts(scriptUrl);
