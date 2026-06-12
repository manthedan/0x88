import { START_FEN } from '../chess/board.ts';
import { MontyEngine, DEFAULT_MONTY_WASM_URL } from './montyEngine.ts';

const params = new URLSearchParams(location.search);
const statusEl = document.getElementById('status')!;
const progressEl = document.getElementById('progress')!;
const outputEl = document.getElementById('output')!;
const wasmUrl = params.get('montyWasm') ?? DEFAULT_MONTY_WASM_URL;
const nodes = Math.max(1, Math.floor(Number(params.get('nodes') ?? '500') || 500));
const contempt = Math.floor(Number(params.get('contempt') ?? '400') || 400);
const fen = params.get('fen') ?? 'r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 3';

function setStatus(text: string, cls?: 'ok' | 'fail'): void {
  statusEl.textContent = text;
  statusEl.className = cls ?? '';
}

function write(value: unknown): void {
  outputEl.textContent = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
}

async function run(): Promise<void> {
  const started = performance.now();
  const engine = new MontyEngine({ nodes, hashMb: 32 }, wasmUrl);
  const progressByUrl = new Map<string, { loadedBytes: number; totalBytes: number }>();
  engine.onDownloadProgress = (url, loadedBytes, totalBytes) => {
    progressByUrl.set(url, { loadedBytes, totalBytes });
    progressEl.textContent = [...progressByUrl.entries()]
      .map(([u, p]) => `${u.split('/').pop()}: ${(p.loadedBytes / 1e6).toFixed(0)}/${p.totalBytes ? (p.totalBytes / 1e6).toFixed(0) : '?'}MB`)
      .join(' · ');
  };
  try {
    setStatus('Prewarming Monty worker (first run downloads ~950MB of nets)…');
    await engine.prewarm();
    const prewarmMs = performance.now() - started;

    setStatus('Searching start position (contempt 0)…');
    const startBestmove = await engine.bestMove(START_FEN);
    const startInfo = engine.lastInfo();

    setStatus(`Searching start position (contempt ${contempt})…`);
    engine.setOptions({ contempt });
    await engine.newGame();
    const contemptBestmove = await engine.bestMove(START_FEN);
    const contemptInfo = engine.lastInfo();

    setStatus('Searching non-startpos FEN…');
    engine.setOptions({ contempt: 0 });
    await engine.newGame();
    const fenBestmove = await engine.bestMove(fen);
    const fenInfo = engine.lastInfo();

    const report = {
      ok: true,
      wasmUrl,
      nodes,
      contempt,
      crossOriginIsolated: (globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated === true,
      runtime: engine.runtimeStatus(),
      runtimeLabel: engine.runtimeLabel(),
      prewarmMs,
      elapsedMs: performance.now() - started,
      startBestmove,
      startInfo: startInfo[0] ?? null,
      contemptBestmove,
      contemptInfo: contemptInfo[0] ?? null,
      fen,
      fenBestmove,
      fenInfo: fenInfo[0] ?? null,
    };
    write(report);
    if (!startBestmove || !contemptBestmove || !fenBestmove) throw new Error('missing bestmove in at least one search');
    setStatus(`OK (${engine.runtimeLabel()}): ${startBestmove}, contempt ${contemptBestmove}, fen ${fenBestmove}`, 'ok');
  } finally {
    engine.dispose();
  }
}

run().catch((error) => {
  const report = { ok: false, message: error instanceof Error ? error.message : String(error), stack: error instanceof Error ? error.stack : undefined };
  write(report);
  setStatus(`Failed: ${report.message}`, 'fail');
});
