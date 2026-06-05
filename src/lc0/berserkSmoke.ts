import { START_FEN } from '../chess/board.ts';
import { BerserkEngine, DEFAULT_BERSERK_EMSCRIPTEN_JS_URL } from './berserkEngine.ts';

const params = new URLSearchParams(location.search);
const statusEl = document.getElementById('status')!;
const outputEl = document.getElementById('output')!;
const jsUrl = params.get('berserkJs') ?? DEFAULT_BERSERK_EMSCRIPTEN_JS_URL;
const depth = Math.max(1, Math.floor(Number(params.get('depth') ?? '1') || 1));
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
  const engine = new BerserkEngine({ depth, hashMb: 16, threads: 1 }, jsUrl);
  try {
    setStatus('Prewarming Berserk worker…');
    await engine.prewarm();
    await engine.newGame();
    setStatus('Searching start position…');
    const startBestmove = await engine.bestMove(START_FEN);
    const startInfo = engine.lastInfo();
    await engine.newGame();
    setStatus('Searching non-startpos FEN…');
    const fenBestmove = await engine.bestMove(fen);
    const fenInfo = engine.lastInfo();
    const report = {
      ok: true,
      jsUrl,
      depth,
      elapsedMs: performance.now() - started,
      runtime: engine.runtimeStatus(),
      startBestmove,
      startInfo: startInfo[0] ?? null,
      fen,
      fenBestmove,
      fenInfo: fenInfo[0] ?? null,
    };
    write(report);
    setStatus(`OK: ${startBestmove}, ${fenBestmove}`, 'ok');
  } finally {
    engine.dispose();
  }
}

run().catch((error) => {
  const report = { ok: false, message: error instanceof Error ? error.message : String(error), stack: error instanceof Error ? error.stack : undefined };
  write(report);
  setStatus(`Failed: ${report.message}`, 'fail');
});
