import { START_FEN } from '../chess/board.ts';
import { BerserkEngine, DEFAULT_BERSERK_EMSCRIPTEN_JS_URL } from './berserkEngine.ts';

const params = new URLSearchParams(location.search);
const statusEl = document.getElementById('status')!;
const outputEl = document.getElementById('output')!;
const jsUrl = params.get('berserkJs') ?? DEFAULT_BERSERK_EMSCRIPTEN_JS_URL;
const depth = Math.max(1, Math.floor(Number(params.get('depth') ?? '1') || 1));
const abortDepth = Math.max(depth + 1, Math.floor(Number(params.get('abortDepth') ?? '10') || 10));
const fen = params.get('fen') ?? 'r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 3';

function setStatus(text: string, cls?: 'ok' | 'fail'): void {
  statusEl.textContent = text;
  statusEl.className = cls ?? '';
}

function write(value: unknown): void {
  outputEl.textContent = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
}

async function expectMissingAssetFailure(): Promise<string> {
  const engine = new BerserkEngine({ depth }, `/berserk/missing-berserk-emscripten-${Date.now()}.js`);
  try {
    await engine.prewarm();
    throw new Error('Missing Berserk JS unexpectedly loaded');
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    if (/unexpectedly loaded/.test(message)) throw error;
    return message;
  } finally {
    engine.dispose();
  }
}

async function expectAbortAndRecovery(): Promise<{ abortErrorName: string; abortElapsedMs: number; recoveryBestmove: string | null }> {
  const engine = new BerserkEngine({ depth: abortDepth, hashMb: 16, threads: 1 }, jsUrl);
  const controller = new AbortController();
  const started = performance.now();
  const search = engine.bestMove(START_FEN, controller.signal);
  window.setTimeout(() => controller.abort(), 10);
  try {
    await search;
    throw new Error('Abort search unexpectedly completed');
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    if (/unexpectedly completed/.test(message)) throw error;
    const abortElapsedMs = performance.now() - started;
    engine.setOptions({ depth, hashMb: 16, threads: 1 });
    await engine.newGame();
    const recoveryBestmove = await engine.bestMove(fen);
    return { abortErrorName: error instanceof Error ? error.name : 'Error', abortElapsedMs, recoveryBestmove };
  } finally {
    engine.dispose();
  }
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
    await engine.newGame();
    setStatus('Running repeated search…');
    const repeatBestmove = await engine.bestMove(START_FEN);
    await engine.newGame();
    setStatus('Running MultiPV analysis…');
    const multipvInfo = await engine.analyze(fen, { multipv: 2, depth });
    engine.dispose();

    setStatus('Checking abort/recovery…');
    const abort = await expectAbortAndRecovery();
    setStatus('Checking missing asset failure…');
    const missingAssetMessage = await expectMissingAssetFailure();

    const report = {
      ok: true,
      jsUrl,
      depth,
      abortDepth,
      elapsedMs: performance.now() - started,
      runtime: engine.runtimeStatus(),
      startBestmove,
      startInfo: startInfo[0] ?? null,
      fen,
      fenBestmove,
      fenInfo: fenInfo[0] ?? null,
      repeatBestmove,
      multipvCount: multipvInfo.length,
      multipvInfo,
      abort,
      missingAssetMessage,
    };
    write(report);
    setStatus(`OK: ${startBestmove}, ${fenBestmove}, repeat ${repeatBestmove}, multipv ${multipvInfo.length}, abort ${abort.abortErrorName}`, 'ok');
  } finally {
    engine.dispose();
  }
}

run().catch((error) => {
  const report = { ok: false, message: error instanceof Error ? error.message : String(error), stack: error instanceof Error ? error.stack : undefined };
  write(report);
  setStatus(`Failed: ${report.message}`, 'fail');
});
