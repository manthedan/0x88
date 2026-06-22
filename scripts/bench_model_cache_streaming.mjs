#!/usr/bin/env node
import { loadLc0ModelForOrt } from '../src/lc0/modelCache.ts';

function parseArgs(argv) {
  const args = { mb: 32, chunkKb: 256, repeats: 3 };
  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    const next = argv[i + 1];
    if (arg === '--mb' && next) { args.mb = Number(next); i += 1; continue; }
    if (arg === '--chunk-kb' && next) { args.chunkKb = Number(next); i += 1; continue; }
    if (arg === '--repeats' && next) { args.repeats = Number(next); i += 1; continue; }
    if (arg === '-h' || arg === '--help') {
      console.log('Usage: node --experimental-strip-types scripts/bench_model_cache_streaming.mjs [--mb 32] [--chunk-kb 256] [--repeats 3]');
      process.exit(0);
    }
    throw new Error(`Unknown argument: ${arg}`);
  }
  if (!Number.isFinite(args.mb) || args.mb <= 0) throw new Error('--mb must be positive');
  if (!Number.isFinite(args.chunkKb) || args.chunkKb <= 0) throw new Error('--chunk-kb must be positive');
  if (!Number.isFinite(args.repeats) || args.repeats <= 0) throw new Error('--repeats must be positive');
  return args;
}

function makeStream(totalBytes, chunkBytes) {
  let emitted = 0;
  return new ReadableStream({
    pull(controller) {
      if (emitted >= totalBytes) {
        controller.close();
        return;
      }
      const size = Math.min(chunkBytes, totalBytes - emitted);
      const chunk = new Uint8Array(size);
      chunk.fill(emitted & 0xff);
      emitted += size;
      controller.enqueue(chunk);
    },
  });
}

async function main() {
  const args = parseArgs(process.argv);
  const totalBytes = Math.floor(args.mb * 1024 * 1024);
  const chunkBytes = Math.floor(args.chunkKb * 1024);
  const modelUrl = 'http://localhost/models/lc0/bench.onnx';
  const manifestUrl = 'http://localhost/models/lc0/manifest.json';
  const previousFetch = globalThis.fetch;
  const previousLocation = globalThis.location;
  globalThis.location = { href: 'http://localhost/' };
  const rows = [];
  try {
    globalThis.fetch = async (input) => {
      const url = typeof input === 'string' ? input : input.url;
      if (url === manifestUrl) {
        return new Response(JSON.stringify({ models: [{ file: 'bench.onnx', url: modelUrl, bytes: totalBytes }] }), {
          headers: { 'content-type': 'application/json' },
        });
      }
      if (url === modelUrl) {
        return new Response(makeStream(totalBytes, chunkBytes), {
          headers: { 'content-length': String(totalBytes) },
        });
      }
      return new Response(null, { status: 404 });
    };

    for (let i = 0; i < args.repeats; i += 1) {
      let progressEvents = 0;
      const result = await loadLc0ModelForOrt(modelUrl, {
        cache: false,
        manifestUrl,
        onProgress: () => { progressEvents += 1; },
      });
      rows.push({
        repeat: i + 1,
        bytes: result.bytes,
        progressEvents,
        elapsedMs: Number(result.elapsedMs.toFixed(3)),
        downloadMs: Number((result.telemetry.downloadMs ?? 0).toFixed(3)),
        preallocatedDownload: result.telemetry.preallocatedDownload === true,
      });
    }
  } finally {
    globalThis.fetch = previousFetch;
    globalThis.location = previousLocation;
  }

  const avgDownloadMs = rows.reduce((sum, row) => sum + row.downloadMs, 0) / rows.length;
  console.log(JSON.stringify({
    schema: 'lc0_browser.model_cache_streaming_bench.v1',
    totalBytes,
    chunkBytes,
    repeats: args.repeats,
    avgDownloadMs: Number(avgDownloadMs.toFixed(3)),
    rows,
  }, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
