# ORT WebGPU batching and JSPI (1.27)

Date: 2026-10-11. Branch: `perf/ort-batching-jspi`. Base: `2ea6fa1`. Runtime: `onnxruntime-web` **1.27.0**. ORT 1.30 stays rejected (`docs/ort_1_30_ab_2026-10.md`) and was not remeasured.

## Decision

| Experiment | Result |
|---|---|
| LQO physical batching (dynamic first dim, one `session.run` or a padded physical size) | **Rejected.** Physical 8 and an unpadded logical batch are much faster than today's serial batch-1 runs, and they change the 32-visit Play move. The evaluator change is reverted. |
| JSPI WebGPU pair | **Shipped** for production builds when `WebAssembly.Suspending` and `WebAssembly.promising` are both functions. Otherwise the asyncify pair. The CPU wasm pair is unchanged. The staging allowlist now includes the JSPI glue and binary. Nothing was published. |
| Batch-8 sibling of the default t1-256x10 small net | **Rejected as a default.** The 400-visit search is about 76% faster and no measured Play level is slower, but legal priors drift and visits 8 and 32 play a different move. Play, Analysis, and Arena stay on the batch-1 file with search batch 1. |

## Environment and protocol

- Apple M4. Headless Chrome through `agent-browser` with `--enable-unsafe-webgpu` (real Metal, not SwiftShader).
- UA: `Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/149.0.0.0 Safari/537.36`.
- `crossOriginIsolated=true`, secure loopback, `navigator.gpu` present, `WebAssembly.Suspending` and `WebAssembly.promising` present. Console errors during the cohort: none.
- Product client built with `BUILD_SCOPE=product npm run build:client`, served by `PORT=5184 node scripts/serve_isolated_static.mjs dist-client`. The dev server was not used (it 404s the wasm sidecars).
- Browser timing held `scratchpad/benchlock`. Other agents' locks were left alone. This cohort ran 2026-10-11T09:02:30Z–09:05:23Z (172.7s of bench work after the wait). Load average 3.06 / 2.50 / 4.38 at start and 2.24 / 2.40 / 3.99 at end, so the machine was busy.
- Three rounds. LQO physical sizes were interleaved p1 / p4 / p8 / logical. JSPI and the small-net search were ABAB. Each arm got a fresh worker. Strict `ep=webgpu`, one wasm thread, eval cache off, `reuseTree` false, pipeline depth 1, `yieldEveryMs` 16. LQO and the small-net arms pinned `runtimeArtifact: 'asyncify'` so JSPI auto-select did not mix into those tables. The JSPI arms pinned `asyncify` or `jspi` and the ready diagnostics matched the pin on every sample (`jspiFallback` unset).
- Search tables are the median [min, max] of the three `elapsedMs` samples. Warm-eval tables are the median of the three per-session medians (5 timed evals after 2 warmups). Session-create is the median of the three `InferenceSession` create samples, after model bytes were already in hand.
- A lab page drove the matrix and was deleted afterwards. Its screenshot timed out at 30s; the page is not part of the ship. The loader check for the shipped JSPI path is the built-app smoke recorded at the end of this note.

The node batch tolerance in `tests/lc0_onnx_batch.test.mjs` (non-QDQ float16) is: same best move, `|q| < 1e-5`, `|mlh| < 1e-3`, top-5 legal prior `|prior| < 1e-5`. GPU QDQ below is far outside that.

## Confirmed input shapes

Live session metadata (`inputShape` on the ready message) and a protobuf walk of the ONNX graph agree.

| File | Bytes | Input `/input/planes` | Outputs |
|---|---:|---|---|
| `public/models/lc0/lqo_v2.f16.qdq8.onnx` | 95,662,716 | float16, `['batch', 112, 8, 8]` | policy `['batch', 1858]`, wdl `['batch', 3]`, mlh `['batch', 1]` |
| `public/models/lc0/t1-256x10-distilled-swa-2432500.batch1.f16.qdq8.onnx` | 20,558,031 | float16, `[1, 112, 8, 8]` | policy still has a symbolic batch |
| `public/models/lc0/t1-256x10-distilled-swa-2432500.batch8.f16.qdq8.onnx` | 20,558,031 | float16, `[8, 112, 8, 8]` | wdl and mlh leading dim 8 |

SHA-256 (unchanged bytes): LQO `4effd3079add6ab04043f066057f987b54e90e7e54cdf62ac0931b300769fc0e`, batch-1 small `9396c11226de7f29dbad8c60147b1082b8093adc7a0b77819cd0fb33103a095f`, batch-8 small `203237314baa01a05edc41f1a59f3e34c8f86c445bc47156ad8cab0c379d9273`, Maia3 `4141faeca4b0aa9f99073d70abb43211264ed2eb04cb96214ba85a86dd455a10` (28,138,813 bytes, weights live under the shared models directory, not in `public/`).

`sessionFixedInputBatchSize` treated a symbolic first dim as 1, so an LQO search with `recommendedBatchSize` 8 issued eight serial `session.run` calls. The live search confirmed that: pin 1 reported `ortPhysicalBatchSize` 1, pin 4 reported 4, pin 8 reported 8. With no pin, the mean physical size over eval calls was 4.5 / 6.6 / 7.21 / 7.86 at 8 / 32 / 100 / 400 visits (search batch cap 8, real batches smaller than the cap).

## 1. LQO physical batching — rejected

Play queen-odds, human white, bot missing the queen. FEN `rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1` with the existing color swap, so the net saw `rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR b KQkq - 0 1`. Search options were Play's: `drawScore=-0.6`, `searchContemptLimit=32`, `cpuct=1.5`, `fpu=0.4`, `batchSize=8`. Color swap, legal priors, and contempt were left as they are; only the physical batch fed to `session.run` changed, in a lab worker.

Search wall, milliseconds, median [min, max], n=3:

| Visits | Physical 1 (today) | Physical 4 | Physical 8 | Logical (no pad) |
|---:|---:|---:|---:|---:|
| 8 | 177.885 [175.675, 189.08] | 87.43 [86.8, 88.54] | 96.73 [96.19, 97.055] | 85.61 [84.07, 88.765] |
| 32 | 528.6 [513.185, 548.475] | 256.58 [255.35, 257.695] | 238.945 [235.635, 240.315] | 210.515 [209.415, 212.335] |
| 100 | 1559.705 [1550.945, 1623.09] | 735.83 [729.67, 736.87] | 671.375 [649.635, 677.96] | 613.51 [603.295, 623.32] |
| 400 | 6231.04 [6192.545, 6235.82] | 2871.43 [2838.795, 2872.37] | 2446.995 [2366.9, 2463.25] | 2413.34 [2334.67, 2415.125] |

Physical 8 is about 46% / 55% / 57% / 61% faster than physical 1. Ranges do not overlap. Logical is as fast or faster than a pad to 8. 1600 visits were not measured. The speed gate alone would pass.

Parity against five serial batch-1 rows (startpos, e4 reply, Nf3/Nf6, the queen-odds FEN, an Italian-ish position). Policy argmax matched on every row for physical 4, 5, and 8: `e2e4`, `d7d5`, `f3e5`, `d2d4`, `f3g5`. Max absolute deltas, the same on all three physical sizes: prior 0.01684712796989385, q 0.006103515625, wdl 0.003173828125, mlh 0.03271484375. That fails the existing tolerance.

Root moves were identical on all three rounds. Physical 1 played `g8f6` / `d7d5` / `d7d5` / `d7d5` at 8 / 32 / 100 / 400. Physical 4, physical 8, and logical all played `g8f6` / `e7e5` / `d7d5` / `d7d5`. The 32-visit Play level changes move (`d7d5` → `e7e5`) and the root value changes (−0.236 → −0.221). Within an arm, visits 8 / 32 / 100 reproduced the same value on every round. Visit 400 kept the move and wobbled in the fourth digit (physical 1 values −0.214557, −0.214838, −0.215392).

Killed. `src/lc0/onnxEvaluator.ts` is back to treating a symbolic leading dim as batch 1. No Play or `bt4Engine` change.

## 2. JSPI — shipped

ORT 1.27 ships a matching pair next to the asyncify pair:

| File | Raw bytes | SHA-256 | Brotli q11 | gzip -9 |
|---|---:|---|---:|---:|
| `ort-wasm-simd-threaded.asyncify.wasm` | 24,254,953 | `7e83cd6cee77e478bc96a7e91b198144fb5e4126287daf1f9b54bb195ebcd55a` | 3,439,945 | 5,878,533 |
| `ort-wasm-simd-threaded.jspi.wasm` | 15,046,878 | `7c28cdb40958a998f5aa0981d5cb8e57ac1e7e9b4d2f18a7d74e00dd9629d7a3` | 2,333,297 | 3,656,847 |
| `ort-wasm-simd-threaded.asyncify.mjs` | 47,507 | `7236653b8565da4046e459cd0e274123419a1d9f1f8f18fd36c28058346ca655` | 15,247 | |
| `ort-wasm-simd-threaded.jspi.mjs` | 44,523 | `47f5232865f07cc6a11e825cac8dfd9b75bdf5f4a0889bde077e753ae93d3913` | 14,643 | |
| `ort-wasm-simd-threaded.wasm` (CPU) | 13,479,978 | `d1ab1b94b16a65b29d710d0b587b29e7bed336827577623913479b8afe8113e6` | | |
| `ort.webgpu.min.mjs` | 67,169 | `46988a5a025f49449850f39f95eb0d21e40e67b3beb13a0b54efd3ab5d83f60e` | | |
| `ort.jspi.min.mjs` | 67,275 | `98ec6626b872143ee084d1685a467bdc5c0a29e41c23b6ab0c5dbbf9a17476a7` | | |

Brotli q11 gap on the GPU binary is 1,106,648 bytes. The glue gap is a few hundred bytes. The wasm gap is large enough to ship.

The package `exports` map includes `./jspi` and `./webgpu`. Under the `onnxruntime-web-use-extern-wasm` condition (what Vite resolves), those entries are `dist/ort.jspi.min.mjs` and `dist/ort.webgpu.min.mjs`. The JSPI orchestrator hardcodes the glue name `ort-wasm-simd-threaded.jspi.mjs` and throws `WebAssembly JSPI is not supported` when `"Suspending" in WebAssembly` is false. The WebGPU orchestrator hardcodes `ort-wasm-simd-threaded.asyncify.mjs` and does not mention `Suspending`. The file this app imports is still `onnxruntime-web/webgpu`. The JSPI emscripten glue then calls `locateFile("ort-wasm-simd-threaded.jspi.wasm")`, wraps imports with `new WebAssembly.Suspending`, wraps `OrtRun` / `OrtRunWithBinding` / `OrtBindInput` with `WebAssembly.promising`, and contains the `em-pthread` bootstrap (`new Worker(new URL(import.meta.url), { name: 'em-pthread' })`). Production `wasmPaths` points that webgpu bundle at the JSPI `.mjs` and `.wasm` together. A second import of `onnxruntime-web/jspi` is unnecessary: that orchestrator throws when the API is missing, so it cannot be the fallback. The dev server cannot swap glue either. A `/ort/` prefix is blocked for source-mode imports, and the webgpu bundle's own glue is the asyncify module. Pointing that glue at the JSPI or CPU binary is a `LinkError`.

The in-process WebGPU-to-wasm fallback stays on whichever GPU binary the thread already selected, the same way it stays on asyncify today. This cohort used strict `ep=webgpu` and did not exercise that fallback.

Selection (`src/nn/ortRuntime.ts`):

- Feature detect requires both `WebAssembly.Suspending` and `WebAssembly.promising` to be functions.
- Production GPU threads (`import.meta.env.PROD`, WebGPU still in the resolved provider list, or strict `ep=webgpu`) load JSPI when the API exists and the preference is `auto` or `on`.
- Missing API, preference `off`, a non-production bundle, Node (inlined asyncify glue), and the experimental fixed/relaxed artifacts stay on asyncify. A forced `jspi` kind with no API is coerced to asyncify with fallback reason `WebAssembly.Suspending or WebAssembly.promising is unavailable`.
- The CPU wasm EP stays `ort-wasm-simd-threaded.wasm`. WebGPU threads stay at 1 thread.
- Preference is read on the thread that owns `location`: `?ortJspi=` (`0` / `false` / `off` / `asyncify` → off, `1` / `true` / `on` / `jspi` → on, empty / `auto` → auto), else `ORT_JSPI` or `TINY_LEELA_ORT_JSPI`. Product workers (Play, Analysis, Arena, Maia) do not copy the page query and no caller forwards a pin, so they follow auto-detect. `?ortJspi=0` on a Play URL does not switch the worker. A main-thread session, such as `/single-engine`, does see the page query.

### Session and eval

Fresh worker, ABAB, n=3. Backend `webgpu->webgpu`, 1 thread.

| Lane | Session create ms | First eval ms | Warm median ms |
|---|---:|---:|---:|
| Maia3 asyncify | 443.15 [442.7, 444.33] | 39.215 [38.54, 146.5] | 11.325 [11.205, 11.495] |
| Maia3 JSPI | 369.605 [366.415, 474.35] | 34.8 [31.615, 53.445] | 11.315 [11.295, 11.43] |
| Small Lc0 asyncify | 355.81 [353.895, 356.795] | 29.195 [27.135, 31.745] | 10.05 [10.025, 10.575] |
| Small Lc0 JSPI | 322.345 [315.975, 323.665] | 27.7 [27.07, 29.185] | 9.535 [9.51, 9.605] |

Maia3 is the QDQ model, Elo 1500/1500, startpos tokens. Small Lc0 is the batch-1 QDQ file, startpos, best move `d2d4`, q `0.043212890625` on every session. One Maia JSPI create sample (474.35 ms, the first JSPI arm) was slower than every asyncify sample. The later two (369.605, 366.415) do not overlap asyncify, and the median is faster. Small-Lc0 create has no overlap and JSPI is faster. Warm Maia medians tie. Warm small-Lc0 medians favor JSPI with no overlap. The asyncify Maia first-eval max (146.5 ms) is the first shader compile of that cohort; it is not a warm number.

Logits. Maia: 4352 move logits and 3 value logits, max abs delta 0 and mean 0 on every compare, including later asyncify sessions against the first asyncify store. Small Lc0: 20 legal-prior logits and 3 WDL outputs, max abs delta 0 and mean 0, best move unchanged. The GPU was deterministic across these sessions.

Wasm compile, one shot each after the session cohort, asyncify then JSPI, so the disk cache may favor JSPI. These are single observations, not medians: asyncify fetch 44.575 ms, compile 15.52 ms, 24,254,953 bytes; JSPI fetch 35.705 ms, compile 11.18 ms, 15,046,878 bytes.

Kill criteria are not met. Median session create is faster, warm eval is not slower, logits match, the missing-API fallback is unit-tested, and the Brotli gap is about 1.06 MB.

### Staging (not published)

`public/ort` is a symlink to `node_modules/onnxruntime-web/dist`, which already contains the JSPI pair. The content-addressed artifact manifest (`artifacts/sha256/...`, `scripts/write_artifact_release_manifests.mjs`) does not list ORT wasm. Same-origin `/ort/` files are chosen by `scripts/ort_runtime_assets.mjs` and copied by `scripts/prepare_netlify_r2_public_assets.mjs`. `scripts/check_ort_runtime_assets.mjs` requires every allowlisted name and the pthread markers on every glue module. `scripts/check_ort_wasm_dedup.mjs` requires each allowlisted wasm once and rejects a Vite-emitted copy.

The allowlist was four files (asyncify mjs/wasm and the CPU mjs/wasm). It is now six: those four plus `ort-wasm-simd-threaded.jspi.mjs` and `ort-wasm-simd-threaded.jspi.wasm`. JSEP and `ort.webgpu.min.mjs` stay excluded (Vite bundles the JS entry). Auto-detect does not fall back when the JSPI file 404s, so a future publish has to include the pair. This branch does not run the release script and does not upload anything.

`public/ort-experimental/manifest.json` is the fixed/relaxed asyncify experiment and is unchanged. Those builds stay on the asyncify identity.

## 3. Batch-8 small net — speed passes, default unchanged

Same bytes as the batch-1 file (20,558,031). Lab A/B, asyncify pinned, three rounds ABAB. Batch-1 file with search batch 1 (what Play and Analysis do today: Analysis hardcodes batch 1, Play's small-net search omits `batchSize` and PUCT defaults to 1) versus the batch-8 file with search batch 8. `ortPhysicalBatchSize` was 1 versus 8.

The search FEN was the queen-odds position `rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1` (black missing the queen, white to move), with no color swap. It is the position the harness already had open. It is not startpos. The five-FEN prior check does include startpos.

| Visits | Batch-1 file, search batch 1 | Batch-8 file, search batch 8 | Median change |
|---:|---:|---:|---:|
| 8 | 93.895 [90.725, 103.79] | 39.07 [38.5, 39.18] | −58% |
| 32 | 324.02 [315.52, 333.635] | 90.955 [90.39, 91.39] | −72% |
| 100 | 971.615 [895.22, 997.57] | 253.28 [247.48, 265.405] | −74% |
| 400 | 3738.515 [3584.295, 4383.265] | 899.91 [891.36, 917.075] | −76% |

No level is slower. Ranges do not overlap. The 400-visit gain is 75.9%, above the 10% bar. 1600 was not measured.

Cross-file prior check, five FENs, batch-1 serial rows versus one batch-8 `evaluateBatch`. Policy argmax matched: `d2d4`, `e7e5`, `d2d4`, `d2d4`, `f3g5`. Max abs deltas: prior 0.01609116123212606, q 0.0040283203125, wdl 0.0029296875, mlh 0.5. Outside the existing tolerance. The mlh gap of 0.5 is on its own enough to refuse a silent model swap.

Root moves on the queen-odds FEN were identical on all three rounds. Batch 1 played `e2e4` / `e2e4` / `e2e4` / `d2d4`. Batch 8 played `b2b4` / `d2d4` / `e2e4` / `d2d4`. Play levels 8 and 32 change move. At 400 visits the move agrees and the values are almost equal (0.996158 vs 0.996156) but the low-visit change is stable.

The speed gate passes. The nets are not interchangeable, so the product default stays the batch-1 file and search batch 1. No Play, Analysis, or Arena edit.

## What is in the tree

`src/nn/ortRuntime.ts` selects the JSPI pair on a production GPU thread. `scripts/ort_runtime_assets.mjs` stages it. Tests cover the preference matrix, the missing-API coercion, the six-file allowlist, the pthread-marker check, the dedup check, and the R2 public-asset copy. The dynamic-batch evaluator, the lab route, and the worker init pins used for the A/B were removed.

## Built-app loader check

`BUILD_SCOPE=product npm run build:client` exited 0 after the ship commit. The built `searchWorker`, `maia3Worker`, and the main ORT chunk call the selector with `prod:!0`. `dist-client/lab/ort-batching/` is absent (HTTP 404). `/ort/ort-wasm-simd-threaded.jspi.wasm` is staged and returns 200 (15,046,878 bytes).

Benchlocked open of `http://127.0.0.1:5184/single-engine/?ep=webgpu` at 2026-10-11T09:27:19Z–09:28:21Z. Load average 2.88 / 4.41 / 4.01 at start. The page reached `Ready. Drag a legal move or ask the engine to move.` GPU status `active`, backend `webgpu->webgpu`, model `/models/lc0/t1-256x10-distilled-swa-2432500.batch1.f16.qdq8.onnx`. Main-thread resource timing fetched only:

- `http://127.0.0.1:5184/ort/ort-wasm-simd-threaded.jspi.mjs`
- `http://127.0.0.1:5184/ort/ort-wasm-simd-threaded.jspi.wasm`

The asyncify wasm was not requested. A second open reproduced the same ready state. `agent-browser screenshot` did not return (killed at 60s for the full page and at 25s for `#gpuStatus`), so there is no PNG. The browser session was closed after each attempt.
