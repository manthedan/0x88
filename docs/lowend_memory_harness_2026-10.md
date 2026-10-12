# Low-end memory harness — October 2026

There was no measurement of resident memory per surface, or of what stays allocated after the user leaves that surface. This note is that measurement on this Mac, plus the harness and the steps for a phone or a small desktop. It does not change product behavior. The machine is an Apple M4 with 10 cores and 24 GiB of RAM. These numbers are a high-end baseline, not a low-end survival test.

## What was measured

`npm run mem:harness` drives the built app with Playwright (already a dependency). Each sample calls `performance.measureUserAgentSpecificMemory()`, reads `performance.memory` (JS heap), and counts workers. The page is cross-origin isolated because `scripts/serve_isolated_static.mjs` sends COOP/COEP/CORP. The memory call waits for garbage collection. On this Chrome that wait is about 11–19 s for the first samples in a page and then about 60 s, which is the API's own spacing, not engine time. `measure latency` in the table is that wait.

Playwright's default headless binary is `chrome-headless-shell`. That shell exposes the function and then rejects it with "is not available" because Performance Manager instrumentation is off. The harness launches Playwright's `channel: 'chromium'` build, Chrome for Testing 151.0.7922.34, including when headless, and passes `--enable-precise-memory-info` so the JS heap is not quantized to the default coarse buckets. JS heap still misses worker WASM: on Play it stays under 7 MiB while user-agent memory is over 1 GiB.

Each phase is a fresh browser context, so Play's leftover workers do not leak into Analysis. Inside a phase the page is one session, so retention is visible. Measured runs do not add `?memprobe=1`. The lab overlay is a separate check. Medians are three runs. A checkpoint counts as unreleased when the peak gained since the comparison baseline is above 8 MiB and more than half of that gain is still present. Landing the threshold exactly does not flag. The summary calls that finding repeated only when a strict majority of runs flag it; a tie does not. Medians, spreads, retention counts, and phase durations are in `docs/lowend_memory_harness_2026-10.json`. The per-run checkpoint readings those numbers come from are the twelve phase files in `docs/lowend_memory_harness_2026-10_runs/`.

Phases, one bench-lock chunk each: Play (landing, Maia3 first move, then sf-lite, sf-full, Reckless, Viridithas, Berserk, PlentyChess, Stormphrax, then Lc0 small, then in-app return to landing); Analysis (shell, Lc0 held for 30 s at 100000 visits, Stop, then each CPU family at depth 2, then the row back on Lc0); Arena (Stockfish Lite vs Reckless, 200 ms, one game, then Stop); Maia (Analysis human-move panel). Wall time inside the lock: Play 495–502 s, Analysis 450–463 s, Arena 138–141 s, Maia 32–38 s. All four stay under the 20 minute chunk. One-minute load average at phase start ranged from 0.88 to 3.69.

This worktree's `dist-client` did not contain the gitignored Maia3 ONNX files or the Reckless wasm/nnue blobs. For the run they were symlinked into `dist-client` from the models cache and the main checkout's `public/reckless`. Without those files those engines 404 and the checkpoint fails. The symlinks are not part of the commit.

## Results

User-agent bytes are Chrome's attribution, not OS RSS, and not the WASM `INITIAL_MEMORY` reservation. The big steps land in the same range as those reservations (Stockfish full +320 MiB, Berserk +281, PlentyChess +307, Stormphrax +566).

| Checkpoint | Resident | UA median | UA min–max | Delta | JS heap | Workers | Measure latency |
|---|---|---:|---:|---:|---:|---:|---:|
| landing | | 1.5 MiB | 1.5–1.5 | | 1.5 MiB | 0 | 15.0 s |
| play-default-after-first-move | maia3 | 116.6 MiB | 116.6–116.6 | +115.1 | 3.6 MiB | 1 | 16.0 s |
| play-cpu-sf-lite | sf-lite | 245.5 MiB | 245.5–245.5 | +128.9 | 3.7 MiB | 2 | 17.0 s |
| play-cpu-sf-full | sf-full | 565.0 MiB | 565.0–565.1 | +319.5 | 3.8 MiB | 3 | 16.0 s |
| play-cpu-reckless | reckless | 565.2 MiB | 565.2–565.2 | +0.2 | 3.9 MiB | 4 | 60.0 s |
| play-cpu-viridithas | viridithas | 565.4 MiB | 565.4–565.4 | +0.2 | 4.0 MiB | 5 | 60.0 s |
| play-cpu-berserk | berserk | 846.1 MiB | 846.1–846.1 | +280.7 | 4.1 MiB | 6 | 60.0 s |
| play-cpu-plentychess | plentychess | 1153.4 MiB | 1153.4–1153.4 | +307.2 | 4.0 MiB | 7 | 60.0 s |
| play-cpu-stormphrax | stormphrax | 1719.4 MiB | 1719.4–1719.4 | +566.1 | 4.1 MiB | 8 | 60.0 s |
| play-neural-lc0-small | lc0-small | 1255.4 MiB | 1255.4–1255.4 | −464.0 | 6.7 MiB | 7 | 60.0 s |
| landing-after-play | | 1255.3 MiB | 1255.3–1255.3 | −0.2 | 6.7 MiB | 7 | 60.0 s |
| analysis-shell | | 247.0 MiB | 247.0–247.0 | | 3.3 MiB | 1 | 16.0 s |
| analysis-lc0-30s | lc0 | 132.4 MiB | 132.3–133.2 | −114.7 | 3.4 MiB | 1 | 13.0 s |
| analysis-lc0-stopped | lc0 | 115.3 MiB | 115.3–115.3 | −17.1 | 3.4 MiB | 1 | 15.0 s |
| analysis-cpu-sf | sf | 115.2 MiB | 115.2–115.3 | −0.1 | 5.3 MiB | 2 | 18.0 s |
| analysis-cpu-reckless | reckless | 460.1 MiB | 460.0–460.1 | +344.8 | 3.7 MiB | 3 | 60.0 s |
| analysis-cpu-viridithas | viridithas | 460.4 MiB | 460.4–461.7 | +0.4 | 3.8 MiB | 3 | 60.0 s |
| analysis-cpu-berserk | berserk | 460.4 MiB | 460.2–460.7 | −0.2 | 3.8 MiB | 3 | 60.0 s |
| analysis-cpu-plentychess | plentychess | 460.6 MiB | 460.6–460.7 | +0.2 | 3.9 MiB | 3 | 60.0 s |
| analysis-cpu-stormphrax | stormphrax | 463.6 MiB | 460.5–463.8 | +3.1 | 4.0 MiB | 3 | 60.0 s |
| analysis-after-cpu | lc0 | 460.1 MiB | 460.1–460.2 | −3.6 | 4.0 MiB | 2 | 60.0 s |
| arena-shell | | 4.0 MiB | 4.0–4.0 | | 3.4 MiB | 0 | 18.0 s |
| arena-cpu-match | sf+reckless | 133.4 MiB | 133.4–133.4 | +129.4 | 4.3 MiB | 2 | 60.0 s |
| arena-cpu-stopped | | 133.8 MiB | 133.6–133.8 | +0.4 | 4.3 MiB | 2 | 60.0 s |
| maia-before | | 3.8 MiB | 3.8–3.8 | | 3.3 MiB | 1 | 12.0 s |
| maia-panel | maia3 | 218.7 MiB | 218.7–218.7 | +214.9 | 24.3 MiB | 2 | 16.0 s |
| landing-after-maia | | 4.4 MiB | 4.4–4.5 | −214.3 | 4.0 MiB | 0 | 1.7 s |

Spread on the large Play and Arena rows is under 0.2 MiB across three runs. Analysis Stormphrax is the wide CPU row (460.5–463.8 MiB).

## Largest residents, and what never comes back

The largest resident is Play after Stormphrax: 1719.4 MiB and 8 live workers. Switching the selection to Lc0 small drops the Stormphrax worker (about 566 MiB, 8 workers to 7) and leaves 1255.4 MiB. In-app navigation back to the landing page leaves 1255.3 MiB and the same 7 workers. That is 1254 MiB above the fresh landing page, on 3/3 runs. The module caches and workers survive leaving Play.

On the run-1 breakdown, the workers still present on the landing page after Play were Stockfish full (~319 MiB, `/stockfish/stockfish-19-single.js`), two dedicated workers with opaque URLs (~307 MiB and ~281 MiB, matching the PlentyChess and Berserk steps), Stockfish lite (~129 MiB), the Maia3 worker (~112 MiB), and the window itself (~104 MiB once Lc0 small had loaded). Reckless and Viridithas each added a live worker and about 0.2 MiB. Their wasm did not show up as a new large resident at the sample.

Analysis is different. Opening the page and reaching the shell sample already holds a 247.0 MiB search worker. Thirty seconds later, with Lc0 small still in a search (about 4.6k–5.4k of 100000 visits, ~112 nodes/s), that worker is 129–133 MiB and the page total is 132.4 MiB. Stop releases about 17 MiB (to 115.3) and the worker stays. The 30 s search is not the memory cost. The loaded worker is. The flag for "Stop after Lc0" does not fire, because the running sample is below the shell sample, not above it.

The Analysis CPU tour does stay. Reckless adds 344.8 MiB. Viridithas, Berserk, PlentyChess, and Stormphrax add about 0–3 MiB on top of that. Switching the row back to Lc0 drops 3.6 MiB and one worker (3 to 2) and leaves 460.1 MiB, 345 MiB above the post-Stop Lc0 sample, on 3/3 runs. In the Reckless sample the new bytes are attributed to the dedicated worker whose URL is `/stockfish/stockfish-19-lite.js`, while the status line says Reckless finished. Later families do not add another pile of that size.

Arena's shell is 4.0 MiB. The Lite vs Reckless match is 133.4 MiB and 2 workers. Stop leaves 133.8 MiB and both workers, on 3/3 runs. That comparison is Stop, not leaving the page. The harness does not navigate away from Arena.

The Maia3 human-move panel is the case that does release. The grid painted (ratings 1100–2200). The sample is 218.7 MiB and 2 workers. In-app navigation home drops it to 4.4 MiB and 0 workers, on 0/3 retention flags. Maia on Play does not get the same treatment: the Play Maia3 worker is still in the 1255 MiB leftover.

`performance.memory` is the wrong tool for this question. Use the user-agent measurement, and expect the 60 s gap once a page has been sampled a few times.

## Running the same checkpoints on a phone or a small desktop

Do this on a machine you can watch. The harness above was not run on a phone.

Build the client (`BUILD_SCOPE=product npm run build:client`) and serve it with the isolated static server so the document is cross-origin isolated. `window.crossOriginIsolated` must be true. Otherwise `measureUserAgentSpecificMemory()` throws a SecurityError and the sample is only the JS heap. Chrome on Android supports the API on an isolated page. A normal `vite dev` server is the wrong target: wasm assets 404 there, and this server caches built assets for an hour.

Android, over USB:

1. On the phone, enable Developer options and USB debugging. Plug it in. `adb devices` should list it.
2. Reverse the desktop server onto the phone's loopback: `adb reverse tcp:5186 tcp:5186` (use the port you actually served). Loopback is a secure context, which this API requires. A LAN address such as `http://192.168.x.x` is not a secure context, so COOP/COEP there still leave `crossOriginIsolated` false and the probe reports "not cross-origin isolated". An https origin you already trust is the other way to get a secure context.
3. On the desktop, open `chrome://inspect/#devices` and confirm the phone's tab. If you need the raw inspect socket, `adb forward tcp:9222 localabstract:chrome_devtools_remote` and attach a desktop DevTools client to `localhost:9222`.
4. On the phone, open `http://127.0.0.1:5186/?memprobe=1` once. The root layout loads the probe only when that query is `1`, and the layout survives client navigations, so the overlay stays as you move between Home, Play, Analysis, and Arena. A full reload without the query removes it.
5. At each checkpoint below, scroll the black "Sample memory" button into view, or in the inspect console run `await window.__memprobe.sample('label')`. The call waits on GC. Write down `userAgentSpecificBytes`, `usedJSHeapSize`, `hookedWorkersLive` or `workerScopes`, `measureLatencyMs`, and the error string if there is one. Also write down Chrome's own site memory or `adb shell dumpsys meminfo` for the Chrome process if you want OS RSS next to the page number.

Checkpoints, same order as the table: landing; Play with the default engine after it has played a move; Play after each CPU engine and then Lc0 small; leave Play back to Home; Analysis with Lc0 left running about 30 s, then Stop; Analysis after each CPU engine, then back to Lc0; Arena, a short CPU match, then Stop; Analysis, open Human moves, Load Maia3, wait until the rating rows appear; leave that page back to Home. The question to answer on the device is which of those last "leave" samples are still near the peak.

A low-memory desktop is the same page and the same `?memprobe=1` samples, in a fresh Chrome profile (`--user-data-dir` pointing at an empty directory) so an old service worker and an old engine cache are not in the number. Record Chrome's Task Manager (Shift+Esc) process footprint beside the probe. A 4 GiB machine or a profile with a tight OS limit is the useful comparison against the 24 GiB table above. Headless shell will not produce the user-agent number; use ordinary Chrome or Chrome for Testing.

The overlay is `src/lib/memprobe.ts` (80 lines). It mounts only from the `?memprobe=1` dynamic import in the root layout. With the query absent, the module is not imported.
