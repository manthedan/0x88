# Low-end memory harness — October 2026 rerun

**This is an Apple M4 desktop with 10 cores and 24 GiB of RAM, NOT a phone.** No phone or deliberately memory-constrained device was measured. This is a desktop baseline, not a mobile survival, OS-RSS, or crash-limit result. Product behavior and the harness are unchanged by this rerun.

## Provenance and protocol

The committed table and all twelve phase artifacts were regenerated on **13 October 2026**, against a product client built from **`4c8193f`**. That includes the family-settling/visit-timer fixes, engine failures captured during the memory call (including Maia), and Stop timeouts recorded as unsuccessful checkpoints. The previous table, preserved in `git show 4c8193f:docs/lowend_memory_harness_2026-10.json`, predates those harness fixes; it is superseded, not pooled with this sample set.

Host: Apple M4, 10 cores, 25,769,803,776 bytes physical RAM (24 GiB), macOS/darwin arm64, Node v24.16.0, Playwright Chromium / Chrome for Testing **151.0.7922.34**. The old run used Node v25.9.0 and the same browser version. Viewport: 1280×800; headless full Chromium, `--enable-precise-memory-info`; fresh browser/context per phase, service workers allowed. Do not use Playwright's default headless shell: it can expose the memory API but reject the call.

Each sample calls `performance.measureUserAgentSpecificMemory()`, then reads the JS heap, worker hook, and page status. These are **Chrome-attributed user-agent bytes**, not physical resident memory/RSS or WASM reservation sizes. JS heap excludes worker WASM and can include transient allocations: the default Play heap ranged 3.6–30.5 MiB while UA bytes barely moved. Worker count is main-thread Worker/SharedWorker objects not yet terminated; nested workers are invisible.

Memory calls wait for GC. Many persistent-worker CPU checkpoints took about 60 s; other calls completed sooner. The recorded status and bytes are at the **end** of that wait, not exactly at the trigger. The Analysis timer starts on Lc0 visit progress, runs for 30 s at a 100000-visit target, then calls the memory API; it is not a completed 100000-visit search. A Stop timeout would have `ok: false` and be excluded from retention. A failure that arrives during the wait would remain next to its byte count. This suite had **81/81 successful byte samples, no engine failures, no memory-API errors, and no page errors**; every checkpoint has 3/3 samples.

Three repeats of Play, Analysis, Arena, and Maia, ordered Analysis → Maia → Play → Arena within each repeat. Every phase invocation held `SP/benchlock` separately, with `--runs 1`; no build overlapped our lock. Total measurement wall time inside those twelve chunks: **44.6 min**, excluding lock waits and separate browser checks. Largest chunk: 8.3 min, below the 20-minute limit.

| Phase | Median wall time | Min–max |
|---|---:|---:|
| play | 317.9 s | 308.9–328.6 s |
| analysis | 498.4 s | 495.6–499.7 s |
| arena | 40.3 s | 33.3–56.8 s |
| maia | 32.2 s | 31.6–33.1 s |

### Machine noise

One-minute load at phase start: **2.22–9.31**; considering both start and end: **2.22–10.47**. The old start range was 0.88–3.69, so this was a substantially busier desktop. A process/VM snapshot during the run showed a busy Brave renderer, WindowServer, other desktop apps, and compressed/swap-backed system memory. Other agents interleaved timing chunks via the mutex; their loads were not deliberately overlapped. Free memory reported at phase end was 0.54–2.10 GiB (free, not a macOS “available RAM” estimate).

Stable large Play totals have sub-0.1 MiB spreads; Analysis CPU rows have sub-0.5 MiB spreads. In contrast, the first Analysis shell sample varied by **220.0 MiB**, and running Lc0 by **7.4 MiB**. Do not infer startup/search speedups from these wall times or treat a first-shell baseline as steady-state memory. This is a historical old/new comparison, **not an interleaved A/B experiment isolating the last two fixes**.

## Updated byte table

All byte columns below are **MiB** (bytes / 1048576), rounded to one decimal. UA min–max is the spread over three runs. Delta is the median of each run's change from its previous checkpoint, not necessarily the difference between the two displayed medians. Unrounded bytes, heap/worker/latency spreads, breakdowns, status, and errors are in `docs/lowend_memory_harness_2026-10.json` and `docs/lowend_memory_harness_2026-10_runs/`.

| Checkpoint | Resident | UA median | UA min–max | Delta | JS heap | Workers | Measure latency |
|---|---|---:|---:|---:|---:|---:|---:|
| landing |  | 1.5 | 1.5–1.5 | — | 1.5 | 0 | 14.0 s |
| play-default-after-first-move | maia3 | 116.6 | 116.6–116.6 | +115.1 | 30.5 | 1 | 12.0 s |
| play-cpu-sf-lite | sf-lite | 245.5 | 245.5–245.5 | +128.9 | 3.7 | 2 | 15.6 s |
| play-cpu-sf-full | sf-full | 565.0 | 565.0–565.1 | +319.5 | 3.8 | 3 | 16.0 s |
| play-cpu-reckless | reckless | 436.5 | 436.5–436.5 | -128.6 | 3.9 | 3 | 60.0 s |
| play-cpu-viridithas | viridithas | 117.2 | 117.2–117.2 | -319.3 | 4.0 | 3 | 60.0 s |
| play-cpu-berserk | berserk | 397.9 | 397.9–397.9 | +280.7 | 4.1 | 3 | 60.0 s |
| play-cpu-plentychess | plentychess | 705.1 | 705.1–705.1 | +307.2 | 4.0 | 3 | 18.0 s |
| play-cpu-stormphrax | stormphrax | 990.5 | 990.5–990.5 | +285.4 | 4.1 | 3 | 14.0 s |
| play-neural-lc0-small | lc0-small | 526.6 | 526.6–526.6 | -464.0 | 6.7 | 2 | 16.0 s |
| landing-after-play |  | 526.4 | 526.4–526.4 | -0.1 | 6.7 | 2 | 16.0 s |
| analysis-shell |  | 27.0 | 27.0–247.0 | — | 3.3 | 1 | 14.0 s |
| analysis-lc0-30s | lc0 | 130.7 | 124.0–131.4 | +97.0 | 3.4 | 1 | 14.0 s |
| analysis-lc0-stopped | lc0 | 115.3 | 115.0–115.3 | -15.5 | 3.4 | 1 | 11.3 s |
| analysis-cpu-sf | sf | 459.9 | 459.8–459.9 | +344.6 | 5.3 | 2 | 60.0 s |
| analysis-cpu-reckless | reckless | 460.0 | 459.7–460.0 | +0.1 | 3.7 | 3 | 60.0 s |
| analysis-cpu-viridithas | viridithas | 460.0 | 459.9–460.1 | +0.1 | 3.9 | 3 | 60.0 s |
| analysis-cpu-berserk | berserk | 740.6 | 740.4–740.7 | +280.6 | 3.9 | 3 | 60.0 s |
| analysis-cpu-plentychess | plentychess | 818.6 | 818.4–818.7 | +78.0 | 3.9 | 3 | 60.0 s |
| analysis-cpu-stormphrax | stormphrax | 1026.1 | 1025.9–1026.1 | +207.5 | 4.0 | 3 | 60.0 s |
| analysis-after-cpu | lc0 | 460.1 | 459.9–460.2 | -566.0 | 4.0 | 2 | 60.0 s |
| arena-shell |  | 4.0 | 4.0–4.0 | — | 3.4 | 0 | 10.0 s |
| arena-cpu-match | sf+reckless | 4.7 | 4.6–4.7 | +0.6 | 4.8 | 0 | 18.9 s |
| arena-cpu-stopped |  | 4.8 | 4.7–5.0 | +0.2 | 4.2 | 0 | 6.8 s |
| maia-before |  | 247.0 | 247.0–247.0 | — | 3.3 | 1 | 10.0 s |
| maia-panel | maia3 | 218.7 | 218.7–218.8 | -28.3 | 19.5 | 2 | 14.0 s |
| landing-after-maia |  | 4.4 | 4.4–4.5 | -214.3 | 3.9 | 0 | 4.7 s |

## What changed versus the old table

| Checkpoint | Old UA median | New UA median | New − old |
|---|---:|---:|---:|
| play-cpu-reckless | 565.2 | 436.5 | -128.7 |
| play-cpu-viridithas | 565.4 | 117.2 | -448.2 |
| play-cpu-berserk | 846.1 | 397.9 | -448.2 |
| play-cpu-plentychess | 1153.4 | 705.1 | -448.3 |
| play-cpu-stormphrax | 1719.4 | 990.5 | -728.9 |
| play-neural-lc0-small | 1255.4 | 526.6 | -728.9 |
| landing-after-play | 1255.3 | 526.4 | -728.8 |
| analysis-shell | 247.0 | 27.0 | -220.0 |
| analysis-lc0-30s | 132.4 | 130.7 | -1.6 |
| analysis-lc0-stopped | 115.3 | 115.3 | -0.0 |
| analysis-cpu-sf | 115.2 | 459.9 | +344.7 |
| analysis-cpu-berserk | 460.4 | 740.6 | +280.2 |
| analysis-cpu-plentychess | 460.6 | 818.6 | +358.1 |
| analysis-cpu-stormphrax | 463.6 | 1026.1 | +562.4 |
| analysis-after-cpu | 460.1 | 460.1 | +0.0 |
| arena-cpu-match | 133.4 | 4.7 | -128.8 |
| arena-cpu-stopped | 133.8 | 4.8 | -128.9 |
| maia-before | 3.8 | 247.0 | +243.2 |
| maia-panel | 218.7 | 218.7 | -0.0 |
| landing-after-maia | 4.4 | 4.4 | -0.0 |

- Landing, default Play Maia3, Play Stockfish Lite/full, the loaded Maia panel, and Analysis after Stop/after the CPU tour remain close to the old byte levels. Lower Play totals after the first CPU switches are consistent with the **already-merged inactive-CPU cap** in `src/lc0/playBrowser.ts` / `src/lc0/cpuEngineResidency.ts`: after Reckless, Lite is evicted; after Viridithas, full is evicted. Stormphrax switch-away disposal is still present. No product eviction policy was changed here. After Lc0 and after leaving Play, 2 workers remain instead of 7; about 729 MiB less is held. Run 1's landing-after-Play breakdown is primarily the retained PlentyChess worker (~307 MiB), Maia worker (~112 MiB), and window (~103 MiB). This is still substantial retention.
- Analysis now records Stockfish's ~345 MiB step **on the Stockfish row**, not one row late on Reckless. Run 1 attributes it to `/stockfish/stockfish-19-lite.js`, with `Analysis complete · SF Lite`. Family-settling waits are now part of the measured protocol. Berserk, PlentyChess, and Stormphrax samples show their actual CPU residents above the retained Stockfish/Lc0 base, rather than staying near 460 MiB. The largest UA median is Analysis Stormphrax (1026.1 MiB), followed by Play Stormphrax (990.5 MiB).
- The first Analysis worker footprint is not stable: shell runs 1/2 were about 27 MiB, run 3 about 247 MiB. Their leading search-worker breakdowns were ~23 and ~243 MiB respectively, even though the sampled UI was Ready in all cases. This is consistent with startup/GC attribution timing, not a failed Stop. It prevents a clean causal “220 MiB saved” conclusion for the shell median.
- **Arena is post-completion in this rerun.** In all three `arena-cpu-match` readings, the status at memory-call completion is `Match done (1 game).`; 2 workers were created and 0 remain. Stop is clicked only after that reading, so the stopped row does **not** measure an active-match cancellation. The old 133.4/133.8 MiB readings had 2 workers alive. The harness explicitly allows a move **or a result** and keeps the original one-game, 200 ms query; GC timing and completion/disposal differ. The new low numbers are not proof of improved active-match Stop retention, nor of peak Arena memory. An active-match Stop/peak experiment remains separate work.

## Retention and the two previously “not repeated” scenarios

A flag means peak gain over baseline exceeds 8 MiB, and the later sample keeps more than both 8 MiB and half that gain. Exactly meeting a threshold does not flag; repeated means a strict majority. **0/3 flags does not mean 0/3 successful runs.** All comparisons below have 3/3 comparable byte samples. A retained worker/model is not by itself a leak verdict, and a quiet flag does not mean all bytes were released.

| Comparison | Old flags | New flags | Comparable runs | Peak gain | Still held |
|---|---:|---:|---:|---:|---:|
| play-cpu-tour | 3/3 | 0/3 | 3/3 | +873.9 | +409.9 |
| play-left-surface | 3/3 | 3/3 | 3/3 | +989.1 | +525.0 |
| analysis-lc0-stop | 0/3 | 2/3 | 3/3 | +97.0 | +88.0 |
| analysis-cpu-tour | 3/3 | 0/3 | 3/3 | +910.8 | +344.9 |
| arena-stop | 3/3 | 0/3 | 3/3 | +0.6 | +0.8 |
| maia-left-surface | 0/3 | 0/3 | 3/3 | -28.3 | -242.6 |

### Analysis Lc0 Stop: succeeded 3/3

All timed runs reached visits and were still running when the memory call began; Stop then settled successfully, and every CPU tour subsequently completed. No Stop-timeout or engine-failure flag appeared. The running median is 130.7 MiB; stopped is 115.3 MiB, with 1 worker still present. The old 0/3 retention flags were not failed runs: its shell baseline was ~247 MiB, above the running and stopped totals. Now the lower shell baseline on runs 1/2 produces 2/3 flags; the ~247 MiB baseline on run 3 still does not. The flag change is baseline-sensitive, **not evidence that Stop became worse**. Stop releases a median 15.5 MiB from the running sample while the loaded search worker remains.

A separate built-app agent-browser check verified `searchActive: true → false`, then `analyzing: false`, disabled Stop, unchanged visit-progress counters and FEN. The last visit text can remain visible after Stop; that is not continued search. No harness fix or product change was needed to make this scenario run.

### Maia leave: succeeded 3/3, no retention flag

The model loaded and all six rating rows (1100–2200) painted on every run, with empty panel-failure status before and after the memory wait. Navigation home reduced 218.7 MiB / 2 workers to 4.4 MiB / 0 workers. The direct panel-to-home drop is about 214 MiB. The old 0/3 flag likewise meant “not retained,” not a failure to load Maia.

The current Maia-before sample also contains the initial Lc0 worker and is 247.0 MiB, higher than the loaded-panel total. Its negative peak gain therefore cannot estimate the isolated Maia allocation. Prefer the direct panel-to-home drop and worker termination as the release evidence. The shell/GC effect is documented rather than changing the baseline or forcing extra GC to obtain a particular flag.

### Other comparisons

Play-left-surface remains 3/3, about 525.0 MiB above landing. Play-cpu-tour is now unflagged despite about 409.9 MiB held: the cap lowered its retained fraction below 50%. Analysis-cpu-tour is also unflagged despite about 344.9 MiB held: its newly measured CPU peak is much higher, so the retained fraction is below 50%. Neither is a “no retention” conclusion. Arena-stop is unflagged only for the completed-match case described above.

## Reproducing the desktop suite

Build before acquiring the lock, serve only the built product, and provide all engine/model assets:

```sh
BUILD_SCOPE=product npm run build:client
PORT=5189 npm run web:isolated:static
# In another terminal, one phase per lock acquisition:
SP/benchlock npm run mem:harness -- --phase analysis --runs 1 --run-index 1 \
  --base-url http://127.0.0.1:5189 --out /tmp/mem-rerun/run-1-analysis.json
# Repeat phases analysis, maia, play, arena for run indexes 1, 2, 3.
npm run mem:harness -- --combine /tmp/mem-rerun \
  --summary-json /tmp/mem-summary.json --summary-md /tmp/mem-summary.md
```

`SP/benchlock` denotes the shared scratchpad lock executable; use its actual absolute path. Do not combine new and old phase files in one directory. Each phase must complete before the next measurement starts. The JSON is regenerated from raw files, not hand-edited.

This worktree needed gitignored Maia ONNX files and Reckless WASM/NNUE blobs linked into **dist-client only** from the shared model cache / existing local engine assets. Those symlinks are not committed. In particular, check `/models/maia3/maia3_simplified.qdq8.onnx`, `/models/maia3/maia3_simplified.onnx`, and the selected `/reckless/*.wasm` files. A checkout missing them gets 404s and is not comparable to this table. Force a cache reload after rebuilding: this server caches JS/WASM for an hour.

## Verification and limits

`npm test`: typecheck 457 files, 0 errors/warnings; 1069 tests, 1066 passed, 3 environment-dependent skips, 0 failures. Focused memory-harness tests: 20/20. No `src/`, engine, runtime, harness, dependency, or build-config changes were made; product equivalence here is the unchanged built product revision, not a newly optimized engine result.

Separate `?memprobe=1` browser check: cross-origin isolated, UA 1.46 MiB, no API error, 0 worker scopes; overlay screenshot also showed a successful UA sample. This check used agent-browser's Chrome 149 and is **not pooled into the Chromium 151 table**. Screenshots/evidence are in `SP/memtable-rerun/` (`shots/memprobe-sample.png`, `shots/analysis-stopped.png`, `shots/maia-panel-run1.png` through `run3.png`, and the overlay/Stop JSON). The probe overlay persists across in-app navigation and is absent on a fresh reload without `?memprobe=1`.

No real phone or low-memory-device result, peak allocation trace, full 100000-visit search, or active Arena Stop result is claimed. Do not use the table to set a mobile memory budget or identify a crash threshold. Nothing was pushed, merged, published, or deployed.

## Running the same checkpoints on a phone or a small desktop

Do this on a machine you can watch. The harness above was not run on a phone.

Build the client (`BUILD_SCOPE=product npm run build:client`) and serve it with the isolated static server so the document is cross-origin isolated. `window.crossOriginIsolated` must be true. Otherwise `measureUserAgentSpecificMemory()` throws a SecurityError and the sample is only the JS heap. Chrome on Android supports the API on an isolated page. A normal `vite dev` server is the wrong target: wasm assets 404 there, and this server caches built assets for an hour.

Android, over USB:

1. On the phone, enable Developer options and USB debugging. Plug it in. `adb devices` should list it.
2. Reverse the desktop server onto the phone's loopback: `adb reverse tcp:5189 tcp:5189` (use the port you actually served). Loopback is a secure context, which this API requires. A LAN address such as `http://192.168.x.x` is not a secure context, so COOP/COEP there still leave `crossOriginIsolated` false and the probe reports "not cross-origin isolated". An https origin you already trust is the other way to get a secure context.
3. On the desktop, open `chrome://inspect/#devices` and confirm the phone's tab. If you need the raw inspect socket, `adb forward tcp:9222 localabstract:chrome_devtools_remote` and attach a desktop DevTools client to `localhost:9222`.
4. On the phone, open `http://127.0.0.1:5189/?memprobe=1` once. The root layout loads the probe only when that query is `1`, and the layout survives client navigations, so the overlay stays as you move between Home, Play, Analysis, and Arena. A full reload without the query removes it.
5. At each checkpoint below, scroll the black "Sample memory" button into view, or in the inspect console run `await window.__memprobe.sample('label')`. The call waits on GC. Write down `userAgentSpecificBytes`, `usedJSHeapSize`, `hookedWorkersLive` or `workerScopes`, `measureLatencyMs`, and the error string if there is one. Also write down Chrome's own site memory or `adb shell dumpsys meminfo` for the Chrome process if you want OS RSS next to the page number.

iOS, over USB to a Mac:

1. Enable **Web Inspector** in the iPhone/iPad's Safari settings under **Advanced** (on versions with an Apps settings group, open Apps → Safari first). On the Mac, enable Safari Settings → Advanced → **Show features for web developers**. Connect the device, unlock it, and trust the Mac if prompted.
2. Open the built app at a **trusted HTTPS origin reachable from the device**, with the same COOP/COEP headers and `?memprobe=1`. The iPhone's `127.0.0.1` is not the Mac; Android's `adb reverse` recipe does not apply to iOS. A local HTTPS setup is sufficient; publishing/deploying is not required.
3. Choose **Develop → [device] → [page]** in desktop Safari. Use the overlay or `await window.__memprobe.sample('label')` in the inspected page's console. Check `isSecureContext`, `crossOriginIsolated`, and `typeof performance.measureUserAgentSpecificMemory` first.
4. Keep any API-unavailable/error fields in the record. If Safari cannot provide user-agent bytes or `performance.memory`, use Web Inspector's memory tools and device/process diagnostics as separately labelled evidence, not as equivalent UA bytes. Record tab reloads, crashes, and whether the next move succeeds; do not substitute an iOS Simulator for a real low-memory phone.

Remote-inspection references: [Chrome Android debugging](https://developer.chrome.com/docs/devtools/remote-debugging) and [WebKit: enabling Web Inspector](https://webkit.org/web-inspector/enabling-web-inspector/).

Checkpoints, same order as the table: landing; Play with the default engine after it has played a move; Play after each CPU engine and then Lc0 small; leave Play back to Home; Analysis with Lc0 left running about 30 s, then Stop; Analysis after each CPU engine, then back to Lc0; Arena, a short CPU match, then Stop; Analysis, open Human moves, Load Maia3, wait until the rating rows appear; leave that page back to Home. The question to answer on the device is which of those last "leave" samples are still near the peak.

A low-memory desktop is the same page and the same `?memprobe=1` samples, in a fresh Chrome profile (`--user-data-dir` pointing at an empty directory) so an old service worker and an old engine cache are not in the number. Record Chrome's Task Manager (Shift+Esc) process footprint beside the probe. A real 4 GiB machine or a browser process under a verified OS memory limit is the useful comparison against the 24 GiB table above. Changing the viewport, overriding `navigator.deviceMemory`, or throttling CPU/network in DevTools does not impose a RAM limit. Record physical RAM, browser/OS versions, background apps, temperature/power state, API errors, and crashes; repeat the sequence three times, using fresh sessions between surface tours but in-app navigation within each tour. Headless shell will not produce the user-agent number; use ordinary Chrome or Chrome for Testing.

The overlay is `src/lib/memprobe.ts` (80 lines). It mounts only from the `?memprobe=1` dynamic import in the root layout. With the query absent, the module is not imported.
