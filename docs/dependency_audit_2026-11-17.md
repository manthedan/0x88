# Dependency audit — 2026-11-17

## Scope and outcome

Audited every original `dependencies` and `devDependencies` entry at production
`c808bad0e2955d3841daff5372ee0c0598ec4d56`, plus third-party assets under `public/`.
Work was confined to `chore/deps-upgrade`; `node_modules` is a real, worktree-local
installation, not the shared main-checkout directory. No deployment or upload.

- Upgraded SvelteKit to **3.3.0**, adapter-static **4.0.1**, explicit peer
  vite-plugin-svelte **7.3.1**, Svelte **5.57.3**, svelte-check **4.7.8**,
  TypeScript **6.0.3**, Vite **8.3.8**, Playwright **1.65.1**, Wrangler **4.164.0**,
  and Valibot **1.5.0**. smol-toml already resolves to current **1.9.1**.
- Removed **@tanstack/query-core**, **@tanstack/store**, and **binaryen** only.
  No replacements or consolidations were implemented.
- Kept ONNX Runtime at **1.27.0** and tightened its former caret to an exact pin;
  its checked-in/staged glue, WASM and measured kernel contracts are unchanged.
- **Audit is not green:** one unpatched advisory, GHSA-fc5q-39xc-vjpf, propagates
  into eight high-severity package records. No ignore applied. Main has since made
  the CI audit job report-only (owner decision 2026-11-17, `continue-on-error`), so
  this residual shows in CI without failing it.
- Clean `npm ci`, full tests, Biome, product build and interactive static-browser
  verification pass. Detailed verification and owner decisions follow.

The actual starting lockfile was slightly newer than the brief: Kit 2.70.4,
adapter 3.0.11, Vite 8.3.7, plugin 7.1.2, TypeScript 5.9.3, Wrangler 4.159.0.
These resolved versions, not the package.json lower bounds, are the baseline.

## Migration applied

Read the complete [official Kit 3 migration guide](https://svelte.dev/docs/kit/migrating-to-sveltekit-3).
This static, universal-prerendered app has no remote functions, server actions,
server hooks, parameter matchers or custom adapters requiring those migrations.

1. Replaced unsupported `svelte.config.js` with shared `kit.config.js`, passed
   as flat options to `sveltekit(kitConfig)` in `vite.config.ts:27`. Retained the
   exact CSP source lists, static assets/output environment overrides and hash
   mode. Deployment CSP checks, fixtures and release input hashing use the same
   configuration; the security gate was not removed or loosened.
2. Replaced `$lib` imports with explicit-extension `#lib/*` package imports;
   replaced `$app/environment` with `$app/env` in the PWA registration helper.
3. Replaced `$app/stores` with `$app/state`; SiteHeader uses runes for pathname
   reactivity, props and local state. Its active navigation changes correctly
   across client-side Home → Docs → Play navigation.
4. Extended `$app/tsconfig`, included `$app/types` and generated ambient/route
   declarations. Preserved the narrow experimental GPU interfaces using
   `Omit<Navigator, 'gpu'>` rather than intersecting them with TS6's newly typed
   native GPU member; no engine runtime logic changed.
5. Removed obsolete global `cookie: 0.7.2` override: Kit 3 requires cookie v2
   (resolves 2.0.2), while Wrangler's separate cookie resolves 1.1.1. Retained
   the existing undici 7.30.2 security override.
6. Adapted the post-build runtime guard: Kit 3 calls `kit.init(payload)` instead
   of embedding its global id in every client chunk. Verify each HTML payload
   against `_app/version.json`, initialization and the start/app entries plus
   every module they import statically or by literal dynamic import (Kit 3's
   start entry statically imports `payload.*.js`; a missing module would break
   hydration), confined to `_app/immutable`. Added seven tests for valid/nested pages, stale versions, regexp-safe
   comparison, missing initialization and missing entry files; retained Kit 2
   mismatch tests. Deployment still runs the guard.
7. Fixed import ordering/formatting with the same `biome check` used in CI.
   TS6 also exposed missing `arrowPreview` and `lc0ServedModelUrl` bindings in
   source-slicing test harnesses; added those bindings, without modifying the
   production behavior or weakening assertions.

## Security audit: exact residual and owner option

`npm audit --audit-level=high` exits **1**. `npm audit --json` reports **8 high,
0 moderate, 0 low, 0 critical**. All eight records originate from:

- [GHSA-fc5q-39xc-vjpf](https://github.com/advisories/ghsa-fc5q-39xc-vjpf):
  picomatch ReDoS through single-branch repeated extglobs, affected **<=4.0.7**.
- `npm view picomatch version` returns **4.0.7**, the latest published version.
  There is **no published patched version** on the audit date. An override to
  4.0.7, or to an older 2.x/3.x version, does not fix this advisory.
- Installed paths: `vite@8.3.8 → picomatch@4.0.7`,
  `vite → tinyglobby@0.2.17 → picomatch / fdir@6.5.0 → picomatch`, and
  `svelte-check@4.7.8 → fdir → picomatch` (deduped).
- Affected aggregate package records: **picomatch, fdir, tinyglobby, vite,
  svelte-check, @sveltejs/vite-plugin-svelte, @sveltejs/kit,
  @sveltejs/adapter-static**. The latter Kit/adapter records are inherited Vite
  findings, not a recurrence of the four direct Kit advisories.
- GHSA-pfhg-q3qv-9q6r, GHSA-27mw-82g8-phjj, GHSA-3r8g-jcqq-3p5j and
  GHSA-6cf5-gc4q-fm3x are absent after the Kit upgrade.
- npm suggests a forced Vite 6 downgrade; **rejected**. Vite 8 is retained.

**Owner decision, not applied:** the CI audit job is now report-only on main, so
no exception is needed to keep CI green. When a patched upstream release exists,
update the lockfile (use a narrowly scoped
picomatch override only if the fixed version satisfies the actual consumer API).
If the audit job is ever made blocking again before then, approve only **GHSA-fc5q-39xc-vjpf**,
with an expiry of **2026-12-01**, named owner and weekly upstream checks. Any
future filter must inspect the advisory graph: remove only this advisory leaf,
retain inherited records that have another active cause, fail for every other
high/critical advisory and fail when the exception expires. Preserve the full
unfiltered JSON as evidence. Do not whitelist these eight package names.

Justification for considering that temporary exception: globs are trusted
repository/build-tool configuration, not visitor input, and adapter-static
ships no Node request server. This limits production exposure, not the existence
of the defect; untrusted build input can still hang a tool. No blanket clean bill
of health is implied. `npm audit` does not assess vendored WASM, model parsers,
remote runtime assets or licensing.

## Direct dependency inventory

Sizes below are decimal MB of the registry's **unpacked package alone** from
`npm view name@version dist.unpackedSize`; not compressed browser transfer sizes,
transitive installs, browser downloads or tree-shaken bundle attribution. Usage
references are representative, not exhaustive. “No direct finding” means the
current npm audit snapshot only, not a comprehensive security review.

### Runtime dependencies

| Dependency (resolved) | Usage evidence | Size / maintenance / security | Needed? Verdict |
| --- | --- | --- | --- |
| `@bjorn3/browser_wasi_shim` 0.4.2 | `src/lc0/recklessWasiWorker.ts:1`, `src/lc0/recklessBrowserApiWorker.ts:1`, `scripts/viridithas_wasi_smoke.mjs:3` | 0.115 MB; small specialized WASI shim, current registry release; no direct finding. Runs native-engine ports in workers; sandbox is not a substitute for asset integrity. | **Keep.** Required imports, not interchangeable with TVM's embedded WASI ABI without testing. |
| `@tanstack/query-core` 5.101.0 | No import, dynamic import, script, config or build reference found; original `package.json:206` and lock declarations only. | 2.313 MB; maintained upstream, but contributes install surface with no consumer. Never bundled in baseline. | **Remove — done.** Existing explicit model caches do not use it. Do not replace those caches as part of removal. |
| `@tanstack/store` 0.11.0 | No import, script or config reference; original `package.json:207` and lock declarations only. `src/lib/stores/theme.ts:1` uses `svelte/store`, not TanStack. | 0.131 MB; maintained upstream, unused here. Never bundled in baseline. | **Remove — done.** No store consolidation required. |
| `chessground` 9.2.1 | `src/lc0/playBrowser.ts:3`, `analysisBrowser.ts:1`, `arenaBrowser.ts:1`, `policyOnlyBrowser.ts:1`; `vite.config.ts:18` patches SVG arrows; `src/app.html:9-11` loads vendored CSS. | 0.371 MB; **npm explicitly marks this unscoped package unsupported/deprecated**. No direct npm vulnerability finding. Patched arrow implementation and GPL assets increase migration risk. | **Keep for this change; propose replace** with maintained scoped `@lichess-org/chessground` (registry 10.5.0, 0.461 MB), see proposals. |
| `onnxruntime-web` 1.27.0 | `src/nn/ortRuntime.ts:3-5`, `src/nn/ortGpuIo.ts:24`; `public/ort` links to the local package's dist; `vite.config.ts:34` selects external-WASM export. | 137.517 MB package; active upstream, behind 1.31.1 deliberately. Large JS/WASM and binary/model-input attack surface; current npm snapshot has no direct finding. Staged runtime/glue hashes and kernel measurements matter more than semver alone. | **Keep, exact pin.** No update, runtime artifact change or consolidation authorized here. |
| `stockfish` 19.0.1 | Six `public/stockfish` JS/WASM links target `node_modules/stockfish/bin`; `src/lc0/stockfishEngine.ts:53-61`; `scripts/write_engine_source_archive.mjs:218-223`. | 205.117 MB; current pinned upstream browser engine; GPL-3.0 corresponding-source obligation. No direct npm finding; native WASM/glue remain outside audit's meaningful coverage. | **Keep.** Public symlinks and source packaging prove it is used despite no ordinary package import. |
| `valibot` 1.5.0 | `src/lc0/modelCache.ts:1`, `src/lc0/gameImportSchema.ts:8`, `src/nn/tvmjsManifestSchema.ts:10`; `tests/fetch_boundary_schemas.test.mjs:3`. | 1.865 MB; actively maintained/current; no direct finding. Tree-shakeable boundary validation protects downloaded metadata and imports. | **Keep.** Replacing with handwritten validation would remove useful safeguards. |

### Development dependencies

| Dependency (resolved) | Usage evidence | Size / maintenance / security | Needed? Verdict |
| --- | --- | --- | --- |
| `@biomejs/biome` 2.5.6 | `package.json:189-191` format/lint/check scripts, `biome.json:1`, `.github/workflows/ci.yml:37` | 0.737 MB wrapper, plus platform binary; maintained, 2.6.8 available; no direct finding. | **Keep, pin 2.5.6.** 2.6 begins formatting existing Markdown/YAML across the repo; tested upgrade produced widespread unrelated document diffs. Defer that formatting adoption rather than weaken configuration or reformat the entire repository here. |
| `@sveltejs/adapter-static` 4.0.1 | `kit.config.js:1,33`; output is `dist-client`. | 0.012 MB; current first-party adapter; only inherited picomatch high record remains. | **Keep.** Static prerender deployment contract. |
| `@sveltejs/kit` 3.3.0 | `vite.config.ts:1,27`, `package.json:15,187`, `src/routes/`; client navigation APIs in workspace pages. | 1.564 MB; current first-party framework; direct Kit advisories resolved; inherited Vite/picomatch high remains. | **Keep.** Replacing routing/SSR is a separate high-risk application rewrite. |
| `@sveltejs/vite-plugin-svelte` 7.3.1 (new explicit peer) | Consumed through `sveltekit()` in `vite.config.ts:27`; Kit peer requires 7.x. | 0.148 MB; current first-party compiler integration; inherited Vite/picomatch high remains. | **Keep.** Declare matching peer explicitly, rather than rely on implicit npm peer installation. |
| `agent-browser` 0.31.2 | `scripts/batching_strength_match.mjs:22` explicitly names local binary; many `lc0_browser_*` CLI defaults; `scripts/lib/process.mjs:62-89` executes it. | 84.951 MB; upstream active at 0.43.0, local CLI intentionally behind. No direct audit finding; browser automation has filesystem/network authority. | **Keep; propose consolidation** with Playwright for overlapping harnesses, not removal. New 0.x minor is potentially breaking, requiring output/session lifecycle qualification. Native Pi browser verification uses the harness-managed tool, not proof that this old project CLI is compatible with 0.43. |
| `binaryen` 130.0.0 | No live script/config/import/binary reference; original declaration `package.json:195`. `docs/reckless_wasm_opt_experiment.md:6,22` records a past negative `wasm-opt` experiment, not a current command. | 96.485 MB; maintained upstream (132 available), but unused locally. | **Remove — done.** Restore as an explicit tool dependency only if a new optimization experiment needs it. |
| `playwright` 1.65.1 | `scripts/dist_browser_smoke.mjs:7`, `scripts/production_browser_smoke.mjs:6`, `scripts/stockfish_upgrade_browser.mjs:6`; CI browser smoke. | 5.064 MB wrapper, plus playwright-core/browser binaries; maintained/current, no direct finding. Keep browser binaries patched independently. | **Keep.** Existing scripted smoke and multi-browser coverage are real uses. |
| `smol-toml` 1.9.1 | `scripts/netlify_headers.mjs:16`; header/CSP generators and tests. | 0.140 MB; current maintained parser; no direct finding. Replaced earlier hand parsing intentionally. | **Keep.** Do not replace a real TOML parser with regex for a tiny install saving. |
| `svelte` 5.57.3 | `src/lib/stores/theme.ts:1`, Svelte components, `src/routes/+layout.svelte:2`. | 2.947 MB; current compiler/runtime, no direct finding. | **Keep.** Required Kit peer and application runtime. |
| `svelte-check` 4.7.8 | `package.json:15` typecheck, called by full tests. | 5.957 MB; current Svelte tooling; inherited fdir/picomatch high remains. | **Keep.** Neither `tsc` alone nor Biome checks Svelte templates equivalently. |
| `typescript` 6.0.3 | `package.json:15-16`, `tsconfig.json`; compiler API in `tests/analysis_review_cache.test.mjs:4` and other source-slicing tests. | 24.347 MB; maintained; Kit requires 6.x and svelte-check supports 5/6, not latest 7.0.2. No direct finding. | **Keep ~6.0.3.** Required major verified; defer incompatible 7.x. |
| `vite` 8.3.8 | `package.json:26,29` dev/build scripts; `vite.config.ts`; `kit.config.js:2` loadEnv. | 2.405 MB plus Rolldown/native toolchain; current Vite 8; unpatched picomatch high. | **Keep Vite 8.** Do not accept npm's forced downgrade. |
| `wrangler` 4.164.0 | `package.json:47` `deploy:artifact-worker`, `cloudflare/artifacts.wrangler.toml`. | 15.825 MB plus workerd/miniflare; actively maintained/current, no direct finding in this snapshot. Publish-capable development tool, not shipped app JS. | **Keep.** Script/config reference means not unused. Upgraded installation only; no Wrangler command/deployment was executed. |

### Removal proof

Searched the complete tracked source/config/script/test tree for `query-core`,
`@tanstack/store`, `binaryen`, `wasm-opt` and binary entrypoint spellings, including
hidden CI/config files; excluded installed/generated bundles from usage claims.
Only the package declarations and historical Binaryen experiment docs referenced
those packages. Checked dynamic imports, spawned tools and public symlink targets,
not just ordinary imports. Removed declarations, regenerated lock, ran a fresh
`npm ci`, all tests and the full static build without them. NOTICE no longer lists
TanStack as shipped. Approximate registry payload removed: **98.930 MB**; measured
local `du -sk node_modules` fell **898,472 → 798,092 KiB**. Neither number is a
browser-download saving: those three packages were absent from the baseline app
bundle already.

## Vendored and staged third-party assets

Inventory includes regular files and local symlink targets. Directory byte totals
are raw logical file sums, include source archives/model data and all runtime
variants, and are **not** homepage download sizes. Every binary/compiled runtime
below is outside npm advisory coverage; “keep” does not assert a full binary CVE
review. Existing artifact manifests, pins, source archives and integrity checks
are the security/maintenance controls; release promotion still needs engine parity.

| Public asset / provenance | What uses it (file:line) | Raw size / maintenance and security posture | Verdict |
| --- | --- | --- | --- |
| `stockfish/` — stockfish.js 19.0.1, custom fixed/relaxed single-thread builds, npm-linked threaded/full builds, source tarball | `src/lc0/stockfishEngine.ts:53-61`, `scripts/engine_artifact_registry.mjs:160-171` | 284.541 MB incl. ~80.6 MB source; active upstream, pinned manifests/GPL sources. Worker glue and WASM must move together. | **Keep.** Not a duplicate unused npm dependency. Propose staged-release footprint work, not deletion. |
| `berserk/` — Berserk 14, Emscripten scalar/SIMD/relaxed tiers, canonical NN data, source | `src/lc0/berserkEngine.ts:16`, `src/lc0/engineProvision.ts:87-88` | 44.293 MB; pinned port (upstream `8ae895a6`), GPL-3.0; generated Emscripten JS is a third-party runtime too. No npm CVE visibility. | **Keep** every feature-detected tier and corresponding source. |
| `plentychess/` — 8.0.0, Emscripten/SSE4.1/relaxed tiers, shared NN data/source | `src/lc0/plentychessEngine.ts:15`, `engineProvision.ts:91-92` | 85.282 MB; pinned `50ad9f1b`, GPL-3.0; shared preload data already avoids tier duplication. | **Keep.** No tier removal without device coverage. |
| `stormphrax/` — 8.0.0, Emscripten fixed/relaxed tiers, NN data/source | `src/lc0/stormphraxEngine.ts:16`, `engineProvision.ts:95-96` | 94.724 MB; pinned `58296551`, GPL-3.0-or-later; versionless logical paths require release/hash consistency. | **Keep.** Network licence confirmation remains an existing owner obligation in NOTICE. |
| `viridithas/` — 20.0.0 WASI scalar/SIMD/relaxed binaries and source | `src/lc0/viridithasEngine.ts:7`, `engineProvision.ts:79-84` | 227.711 MB; pinned `0631113e`, MIT engine/CC0 network; uses browser WASI shim. | **Keep.** High size is NN weights/variants, not proof of dead code. |
| `reckless/` — pinned WASI/browser-API port, source archives/manifest | `src/lc0/recklessEngine.ts:51`, `engineProvision.ts:70-75`, `recklessBrowserApiWorker.ts:1` | 178.298 MB here, mostly multiple corresponding-source archives; executable assets are served via release host/local builds. AGPL-3.0, pinned `00106174`, v60 net. | **Keep.** Missing local binaries do not make source archives unused; source-retention and hosted-release obligations remain. |
| `ort/` → local onnxruntime-web dist | `src/nn/ortRuntime.ts:673`, `vite.config.ts:34`, `public/ort` symlink | 134.361 MB staged dist; 1.27.0 JS/glue/WASM, MIT, exact runtime pairing; multiple EP artifacts. | **Keep pinned.** Propose selective staging through the existing release pipeline, not ad hoc symlink edits or runtime upgrades. |
| `ort-experimental/{fixed,relaxed}/` JS/WASM + manifest | `src/lc0/policyOnlyBrowser.ts:1123`, `src/nn/ortRuntime.ts:613` | 53.814 MB; custom ORT builds, separately maintained/manifested; research path remains reachable. | **Keep; propose separating research distribution.** Not provably unused. |
| `runtimes/centipawn-tvmjs-webgpu/.../v1-baseline,v2-shape-k16,v3-subgroups/` — TVMJS bundles, TVM runtime/model WASM and weights; `runtimes/squareformer-tvm-hybrid/...` generated WGSL | `src/nn/squareformerTvmjsWebgpuEvaluator.ts:142,250,269,291`, `src/nn/squareformerTvmHybridEvaluator.ts:274`; manifests select bundle/WASM and fallback siblings | 88.148 MB combined runtime tree (359 files). Apache TVM Apache-2.0; pinned generated artifacts, SHA-verified bundle load. TVM's embedded WASI polyfill is part of its runtime ABI, not trivially interchangeable with browser_wasi_shim. Experimental subgroup versions have deliberate fallback contracts. | **Keep.** Propose research/production staging separation and shared immutable runtime bytes, only after manifest/hash/fallback validation. |
| `chessground.{base,brown,cburnett}.css` including embedded piece SVGs | `src/app.html:9-11` | Small CSS/artwork, GPL package-derived; base/brown locally modified, piece sheet copied from 9.2.1. Unsupported unscoped package and custom arrow patch must migrate together. | **Keep; merge provenance/staging** with scoped Chessground upgrade proposal, not overwrite modified CSS. |
| `models/lc0`, `models/maia3`, other `.onnx`/meta and manifests | `src/lc0/playBrowser.ts:74-82`, `src/lc0/bt4Engine.ts`, `public/models/maia3/manifest.json:1` | 533.199 MB locally available model tree; neural assets, not JS libraries. Hash/size pins; Maia/LC0/odds upstream provenance and outstanding network licensing questions documented in NOTICE. Model parsers and memory consumption remain attack surfaces. | **Keep.** No model/artifact upgrades in this task. |
| `openings/eco.json`, COPYING/README — lichess chess-openings data | `src/lc0/playBrowser.ts:1522`, `analysisBrowser.ts:393` | 0.472 MB; pinned upstream `3ba6b20d`, CC0-1.0; inert JSON validated by consumer. | **Keep.** Removing breaks opening names; not a replacement for the chess rules engine. |
| `lc0/opening_books/queen500.epd`, native fixture JSON/JSONL | `src/lc0/batchingStrengthOpenings.ts:10`, `src/lc0/policyOnlyBrowser.ts` | Included in 0.129 MB `lc0/` directory; book/fixture data, not an executable library. | **Keep** research fixtures/book; optional staging separation proposal only. |
| `lc0/lc0_input_encoder.wasm`, `lc0_legal_priors.wasm` | `src/lc0/wasmInputEncoder.ts:38-44`, `src/lc0/wasmLegalPriors.ts:127` | Project-generated helper kernels, not independent npm libraries; reproducible build/check scripts and URL/hash pins. | **Keep.** JS/WASM parity is deliberate, not unnecessary library duplication. |
| `engine-logos/` upstream marks/artwork | `src/lc0/engineCatalog.ts`, `public/engine-logos/README.md:1`, `NOTICE.md:90` | 0.407 MB; static images, not libraries; trademark/licence provenance still requires owner confirmation for some artwork. | **Keep.** No executable dependency to remove. |

Also inspected top-level `app-shell.css`, `sw.js`, web manifest, generated
artifact/release/channel indexes and static site metadata: these are project code
or data, not additional vendored UI libraries. No hidden jQuery/framework bundle
was found. Generated Emscripten glue and TVMJS are accounted for above rather than
misclassified as first-party JS. Existing source/archive and model-licence caveats
in `NOTICE.md` are not resolved by npm audit.

### Vendored Chessground CSS differs from 9.2.1

`public/chessground.cburnett.css` is byte-identical to the 9.2.1 package asset.
`public/chessground.base.css` and `public/chessground.brown.css` are **local edits**
(history: 3433f659 theming, 5509e47a/6abf54f2/50d16708 coordinate fixes), so a
Chessground upgrade must not overwrite them with package assets:

- base: coordinates restyled lichess-like: ranks on the right edge and files on the
  bottom edge as 12.5% strips inside the board, lowercase bold labels, no 0.8
  opacity, no `.ranks.left` variant.
- brown: board squares from a `repeating-conic-gradient` over `--board-light` /
  `--board-dark` (set in app-shell.css for theme/board-style switching) instead of
  the embedded base64 SVG; coordinate colors via `--coord-dark` / `--coord-light`.

Re-apply these as overrides (or keep the files vendored) during the scoped
`@lichess-org/chessground` migration proposed below.

## Replacement / consolidation proposals — not implemented

1. **Move Chessground to `@lichess-org/chessground` 10.5.0.** Highest maintenance
   priority: unscoped 9.2.1 is unsupported. Estimate **1–3 days, medium risk**.
   Check API/type/CSS changes, update the build-time SVG-arrow patch matcher and
   geometry tests, reconcile modified board CSS and piece licences, then test
   drag, keyboard, annotations, promotion, touch and board resizing on every
   surface. Scoped registry package is 0.461 MB, not necessarily smaller. Do not
   silently point the current patch at unverified new internals.
2. **Consolidate overlapping browser harness plumbing, not test coverage.**
   Prefer Playwright for deterministic CI/multi-browser drivers; retain
   agent-browser where interactive research is useful. Inventory the LC0/GPU
   scripts before changing them. Estimate **3–5 days, medium/high risk** because
   session ownership, process cleanup, native JSON envelopes, GPU flags and
   timeout semantics differ. Potentially avoids the 85 MB CLI install and a
   second automation protocol only if every current CLI caller is migrated.
3. **Separate research and product asset staging; deduplicate by content hash.**
   Reuse `prepare_netlify_r2_public_assets.mjs`, release manifests and existing
   ORT external-WASM selection. Estimate **2–4 days, medium/high risk**. Product
   build still copies research runtime variants despite BUILD_SCOPE gating
   application routes; source archives/fallbacks cannot just be deleted.
   Measure cache-first/offline behavior, artifact retention, COOP/COEP/CSP,
   integrity pins, WASM/glue pairing and all feature ladders. No runtime/library
   merger into a single engine is sensible: independent engine opinions are a
   product feature. No TVM/WASI/ORT consolidation without ABI/parity evidence.
4. **Do not add a query/state library to replace existing caches.** The removed
   TanStack packages had no consumers. A proposed Svelte-store/model-cache
   rewrite would be separate, **several days and high correctness risk** around
   persistence, cancellation and GPU lifetime, with no demonstrated size win.

Deferred latest releases: **TypeScript 7.0.2** (outside Kit/svelte-check peers),
**agent-browser 0.43.0** (breaking pre-1.0 CLI qualification), **Biome 2.6.8**
(new repo-wide Markdown/YAML formatting adoption), **ORT 1.31.1** (explicit owner
pin). Binaryen 132 is irrelevant after removal. No other outdated entries remain
in the final `npm outdated --json` snapshot.

## Verification and size comparison

Used **Node 24.22.1**, matching the project's Node 24 requirement. Initial install
used the shell's Node 25.9.0; baseline and final comparison builds and final clean
install/tests used Node 24.22.1. No shared node_modules writes or symlinks created.

- `npm ci`: exit 0; real local node_modules, lockfile unchanged by install.
- `npm audit --audit-level=high`: exit 1, exact unpatched residual above.
- `npm run check:biome`: exit 0, no fixes needed after the CI-equivalent formatter run.
- `npm test`: **2,410 tests; 2,403 pass, 7 skip, 0 fail** (456.8 s), including
  svelte-check with **0 errors / 0 warnings**. Skips are existing environment/
  fixture-dependent cases, not new exclusions: three require `.venv-onnx/bin/python`,
  one requires an upstream LC0 `encoder.cc` checkout, and three fold goldens lack
  the expected sibling fixture files (`lqo_v2` and two Maia3 artifacts). No sibling
  worktree/model directories were written to stage these optional fixtures.
- `BUILD_SCOPE=product npm run build:client`: exit 0, static prerender/adapter and
  migrated post-build guard pass. `npm run deploy:csp-check -- dist-client` also
  passes all built pages with unchanged CSP source policy.
- `PORT=64091 npm run web:isolated:static`: agent-browser on local built files,
  fresh owned browser session. Home and Docs render; client navigation updates
  active header links; Play Maia3 **1.e4 e5**, Play Stockfish Lite **1.e4 g6**;
  Analysis displays **Lc0 Small +0.13 / d4** and **Stockfish Lite +0.26 / e4**;
  Arena Stockfish Lite vs Berserk Relaxed SIMD, **50 ms/move, one completed
  game, 267 plies, draw (½–½)**. Both CPU engine outputs and eval charts populated.
- Browser console: **no errors or warnings**, three informational runtime-audit/
  ORT messages; no page errors. Analysis → Play client navigation's known
  `Missing #message` issue was not exercised or changed.
- During manual testing, an immediate typed move after switching an active
  Maia game to Stockfish was rejected as “still loading”; a full Play reload
  restored Stockfish and accepted the move/reply. This UI transition was not
  attributed to dependencies or fixed here. A wait on hidden, ungenerated PGN
  output also timed out after Arena had completed; the visible final score and
  saved tournament state were inspected instead. These are recorded rather
  than misrepresented as additional successful assertions.

Raw output bytes, counted recursively with file contents rather than filesystem
allocation (same local asset set, no baseline copies added to dist):

| Metric | Before | After | Delta |
| --- | ---: | ---: | ---: |
| Total `dist-client` | 1,730,335,581 | 1,730,386,295 | +50,714 (+0.0029%) |
| Files | 639 | 641 | +2 |
| All `.js` incl. vendored/staged | 21,302,892 | 21,313,449 | +10,557 |
| App `_app` JS | 3,576,755 / 71 files | 3,587,312 / 73 files | +10,557 (+0.295%) |
| Shared JS chunks | 1,436,031 / 48 | 1,450,785 / 49 | +14,754 |
| Worker JS | 1,877,106 / 8 | 1,875,691 / 8 | −1,415 |
| Route-node JS | 257,566 / 13 | 256,457 / 13 | −1,109 |
| Entry JS | 6,052 / 2 | 4,379 / 3 | −1,673 |
| Largest shared chunk | 273,807 | 273,807 | 0 |
| Hybrid search worker | 583,584 | 583,220 | −364 |
| Search lab worker | 477,664 | 477,664 | 0 |
| Main search worker | 219,683 | 219,318 | −365 |
| Maia3 worker | 115,764 | 115,443 | −321 |

Most output is models, runtimes, engine networks and corresponding source, not
Svelte. New Kit payload/chunk splitting accounts for two more app JS files;
removing never-imported development packages cannot shrink an existing bundle.
Full per-chunk filenames/sizes are in baseline-size.json and final-size.json.

Evidence directory (outside Git):
`$SCRATCH/deps-upgrade/`.
Contains baseline/final build and install logs, audit JSON/text, outdated and
registry metadata, full test logs, size inventories, browser diagnostics/result
notes and verified PNGs for Home, Docs, both Play opponents, Analysis and Arena.

## Closeout / owner decisions

No suppression was applied; the audit CI job (report-only on main since
2026-11-17) still reports the single upstream picomatch mechanism. Owner decisions:
whether a narrow expiring advisory exception is wanted if the job becomes blocking again; schedule Chessground scoped migration,
browser-tool consolidation and asset staging work; retain/confirm existing
third-party model/artwork licensing obligations. No push, merge, deploy or upload.

Requested closeout review command:

```sh
~/.pi/agent/skills/autoreview/scripts/autoreview --mode branch --base main --engine codex --model gpt-6.1-sol --thinking xhigh
```

Review completed on this branch: **autoreview clean — no accepted/actionable
findings reported**, overall “patch is correct” (0.88). No findings were accepted
or rejected and no code changes were requested. Read-only reviewer inspection
used the dependency contracts and existing build output; independent tests and
browser verification above were performed by the implementation session. Full
output: `autoreview.log` in the evidence directory.
