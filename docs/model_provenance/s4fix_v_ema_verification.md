# s4fix EMA local verification

This is a local ship check, not a publication or browser/GPU qualification.
Node was v25.9.0. The work stayed on `ship/centipawn-s4fix-v`.
See `s4fix_v_ema.md` for provenance and the encoder mismatch investigation.

## Search comparison

| Option | Native fixture | Play | Analysis | Arena | Change |
| --- | --- | --- | --- | --- | --- |
| Preset/policy | monty_lc0both, Monty widening | montyLitePuctPolicy | same | same, including warmup | Already equivalent widening: top-p .85, min 6, +2 actions, exponential threshold power 2, forcing moves, sharp Gini .5, scale .5–1.5 |
| cpuct | 1.5 | 1.5 | 1.5 | 1.5 | Explicit shared Centipawn setting; generic PUCT unchanged |
| cpuct schedule | lc0-log | lc0-log | lc0-log | lc0-log | Changed Centipawn from constant |
| cpuct base / factor | 38739 / 3.8940000534 | 38739 / 3.894 | same | same | Made explicit; float32 fixture rounding is immaterial |
| FPU / strategy | 0 / lc0-reduction | 0 / lc0-reduction | same | same | Changed Centipawn strategy from constant |
| FPU reduction | .3300000131 | .33 | .33 | .33 | Made explicit |
| Policy temperature | 2 | 2 | 2 | 2 | Model registry setting, evaluator legal policy only |
| AV/rank/regret/risk/uncertainty weights | all 0 | all 0 | all 0 | all 0 | AV explicitly 0 instead of generic .25; no AV head in this model |
| Auxiliary WDL search weight | 0 | 0 | 0 | 0 | Unchanged; ordinary WDL only |
| Final-move temperature | 0 | 0 | 0 | 0 | Existing chooseMove visit-argmax default |
| Root reuse | off | off | off | off | No Centipawn caller passes a root; no change |
| Eval cache | none | 4096 | 4096 | UI-selected, default 2048 | Retained product feature; Arena's existing wrapper has a minimum of 1 even if UI selects 0 |
| Search batch | 1 | min(32, visits) | 32, URL-selectable 1–256 | 32, URL-selectable 1–256 | Retained product batching; not strength-qualified by the bundle |
| Visits | 256 | 8/32/100/400/1600 | chosen, default 400 | chosen, default 100, or movetime | Retained product budgets |
| Early stop | off | root-dominance | off | off, or chosen movetime | Retained Play optimization, documented as preserving the budget winner |
| Pipeline depth | 1 | 1 | 1 | 1 | Unchanged |
| Search drawn root | true | true | true | true | Existing root draw exemption |
| Other tactical/variance/butterfly/qsearch weights | disabled | disabled | disabled | disabled | Unchanged |

There is no product Centipawn search-worker path: these three surfaces call
`chooseMove` directly. The worker search paths in `src/lc0/` are for other
engines and were not changed. The generic defaults in `src/search/puct.ts`
remain constant-cpuct / constant-FPU for other callers.

The standalone search check received each complete root-FEN/move line and
ran fresh searches with batch 1, 256 completed visits, no evaluation cache and
no root reuse. Agreement was **50/51 (98.0392156863%)**, including **2/2**
positions with absolute native value at least 0.8. Elapsed time was 74.628 s,
so the full search check is a script, not an automatic test. A fast test
checks shared constants and whole-history cache behavior.

Only disagreement:

| Position | Native move | Native visits | Native root value / cp | Site move | Site visits / value |
| --- | --- | --- | --- | --- | --- |
| game 39b ply 38 | d4b5 | 256 total; per-move counts not supplied | .0075 / +3 cp | e2f3 | e2f3: 102 / .0469122059; d4b5: 94 / .0087363709 |

The remaining site top moves were d4f3 (47 visits, Q -.0379355315), d4c2
(10, -.0724937928), and d4b3 (3, -.1972782179). Do not invent native move
counts: the supplied search fixture contains only the total budget, best move,
score and root value. This low-value flip passes the documented loose gate.

## Parity

All 51 positions match at every integer input: 78,336 token integers and
91,392 attack-summary integers. All legal UCI/policy-index mappings match,
including Black normalization and promotions. Maximum absolute ORT Web WASM
errors against supplied expected outputs:

| Output | Max absolute error |
| --- | ---: |
| Legal policy logits | 0.00001239776611328125 |
| WDL logits | 0.0000034570693969726562 |
| WDL probability (ordinary and auxiliary checks) | 0.0000011927414168555472 |
| q | 0.0000010728836059570312 |
| wdl_sf18 logits | 0.000006318092346191406 |
| Legal prior (T=2 and sampled T=1 checks) | 0.000000679524329971759 |

Every position checks evaluator T=2 policy, ordinary WDL and auxiliary WDL;
four also compare T=1 versus T=2 and require exactly equal WDL outputs.
Hidden is intentionally not fetched. The model and metadata match the bundle's
SHA256SUMS and the manifest; the ONNX is staged as a Git LFS pointer.

## Prior consumers and UI

`SquareFormerEvaluator` produces the tempered `Evaluation.policy` map.
`CachedEvaluator` stores/copies that map. Generic PUCT normalizes legal
entries into edge priors. Centipawn's Monty policy consumes them for top-p
widening, Gini exploration, PUCT U, visited-policy-mass FPU reduction and
final equal-visit/equal-Q tie-breaking. Priors also travel in returned root
policy, PV/MultiPV entries and optional search traces. No WDL temperature
or auxiliary-value weight changes.

Play consumes the resulting move/progress/PV; Analysis consumes searched
scores and PV/MultiPV through `centipawnPuctAnalysisLines`; Arena consumes
moves, searched Q and visit/evaluation counts. These can visibly change due
to the model, prior and aligned constants. No current product component
prints a direct Centipawn policy percentage. Arena's root-chart `P %` and
reply-policy ranking use `lastLc0SearchResults` only; Analysis Human and retry
likelihood percentages are Maia3 only. Those percentages do not change here.

No numeric Centipawn Elo calibration table or numeric Elo label was found in
the active UI. The existing Play budgets and effort names Fastest, Quick,
Standard, Strong and Deep remain, along with the generic “full-strength
engine, strong even at level 1” blurb. They are inherited qualitative labels,
not a new strength calibration. Historical model-specific provenance and
dated lab/evidence documents remain unchanged and describe their old models.

## Commands and outcomes

`node-test` below means `node --experimental-strip-types --test`.
The focused set is these seven files:

```text
tests/centipawn_s4fix_parity.test.mjs
tests/centipawn_s4fix_runtime.test.mjs
tests/centipawn_s4fix_search.test.mjs
tests/centipawn_release.test.mjs
tests/runtime_registry.test.mjs
tests/centipawn_search_history.test.mjs
tests/engine_catalog.test.mjs
```

| Command/run | Pass / fail / skip | Result |
| --- | --- | --- |
| Initial `node-test tests/centipawn_s4fix_parity.test.mjs` | 0 / 1 / 0 | Rejected a new constructor parameter property in strip-only Node; replaced with a normal field |
| Same parity command after correction | 2 / 0 / 0 | Passed |
| `node-test tests/runtime_registry.test.mjs tests/centipawn_release.test.mjs tests/centipawn_search_history.test.mjs tests/engine_catalog.test.mjs` | 14 / 1 / 0 | Browser-loader import reached pre-existing legacy parameter properties; moved integration coverage to the repo's transform-types child-test pattern |
| `node-test` with the seven-file focused set | 22 / 0 / 0 | Passed |
| `node-test --test-timeout=120000` with all 33 matching test files below | 289 / 0 / 0 | Passed, including old TVMJS/subgroup tests |
| `npm test` (exactly once) | 1981 / 1 / 4 | One v1-only URL assertion in stage_lc0_test_assets; no timeout failures |
| `node-test --test-concurrency=1 --test-timeout=120000 tests/stage_lc0_test_assets.test.mjs` plus the focused set | 28 / 0 / 0 | Passed after using the normalized v2 representation URL in that assertion |
| `node-test --test-timeout=120000` with the 33 matching files plus `tests/stage_lc0_test_assets.test.mjs` | 295 / 0 / 0 | Passed after the full-suite assertion correction |
| Same 34-file command after the research-override guard and regression test | 296 / 0 / 0 | Final source-tree targeted check passed |
| `npm run typecheck` | 0 errors / 0 warnings | All three standalone runs passed; npm test's embedded typecheck also passed |
| `node --experimental-strip-types scripts/check_centipawn_s4fix_search.mjs .local-dev-artifacts/centipawn-s4fix/search.json` | 50 / 51 move agreement | Gate passed, including all decisive positions |
| `BUILD_SCOPE=product npm run -s build:client` (exactly once) | Passed | dist-client contains byte-identical new ONNX, metadata and Centipawn manifest; no fp16/batch16 variant |
| `node scripts/write_artifact_index.mjs --check` | Passed | Generated index current |
| Release generator command below, also with `--check` | Passed | 92 artifact entries; no upload |
| `git diff --check` | Passed after removing an extra test-file EOF blank line | No whitespace errors |

The four full-suite skips are the split-K Python rewrite test, two dynamic
MatMulInteger Python transformer tests (local `.venv-onnx` unavailable), and
the generated LC0 policy table/source comparison (external C++ source
unavailable). The full-suite failure was not ignored: its six-test file and
all focused tests passed serially after correction. The full suite was not
run a second time, as requested. The build emitted a Node localstorage-file
warning and informational plugin timing output, not a build failure. A final
inspection then corrected Analysis/Arena research URL overrides so other models
do not inherit the shipped model id/settings. That guard and its regression
test passed typecheck and the final 296-test run. The default shipped route
is unchanged; the product build was not repeated after this guard, honoring
the one-build limit.

The matching test command was built from
`git grep -il -e centipawn -e squareformer -e runtimeRegistry -- tests/`,
filtered to `.test.mjs`, plus the three new tests, sorted uniquely:

```text
tests/analysis_bignet_served_model.test.mjs
tests/analysis_progress.test.mjs
tests/analysis_stockfish_release.test.mjs
tests/arena_parallel_games.test.mjs
tests/artifact_release_manifest.test.mjs
tests/browser_runtime_evaluator.test.mjs
tests/centipawn_evaluator_cache.test.mjs
tests/centipawn_release.test.mjs
tests/centipawn_s4fix_parity.test.mjs
tests/centipawn_s4fix_runtime.test.mjs
tests/centipawn_s4fix_search.test.mjs
tests/centipawn_search_history.test.mjs
tests/centipawn_subgroups_evidence.test.mjs
tests/centipawn_tree_reuse_probe.test.mjs
tests/centipawn_tvmjs_evaluator.test.mjs
tests/centipawn_tvmjs_subgroups.test.mjs
tests/engine_catalog.test.mjs
tests/engine_logos.test.mjs
tests/fetch_boundary_schemas.test.mjs
tests/game_review_hardening.test.mjs
tests/kernel_hillclimb.test.mjs
tests/lc0_analysis_format.test.mjs
tests/move_encodings.test.mjs
tests/numerics_parity.test.mjs
tests/ort_output_readback.test.mjs
tests/play_preferences.test.mjs
tests/production_browser_smoke.test.mjs
tests/runtime_registry.test.mjs
tests/service_worker.test.mjs
tests/squareformer_input.test.mjs
tests/squareformer_tvm_hybrid_evaluator.test.mjs
tests/squareformer_tvm_kernel_assets.test.mjs
tests/squareformer_tvm_readback.test.mjs
```

## Release and reviewer notes

The repository generator now writes v2 release manifests. Generation was:

```sh
node scripts/write_artifact_release_manifests.mjs \
  --release-id 2026-11-13.centipawn-s4fix-v-ema.1 \
  --generated-at 2026-11-13T00:00:00.000Z \
  --manifest public/models/centipawn.manifest.json \
  --base-release public/releases/2026-07-14.centipawn-tvmjs.1.json \
  --out-dir .local-dev-artifacts/centipawn-s4fix/release --no-brotli
```

Only the resulting new release and stable channel JSON were copied to
`public/releases/` and `public/channels/`. All old release JSON and model
files remain untouched. No remote publication, browser, GPU work or
dev/preview server was run. The release script's default old-runtime manifest
entry is intentionally retained for historical artifact publication; it is
not an application runtime selection.

The new release carries forward the prior catalog and uses identity-only
representations. Publishing must account for v1-to-v2 identity migration
sources, not just upload the two new model files. Publish/validate immutable
bodies and the new release before promoting the stable channel or app shell.
Nothing was uploaded here.

The model id for Analysis/Arena research URL overrides is derived from the
chosen ONNX path, so unrelated custom model/meta pairs retain temperature 1
and legacy history encoding. The shipped URL retains temperature 2 and the
ORT-only guard.

Remaining risks are untested browser/provider performance, unqualified
batch>1 strength, longer-history cache-key/encoding overhead, the absence of
an old-served-model head-to-head comparison or new Play calibration, and the
bundle's unspecified model licence. FEN-only padding rules beyond the shapes
covered by the fixture are not independently established. Lc0/Maia/other
engine search paths, the analysis scheduler, GPU/session implementation and
compiled artifacts were not changed.
