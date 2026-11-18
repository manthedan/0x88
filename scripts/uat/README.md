# UAT journey recorder

Records what a user sees *while* using the app, so a vision model can judge motion and state
transitions rather than single still screenshots. Still screenshots missed things like the Analysis
Stop/progress button overflowing, the toolbar re-stacking as progress changed, engine cards stuck on
"Searching…" while the status line showed live progress, and a clipped tab badge.

## Run

The journeys need real model files, which are gitignored: Lc0 Small (Play, Arena, Analysis) and BT4
(`analysis-engines`). A plain `build:client` in a fresh checkout serves 404s for them. Stage the
pinned models, then build an archive of the commit with the models linked in. This is the same
build `perf:routes` measures. Run from the worktree root:

```sh
npm run test:stage-lc0-assets      # pinned Lc0 Small, BT4, Maia3 into ../models/lc0-bestnets (hash-checked)
npm run perf:routes -- --baseline HEAD --candidate HEAD --build-only --out .local-dev-artifacts/uat-build
```

Then record with `--dist .local-dev-artifacts/uat-build/builds/baseline/dist-client`. A checkout whose
`public/models` already links the models can use `BUILD_SCOPE=product npm run build:client` and the
default `--dist dist-client`.

```sh
node scripts/uat/record_journeys.mjs                     # all journeys × 1280x800, 390x844, 320x640 in Chrome
node scripts/uat/record_journeys.mjs --journeys analysis-engines --viewports 390x844 --video
node scripts/uat/record_journeys.mjs --browser safari    # needs `safaridriver -p 4444` running
node scripts/uat/record_journeys.mjs --base-url http://127.0.0.1:5181 --memory-class phone
node scripts/uat/record_journeys.mjs --dry-run           # print the plan only
node scripts/uat/record_journeys.mjs --sheets-only .local-dev-artifacts/uat/<run>   # re-render review pages
```

`--help` lists every option. Exit code 0 means every journey completed, and 2 means at least one
step failed. A failed journey still produces its frames, because the failure is UAT evidence too.

## Journeys

| name | what it does |
|---|---|
| `play-game` | five moves against Lc0 Small at level 0 from the start position |
| `play-promotion` | Analysis on a K+P position, **Play from here**, a7-a8 through the promotion picker, then two more moves |
| `analysis-engines` | two engines streaming; add Berserk; Lc0 Small → BT4 through the consent dialog; download; remove an engine; Stop, then Analyze; a board move; result tabs |
| `arena-match` | one 100 ms/move game, Lc0 Small vs Stockfish lite |
| `route-tour` | client-side header navigation Analysis → Play → Arena → Home → Analysis (uses the phone menu when the header is collapsed) |

Selectors, readiness checks and move helpers come from `scripts/perf_routes/journeys.mjs` and
`human.mjs`. A UI change that breaks those breaks both harnesses. Clicks use real pointer input
(Playwright `page.click` in Chrome, a WebDriver element click in Safari). If the pointer can't reach a
control, the recorder logs a `warning` event and falls back to a DOM click.

Viewports narrower than 700 px are emulated as touch phones (`isMobile`, `hasTouch`, DPR 2) but keep
a desktop UA. Add `--memory-class phone` to exercise the phone large-net policy
(`?memoryClass=phone`). Safari can't always shrink its window to phone widths. In that case the
recorder stores the viewport it actually got (`meta.measuredViewport`), and the index, review pages
and sheets show it next to the requested size.

`--sheets-only` refuses a symlinked recording directory and never writes through symlinks inside
it. Review pages and sheets only use the recorder's own `frames/*` and `video.webm` names, and only
when they are regular files inside the recording, so a recording from someone else is safe to
re-render.

## How it records

- Each run gets a fresh counting proxy (`scripts/perf_matrix/proxy.mjs`), so it starts on a fresh
  origin with empty storage. The proxy injects `scripts/perf_routes/probe.js` plus the UAT probe
  (`probe.mjs`) at the top of every HTML document. Full navigations, such as Play from here, are
  covered too.
- **Frames.** The recorder takes a screenshot every `--cadence` ms (250 by default). It also takes
  one at every step start and end, and one at the start of each DOM-mutation burst (the first
  mutation after 150 ms of quiet; Chrome only, through a binding). Before each screenshot, the probe
  returns:
  - layout-shift entries (Chrome only)
  - overflow findings from a cheap DOM walk (`detectOverflows`: clipped, ellipsis, text-spill,
    escapes, offscreen and page-scroll-x)
  - the rects of on-screen controls, which the recorder diffs into resized/moved events
  - status and live-region texts
  - console and worker errors (Safari only; Chrome gets these from CDP)
- **Size.** Identical screenshots are stored once (content hash). After `--soft-cap` unique frames
  the cadence drops 4×. After `--hard-cap`, only step and burst frames are kept. JPEG quality 70 is
  the default, and Safari is always PNG. Mutation-burst captures keep going past the hard cap, at
  most one every 2 s, so late transitions in a long Arena game still show up. They stop at 1.5×
  the hard cap.

## Output

```
.local-dev-artifacts/uat/<timestamp>/
  index.html                       all runs: status, unique frames, overflow spans, resized controls, errors
  run.json
  <journey>--<WxH>/
    index.html                     steps, automatic leads (overflow spans, control resizes, errors), full filmstrip
    timeline.json                  meta, steps, every sample (frame file, step, overflow diff, control changes,
                                   layout shift, changed texts, events) and a digest
    frames/0001.jpg …              unique frames
    sheets/sheet-01.jpg …          contact sheets (Chromium-rendered grids) to hand to a vision model
    video.webm                     with --video (Chrome)
```

Contact sheets list consecutive identical frames once, labelled with how long the screen stayed
unchanged ("held 4.20s"). Use them to spot stale state. A hold stops where capture paused for more
than 5 s; the next frame is marked "unobserved Ns before", since nobody saw the screen then.

## Review

Give a reviewer model `REVIEWER_PROMPT.md` along with the run's contact sheets, `index.html` digest
and `timeline.json`. The automatic findings are leads to check, not verdicts.

The pure parts are unit-tested in `tests/uat_recorder.test.mjs`: frame dedupe, overflow detection on
synthetic rects, control diffs, timeline assembly, rendering, and the probe against a fake DOM.
