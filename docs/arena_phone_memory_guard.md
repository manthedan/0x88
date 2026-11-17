# Phone large-net consent and Arena memory guard

BT4 and t3 are **off by default on memory-constrained devices** in Analysis,
Play and Arena. **Queen Odds is exempt**: it runs alone in Play, not in
multi-analysis or Arena. The owner's iPhone headroom measurements put it about
0.8–1 GB below the 2 GB kill line, versus under 512 MB for BT4/t3. This is a
policy decision, not a guarantee that Queen Odds cannot crash. iPhone Safari can kill a tab even when switching between nets
that each load alone; Android WASM can run at only ~1–4 visits/s. Download sizes
do not describe runtime memory use. Device classification is a conservative
policy, not a RAM measurement.

## Device-wide opt-in

BT4 and t3 require opt-in. Queen Odds remains available by default with its
ordinary asset checks and existing size/download warning behavior, without a
phone opt-in dialog; accepting its ordinary warning never enables BT4/t3. Picker options stay
visible as “Large net — off on phones”. Choosing one opens an in-app native
modal dialog before any worker acquisition or model download. It explains tab
crashes, slow operation without WebGPU and the selected model's MB download.
“Use anyway” enables BT4 and t3 on this device; “Cancel” and Escape preserve
the previous selection. Native modal inertness traps focus, and focus is restored
on dismissal. The dialog follows Analysis's existing native-dialog pattern.

`phoneLargeNetPolicy.ts` remembers consent under `0x88-phone-large-nets-v1` in
localStorage. Access and writes are guarded; a session override keeps opt-in
and revocation working when storage fails. Revoking Play consent replaces a large
live opponent with Small, but only dismisses a blocked restart draft when the live
opponent is unaffected; unresolved saved-large-engine probes are also cancelled.
A line beside each picker shows
“BT4 and t3 on this device: on · turn off”. Turning off cancels gated large-net work,
releases its sessions, and falls back to a small engine. Arena's setup fieldset
locks this control during a run, like its other engine settings.

Saved profiles, manual rows, URL seats, restored Play opponents and checkpoint
resume never grant consent. Without opt-in BT4/t3 fall back to Lc0 Small and show
a persistent explanation by the picker. Checkpoint resume refuses to mix results
if that fallback changes the saved engine. The session-pool acquisition boundary
also rejects BT4/t3 without consent, protecting missed/programmatic paths.
Known large-net `?model=` overrides (including shipped split-K variants) on phones
always fall back to Small, with an
explanation to use the real large-net picker instead. This prevents a large net
from being disguised as Small and remaining resident after consent is revoked.
Critical-moments review offers only Lc0 Small (plus Stockfish and Maia), so it has
no large-net option to gate; Analysis review uses its consent-checked engine rows.

Desktop keeps its existing selection, availability and restore policies, including
the Analysis/Arena download warning (now rendered as an accessible in-app dialog).
Play on desktop still has no extra consent prompt. Constrained Arena can use the
WASM fallback after consent; desktop Arena's WebGPU requirement is unchanged.

## Device rule

`src/lc0/deviceMemoryClass.ts` is the shared classifier:

- All iPhone, iPad and iPod user agents count as constrained.
- A Macintosh user agent with `maxTouchPoints > 1` counts too (iPad desktop mode).
- Android counts when `deviceMemory <= 4` GB or memory is unreported/invalid.
  Missing memory is treated conservatively because not every mobile browser
  exposes it.
- Other desktops never count, even with low reported RAM or a touchscreen.
- `?memoryClass=phone` and `?memoryClass=desktop` override the rule for lab tests.
  Unknown values use normal detection. Both Arena and the session pool use the
  same helper, and importing it without browser globals is safe.

## Pairings and residency

`src/lc0/arenaMemoryGuard.ts` checks actual tournament pairings against `BIG_NETS`
(including Queen Odds, BT4 and t3). Seats count separately even when the same net
is on both sides. Match considers only the first two seats. A CPU-led gauntlet
can have multiple large nets in the field because they never face each other.

Arena disables large-net options against a large-net opponent and explains:
“Two large nets at once can crash this device.” After device opt-in, a conflicting URL, restored
roster or mode change remains visible, but cannot start; the start entry also
checks the rule before any asset checks or warmup. The block has no dialog.
Device-wide consent never bypasses this two-large-net block. Asset checks still
apply after consent; lack of WebGPU is allowed only on constrained Arena devices,
with the slow-WASM warning described above.
Queen Odds is still counted as large for the two-large-net pairing rule if it
is ever offered; its consent exemption does not bypass that rule or add it to
Arena's existing menu.

On constrained devices, Arena requests exclusive residency from
`bigNetSessionPool.ts`. Acquiring a different net disposes the previous worker
**before** constructing its replacement, even if the previous session was retained
by the three-minute idle TTL. Async acquisition awaits the bounded graceful
session/device release from `main` (matching acknowledgement or 1500 ms timeout,
then worker termination). Release/transport failures propagate to awaiting owners.
Releasing a net detaches it immediately; its worker drains asynchronously. Reacquiring
the same active net still reuses it. Shared callers such as Analysis keep ownership
of their active sessions: without an exclusive request, only retired sessions are
evicted. This avoids interrupting Analysis's existing parallel initialization. Analysis's
TTL-0 retirement and identity-checked detach remain intact. Analysis and Play also
await pending pool retirements on phones before constructing another net; a
searcher's own reinitialization awaits its detached worker's cleanup.
Desktop retains the existing multi-net pool and idle TTL, even for exclusive
requests.

Arena diagnostics and asset checks never acquire searchers. Phone warmup prepares
only the current game's engines, including when advancing through a safe gauntlet.
Desktop keeps the existing whole-roster warmup.

The phone modal is intentionally compact: “Use a large net on this phone?”,
two sentences covering crash/slow-WASM risk and the selected download size
(BT4 ~178 MB, t3 ~163 MB), plus small print about turning it off below the picker.
Cancel/Use anyway, native inertness, focus restoration, Tab trapping and Escape
behavior are unchanged.

## Verification

- Shared policy matrix, storage failures, restore fallback and acquisition boundary:
  `tests/phone_large_net_policy.test.mjs`
- Saved-profile consent: `tests/engine_setups.test.mjs`
- Classifier: `tests/device_memory_class.test.mjs`
- Pairings, start-entry block, warmup routing, passive diagnostics:
  `tests/arena_memory_guard.test.mjs`
- Residency, disposal order, idle TTL and same-net reuse:
  `tests/big_net_session_pool.test.mjs`
- Browser: build with `BUILD_SCOPE=product npm run build:client`, serve with
  `PORT=<free port> npm run web:isolated:static`, compare `/app/arena/` with and
  without `?memoryClass=phone`. No phone or Safari automation is needed.
- Built-preview evidence at 390×844: scratchpad `mobile-soak/phone-bignet-optin/`.
  Each surface has desktop-policy, phone-picker, warning-dialog and opted-in
  screenshots. Browser checks also cover Escape/focus restoration, Tab trapping,
  revocation, Arena's pair block after opt-in, persisted consent, URL-seat fallback
  and Play's restored-opponent fallback. The local Maia3 model is absent (404),
  unrelated to the policy; large-net picker/consent checks use the hosted BT4 asset.

### Release 2 rebase verification

Rebased onto `main` including mobile-memory Phase 4. Full
`LC0_MEMORY_MODEL_DIR=<worktree>/.local-dev-artifacts/release2/models npm test -- --test-concurrency=2`:
**2463 passed, 4 existing optional skips**, including all seven folding goldens
and real Small inference parity. Typecheck: zero errors/warnings; repository lint,
targeted Biome checks and `BUILD_SCOPE=product npm run build:client` pass.

Agent-browser product-build evidence is in scratchpad `mobile-soak/release2/`:

- At **390×844**, `?memoryClass=phone`: Play Queen Odds is ready with no dialog
  and no consent; Analysis BT4 shows the compact ~178 MB modal; Arena shows the
  phone policy and the t3 ~163 MB modal. Tab cycles between the two actions;
  Escape cancels without granting consent or changing the selected net.
- Arena acceptance enables t3 against a small engine; changing the other seat to
  Lc0 leaves its t3/BT4 options disabled (“one large net at a time”). Revocation
  returns the selected t3 seat to Small.
- Same viewport with `?memoryClass=desktop`: Queen Odds has no dialog; Play BT4
  also requires no opt-in, even with consent off; Analysis retains its ordinary
  download warning; Arena has no phone control or large-pair restriction. An
  additional **1280×900** Arena screenshot covers the desktop layout.
- The optional Reckless assets are not staged locally, so adding that Analysis
  row produces an unrelated 404/failure row. No actual iPhone, Safari automation,
  physical-memory measurement, deployment or upload was performed.
