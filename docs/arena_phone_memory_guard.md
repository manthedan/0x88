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

BT4 and t3 require remembered crash-risk opt-in. All three large nets also get
an uncached-download warning on phones. Queen Odds uses **download-only** consent,
not crash opt-in: accepting it never enables BT4/t3 or persists a preference.
Picker options stay visible as “Large net — off on phones”. Choosing one opens
an accessible native modal before acquiring a worker or downloading model bytes.
BT4/t3 use “Use BT4 on this phone?” / “Use t3 on this phone?” and
“Large nets can crash the tab on phones.” Uncached models add
“BT4 is a N MB download, cached after the first time.” (or t3).
Cached BT4/t3 still need the crash warning if opt-in is off; once opted in,
only uncached downloads prompt again. Buttons remain Cancel / Use anyway.

Uncached Queen Odds uses “Download Queen Odds?”, “It's a N MB download, cached
after the first time.” and Cancel / Download. Verified cached Queen Odds has no
dialog. “Cancel” and Escape preserve the previous selection. Native modal inertness traps focus, and focus is restored
on dismissal. The dialog follows Analysis's existing native-dialog pattern.

`phoneLargeNetPolicy.ts` remembers consent under `0x88-phone-large-nets-v1` in
localStorage. Access and writes are guarded; a session override keeps opt-in
and revocation working when storage fails. Matching localStorage events reconcile
other tabs' opt-in, revocation and storage clearing, notify controls, and cancel
gated product work; unrelated storage areas cannot overwrite session consent. Revoking Play consent replaces a large
live opponent with Small, but only dismisses a blocked restart draft when the live
opponent is unaffected; unresolved saved-BT4/t3 probes are also cancelled. Every
revocation retires retained BT4/t3 sessions, including leftovers from an earlier
game, while exempt Queen Odds work and saved preferences remain intact.
A line beside each picker shows
“BT4 and t3 are off on phones (crash risk).” or
“BT4 and t3 on for this phone · Turn off”. It appears only beside pickers offering
BT4/t3, not in the lab (which has no engine picker). Turning off cancels gated large-net work,
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
The mobile-soak lab requests the same cache-aware warnings for each selected
large model before any engines or ballast, including `auto=1` and crash
continuation. Declining skips remaining steps without loading a model; leaving
aborts the dialog. Desktop diagnostics need no prompt. Only the actual shipped
Queen Odds model inherits its crash-risk exemption; uncached Queen Odds still
gets the download-only warning. Effective model overrides are inspected, with
known download sizes or an explicit size-unavailable warning. The deliberate
two-net Arena stress step is not a product tournament pairing.

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
with its existing CPU-fallback availability hints.
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

Phone dialogs have no slow-WebGPU sentence or turn-off fine print. Native
inertness, focus restoration, Tab trapping and Escape behavior remain intact.
An aborted or superseded metadata lookup cannot open a stale second dialog.

### Artifact sizes and verified cache presence (v2 copy)

`largeNetDownload.ts` uses the same Apple/WebGPU split-K rule as the worker, then
`inspectLc0ModelDownload()` resolves the loader's pinned manifest/channel artifact.
Download sizes use rounded **decimal MB** from that artifact, including the encoded
Brotli size when the selected channel supplies it. The previous 178 MB BT4 estimate
was actually MiB: the local raw artifact is 186,676,343 bytes, so the real warning
is **187 MB**; Queen Odds is 95,662,716 bytes (**96 MB**). The pinned channel-v2
Brotli representations are about 165 MB and 85 MB respectively. There is no
cellular detection, HEAD/model fetch, inference session or model-byte allocation
in the warning probe. A failed size lookup says the size is unavailable, rather
than inventing an estimate.

“Cached” requires a current matching SHA-256/length/schema/runtime verification
record plus the exact resolved Cache Storage key. Missing/stale metadata,
missing keys or denied storage fail closed to an uncached warning. The probe
reads keys/metadata only; it never reads/hashes a full cached model, writes LRU
metadata or creates a cache bucket. This attests previously verified storage,
not a fresh hash of the body; inference retains its ordinary integrity checks
and repair behavior.

## Verification

- Cache-aware dialog copy, cancellation, Queen Odds downloads, lab overrides and desktop behavior:
  `tests/large_net_consent.test.mjs`
- Read-only verified cache probing, channel representations and device-specific artifacts:
  `tests/large_net_download.test.mjs`
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
**2488 passed, 4 existing optional skips**, including all seven folding goldens
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
- The required phone screenshots were refreshed against the final reviewed build.
  A lab autorun with `steps=lqo` but a BT4 `lqoModel` override shows the ~178 MB
  consent dialog before any step or ballast; Escape leaves the step skipped,
  consent unset and allocated ballast zero (`phone-lab-override-dialog.png`).
- The optional Reckless assets are not staged locally, so adding that Analysis
  row in the earlier browser pass produced an unrelated 404/failure row. No actual iPhone, Safari automation,
  physical-memory measurement, deployment or upload was performed.

Codex closeout used
`~/.pi/agent/skills/autoreview/scripts/autoreview --mode branch --base main --engine codex --model gpt-6.1-sol --thinking xhigh`.
The first attempt failed absolute-path schema validation; subsequent runs added
only repository-relative-path guidance plus JSON output. All **eight** actionable
findings were verified and fixed: reinit resurrection/retry poisoning; retained
Play workers/lab consent; cross-tab revocation/effective lab overrides; hidden
Analysis retirement/unaffected Arena runs. New controlled-worker, actual-pool,
controller and extracted-handler tests cover these paths. None was waived.
`autoreview-6.log/json` is the final same-model whole-branch review: **exit 0,
clean, no accepted/actionable findings**. No redundant review followed.
`npm-test-final.log` and `product-build-final.log` record the final complete suite
and product rebuild; the final source is `008ae26f`. Analysis revocation detaches
gated workers immediately, independent of deferred maintenance; Arena retires
unused gated workers without aborting an unaffected active/starting tournament.

### Release 2 owner copy follow-up (v2)

The v2 rules above supersede the earlier screenshot copy. Full
`LC0_MEMORY_MODEL_DIR=<worktree>/.local-dev-artifacts/release2/models npm test -- --test-concurrency=2`:
**2522 passed, 4 existing optional skips** (`npm-test-v2.log`), with all seven
folding goldens and real Small parity running. Typecheck (zero errors/warnings),
repository lint, targeted Biome checks and the product build pass
(`product-build-v2.log`).

Fresh agent-browser artifacts in scratchpad `mobile-soak/release2/`:

- `phone-analysis-bt4-dialog-v2.png`: 390×844, uncached BT4 crash/download copy.
- `phone-play-queen-odds-uncached-v2.png`: download-only Queen Odds dialog.
- `phone-play-queen-odds-cached-v2.png`: actual verified Cache Storage entry, no
  dialog and no crash opt-in. `cache-proof-v2.json` records the real 95,662,716
  bytes and pinned SHA-256, freshly hashed by `loadLc0ModelForOrt()` without an
  inference session. A temporary local-only bundle of that loader staged the
  cache; it is not an app hook, committed asset or deployment.
- `phone-picker-off-v2.png` / `phone-picker-on-v2.png`: exact shorter hints.
  Turning off with Queen Odds selected preserves that exempt opponent.
- `desktop-play-v2.png`: 1280×900, desktop unchanged with no phone hint/dialog.

Browser checks confirm Cancel/Escape keep the prior picker selection for both
uncached warnings. Cache presence is not simulated with localStorage or a fake
response. No cellular detection, physical-iPhone/Safari automation, push,
deployment or upload was introduced.
