# Arena large-net memory guard

Safari can kill a phone tab when two large nets are resident. Download sizes do
not describe their runtime memory use. This is a residency policy, not a RAM
measurement or a ban on running one large net.

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
“Two large nets at once can crash this device.” A conflicting URL, restored
roster or mode change remains visible, but cannot start; the start entry also
checks the rule before any asset checks or warmup. The block has no dialog.
Existing asset/WebGPU checks and the single-net download warning still apply.
Queen Odds is covered by the policy without adding it to Arena's existing menu.

On constrained devices, `bigNetSessionPool.ts` has one resident slot. Acquiring a
different net disposes the previous worker **before** constructing its replacement,
even if the previous session was retained by the three-minute idle TTL. Releasing
a net disposes it immediately. Reacquiring the same active net still reuses it.
Desktop retains the existing multi-net pool and idle TTL.

Arena diagnostics and asset checks never acquire searchers. Phone warmup prepares
only the current game's engines, including when advancing through a safe gauntlet.
Desktop keeps the existing whole-roster warmup.

## Verification

- Classifier: `tests/device_memory_class.test.mjs`
- Pairings, start-entry block, warmup routing, passive diagnostics:
  `tests/arena_memory_guard.test.mjs`
- Residency, disposal order, idle TTL and same-net reuse:
  `tests/big_net_session_pool.test.mjs`
- Browser: build with `BUILD_SCOPE=product npm run build:client`, serve with
  `PORT=<free port> npm run web:isolated:static`, compare `/app/arena/` with and
  without `?memoryClass=phone`. No phone or Safari automation is needed.
