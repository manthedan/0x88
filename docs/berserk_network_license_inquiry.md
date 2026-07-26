# Berserk network licence inquiry — draft

Status: **draft, not yet sent.**
Target: an issue on `https://github.com/jhonnold/berserk-networks` (the networks
repo, not the engine repo — that is where the missing licence would live).
Context: `docs/engine_artifact_distribution.md` Berserk card, and
`docs/runtime_efficiency_and_release_readiness_audit_2026-07-25.md` §2.2.

Suggested title: **Licence for the released networks?**

---

Hi — first, thanks for Berserk. We ported it to WebAssembly and it runs really
well in the browser.

I'm asking about the licensing of the network files released here, because I
want to make sure we're not redistributing something we shouldn't be.

**Context.** [0x88](https://0x88.app) is an open-source, GPL-3.0-or-later
browser chess lab: several engines compiled to WebAssembly, running entirely
client-side, so people can play and analyse with them without installing
anything. Berserk is one of them — built from tag `14`
(`8ae895a6151695be4a50d4fb65b0c131659c513a`) via Emscripten, with the network
`berserk-9b84c340af7e.nn` preloaded into the module.

**The question.** The engine repo is clearly GPL-3.0, but this networks repo
doesn't carry a licence file, so I can't tell whether the nets are meant to be
covered by the engine's licence or whether they have their own terms.

Concretely: **may the network be redistributed alongside a compiled Berserk
binary** — in our case embedded in an Emscripten `.data` preload package that
we'd host and serve to users?

A "yes, they're GPL-3.0 like the engine" (or any other answer) is all I need.

**What we've done in the meantime.** Rather than guess, we've stopped
redistributing it. We don't ship the network, the preload package, or the
compiled artifacts; our build script fetches the net from your release page at
build time and verifies its SHA-256, so anyone who wants Berserk builds it
locally and gets the file from you rather than from us. That's a bit worse for
our users, so I'd be glad to change it — but I'd rather be over-cautious than
redistribute weights whose terms I've guessed at.

**One suggestion, unrelated to us.** Adding a `LICENSE` file to this repo (or a
line in the README) would settle it permanently for anyone else packaging
Berserk — distro maintainers, GUI bundlers, other web ports. It's the kind of
thing that quietly saves a lot of people this exact question.

Happy to provide any detail about how we're building or distributing it. And if
the answer is "please don't redistribute the nets", that's completely fine —
we'll leave things as they are.

Thanks again for the engine.

---

## If the answer is yes

Reverting is deliberately a small, contained change — the re-enable path was
left documented rather than deleted:

1. `src/lc0/berserkVariants.ts` — re-list the shipped paths in
   `DEPLOYED_BERSERK_PATHS` (kept as an explicit empty set with a comment
   saying exactly this), and drop `berserkArtifactsUnavailable` /
   `BERSERK_ARTIFACT_BUILD_HINT` usage from the four construction sites.
2. `.gitignore` / `.gitattributes` — restore the Berserk artifact tracking and
   LFS rules.
3. `scripts/r2_brotli_publish_assets.mjs` — restore the Berserk publish targets
   (one canonical `.data`, not one per SIMD tier).
4. `scripts/write_artifact_release_manifests.mjs` — restore the Berserk
   manifest to `DEFAULT_SOURCE_MANIFESTS`.
5. `public/artifact-index.json` and
   `public/berserk/berserk-emscripten-single-thread.manifest.json` — restore
   `deploymentStatus`, `primaryUrls`, artifacts, totals, and `sourceArchive`.
6. `NOTICE.md`, `README.md`, `docs/engine_artifact_distribution.md` — record the
   resolution and the answer received.

Record the author's answer verbatim in `docs/engine_artifact_distribution.md`
with a date and a link to the issue, so the provenance is auditable rather than
resting on someone's memory of a conversation.

## If the answer is no, or there is no answer

Leave it exactly as it is. Berserk stays buildable locally and undistributed,
which is already the shipped state. Consider noting in `NOTICE.md` that the
question was asked and when, so the next person doesn't have to re-derive it.
