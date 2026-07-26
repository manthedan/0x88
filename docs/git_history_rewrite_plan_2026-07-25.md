# Git history rewrite plan — 2026-07-25

Prepared on a throwaway clone. **The working repository at
`/Users/macthedan/projects/lc0_browser/leelaweb` was never written to** — it was
cloned with `git clone --no-local` (no hardlinks, no shared object store, no
`alternates` file), and its head/tree/refs were re-verified as unchanged after
the rewrite completed.

Clone used for this work:
`/private/tmp/claude-501/-Users-macthedan-projects-lc0-browser/97470469-5cf0-48cf-a299-db060c846c57/scratchpad/rewrite-clone`

---

## 0. Headline finding — read this before doing anything

**The two 107 MB Stockfish blobs are not on `main`. They never were.**

They were added in `8b82ee8` and removed in `5a39ddf`, but that add/remove pair
lives entirely on the unmerged branch `stockfish-relaxed-full-threaded-variants`.
`git merge-base --is-ancestor 8b82ee8 main` returns false.

Consequence: a contributor who clones the published repo **already never
downloads them**, provided that branch is not published.

| clone shape | original | rewritten | saved |
|---|---:|---:|---:|
| all refs (`git clone --mirror`) | 277.9 MB | 197.3 MB | **80.6 MB (29%)** |
| `main` only (`--single-branch`) | 198.8 MB | 197.3 MB | 1.5 MB (0.7%) |

So there are two ways to get the same 80 MB:

- **Option A (no rewrite):** publish only `main`, or delete the
  `stockfish-relaxed-full-threaded-variants` branch before publishing. Zero risk,
  zero hash churn, nobody has to re-clone.
- **Option B (rewrite, done here):** purge the blobs from all history so the
  branch can be published safely and the local `.git` shrinks too.

**Option A is recommended** unless that branch specifically needs to be public.
The rewrite is real and verified, but it buys only 1.5 MB on a `main`-only
publish and costs every commit hash in the repo.

---

## 1. Inventory

Measured across **all refs** (7 branches + tag `v0.0.1`), 1029 commits on `main`.

Totals: 5017 blobs, 630.0 MB logical, 277.2 MB packed.
Live at `main`: 806 blobs / 255.6 MB logical / 193.5 MB packed.
Dead at every ref tip: 3513 blobs / 350.7 MB logical / **82.0 MB packed**.

Note the gap between *logical* and *packed*. Logical size is what alarms you in a
`git verify-pack` report; **packed size is what a contributor actually
downloads**. Several of the "huge" historical files delta-compress to almost
nothing and are not worth chasing.

### 1a. Blobs >1 MB in history

| logical | packed | live at `main`? | live at any tip? | path |
|---:|---:|:--:|:--:|---|
| 112.96 MB | 76.87 MB | dead | dead | `public/stockfish/stockfish-18-single-relaxed.wasm` |
| 112.88 MB | 2.05 MB | dead | dead | `public/stockfish/stockfish-18-relaxed.wasm` |
| 44.57 MB | 44.59 MB | **LIVE** | LIVE | `public/reckless/reckless-simd128-external-corresponding-source.tar.gz` |
| 44.57 MB | 44.59 MB | **LIVE** | LIVE | `public/reckless/reckless-relaxed-simd128-corresponding-source.tar.gz` |
| 44.57 MB | 44.59 MB | **LIVE** | LIVE | `public/reckless/reckless-simd128-corresponding-source.tar.gz` |
| 44.57 MB | 44.58 MB | **LIVE** | LIVE | `public/reckless/reckless-scalar-corresponding-source.tar.gz` |
| 26.88 MB | 6.54 MB | **LIVE** | LIVE | `public/ort-experimental/relaxed/ort-wasm-relaxedsimd-threaded.asyncify.wasm` |
| 26.86 MB | 0.88 MB | **LIVE** | LIVE | `public/ort-experimental/fixed/ort-wasm-simd-threaded.asyncify.wasm` |
| 7.30 MB | 5.64 MB | **LIVE** | LIVE | `public/stockfish/stockfish-18-lite-single.wasm` |
| 7.29 MB | 0.18 MB | **LIVE** | LIVE | `public/stockfish/stockfish-18-lite-single-relaxed.wasm` |
| 7.09 MB | 0.14 MB | dead | dead | `public/stockfish/stockfish-18-lite-relaxed.wasm` |
| 3.52 MB | 0.14 MB | dead | LIVE | `docs/reckless_browser_benchmark_2026-06-04_..._api_simd_corrected.json` |
| 3.52 MB | 0.00 MB | dead | dead | (duplicate of the above, different blob) |
| 2.33 MB | 0.12 MB | dead | LIVE | `docs/viridithas_reckless_browser_benchmark_2026-06-04_....json` |
| 2.01 MB | 1.50 MB | dead | dead | `Nibbler.standalone.html` |
| 1.85 MB | 0.09 MB | dead | LIVE | `docs/reckless_browser_benchmark_2026-06-04_..._api_simd.json` |
| 1.85 MB | 0.07 MB | dead | LIVE | `docs/reckless_external_nnue_benchmark_2026-06-04_....json` |

Two corrections to the assumptions in the original brief:

- `public/ort-experimental/{fixed,relaxed}/*.asyncify.wasm` are **live in the
  current tree**, not dead history. They cannot be stripped. They are also far
  cheaper than they look (53.7 MB logical → 7.4 MB packed).
- The four `public/reckless/*corresponding-source*.tar.gz` archives are **live in
  the current tree** as well. See §2c.

### 1b. Git LFS (separate from pack size)

`.gitattributes` routes `public/{viridithas,stormphrax,plentychess,berserk}/*`,
the TVM runtimes, the two BT4 `.onnx` nets, and the *Stockfish/Viridithas/
Stormphrax/PlentyChess* corresponding-source archives to LFS. LFS pointers in
history are ~130 bytes each, so **they contribute nothing to pack size**; the
weight is the 750 MB in `.git/lfs`.

196 LFS pointer blobs in history resolve to:

| class | count | size |
|---|---:|---:|
| live at `main` | 186 | **528.0 MB** |
| live only at a non-`main` tip | 3 | 5.2 MB |
| history-only (dead) | 7 | **151.1 MB** |

The 151.1 MB of dead LFS is dominated by four superseded revisions of
`public/stormphrax/stormphrax-emscripten-single-thread-corresponding-source.tar.gz`
(37.3 MB each). This is reclaimable **locally** with `git lfs prune` and does not
require a history rewrite. It is also never uploaded when publishing, because a
push only sends LFS objects reachable from the pushed refs.

Crucially: `public/reckless/*corresponding-source*.tar.gz` is **not** LFS-tracked
(absent from `.gitattributes`), which is why those four archives sit raw in the
pack and account for 178 MB of the 197 MB published pack.

---

## 2. What was stripped, and why

Only blobs dead at **every** ref tip *and* dead in every ref's history were
targeted, so no branch loses content.

### 2a. Stripped

| path | packed | justification |
|---|---:|---|
| `public/stockfish/stockfish-18-single-relaxed.wasm` | 76.87 MB | Build output. Regenerated by `scripts/build_stockfish_relaxed_simd.mjs` (`outputBase: 'stockfish-18-single-relaxed'`). Superseded by the lite-single variant kept at `main`. Confined to one unmerged branch. |
| `public/stockfish/stockfish-18-relaxed.wasm` | 2.05 MB | Same script, `outputBase: 'stockfish-18-relaxed'`. Same commit pair. |
| `public/stockfish/stockfish-18-lite-relaxed.wasm` | 0.14 MB | Same script, `outputBase: 'stockfish-18-lite-relaxed'`. Dropped by `5a39ddf` in favour of `stockfish-18-lite-single-relaxed.wasm`, which stays. |
| `Nibbler.standalone.html` | 1.50 MB | Third-party GUI artifact committed by accident; now listed in `.gitignore:67` at `main`. |

Total reclaimed: **80.6 MB packed**.

All three `.wasm` files are reproducible build outputs of a script that is still
in the tree, so nothing unrecoverable is lost. Only `Nibbler.standalone.html`
touches `main`'s history — the rest are branch-only.

### 2b. Deliberately NOT stripped — still live in the current tree

- `public/ort-experimental/{fixed,relaxed}/*.asyncify.wasm` — live at `main`.
  Removing them breaks the build. Only 7.4 MB packed anyway.
- `public/stockfish/stockfish-18-lite-single.wasm`,
  `stockfish-18-lite-single-relaxed.wasm` — live at `main`.
- All 528 MB of LFS objects reachable from `main`.

### 2c. Deliberately NOT stripped — GPL corresponding-source (flagged, needs your call)

The four `public/reckless/*corresponding-source*.tar.gz` archives (178 MB packed,
**90% of the published pack**) were checked against the brief's instruction to
flag rather than silently drop anything licence-related. They are **live at
`main` and load-bearing**, referenced by:

- `public/reckless/reckless-wasip1.manifest.json` (four `path`/`url` entries)
- `package.json:126` (`reckless:release-manifest` passes
  `--source-archive public/reckless/reckless-scalar-corresponding-source.tar.gz`)
- `public/_headers:82`, `public/reckless/README.md`, `public/reckless/NOTICE.md`
- `docs/hosted_artifacts.md`, `public/releases/2026-06-23.*.json`

Reckless is GPL-licensed, and shipping its WASM binaries obliges you to offer the
corresponding source. **These must not be stripped.** They are the single
biggest remaining cost and the only real lever left on clone size, but pulling
that lever is a licensing decision, not a cleanup decision.

Note the inconsistency worth resolving separately: `.gitignore:37` lists
`public/reckless/*corresponding-source*.tar.gz` and
`public/reckless/README.md` says these archives are "intentionally not
committed" — yet all four are committed. Either the ignore rule or the commits
are wrong. If you decide the archives belong in release assets rather than git
(the sibling engines already route theirs through LFS), that is a **separate,
tree-changing** change: make it as a normal commit first, then rewrite history.
It is out of scope here because this rewrite guarantees a byte-identical tree.

### 2d. Deliberately NOT stripped — poor value

The `docs/*benchmark*.json` files total 13.1 MB logical but only **0.4 MB
packed** (they delta-compress against each other almost perfectly). Four of the
five are still live at a non-`main` branch tip, so stripping them would alter
those branches' trees. 0.4 MB is not worth that. Left alone.

---

## 3. Verification evidence

### 3a. The `main` tree is byte-identical — proven by tree hash

```
before: git rev-parse main^{tree}  ->  9f763b1bcb7c32144ca3d12049953e7237377499
after:  git rev-parse main^{tree}  ->  9f763b1bcb7c32144ca3d12049953e7237377499
```

A git tree hash is a recursive Merkle hash over every entry's mode, name, and
blob OID. Identical tree hashes is the strongest available proof: not one byte,
mode bit, or filename changed anywhere in the tree, **including LFS pointer file
contents** (pointers are ordinary blobs and are covered by the hash).

Corroborated by a full recursive listing diff:

```
$ git ls-tree -r main   # 1003 entries, before and after
$ diff before/main_lstree.txt after/main_lstree.txt
(no output — identical)
$ shasum -a256 before/main_lstree.txt after/main_lstree.txt
6262b83539a657ef0e2126e025d397501922e66e3c7b191cc551e2dacd8451a8  before
6262b83539a657ef0e2126e025d397501922e66e3c7b191cc551e2dacd8451a8  after
```

### 3b. Topology and metadata preserved

- Commit count on `main`: **1029 before, 1029 after**.
- Author name/email, author date, committer name/email, committer date, and
  subject compared for all 1029 commits: **one** difference, explained below.

The single delta is a merge commit whose *message text* embedded an old short
hash:

```
- Merge commit '3689802'
+ Merge commit '71f054c'
```

`git-filter-repo` rewrites commit hashes mentioned inside commit messages so they
keep pointing at the right commit. Verified against `.git/filter-repo/commit-map`:

```
368980272252be5c77c59b0e2c57842a6d5508fd -> 71f054ce1442100a6baf29367575e36b5f26e8d3
$ git log -1 --format='%h %s' 71f054c
71f054c Document Stockfish relaxed SIMD speedup
```

The new hash is the rewritten counterpart of the exact commit the message
referred to, so the reference stays valid. This is a correctness improvement; use
`--preserve-commit-hashes` if you would rather keep the stale text.

### 3c. Purge is complete and the repo is sound

```
$ git rev-list --objects --all | grep -E 'stockfish-18-(single-|lite-)?relaxed\.wasm|Nibbler\.standalone\.html'
(no matches)

$ git fsck --no-dangling
(clean, no output)
```

### 3d. No LFS pointer became dangling

Re-running the LFS census on the rewritten repo returns **exactly** the pre-rewrite
figures: 196 pointer blobs, **186 distinct LFS objects live at `main` totalling
528.0 MB**, 3 at other tips, 7 history-only. None of the four stripped paths was
LFS-tracked (each stripped blob carried its full multi-MB payload inline, not a
130-byte pointer), so no pointer was touched and no `.git/lfs` object lost its
referrer. This also follows from §3a: identical tree hash implies an identical
set of pointer blobs.

### 3e. The source repository is untouched

Re-checked after the rewrite finished:

```
head    = 8709665848b9437c0aa68fe6e64b3a2aaf85ca1e
tree    = 9f763b1bcb7c32144ca3d12049953e7237377499
commits = 1029
.git = 1.0G   .git/objects = 278M   .git/lfs = 750M
refs: identical to the baseline captured before the rewrite
```

---

## 4. Size before / after

Measured as summed `*.pack` bytes on freshly created clones, which is what a
contributor actually transfers.

| measurement | before | after |
|---|---:|---:|
| all-refs clone (pack) | 277.9 MB | **197.3 MB** |
| `main`-only clone (pack) | 198.8 MB | **197.3 MB** |
| local `.git/objects` (all refs) | 278 MB | 189 MB |
| `.git/lfs` locally | 750 MB | 599 MB after `git lfs prune` |

### Cost for a community contributor

`git clone` of the published repo, `main` only:

| component | size | reducible? |
|---|---:|---|
| pack | 197.3 MB | 178 MB of it is the Reckless GPL archives (§2c) |
| LFS objects for the `main` checkout | 528.0 MB | not without changing the tree |
| **total** | **~725 MB** | |

`GIT_LFS_SKIP_SMUDGE=1 git clone` drops that to ~197 MB for anyone who only wants
to read or patch source and does not need the engine binaries — worth putting in
the README.

The history rewrite moves this total by 1.5 MB. **The clone cost is dominated by
files that are live in the tree, not by dead history.** If shrinking the clone is
the actual goal, the lever is §2c (move the 178 MB of Reckless archives to
release assets or LFS) and the LFS payload — not this rewrite.

---

## 5. Reproducing the rewrite

```sh
# 1. Clone WITHOUT hardlinks. --no-local is mandatory: a plain local clone
#    hardlinks .git/objects, and a rewrite in the clone would corrupt the original.
GIT_LFS_SKIP_SMUDGE=1 git clone --no-local \
  file:///Users/macthedan/projects/lc0_browser/leelaweb rewrite-clone
cd rewrite-clone

# 2. Recreate the local branches you want rewritten (a clone only makes `main`).
for b in audit/tiny-origin-deletion-targets fe-ux-polish \
         feature/game-library-review feature/lc0-wasm-relaxed-simd-inference \
         stockfish-relaxed-full-threaded-variants wip/pre-registry-centipawn-play; do
  git branch --track "$b" "origin/$b"
done

# 3. Baseline for verification.
git rev-parse main^{tree} | tee /tmp/tree.before
git ls-tree -r main > /tmp/lstree.before
git rev-list --count main > /tmp/count.before

# 4. Rewrite. git-filter-repo 2.47.0 (pip install git-filter-repo).
git filter-repo --force --invert-paths \
  --path public/stockfish/stockfish-18-relaxed.wasm \
  --path public/stockfish/stockfish-18-single-relaxed.wasm \
  --path public/stockfish/stockfish-18-lite-relaxed.wasm \
  --path Nibbler.standalone.html

# 5. Verify — all three must be clean.
git rev-parse main^{tree} | diff - /tmp/tree.before      # identical tree
git ls-tree -r main       | diff - /tmp/lstree.before    # identical listing
git rev-list --count main | diff - /tmp/count.before     # 1029
git fsck --no-dangling

# 6. Optional: reclaim the 151 MB of history-only LFS objects locally.
git lfs prune
```

The equivalent no-rewrite path (Option A, recommended):

```sh
git push public main            # publish main only; the 107 MB blobs are not on it
# or, if pushing everything:
git branch -D stockfish-relaxed-full-threaded-variants
```

---

## 6. Consequences you must accept before applying Option B for real

1. **Every commit hash changes.** `main` moves
   `8709665…` → `88bbe48…`. All 1029 commits get new SHAs.
2. **Every existing clone and fork must be re-cloned.** `git pull` will not
   reconcile the histories; it will try to merge them and produce a duplicated
   tree. Anyone with a working copy must re-clone, or rebase their work onto the
   new history by hand.
3. **All seven branches need the same treatment or must be abandoned.** They were
   all rewritten together here. If you rewrite `main` alone and later push an
   un-rewritten branch, the old objects — including both 107 MB blobs — come
   straight back into the published repo. The brief mentioned a
   `lc0-webgpu-pivot` branch; **no such branch exists locally.** If it exists on
   a remote, it must be fetched and rewritten in the same `filter-repo` run, or
   dropped.
4. **The tag `v0.0.1` is re-pointed.** `filter-repo` rewrites it automatically,
   but if it was ever published, consumers pinned to the old tag object break.
   Delete and re-push the tag on the remote.
5. **Any published permalink, PR reference, issue cross-link, changelog entry, or
   `git blame` URL that names an old SHA becomes dead.** In-repo commit-message
   references were remapped (§3b); external ones cannot be.
6. **Force-push is required** and will be rejected by branch protection until it
   is temporarily lifted. Any CI keyed to commit SHAs re-runs from scratch.
7. **`filter-repo` removed the `origin` remote** in the rewritten clone, by
   design, to stop an accidental push back to the source. You must add the public
   remote explicitly.
8. **Reflog and old objects are expired** in the rewritten clone. Keep the
   original repository as the rollback path until you are satisfied — this plan
   never deletes it.

Given §0, consequences 1–6 buy 1.5 MB on a `main`-only publish. Take Option A
unless `stockfish-relaxed-full-threaded-variants` must be public.

---

## 7. Recommendation

1. **Publish `main` only** (Option A). No rewrite, no hash churn, no re-clones.
   Delete or leave unpublished `stockfish-relaxed-full-threaded-variants`.
2. Run `git lfs prune` locally to reclaim 151 MB. Safe, local, reversible.
3. Document `GIT_LFS_SKIP_SMUDGE=1 git clone` in the README for source-only
   contributors (725 MB → 197 MB).
4. Resolve the Reckless archive question (§2c) as a normal tree-changing commit.
   That is where the remaining 178 MB lives, and it is a licensing decision.
5. Keep Option B in reserve for if that branch ever needs to be public.
