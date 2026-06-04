# Repository cleanup status

This repository is now tracked as `lc0_webgpu` with `main` as the trunk branch.

## Current trunk

- Local trunk: `main`
- Remote trunk: `origin/main`
- Remote URL: `https://github.com/manthedan/lc0_webgpu.git`
- Trunk base: browser LC0/arena UI work through `089a252 Add dual arena eval bars`

## Branch policy

- Keep `lc0-webgpu-pivot` separate. It is an active research branch and should not be modified, merged, rebased, or cleaned up without explicit approval.
- Do not delete stale local branches until their purpose is confirmed.
- Prefer small cleanup commits on `main` so the active research branch can pull trunk changes when it is ready.

## Cleanup queue

1. Remove inherited `tiny_leela` repository wiring from local Git config. Done locally: only `origin` remains.
2. Rename low-risk project metadata from `tiny-leela` to `lc0-webgpu`.
3. Audit browser-facing pages and docs for stale `Tiny Leela` branding, keeping historical research notes stable unless they are clearly obsolete.
4. Audit inherited training/Rust/self-play assets separately before deleting or renaming paths such as `rust/tiny_leela_core` or `tiny_leela_ops`; many scripts still depend on those names.
5. Remove truly unused legacy pages/scripts/tests incrementally with validation after each small batch.
