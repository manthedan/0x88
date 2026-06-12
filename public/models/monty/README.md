# Monty network assets

Monty is a lab-only browser experiment. Its two raw network files are about
950 MB total, so they are intentionally not tracked and not copied into product
builds.

For local `monty-smoke.html` testing, stage the extracted upstream network files
here (or as local symlinks) using these exact names:

- `nn-09da29a4b6ed.network`
- `nn-6e49a41bd7c0.network`

These filenames are ignored by `.gitignore`; do not commit them.
