# LC0 search-parity strictness

Decision on how strictly the browser PUCT search is held to native LC0 search,
covering roadmap items 5 (higher-visit fixtures) and 6 (parity strictness).

## What we assert

The browser search is validated against native LC0 BLAS fixtures
(`fixtures/lc0/native_search_*_blas_nodes{32,64,128}.jsonl`, generated with
`scripts/lc0_native_fixture_priors.py` from the local `lc0-release-0.32` binary
and the `t1-256x10-distilled-swa` net) at three levels of strictness:

1. **Best move (strict).** The browser's selected root move must equal native
   LC0's `bestmove`, and it must also be the top child by visits. This holds
   across the FEN-only and explicit-history suites at 32, 64, and 128 visits.
2. **Root prior consistency (soft).** The browser's normalized root priors for
   native's top children must match within `< 0.0035` absolute. This validates
   the policy-map + softmax-temperature path, not the search dynamics.
3. **Completed visit budget (exact).** Fixed-visit budgets complete exactly.

## What we deliberately do NOT assert

**Exact per-move visit distributions.** Native LC0's visit counts depend on
internal details we do not replicate bit-for-bit: cpuct schedule constants,
FPU-reduction specifics, first-play-urgency edge cases, tie-breaking order,
collision/virtual-loss handling under batching, and root-move ordering. Holding
the browser to native's exact `N:` per child would be brittle — a single
off-by-one visit on a near-tie would fail the suite without indicating any real
defect. The browser PUCT uses LC0-shaped (`lc0-log` cpuct, `lc0-reduction` fpu)
but not LC0-identical internals.

## Rationale

Best-move parity is the property that matters for a playable engine and is
stable across 32/64/128 visits and both suites today. Prior consistency guards
the evaluator/policy path. Visit-distribution equality is a stronger claim than
the product needs and would couple the test suite to LC0 search internals we
have chosen not to mirror. If exact-distribution parity is ever required, it
should be pursued by porting the specific LC0 internals (cpuct/fpu constants,
tie-breaks, batching collisions) one at a time, each behind its own fixture.

## Regenerating the higher-visit fixtures

```
cd leelaweb
python3 ../scripts/lc0_native_fixture_priors.py \
  --fixtures fixtures/lc0/fen_only.json --nodes 64 \
  --out fixtures/lc0/native_search_fen_only_blas_nodes64.jsonl
# ...and the history suite, and --nodes 128.
```
