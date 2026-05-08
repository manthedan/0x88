# Search auxiliary-head calibration

When augmenting PUCT with extra model heads such as action value (AV), regret, risk, or uncertainty, treat the extra head as a separately calibrated search prior, not as a plug-and-play replacement for tree search.

Classic PUCT combines policy prior, backed-up value, and visit counts. Extra heads usually have different target distributions, different error modes, and different coverage from the policy/value heads. For example, the current ChessBench AV targets were built from candidate move lists and can be out-of-distribution for unscored legal moves. Therefore any term like:

```text
PUCT score + aux_weight * aux_head(move, position)
```

must be calibrated before it is used as a default search mode.

## Required calibration gates

For each auxiliary-head search policy:

1. Verify perspective/sign conventions with tactical sanity checks, especially mates for both sides to move.
2. Measure head quality on held-out candidate sets: MSE/CE, top-k agreement, regret ordering, and calibration by value bucket.
3. Sweep the auxiliary weight against classic PUCT under a fixed protocol.
4. Compare against policy-only and classic PUCT, not just against other auxiliary weights.
5. Track failure cases where the auxiliary term promotes low-prior or out-of-distribution legal moves.
6. Keep classic PUCT as the default until the auxiliary mode wins across multiple protocols.

## Visit-budget dependence

Calibration is visit-budget dependent.

An auxiliary term that helps at 32 visits can hurt at 128 or 512 visits because the relative importance of priors, initial value estimates, and backed-up search statistics changes with tree depth. Low-visit search is dominated by priors and one-step estimates; higher-visit search increasingly trusts backed-up Q. Therefore calibrated settings should be recorded per visit budget, for example:

```text
visits=32:   avWeight ~= 0.10-0.15 candidate
visits=128:  must be re-swept
visits=512:  must be re-swept
```

Do not promote one global `avWeight` from a single visit setting. At minimum, sweep each deployment-relevant visit budget and preserve separate defaults or a schedule.

## Current AV-PUCT note

Initial small sweeps suggest lower AV weights are better than the original `0.25` default. On 32-visit smoke arenas, `64x6 phase2` preferred roughly `0.15`, while `80x5 phase3` preferred roughly `0.10`. These are provisional because the protocol was small and the AV cache was top-8 limited.

The next data-side calibration step is to rebuild ChessBench AV supervision with wider candidate coverage, e.g. C=48, so the AV head sees nearly all legal moves instead of only the teacher top-8.
