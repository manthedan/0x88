# Centipawn v1 Move Encoding

This project uses a deliberately simple fixed action space before adopting any legacy lc0 policy layout.

- Squares are indexed `a1=0 ... h1=7, a2=8 ... h8=63`.
- UCI moves are canonical for tests and fixtures, e.g. `e2e4`, `a7a8q`.
- Action id:

```text
action = (from * 64 + to) * 5 + promotion
promotion: none=0, n=1, b=2, r=3, q=4
```

`ACTION_SPACE = 64 * 64 * 5 = 20480`.

This is larger than lc0's optimized gathered policy, but it is unambiguous and stable across training, export, and browser inference. A later policy-compression lane may replace it only under a new benchmark/fixture version.
