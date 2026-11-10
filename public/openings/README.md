# Opening names

`eco.json` is a generated position → `[ECO, English opening name]` lookup from
[Lichess chess-openings](https://github.com/lichess-org/chess-openings), revision
[`3ba6b20d31b2e6125cb0e6a2e792f400717d9d39`](https://github.com/lichess-org/chess-openings/tree/3ba6b20d31b2e6125cb0e6a2e792f400717d9d39).
Source files: `a.tsv` through `e.tsv` (3,865 entries; 3,865 distinct positions).

Licence: **CC0 1.0 Public Domain Dedication**; see the adjacent `COPYING.txt`
(copied unchanged from upstream). No engine-generated or hand-authored names.

Keys are EPDs: FEN without the halfmove/fullmove counters, with an en-passant
square only if a legal en-passant capture exists. Duplicate positions retain
the last source entry in a–e order. Analysis walks backward from the current
ply to the deepest named position, retaining that name after leaving book.
The page fetches this asset asynchronously, not through the main JS bundle.

Regenerate from the pinned upstream revision at the repository root:

```
node --experimental-strip-types scripts/build_opening_names.mjs
```

For an offline copy of the five pinned TSVs, append their directory path.
