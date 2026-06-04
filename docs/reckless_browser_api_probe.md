# Reckless browser API facade probe

## Result

A first internal facade probe is now scripted with:

```sh
npm run reckless:probe-browser-api -- startpos 4
```

The script copies the ignored patched Reckless source into `.local_engines/reckless-browser-api-probe`, adds a temporary `lib.rs`, adds a `browser_api` module, builds a native `browser_api_probe` binary, and searches through a direct Rust facade instead of UCI command parsing/stdout capture.

Smoke output from 2026-06-04:

```json
{"bestmove":"c2c4","elapsedMs":2,"lines":[{"multipv":1,"depth":4,"scoreCp":55,"mateIn":null,"nodes":210,"nps":72211,"pv":["c2c4","g8f6"]}]}
```

UCI parity spot-check against the existing native Reckless binary at the same FEN/depth:

```text
info depth 4 seldepth 4 multipv 1 score cp 55 nodes 210 time 15 nps 13352 hashfull 0 tbhits 0 pv c2c4 g8f6
bestmove c2c4
```

The probe therefore confirms that Reckless can be driven through an internal structured API facade and can return the same bestmove, score, nodes, and PV fields without UCI text as the primary data path.

## What this does not do yet

- It does not produce a browser-loadable direct API WASM artifact.
- It does not add a JS worker/adapter for direct calls.
- It does not solve graceful browser cancellation yet.
- It does not replace the existing WASI/UCI path.

## Next API step

Promote the temporary facade into the Reckless build patch path behind an opt-in feature, then add a small exported WASM ABI (`new`, `set_fen`, `search_depth`, `result_json`, `free`) and a browser worker that can benchmark direct persistent API calls against persistent WASI/UCI.
