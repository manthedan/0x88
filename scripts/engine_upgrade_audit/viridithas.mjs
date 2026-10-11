import { runNativeBuiltin, runWasiBench } from './bench.mjs';
import { wasiTiers } from './tiers.mjs';

export default {
  tiers: wasiTiers,
  native: runNativeBuiltin,
  bench: runWasiBench,
  nativeBuild: "RUSTFLAGS='-C target-cpu=native' cargo build --release --no-default-features (locked source with required net; not run automatically)",
  parity: null, // Existing Viridithas smoke is not a rotated-FEN parity harness.
};
