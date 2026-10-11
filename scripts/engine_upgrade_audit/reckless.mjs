import { runNativeBuiltin, runWasiBench } from './bench.mjs';
import { wasiTiers } from './tiers.mjs';

export default {
  tiers: wasiTiers,
  native: runNativeBuiltin,
  bench: (context, tier) => tier.api ? { skip: 'browser API facade has no _start/full bench entry point; WASI tiers are checked separately' } : runWasiBench(context, tier),
  nativeBuild: "RUSTFLAGS='-C target-cpu=native' cargo build --release --no-default-features (locked source; not run automatically)",
  parity: 'reckless',
};
