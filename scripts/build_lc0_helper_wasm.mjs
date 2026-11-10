#!/usr/bin/env node
// Rebuilds the small Rust helper modules shipped as committed binaries in public/lc0/:
//
//   npm run lc0:check-helper-wasm   rebuild and report whether each binary is byte-identical
//                                   to the committed one (exit 1 on a mismatch; writes nothing)
//   npm run lc0:build-helper-wasm   rebuild and copy the result over the committed binary
//
// The recipe is the one these binaries were first built with (the original
// lc0:wasm-*:build scripts from 5e5d52b / ec27e6c): wasm32-wasip1, +simd128, the
// crates' release profile. Needs cargo and `rustup target add wasm32-wasip1`.
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { copyFileSync, existsSync, readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { parseArgs } from 'node:util';

const TARGET = 'wasm32-wasip1';
const RUSTFLAGS = '-C target-feature=+simd128';
const HELPER_WASM = [
  { crate: 'wasm/lc0_input_encoder', output: 'lc0_input_encoder_wasm.wasm', artifact: 'public/lc0/lc0_input_encoder.wasm' },
  { crate: 'wasm/lc0_legal_priors', output: 'lc0_legal_priors_wasm.wasm', artifact: 'public/lc0/lc0_legal_priors.wasm' },
];

const USAGE = `Usage: node scripts/build_lc0_helper_wasm.mjs (--check | --write) [--target-dir DIR]

  --check            Rebuild and compare with the committed binaries; exit 1 if any differ.
  --write            Rebuild and replace the committed binaries.
  --target-dir DIR   Cargo target directory (default: each crate's ignored target/).`;

function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

function cargo(args, env = {}) {
  return spawnSync('cargo', args, { stdio: 'inherit', env: { ...process.env, ...env } });
}

function main() {
  const { values } = parseArgs({
    options: { check: { type: 'boolean' }, write: { type: 'boolean' }, 'target-dir': { type: 'string' }, help: { type: 'boolean' } },
  });
  if (values.help || values.check === values.write) {
    console.log(USAGE);
    process.exit(values.help ? 0 : 2);
  }
  const probe = spawnSync('cargo', ['--version'], { encoding: 'utf8' });
  if (probe.error || probe.status !== 0) {
    console.error(`cargo is not available; install Rust and run \`rustup target add ${TARGET}\`.`);
    process.exit(2);
  }

  let mismatches = 0;
  for (const { crate, output, artifact } of HELPER_WASM) {
    const targetDir = values['target-dir'] ? resolve(values['target-dir'], crate.replace(/\W+/g, '-')) : resolve(crate, 'target');
    const manifest = join(crate, 'Cargo.toml');
    const build = cargo(['build', '--locked', '--release', '--target', TARGET, '--manifest-path', manifest, '--target-dir', targetDir], {
      RUSTFLAGS,
    });
    if (build.error || build.status !== 0) {
      console.error(`${crate}: cargo build failed (is the ${TARGET} target installed? \`rustup target add ${TARGET}\`)`);
      process.exit(1);
    }
    const built = join(targetDir, TARGET, 'release', output);
    const rebuilt = readFileSync(built);
    const committed = existsSync(artifact) ? readFileSync(artifact) : null;
    const identical = committed !== null && Buffer.compare(rebuilt, committed) === 0;
    console.log(
      `${identical ? 'identical' : 'DIFFERS  '} ${artifact}  rebuilt ${rebuilt.length} B ${sha256(rebuilt).slice(0, 16)}  committed ${
        committed ? `${committed.length} B ${sha256(committed).slice(0, 16)}` : 'missing'
      }`,
    );
    if (values.write) {
      if (!identical) copyFileSync(built, artifact);
    } else if (!identical) {
      mismatches++;
    }
  }
  if (mismatches) {
    console.error(
      `${mismatches} helper wasm binar${mismatches === 1 ? 'y does' : 'ies do'} not match a rebuild from source. Run \`npm run lc0:build-helper-wasm\` to replace.`,
    );
    process.exit(1);
  }
}

main();
