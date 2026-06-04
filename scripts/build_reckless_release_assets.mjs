#!/usr/bin/env node
import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { basename, dirname, resolve } from 'node:path';
import { execFileSync } from 'node:child_process';

const repo = process.env.RECKLESS_REPO ?? 'https://github.com/codedeliveryservice/Reckless.git';
const ref = process.env.RECKLESS_REF ?? '0010617448bdef4c8cd7d4f4825b7e42c8bc262a';
const outDir = resolve('public/reckless');

function run(cmd, args, options = {}) {
  console.log(`$ ${cmd} ${args.join(' ')}`);
  execFileSync(cmd, args, { stdio: 'inherit', ...options });
}

function output(cmd, args, options = {}) {
  return execFileSync(cmd, args, { encoding: 'utf8', ...options }).trim();
}

function writeRecipe(workdir, variant) {
  const upstreamCommit = output('git', ['rev-parse', 'HEAD'], { cwd: workdir });
  const upstreamSubject = output('git', ['log', '-1', '--format=%ci %s'], { cwd: workdir });
  const recipe = `# Reckless corresponding source for ${variant.label}

This directory is the patched Reckless source tree used to build ${variant.artifact}.

- Upstream project: ${repo}
- Requested ref: ${ref}
- Resolved upstream commit: ${upstreamCommit}
- Upstream commit summary: ${upstreamSubject}
- Artifact: ${variant.artifact}
- Build script in this project: scripts/build_reckless_wasi.mjs
- Release asset script in this project: scripts/build_reckless_release_assets.mjs

## Rebuild commands

From this repository, run:

\`\`\`sh
${variant.command}
\`\`\`

From this extracted corresponding-source directory, run:

\`\`\`sh
rustup target add wasm32-wasip1
${variant.cargoCommand}
\`\`\`

The repository build script clones Reckless, applies the browser/WASI patches encoded in this repository, and builds a \`wasm32-wasip1\` artifact without Syzygy tablebases. The extracted-source command rebuilds from this already-patched source tree.

## License and attribution

Reckless is an external AGPL-3.0 chess engine by its upstream authors. This project does not claim ownership of Reckless; it only carries browser build scripts and local patches needed to produce the deployed WASM artifacts. If you distribute the WASM artifact, provide this corresponding source and comply with Reckless' AGPL-3.0 license.
`;
  writeFileSync(`${workdir}/CORRESPONDING_SOURCE.md`, recipe);
}

function archiveSource(workdir, archivePath) {
  if (!existsSync(workdir)) throw new Error(`missing source dir: ${workdir}`);
  mkdirSync(dirname(archivePath), { recursive: true });
  run('tar', [
    '-czf', archivePath,
    '--exclude', 'target',
    '--exclude', '.git',
    '-C', dirname(workdir),
    basename(workdir),
  ]);
}

function buildVariant(variant) {
  const workdir = resolve(variant.workdir);
  const artifact = resolve(variant.artifact);
  const env = {
    ...process.env,
    RECKLESS_REPO: repo,
    RECKLESS_REF: ref,
    RECKLESS_BUILD_DIR: workdir,
    RECKLESS_WASM_OUT: artifact,
    ...variant.env,
  };
  run(process.execPath, ['scripts/build_reckless_wasi.mjs'], { env });
  writeRecipe(workdir, variant);
  archiveSource(workdir, resolve(variant.sourceArchive));
}

mkdirSync(outDir, { recursive: true });

buildVariant({
  label: 'Reckless Full scalar fallback',
  artifact: 'public/reckless/reckless.wasm',
  sourceArchive: 'public/reckless/reckless-scalar-corresponding-source.tar.gz',
  workdir: '.local_engines/reckless-wasi-src',
  command: 'npm run reckless:build-wasi',
  cargoCommand: 'cargo build --release --no-default-features --target wasm32-wasip1',
  env: {},
});

buildVariant({
  label: 'Reckless Full SIMD',
  artifact: 'public/reckless/reckless-simd128.wasm',
  sourceArchive: 'public/reckless/reckless-simd128-corresponding-source.tar.gz',
  workdir: '.local_engines/reckless-release-src-simd',
  command: 'npm run reckless:build-simd-wasi',
  cargoCommand: "RUSTFLAGS='-C target-feature=+simd128' cargo build --release --no-default-features --target wasm32-wasip1",
  env: {
    RUSTFLAGS: `${process.env.RUSTFLAGS ? `${process.env.RUSTFLAGS} ` : ''}-C target-feature=+simd128`,
    RECKLESS_WASM_SIMD_NNUE: '1',
  },
});

console.log('Reckless release assets ready:');
console.log('  public/reckless/reckless.wasm');
console.log('  public/reckless/reckless-simd128.wasm');
console.log('  public/reckless/reckless-scalar-corresponding-source.tar.gz');
console.log('  public/reckless/reckless-simd128-corresponding-source.tar.gz');
