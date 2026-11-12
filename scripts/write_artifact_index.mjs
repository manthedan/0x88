#!/usr/bin/env node
// Regenerates public/artifact-index.json from the committed per-artifact
// manifests, so the compact index cannot drift from them:
//   - browser engines: every public/<engine>/*.manifest.json (engine manifest schema)
//   - Lc0, Centipawn and Maia 3 models: the model manifests under public/models/
//   - the promoted Centipawn TVMJS runtime: its bundle manifest
// Only labels, statuses and short notes are curated here. Deployment status is
// derived from tracked paths and repository .gitignore rules, so it says what a clean checkout contains.
// Usage: node scripts/write_artifact_index.mjs [--check]
import { spawnSync } from 'node:child_process';
import { readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, posix } from 'node:path';
import { fileURLToPath } from 'node:url';

export const ARTIFACT_INDEX_PATH = 'public/artifact-index.json';
const ENGINE_MANIFEST_SCHEMA = 'lc0-webgpu.browser-engine-artifact-manifest.v1';

// Curated per-engine fields, in index order. Every engine manifest under public/
// must have a row here, and every row must have a manifest.
const ENGINES = [
  {
    engine: 'stockfish',
    uiLabel: 'Stockfish',
    notes:
      'Lite single-thread is the default Stockfish opponent; threaded variants require cross-origin isolation. The full and pthread baseline pairs are symlinks into node_modules/stockfish (restored by npm ci); the lite single-thread and relaxed-SIMD pairs are committed files.',
  },
  {
    engine: 'viridithas',
    uiLabel: 'Viridithas',
    notes: 'Patched wasm32-wasip1 scalar, SIMD and relaxed-SIMD builds with the network embedded in each binary.',
  },
  {
    engine: 'berserk',
    uiLabel: 'Berserk',
    notes:
      'Single-thread Emscripten builds (scalar, SIMD, relaxed SIMD) sharing one preload .data that holds the network. Network licensing reasoning: docs/engine_artifact_distribution.md.',
  },
  {
    engine: 'plentychess',
    uiLabel: 'PlentyChess',
    notes: 'Single-thread Emscripten builds (baseline, SSE4.1, relaxed SIMD); the processed network is preloaded in the shared .data sidecar.',
  },
  {
    engine: 'stormphrax',
    uiLabel: 'Stormphrax',
    notes: 'Single-thread Emscripten builds (WebAssembly SIMD and relaxed SIMD); undertown.nnue is preloaded in the shared .data sidecar.',
  },
  {
    engine: 'reckless',
    uiLabel: 'Reckless',
    notes:
      'The WASM binaries and extracted NNUE are gitignored and built locally by npm run reckless:build-release, which also writes the release manifest. The manifest and all four corresponding-source archives are committed. Variant ladder: docs/hosted_artifacts.md.',
  },
];

// Curated model groups. Each model in a model manifest must fall in exactly one
// group, so a newly added model fails generation until it is described here.
const MODEL_GROUPS = [
  {
    id: 'lc0-small-t1',
    family: 'lc0',
    uiLabel: 'Lc0 · Small net',
    status: 'default',
    versionOrSource: 't1-256x10-distilled-swa-2432500',
    manifest: 'public/models/lc0/manifest.json',
    prefix: 't1-256x10-distilled-swa-2432500.',
    notes:
      'Default Lc0 network: Analysis, Arena and Play load the batch-1 f16 QDQ8 export by default, or its split-K rewrite (batch1.f16.qdq8.splitk) when ORT runs on WebGPU with an Apple adapter; the lc0web pack serves the opt-in custom WebGPU path.',
  },
  {
    id: 'lc0-t3',
    family: 'lc0',
    uiLabel: 'Lc0 · t3-512 distill',
    status: 'option',
    versionOrSource: 't3-512x15x16h-distill-swa-2767500',
    manifest: 'public/models/lc0/manifest.json',
    prefix: 't3-512x15x16h-distill-swa-2767500.',
    notes: 'The t3-512 option loads the batch-8 f16 export (T3_NET in src/lc0/bt4Engine.ts).',
  },
  {
    id: 'lc0-bt4-it332',
    family: 'lc0',
    uiLabel: 'Lc0 · BT4-it332',
    status: 'experimental',
    versionOrSource: 'BT4-1024x15x32h-swa-6147500-policytune-332',
    manifest: 'public/models/lc0/manifest.json',
    prefix: 'BT4-1024x15x32h-swa-6147500-policytune-332.',
    notes:
      'The experimental BT4-it332 option loads the batch-8 f16 QDQ8 export (BT4_NET in src/lc0/bt4Engine.ts). The unquantized f16 exports are research entries and appear in no committed release manifest under public/releases/.',
  },
  {
    id: 'lc0-bt4-legacy',
    family: 'lc0',
    uiLabel: 'Lc0 BT4 legacy',
    status: 'historical-research',
    versionOrSource: 'BT4-1024x15x32h-swa-6147500',
    manifest: 'public/models/lc0/manifest.json',
    prefix: 'BT4-1024x15x32h-swa-6147500.',
    notes: 'Kept in the manifest for research compatibility; no UI option loads it and it appears in no committed release manifest.',
  },
  {
    id: 'lc0-leela-queen-odds',
    family: 'lc0',
    uiLabel: 'Leela Queen Odds',
    status: 'option',
    versionOrSource: 'LeelaQueenOdds v2 (lqo_v2)',
    manifest: 'public/models/lc0/manifest.json',
    prefix: 'lqo_v2.',
    notes: 'The Leela Queen Odds option loads the f16 QDQ8 export (LQO_NET in src/lc0/bt4Engine.ts); lqo_v2.f16.onnx is its source export.',
  },
  {
    id: 'centipawn-best-bt4',
    family: 'centipawn',
    uiLabel: 'Centipawn',
    status: 'default',
    versionOrSource: 'bt4_soap_rem_c19000_final',
    manifest: 'public/models/centipawn.manifest.json',
    prefix: 'bt4_soap_rem_c19000_final.',
    provenancePath: 'docs/model_provenance/bt4_soap_rem_c19000_final.md',
    notes:
      'Centipawn ORT model and the fallback for the promoted TVMJS runtime. BT4/SquareFormer + ThreatGraph TG1 trained in this project from LC0 public data with Stockfish labels.',
  },
  {
    id: 'maia3',
    family: 'maia3',
    uiLabel: 'Maia3',
    status: 'option',
    versionOrSource: 'maia3_simplified (CSSLab/maia-platform-frontend)',
    manifest: 'public/models/maia3/manifest.json',
    prefix: 'maia3_simplified.',
    provenancePath: 'docs/model_provenance/maia3.md',
    notes: 'Play and the Analysis Human tab load the locally derived QDQ8 file by default and fall back to the upstream fp16 file.',
  },
];

const TVMJS_RUNTIMES = [
  {
    id: 'centipawn-tvmjs-webgpu-v2',
    family: 'centipawn',
    uiLabel: 'Centipawn · TVMJS',
    status: 'default',
    manifest: 'public/runtimes/centipawn-tvmjs-webgpu/bt4-soap-rem-c19000-final/f32/v2-shape-k16/manifest.json',
    notes:
      'Promoted TVMJS WebGPU runtime for Centipawn (CENTIPAWN_TVMJS_WEBGPU_V2 in src/nn/runtimeRegistry.ts); falls back to the ORT model when WebGPU is unavailable.',
  },
];

const RESEARCH_ENTRIES = [
  {
    id: 'lc0-tvmjs-webgpu',
    family: 'lc0',
    uiLabel: 'Lc0 TVMJS whole-model WebGPU',
    status: 'research-only',
    versionOrSource: 'TVMJS exports for t1 and BT4-it332 when locally staged',
    manifestUrl: null,
    primaryUrls: ['/runtimes/lc0-tvmjs-webgpu/'],
    notes:
      'Source and tooling are in the repository, but public/runtimes/lc0-tvmjs-webgpu/ is gitignored and no runtime artifacts are committed. Not in the stable runtime registry.',
  },
];

const ROOT = fileURLToPath(new URL('..', import.meta.url));

function readJson(root, path) {
  return JSON.parse(readFileSync(join(root, path), 'utf8'));
}

const publicUrl = (path) => `/${posix.relative('public', path)}`;
const publicPath = (url) => `public${url.replace(/\/$/, '')}`;

function deployablePaths(root) {
  const tracked = (...extra) => {
    const result = spawnSync('git', ['ls-files', '--cached', '-z', ...extra, '--', 'public'], { cwd: root, encoding: 'utf8' });
    if (result.status !== 0) throw new Error(`git ls-files failed: ${result.stderr}`);
    return result.stdout.split('\0').filter(Boolean);
  };
  // Read only repository .gitignore rules. --exclude-standard would also read
  // global excludes and .git/info/exclude, making the result checkout-dependent.
  const ignored = new Set(tracked('--ignored', '--exclude-per-directory=.gitignore'));
  return new Set(tracked().filter((path) => !ignored.has(path)));
}

/** Every public URL an index entry references. */
export function entryUrls(entry) {
  return [entry.manifestUrl, entry.sourceArchiveUrl, ...(entry.additionalSourceArchiveUrls ?? []), ...entry.primaryUrls].filter(Boolean);
}

function deploymentStatus(entry, deployable) {
  const isDeployable = (url) => deployable.has(publicPath(url));
  if (entryUrls(entry).every(isDeployable)) return 'tracked-deployable';
  if (entry.manifestUrl && isDeployable(entry.manifestUrl)) return 'tracked-manifest-ignored-blobs';
  return 'ignored-local-artifacts';
}

function repoSlug(repo) {
  return repo.replace(/^https:\/\/github\.com\//, '').replace(/\.git$/, '');
}

function engineVersion(manifest) {
  const { upstream } = manifest;
  const parts = [repoSlug(upstream.repo)];
  if (upstream.tag) parts.push(`tag ${upstream.tag}`);
  if (upstream.version && upstream.version !== upstream.tag) parts.push(`version ${upstream.version}`);
  parts.push(`commit ${upstream.commit}`);
  if (upstream.nativeTag) parts.push(`native Stockfish ${upstream.nativeTag} commit ${upstream.nativeCommit}`);
  const nets = manifest.assets.map((asset) => asset.name);
  return `${parts.join(', ')}; network${nets.length > 1 ? 's' : ''} ${nets.join(', ')}`;
}

function engineManifestPaths(root) {
  const paths = [];
  for (const dir of readdirSync(join(root, 'public'), { withFileTypes: true })) {
    if (!dir.isDirectory()) continue;
    for (const name of readdirSync(join(root, 'public', dir.name))) {
      if (!name.endsWith('.manifest.json')) continue;
      const path = `public/${dir.name}/${name}`;
      if (readJson(root, path).schema === ENGINE_MANIFEST_SCHEMA) paths.push(path);
    }
  }
  return paths.sort();
}

function engineEntry(path, manifest, curated) {
  const archives = manifest.artifacts.filter((artifact) => artifact.path.endsWith('.tar.gz')).map((artifact) => publicUrl(artifact.path));
  const sourceArchiveUrl = manifest.sourceArchive?.url ?? null;
  const additional = archives.filter((url) => url !== sourceArchiveUrl);
  return {
    id: posix.basename(path, '.manifest.json'),
    family: manifest.engine,
    uiLabel: curated.uiLabel,
    status: manifest.status,
    license: manifest.upstream.license,
    versionOrSource: engineVersion(manifest),
    manifestUrl: publicUrl(path),
    sourceArchiveUrl,
    ...(additional.length ? { additionalSourceArchiveUrls: additional } : {}),
    primaryUrls: manifest.artifacts.filter((artifact) => !artifact.path.endsWith('.tar.gz')).map((artifact) => publicUrl(artifact.path)),
    notes: curated.notes,
  };
}

function modelEntries(root) {
  const byManifest = new Map();
  for (const group of MODEL_GROUPS) {
    if (!byManifest.has(group.manifest)) byManifest.set(group.manifest, readJson(root, group.manifest));
  }
  for (const [path, manifest] of byManifest) {
    const groups = MODEL_GROUPS.filter((group) => group.manifest === path);
    for (const item of [...manifest.models, ...(manifest.packs ?? [])]) {
      const name = item.file ?? item.id;
      const matches = groups.filter((group) => name.startsWith(group.prefix));
      if (matches.length !== 1) throw new Error(`${path}: ${name} matches ${matches.length} index groups; add or fix a MODEL_GROUPS row`);
    }
  }
  return MODEL_GROUPS.map(({ manifest: path, prefix, provenancePath, ...group }) => {
    const manifest = byManifest.get(path);
    const urls = [
      ...manifest.models.filter((model) => model.file.startsWith(prefix)),
      ...(manifest.packs ?? []).filter((pack) => pack.id.startsWith(prefix)),
    ].map((item) => item.url);
    return {
      id: group.id,
      family: group.family,
      uiLabel: group.uiLabel,
      status: group.status,
      versionOrSource: group.versionOrSource,
      manifestUrl: publicUrl(path),
      ...(provenancePath ? { provenancePath } : {}),
      primaryUrls: urls,
      notes: group.notes,
    };
  });
}

function runtimeEntries(root) {
  return TVMJS_RUNTIMES.map(({ manifest: path, ...runtime }) => {
    const manifest = readJson(root, path);
    const dir = posix.dirname(path);
    return {
      id: runtime.id,
      family: runtime.family,
      uiLabel: runtime.uiLabel,
      status: runtime.status,
      versionOrSource: `${manifest.modelFamily} ${manifest.dtype} ${manifest.version} (TVM ${manifest.compilerProvenance.gitDescribe})`,
      manifestUrl: publicUrl(path),
      primaryUrls: manifest.files.map((file) => publicUrl(`${dir}/${file.path}`)),
      notes: runtime.notes,
    };
  });
}

export function buildArtifactIndex(root = ROOT) {
  const manifestPaths = engineManifestPaths(root);
  const manifests = new Map(manifestPaths.map((path) => [path, readJson(root, path)]));
  const engineRows = [];
  for (const curated of ENGINES) {
    const paths = manifestPaths.filter((path) => manifests.get(path).engine === curated.engine);
    if (!paths.length) throw new Error(`No engine manifest under public/ for ${curated.engine}`);
    for (const path of paths) engineRows.push(engineEntry(path, manifests.get(path), curated));
  }
  for (const [path, manifest] of manifests) {
    if (!ENGINES.some((curated) => curated.engine === manifest.engine)) throw new Error(`${path}: add an ENGINES row for ${manifest.engine}`);
  }
  const drafts = [...modelEntries(root), ...runtimeEntries(root), ...RESEARCH_ENTRIES, ...engineRows];
  const deployable = deployablePaths(root);
  const entries = drafts.map((entry) => {
    const { id, family, uiLabel, status, license, versionOrSource, ...rest } = entry;
    return { id, family, uiLabel, status, ...(license ? { license } : {}), versionOrSource, deploymentStatus: deploymentStatus(entry, deployable), ...rest };
  });
  const inputDates = [...manifests.values(), ...TVMJS_RUNTIMES.map((runtime) => readJson(root, runtime.manifest))]
    .map((manifest) => manifest.generatedAt)
    .filter(Boolean)
    .sort();
  return {
    schema: '0x88.hosted-artifact-index.v1',
    generatedAt: inputDates.at(-1),
    generatedBy: 'scripts/write_artifact_index.mjs (npm run artifact-index)',
    documentationPath: 'docs/hosted_artifacts.md',
    distributionPolicyPath: 'docs/engine_artifact_distribution.md',
    notes: [
      'Generated from the per-artifact manifests; do not edit by hand. generatedAt is the newest generatedAt among the input manifests, so regeneration is deterministic.',
      'URLs are public paths. Production builds resolve /models/, engine-directory and Centipawn TVMJS runtime paths against the artifact host (R2_RESOLVED_PREFIXES in src/lc0/assetUrls.ts; VITE_LC0_BROWSER_ASSET_BASE_URL=https://assets.0x88.app in netlify.toml). The R2 publication history is in public/releases/.',
      'deploymentStatus describes a clean checkout, derived from .gitignore: tracked-deployable = every listed file is committed; tracked-manifest-ignored-blobs = the manifest (and any source archives) are committed but the listed binaries or model blobs are gitignored and must be built, staged (npm run lc0:prepare-model-assets) or fetched from the artifact host; ignored-local-artifacts = nothing listed is committed.',
    ],
    entries,
  };
}

export function serializeArtifactIndex(index) {
  return `${JSON.stringify(index, null, 2)}\n`;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const text = serializeArtifactIndex(buildArtifactIndex());
  const target = join(ROOT, ARTIFACT_INDEX_PATH);
  if (process.argv.includes('--check')) {
    if (readFileSync(target, 'utf8') !== text) {
      console.error(`${ARTIFACT_INDEX_PATH} is stale; run npm run artifact-index`);
      process.exitCode = 1;
    } else {
      console.log(`${ARTIFACT_INDEX_PATH} is up to date`);
    }
  } else {
    writeFileSync(target, text);
    console.log(`Wrote ${ARTIFACT_INDEX_PATH}`);
  }
}
