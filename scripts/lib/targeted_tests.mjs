import { readdirSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';

const SCAN_ROOTS = ['src', 'tests', 'eval', 'scripts', 'wasm', 'cloudflare'];
const PARSE_EXT = new Set(['.mjs', '.js', '.cjs', '.ts', '.tsx', '.svelte']);
const SPECIFIER_RE = /\bfrom\s+['"]([^'"]+)['"]|\bimport\s*\(\s*['"]([^'"]+)['"]\s*\)|\bimport\s+['"]([^'"]+)['"]/g;
const PATH_LITERAL_RE = /['"]((?:\.\.\/)+(?:src|eval|scripts|wasm|cloudflare)\/[^'"]+)['"]|['"]((?:src|eval|scripts|wasm|cloudflare)\/[^'"]+)['"]/g;

// Hand map for files the import walk cannot see (workflow YAML, config, generated
// entrypoints). The walk is authoritative for source; this only adds coverage.
const EXACT_TESTS = new Map([
  ['src/lc0/analysisBrowser.ts', ['engine_catalog', 'engine_resource_broker', 'lc0_analysis_format', 'disposable_variant_pool']],
  ['src/lc0/arenaBrowser.ts', ['engine_catalog', 'engine_resource_broker', 'disposable_variant_pool']],
  ['src/lc0/disposableVariantPool.ts', ['disposable_variant_pool']],
  ['src/lc0/browserVariantOption.ts', ['browser_variant_option']],
  ['src/lc0/engineCatalog.ts', ['engine_catalog']],
  ['src/lc0/engineProvision.ts', ['engine_catalog']],
  ['src/lc0/playBrowser.ts', ['engine_catalog', 'play_preferences']],
  ['src/lc0/playPreferences.ts', ['play_preferences']],
  ['src/lc0/resourceBroker.ts', ['engine_resource_broker']],
  ['src/lc0/recklessVariants.ts', ['lc0_reckless_variants']],
  ['src/lc0/recklessWasiWorker.ts', ['lc0_reckless_wasi_worker', 'lc0_reckless_external_nnue']],
  ['src/lc0/viridithasVariants.ts', ['viridithas_variants']],
  ['src/lc0/berserkVariants.ts', ['berserk_variants']],
  ['src/lc0/plentychessVariants.ts', ['plentychess_variants']],
  ['src/lc0/stormphraxVariants.ts', ['stormphrax_variants']],
  ['src/lc0/stockfishEngine.ts', ['lc0_stockfish_engine']],
  ['src/lc0/gameReview.ts', ['game_review']],
  ['scripts/engine_artifact_registry.mjs', ['engine_artifact_registry', 'netlify_r2_release']],
  ['scripts/check_browser_engine_assets.mjs', ['engine_artifact_registry']],
  ['scripts/netlify_r2_release.mjs', ['netlify_r2_release']],
  ['scripts/precompress_engine_artifacts.mjs', ['engine_artifact_registry', 'netlify_r2_release']],
  ['scripts/prepare_netlify_r2_public_assets.mjs', ['engine_artifact_registry', 'netlify_r2_release', 'ort_runtime_assets']],
  ['scripts/prune_external_model_assets.mjs', ['engine_artifact_registry', 'netlify_r2_release']],
  ['scripts/check_ort_wasm_dedup.mjs', ['ort_wasm_dedup']],
  ['scripts/ort_runtime_assets.mjs', ['ort_runtime_assets']],
  ['scripts/check_ort_runtime_assets.mjs', ['ort_runtime_assets']],
  ['scripts/build_reckless_release_assets.mjs', ['check_reckless_external_wasi_assets']],
  ['scripts/check_reckless_external_wasi_assets.mjs', ['check_reckless_external_wasi_assets']],
  ['scripts/r2_brotli_publish_assets.mjs', ['check_reckless_external_wasi_assets']],
  ['scripts/write_engine_artifact_manifest.mjs', ['check_reckless_external_wasi_assets']],
  ['scripts/production_browser_smoke.mjs', ['production_browser_smoke']],
  ['.github/workflows/production-smoke.yml', ['production_browser_smoke']],
  ['vite.config.ts', ['ort_wasm_dedup']],
]);

const SEARCH_PREFIXES = ['puct_', 'lc0_search', 'search_metamorphic', 'root_policy'];
const ENGINE_TOKENS = ['berserk', 'plentychess', 'reckless', 'stockfish', 'stormphrax', 'viridithas', 'monty', 'maia3', 'maia'];

export function isExecutableSource(file) {
  if (file.startsWith('tests/')) return false;
  if (/^src\/.*\.(?:ts|tsx|js|mjs|svelte|wgsl)$/.test(file)) return true;
  if (/^eval\/.*\.(?:mjs|js|ts)$/.test(file)) return true;
  if (/^scripts\/.*\.(?:mjs|js|py|sh)$/.test(file)) return true;
  if (/^wasm\/.*\.(?:rs|toml)$/.test(file)) return true;
  if (/^cloudflare\/.*\.(?:mjs|js|toml)$/.test(file)) return true;
  return false;
}

function snakeCase(value) {
  return value
    .replace(/\.[^.]+$/, '')
    .replace(/([a-z0-9])([A-Z])/g, '$1_$2')
    .replace(/[-\s]+/g, '_')
    .toLowerCase();
}

function walkFiles(root) {
  const files = [];
  for (const dir of SCAN_ROOTS) {
    const absolute = path.join(root, dir);
    let entries;
    try {
      entries = readdirSync(absolute, { recursive: true });
    } catch (error) {
      if (error?.code === 'ENOENT') continue;
      throw error;
    }
    for (const entry of entries) {
      const rel = path.join(dir, String(entry)).split(path.sep).join('/');
      if (rel.includes('/node_modules/')) continue;
      const absoluteFile = path.join(root, rel);
      let info;
      try {
        info = statSync(absoluteFile);
      } catch {
        continue;
      }
      if (info.isFile()) files.push(rel);
    }
  }
  return files;
}

function resolveExisting(fileSet, base) {
  const candidates = [base];
  if (!path.posix.extname(base)) {
    candidates.push(`${base}.ts`, `${base}.tsx`, `${base}.mjs`, `${base}.js`, `${base}.svelte`, `${base}/index.ts`);
  }
  for (const candidate of candidates) if (fileSet.has(candidate)) return candidate;
  return null;
}

function resolveSpecifier(fileSet, fromFile, spec) {
  const clean = spec.split('?')[0].split('#')[0];
  if (!clean.startsWith('.')) return null;
  const base = path.posix.normalize(path.posix.join(path.posix.dirname(fromFile), clean));
  if (base.startsWith('..')) return null;
  return resolveExisting(fileSet, base);
}

function addEdge(reverse, importer, imported) {
  if (!imported || imported === importer) return;
  let importers = reverse.get(imported);
  if (!importers) {
    importers = new Set();
    reverse.set(imported, importers);
  }
  importers.add(importer);
}

/**
 * Reverse import graph: source file → tests that reach it.
 * Walks static imports from tests, eval scripts, and source, plus path literals
 * tests and eval scripts pass to child processes (`eval/puct_core_tests.mjs`).
 */
export function buildImportGraph(root = process.cwd()) {
  const started = performance.now();
  const files = walkFiles(root);
  const fileSet = new Set(files);
  const reverse = new Map();
  for (const file of files) {
    if (!PARSE_EXT.has(path.posix.extname(file))) continue;
    let text;
    try {
      text = readFileSync(path.join(root, file), 'utf8');
    } catch {
      continue;
    }
    SPECIFIER_RE.lastIndex = 0;
    for (const match of text.matchAll(SPECIFIER_RE)) {
      addEdge(reverse, file, resolveSpecifier(fileSet, file, match[1] || match[2] || match[3]));
    }
    if (file.startsWith('tests/') || file.startsWith('eval/')) {
      PATH_LITERAL_RE.lastIndex = 0;
      for (const match of text.matchAll(PATH_LITERAL_RE)) {
        const spec = (match[1] || match[2]).split('?')[0].split('#')[0];
        const resolved = spec.startsWith('.') ? resolveSpecifier(fileSet, file, spec) : resolveExisting(fileSet, spec);
        addEdge(reverse, file, resolved);
      }
    }
  }
  const tests = files.filter((file) => file.startsWith('tests/') && file.endsWith('.test.mjs')).sort();
  return { root, files: fileSet, reverse, tests, testSet: new Set(tests), buildMs: performance.now() - started };
}

function testsReachableFrom(file, graph) {
  const found = new Set();
  const seen = new Set();
  const queue = [file];
  while (queue.length) {
    const current = queue.pop();
    if (!current || seen.has(current)) continue;
    seen.add(current);
    if (graph.testSet.has(current)) found.add(current);
    for (const importer of graph.reverse.get(current) ?? []) queue.push(importer);
  }
  return found;
}

function testsByPrefix(graph, prefixes) {
  return graph.tests.filter((test) => prefixes.some((prefix) => path.posix.basename(test).startsWith(prefix)));
}

function familyTests(file, graph) {
  const selected = new Set();
  if (file.startsWith('src/search/') || file === 'eval/puct_core_tests.mjs') {
    for (const test of testsByPrefix(graph, SEARCH_PREFIXES)) selected.add(test);
  }
  const base = path.posix.basename(file);
  const engine = ENGINE_TOKENS.find((token) => base.toLowerCase().startsWith(token));
  if (engine && /\.(?:ts|tsx|mjs|js|svelte)$/.test(file) && (file.startsWith('src/') || file.startsWith('scripts/') || file.startsWith('eval/'))) {
    const needle = engine.startsWith('maia') ? 'maia' : engine;
    for (const test of graph.tests) if (path.posix.basename(test).includes(needle)) selected.add(test);
  }
  if (file.startsWith('wasm/')) {
    const prefixes = file.includes('input_encoder')
      ? ['lc0_wasm_input', 'lc0_encoder']
      : file.includes('legal_priors')
        ? ['lc0_wasm_legal', 'lc0_wasm_features']
        : ['lc0_wasm'];
    for (const test of testsByPrefix(graph, prefixes)) selected.add(test);
  }
  if (file.startsWith('cloudflare/')) {
    for (const test of graph.tests) if (path.posix.basename(test).includes('artifact')) selected.add(test);
  }
  return selected;
}

function exactTests(file, graph) {
  const selected = new Set();
  for (const name of EXACT_TESTS.get(file) ?? []) {
    const testPath = name.startsWith('tests/') ? name : `tests/${name}.test.mjs`;
    if (graph.testSet.has(testPath)) selected.add(testPath);
  }
  // Filename guesses are for source, not docs. A markdown file that shares a
  // stem with a test must not turn a docs-only change into a green test run.
  if (!isExecutableSource(file)) return selected;
  const stem = snakeCase(path.posix.basename(file));
  for (const candidate of [`tests/${stem}.test.mjs`, `tests/lc0_${stem}.test.mjs`]) {
    if (graph.testSet.has(candidate)) selected.add(candidate);
  }
  return selected;
}

export function normalizeChangedPath(file, cwd = process.cwd()) {
  let rel = String(file).trim().replace(/\\/g, '/');
  if (!rel) return '';
  if (path.isAbsolute(rel)) rel = path.relative(cwd, rel).split(path.sep).join('/');
  return rel.replace(/^\.\//, '').replace(/\/+/g, '/');
}

/**
 * Decide which tests cover `changedFiles`.
 * status `run` — execute `tests`.
 * status `skip` — no executable source; a zero-test result is explicit, not a pass of product code.
 * status `fail` — executable source matched nothing. Callers must exit non-zero or run the full suite.
 */
export function selectTargetedTests(changedFiles, graph) {
  const tests = new Set();
  const covered = [];
  const uncovered = [];
  const ignored = [];
  for (const raw of changedFiles) {
    const file = normalizeChangedPath(raw, graph.root);
    if (!file) continue;
    const matched = new Set();
    if (graph.testSet.has(file)) matched.add(file);
    for (const test of testsReachableFrom(file, graph)) matched.add(test);
    for (const test of exactTests(file, graph)) matched.add(test);
    for (const test of familyTests(file, graph)) matched.add(test);
    if (matched.size) {
      covered.push(file);
      for (const test of matched) tests.add(test);
    } else if (isExecutableSource(file)) uncovered.push(file);
    else ignored.push(file);
  }
  let status = 'run';
  if (uncovered.length) status = 'fail';
  else if (!tests.size) status = 'skip';
  return {
    status,
    tests: [...tests].sort(),
    covered,
    uncovered,
    ignored,
    graphBuildMs: graph.buildMs ?? null,
  };
}

export function formatUnmatched(decision) {
  const lines = [
    'Refusing a zero-test pass. Executable source matched no tests:',
    ...decision.uncovered.map((file) => `- ${file}`),
    'Pass test files on the command line, or rerun with --on-unmatched full to run npm test.',
  ];
  return lines.join('\n');
}
