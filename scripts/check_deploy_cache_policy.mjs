#!/usr/bin/env node
// Deploy cache policy gate.
//
// Header policy is declared once, in netlify.toml, and public/_headers is
// generated from it — so this checks invariants over a single source and then
// confirms the generated file has not drifted. It runs from the Netlify build
// command (scripts/build_netlify_r2.mjs), which is the path every automatic
// deploy takes, so a violation cannot reach production.
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { checkHeadersFileUpToDate } from './generate_netlify_headers_file.mjs';
import {
  cacheControlDirectives,
  conflictingDirectives,
  effectiveValue,
  maxSeconds,
  parseBuildTable,
  parseHeaderBlocks,
  parseRedirects,
  valuesByName,
} from './netlify_headers.mjs';

const INVARIANTS = [
  'header policy is declared only in netlify.toml',
  'public/_headers is generated from it and up to date',
  'no forced .br redirects',
  'no custom Content-Encoding headers',
  'one-year immutable only on write-once paths (content-addressed artifacts, release manifests, Vite immutable bundles)',
  'HTML and channel pointers revalidate',
  'no path declared twice',
  'content-addressed artifacts carry cross-origin CORP under COEP',
  'cross-origin isolation (COOP/COEP) is declared site-wide',
  'content security policy is declared site-wide',
  'mutable model and engine aliases revalidate',
  'the configured build command runs this gate',
  'no forced .br rewrite in publish-directory _redirects',
];

// One-year immutable is legal only where the bytes behind a URL can never
// change. /artifacts/sha256/* is content-addressed; /_app/immutable/* is Vite's
// content-hashed bundle output; /releases/* names carry the release SHA and
// publish_content_addressed_release.mjs refuses to overwrite an existing
// release manifest, so they are write-once by construction and by enforcement.
const ONE_YEAR_SECONDS = 31536000;

const immutableAllowed = (path) => path === '/artifacts/sha256/*' || path === '/_app/immutable/*' || path === '/releases/*';
const CONTENT_SECURITY_POLICY =
  "default-src 'self'; base-uri 'self'; object-src 'none'; frame-ancestors 'self'; form-action 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval' 'wasm-unsafe-eval' blob: https://assets.0x88.app; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self' data:; connect-src 'self' https:; worker-src 'self' blob:; child-src 'self' blob:; frame-src 'self'; manifest-src 'self'";

// Directive order and case are not significant in these token-list headers, so
// requirements compare the directive set rather than one exact spelling.
function directiveSet(value) {
  return value === undefined
    ? undefined
    : [...cacheControlDirectives(value)]
        .flatMap(([name, args]) => args.map((argument) => (argument === '' ? name : `${name}=${argument}`)))
        .sort()
        .join(', ');
}

// Extra Cache-Control directives that cannot weaken a required policy. The
// requirement is a subset test rather than an exact set so that adding one of
// these does not fail every deploy; anything outside this list is a change to
// the policy and has to be stated here deliberately.
const ALLOWED_EXTRA_DIRECTIVES = new Set(['no-transform', 'stale-while-revalidate', 'stale-if-error']);

// npm scripts that run the gate. Whatever `[build].command` selects must be one
// of these, or an automatic Netlify build silently bypasses the check — which is
// exactly how the gate came to be uncovered in the first place.
const GATED_BUILD_SCRIPTS = ['build:netlify:r2', 'build:netlify'];

// Netlify's own build command decides whether this gate runs at all.
function parseBuildCommand(tomlText) {
  const build = parseBuildTable(tomlText);
  return typeof build?.command === 'string' ? build.command : undefined;
}

// `_redirects` in the publish directory is a second redirect source that the
// TOML parser cannot see. Format: `from  to  status`, where a trailing `!` on
// the status forces the rewrite.
function publishRedirects(rootPath) {
  const path = resolve(rootPath, 'public/_redirects');
  if (!existsSync(path)) return [];
  return readFileSync(path, 'utf8')
    .split(/\r?\n/)
    .flatMap((line) => {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith('#')) return [];
      const [from, to, status] = trimmed.split(/\s+/);
      if (!from || !to) return [];
      return [{ from, to, forced: (status ?? '').endsWith('!') }];
    });
}

const REQUIRED = [
  // Cross-origin isolation is what makes SharedArrayBuffer available, and every
  // threaded engine and WASM worker path depends on it. Losing this pair does
  // not break the build or any test — it breaks the product in the browser — so
  // it is the single most valuable thing this gate can hold.
  ['/*', 'cross-origin-opener-policy', 'same-origin', 'cross-origin isolation COOP'],
  ['/*', 'cross-origin-embedder-policy', 'require-corp', 'cross-origin isolation COEP'],
  ['/*', 'content-security-policy', CONTENT_SECURITY_POLICY, 'site-wide Content-Security-Policy'],
  ['/*.html', 'cache-control', 'public, max-age=0, must-revalidate', 'HTML revalidation Cache-Control'],
  ['/channels/*', 'cache-control', 'public, max-age=0, no-cache', 'channel revalidation Cache-Control'],
  ['/artifacts/sha256/*', 'cache-control', 'public, max-age=31536000, immutable', 'content-addressed immutable Cache-Control'],
  // The site sets COEP: require-corp on /*, so hashed blobs need an explicit
  // cross-origin CORP or they are blocked once served from the asset host.
  ['/artifacts/sha256/*', 'cross-origin-resource-policy', 'cross-origin', 'cross-origin CORP'],
  ['/releases/*', 'cache-control', 'public, max-age=31536000, immutable', 'write-once release manifest Cache-Control'],
  // Logical aliases whose bytes change between releases. Without these the gate
  // only rejected the exact immutable combination, so a plain `max-age=86400`
  // would have let a stale model or engine stay fresh in the browser for a day.
  ['/models/*', 'cache-control', 'public, max-age=0, must-revalidate', 'model alias revalidation Cache-Control'],
  ['/engines/*', 'cache-control', 'public, max-age=0, must-revalidate', 'engine alias revalidation Cache-Control'],
];

export function checkDeployCachePolicy(rootPath = process.cwd()) {
  const failures = [];
  const fail = (message) => failures.push(message);
  const netlify = readFileSync(resolve(rootPath, 'netlify.toml'), 'utf8');

  for (const redirect of parseRedirects(netlify)) {
    const target = typeof redirect.to === 'string' ? redirect.to : '';
    if (redirect.force === true && /\.br(\?|$)/.test(target)) {
      fail(`netlify.toml must not force a stable URL to a .br sidecar: ${redirect.from ?? '(unknown)'} -> ${target}`);
    }
  }

  const blocks = parseHeaderBlocks(netlify, (message) => {
    fail(`netlify.toml declares a header this policy cannot verify — ${message}`);
  });
  const headers = new Map(blocks.map((block) => [block, valuesByName(block.entries)]));

  const seen = new Set();
  for (const block of blocks) {
    if (block.path === undefined) {
      fail('netlify.toml has a [[headers]] block with no `for` path');
      continue;
    }
    if (seen.has(block.path)) fail(`netlify.toml declares ${block.path} more than once; merge the blocks so the effective headers are unambiguous`);
    seen.add(block.path);

    const values = headers.get(block);
    if (effectiveValue(values, 'content-encoding') !== undefined) {
      fail(`netlify.toml must not declare Content-Encoding on ${block.path}; serve compressed variants via CDN/object metadata or negotiated edge code`);
    }
    const directives = cacheControlDirectives(effectiveValue(values, 'cache-control'));
    const conflicts = conflictingDirectives(directives);
    if (conflicts.length) {
      fail(`netlify.toml states ${conflicts.join(', ')} more than once with different arguments on ${block.path}, so its Cache-Control has no single meaning`);
    }
    // `immutable` tells a browser never to revalidate while the response is
    // fresh, which is wrong on a mutable alias at ANY ttl; and a ttl at or over
    // a year pins the URL regardless of whether `immutable` is stated. Both are
    // prohibited off the write-once paths, and the ttl test is a threshold
    // rather than an equality so `max-age=31536001` cannot slip past.
    const ttl = maxSeconds(directives, 'max-age');
    if (!immutableAllowed(block.path)) {
      if (directives.has('immutable')) {
        fail(`netlify.toml has one-year immutable Cache-Control outside content-addressed artifacts: ${block.path}`);
      } else if (ttl !== undefined && ttl >= ONE_YEAR_SECONDS) {
        fail(`netlify.toml pins ${block.path} for ${ttl}s, at or beyond the one-year ceiling reserved for write-once paths`);
      }
    }
  }

  for (const [path, name, expected, description] of REQUIRED) {
    const block = blocks.find((candidate) => candidate.path === path);
    const actual = block ? effectiveValue(headers.get(block), name) : undefined;
    if (actual === undefined) {
      fail(`netlify.toml missing ${description} on ${path}`);
      continue;
    }
    if (name !== 'cache-control') {
      // Single-token policy headers: any deviation is a different policy.
      if (directiveSet(actual) !== directiveSet(expected)) fail(`netlify.toml missing ${description} on ${path}`);
      continue;
    }
    const want = cacheControlDirectives(expected);
    const got = cacheControlDirectives(actual);
    for (const [directive, args] of want) {
      const mine = got.get(directive) ?? [];
      if (!args.every((argument) => mine.includes(argument))) {
        fail(`netlify.toml missing ${description} on ${path}`);
        break;
      }
    }
    const extra = [...got.keys()].filter((directive) => !want.has(directive) && !ALLOWED_EXTRA_DIRECTIVES.has(directive));
    if (extra.length) {
      fail(`netlify.toml adds ${extra.join(', ')} to the required Cache-Control on ${path}; that changes the policy and must be stated in REQUIRED`);
    }
  }

  // F3: a more specific route may not weaken cross-origin isolation. Netlify
  // applies overlapping rules cumulatively (verified against production), so a
  // route redeclaring COEP as unsafe-none loses SharedArrayBuffer for that page
  // while the /* block still reads correctly.
  for (const block of blocks) {
    if (block.path === '/*' || block.path === undefined) continue;
    const values = headers.get(block);
    for (const [header, required] of [
      ['cross-origin-opener-policy', 'same-origin'],
      ['cross-origin-embedder-policy', 'require-corp'],
    ]) {
      const declared = effectiveValue(values, header);
      if (declared !== undefined && declared.toLowerCase() !== required) {
        fail(`netlify.toml overrides ${header} to "${declared}" on ${block.path}, which drops cross-origin isolation for that route`);
      }
    }
  }

  // F1: the gate only protects deploys whose build command actually runs it.
  const buildCommand = parseBuildCommand(netlify);
  if (buildCommand !== undefined && !GATED_BUILD_SCRIPTS.some((script) => buildCommand.includes(script))) {
    fail(
      `netlify.toml [build].command does not run a gated build script (${GATED_BUILD_SCRIPTS.join(', ')}), so automatic deploys would bypass this check: ${buildCommand}`,
    );
  }

  // F2: Netlify also reads a _redirects file from the publish directory, which
  // parseRedirects cannot see.
  for (const redirect of publishRedirects(rootPath)) {
    if (redirect.forced && /\.br(\?|$)/.test(redirect.to)) {
      fail(`public/_redirects must not force a stable URL to a .br sidecar: ${redirect.from} -> ${redirect.to}`);
    }
  }

  try {
    checkHeadersFileUpToDate(rootPath);
  } catch (error) {
    fail(error.message);
  }

  if (failures.length) {
    const error = new Error(`Deploy cache policy violations:\n  ${failures.join('\n  ')}`);
    error.failures = failures;
    throw error;
  }

  return { ok: true, checked: ['netlify.toml', 'public/_headers'], invariants: INVARIANTS };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  try {
    console.log(JSON.stringify(checkDeployCachePolicy(), null, 2));
  } catch (error) {
    console.error(JSON.stringify({ ok: false, failures: error.failures ?? [error.message] }, null, 2));
    process.exit(1);
  }
}
