#!/usr/bin/env node
// Content Security Policy gate.
//
// Every prerendered page is governed by TWO enforced policies: the header from
// netlify.toml's `/*` block, and the <meta http-equiv> policy SvelteKit writes
// from svelte.config.js (kit.csp, hash mode). A browser runs an inline script
// only when every policy in force at that point allows it, and a <meta> policy
// is only in force for elements parsed after the <meta> itself.
//
// That is why the header keeps script-src 'unsafe-inline':
//   - the theme / ORT-console-filter script from src/app.html sits before
//     %sveltekit.head%, so only the header governs it;
//   - SvelteKit's boot script is hashed in the meta policy, but its text embeds
//     the build id, hashed entry chunk names and the page's node ids, so its
//     sha256 differs on every page and on every build. A static header cannot
//     list those hashes, and a header that lists any hash makes browsers ignore
//     'unsafe-inline' — which would block every boot script and stop hydration.
// The meta policy (no 'unsafe-inline') is what restricts inline script on
// everything parsed after it, including anything injected later.
//
// This gate keeps that arrangement true instead of conventional:
//   1. the meta directives and the header directives agree, with the
//      differences listed in HEADER_ONLY_DIFFERENCES and nowhere else;
//   2. with a build directory, every inline script on every built page is
//      allowed by both policies that actually apply to it.
import { createHash } from 'node:crypto';
import { readdirSync, readFileSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { loadEnv } from 'vite';
import { effectiveValue, parseHeaderBlocks, valuesByName } from './netlify_headers.mjs';

const KEYWORDS = new Set(['self', 'none', 'unsafe-inline', 'unsafe-eval', 'wasm-unsafe-eval', 'strict-dynamic', 'unsafe-hashes', 'report-sample']);

// SvelteKit takes keywords unquoted in svelte.config.js and quotes them on
// output; hash and nonce sources are quoted as well.
function quoteSource(source) {
  return KEYWORDS.has(source) || /^(sha256|sha384|sha512|nonce)-/.test(source) ? `'${source}'` : source;
}

/** Parse a serialized policy into directive -> source list (names lowercased). */
export function parseCsp(text) {
  const directives = new Map();
  for (const part of String(text).split(';')) {
    const [name, ...sources] = part.trim().split(/\s+/);
    if (!name) continue;
    const key = name.toLowerCase();
    // Per CSP3 a repeated directive is ignored after its first occurrence.
    if (!directives.has(key)) directives.set(key, sources);
  }
  return directives;
}

/** kit.csp.directives from svelte.config.js, in the same shape as parseCsp. */
export function metaDirectivesFromSvelteConfig(config) {
  const directives = new Map();
  for (const [name, sources] of Object.entries(config?.kit?.csp?.directives ?? {})) {
    directives.set(name.toLowerCase(), Array.isArray(sources) ? sources.map(quoteSource) : []);
  }
  return directives;
}

/** The site-wide header policy declared on `/*` in netlify.toml. */
export function headerPolicyFromNetlifyToml(tomlText) {
  const block = parseHeaderBlocks(tomlText).find((candidate) => candidate.path === '/*');
  return block ? effectiveValue(valuesByName(block.entries), 'content-security-policy') : undefined;
}

// The only places the header may say something the meta policy does not.
//   - script-src 'unsafe-inline': see the header comment of this file.
//   - form-action: SvelteKit's hash mode does not need it in the meta policy
//     and a header-only restriction is strictly tighter.
//   - connect-src: the header (production) may name a SUBSET of what the meta
//     policy allows. The meta policy is also what dev and local static serving
//     enforce, where documented ?assetBase= overrides to other https hosts must
//     keep working; both policies apply in production, so the header decides.
const HEADER_ONLY_DIFFERENCES = {
  'script-src': { extra: ["'unsafe-inline'"] },
  'form-action': { headerOnly: true },
  'connect-src': { subset: true },
};

// Whether a source list allows everything `source` allows: listed verbatim, or
// a host source under a scheme source (`https:` covers `https://a.example`).
function sourceCovered(source, sources) {
  if (sources.has(source)) return true;
  const scheme = /^([a-z][a-z0-9+.-]*):\/\//i.exec(source)?.[1];
  return scheme !== undefined && sources.has(`${scheme.toLowerCase()}:`);
}

/** Required fetch origins from the same public environment Vite bakes into the app. */
function checkConnectOrigins(header, env) {
  const required = new Map([
    ['https://lichess.org', 'Lichess game imports'],
    ['https://api.chess.com', 'Chess.com game imports'],
    ['https://cloudflareinsights.com', 'Cloudflare Web Analytics'],
  ]);
  for (const [setting, fallback] of [
    ['VITE_LC0_BROWSER_ASSET_BASE_URL', 'https://assets.0x88.app'],
    ['VITE_LC0_ARTIFACT_CHANNEL_URL', 'https://assets.0x88.app/channels/stable.json'],
  ]) {
    const value = env[setting]?.trim() || fallback;
    // Channel URLs may be relative to the app; those use the existing 'self'.
    const url = setting === 'VITE_LC0_ARTIFACT_CHANNEL_URL' ? new URL(value, 'https://same-origin.invalid/') : new URL(value);
    const origin = url.origin === 'https://same-origin.invalid' ? "'self'" : url.origin;
    required.set(origin, setting);
  }
  const sources = new Set(header.get('connect-src') ?? header.get('default-src') ?? []);
  return [...required].flatMap(([origin, setting]) =>
    sourceCovered(origin, sources) || sources.has('*')
      ? []
      : [
          `header connect-src blocks ${origin} required by ${setting}; add ${origin} to Content-Security-Policy connect-src in netlify.toml [[headers]] for "/*", then run \`npm run deploy:generate-headers\` to update public/_headers`,
        ],
  );
}

/** Directives that differ between the header and meta policies beyond the documented exceptions. */
export function checkPolicyConsistency(header, meta) {
  const failures = [];
  const names = new Set([...header.keys(), ...meta.keys()]);
  for (const name of names) {
    const rule = HEADER_ONLY_DIFFERENCES[name] ?? {};
    const headerSources = header.get(name);
    const metaSources = meta.get(name);
    if (headerSources === undefined) {
      failures.push(`header policy is missing ${name}, which the meta policy declares`);
      continue;
    }
    if (metaSources === undefined) {
      if (!rule.headerOnly) failures.push(`header policy declares ${name}, which the meta policy does not; add it to svelte.config.js or document why`);
      continue;
    }
    const headerSet = new Set(headerSources);
    const metaSet = new Set(metaSources);
    const extraInHeader = [...headerSet].filter(
      (source) => !(rule.subset ? sourceCovered(source, metaSet) : metaSet.has(source)) && !(rule.extra ?? []).includes(source),
    );
    const missingFromHeader = [...metaSet].filter((source) => !headerSet.has(source));
    if (extraInHeader.length) failures.push(`header ${name} allows ${extraInHeader.join(' ')} that the meta policy does not`);
    if (missingFromHeader.length && !rule.subset) failures.push(`meta ${name} allows ${missingFromHeader.join(' ')} that the header policy does not`);
  }
  return failures;
}

function scriptSources(policy) {
  return policy.get('script-src-elem') ?? policy.get('script-src') ?? policy.get('default-src');
}

/**
 * Whether a policy lets an inline <script> with this sha256 run. CSP Level 2+
 * ignores 'unsafe-inline' as soon as a hash or nonce source is present.
 */
export function inlineScriptAllowed(policy, sha256) {
  const sources = scriptSources(policy);
  if (sources === undefined) return true;
  if (sources.includes(`'sha256-${sha256}'`)) return true;
  const hasHashOrNonce = sources.some((source) => /^'(sha256|sha384|sha512|nonce)-/.test(source));
  return sources.includes("'unsafe-inline'") && !hasHashOrNonce && !sources.includes("'strict-dynamic'");
}

/** Inline scripts and the meta CSP of one HTML document, with document offsets. */
export function inspectHtml(html) {
  const scripts = [];
  const scriptPattern = /<script(\s[^>]*)?>([\s\S]*?)<\/script\s*>/gi;
  for (let match = scriptPattern.exec(html); match; match = scriptPattern.exec(html)) {
    if (match[1] && /\ssrc\s*=/i.test(match[1])) continue;
    scripts.push({ offset: match.index, sha256: createHash('sha256').update(match[2], 'utf8').digest('base64') });
  }
  const metaMatch = /<meta\s+http-equiv="content-security-policy"\s+content="([^"]*)"\s*\/?>/i.exec(html);
  const meta = metaMatch ? { offset: metaMatch.index, policy: parseCsp(metaMatch[1].replaceAll('&#39;', "'").replaceAll('&quot;', '"')) } : undefined;
  return { scripts, meta };
}

/** Every inline script must be allowed by the header and, once parsed past it, by the meta policy. */
export function checkBuiltPage(html, headerPolicy, label = 'page') {
  const failures = [];
  const { scripts, meta } = inspectHtml(html);
  if (!meta) failures.push(`${label}: no <meta http-equiv="content-security-policy"> was emitted, so inline script is restricted only by the header`);
  for (const script of scripts) {
    const where = `${label}: inline script at offset ${script.offset} (sha256-${script.sha256})`;
    if (!inlineScriptAllowed(headerPolicy, script.sha256)) failures.push(`${where} is blocked by the netlify.toml header policy`);
    if (meta && script.offset > meta.offset && !inlineScriptAllowed(meta.policy, script.sha256)) {
      failures.push(`${where} is blocked by the page's own meta policy`);
    }
  }
  return failures;
}

function htmlFiles(dir) {
  return readdirSync(dir, { withFileTypes: true, recursive: true })
    .filter((entry) => entry.isFile() && entry.name.endsWith('.html'))
    .map((entry) => join(entry.parentPath, entry.name))
    .sort();
}

export async function checkCspPolicy(rootPath = process.cwd(), distDir = undefined, env = loadEnv('production', rootPath, 'VITE_')) {
  const failures = [];
  const headerText = headerPolicyFromNetlifyToml(readFileSync(resolve(rootPath, 'netlify.toml'), 'utf8'));
  if (headerText === undefined) throw new Error('netlify.toml declares no Content-Security-Policy on /*');
  const header = parseCsp(headerText);
  const svelteConfig = (await import(pathToFileURL(resolve(rootPath, 'svelte.config.js')).href)).default;
  failures.push(...checkPolicyConsistency(header, metaDirectivesFromSvelteConfig(svelteConfig)));
  failures.push(...checkConnectOrigins(header, env));
  const pages = [];
  if (distDir !== undefined) {
    const dist = resolve(rootPath, distDir);
    for (const file of htmlFiles(dist)) {
      const label = relative(dist, file);
      pages.push(label);
      failures.push(...checkBuiltPage(readFileSync(file, 'utf8'), header, label));
    }
    if (!pages.length) failures.push(`${distDir} contains no built HTML to check`);
  }
  if (failures.length) {
    const error = new Error(`Content-Security-Policy violations:\n  ${failures.join('\n  ')}`);
    error.failures = failures;
    throw error;
  }
  return { ok: true, checked: ['netlify.toml', 'svelte.config.js', ...pages] };
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? '').href) {
  try {
    console.log(JSON.stringify(await checkCspPolicy(process.cwd(), process.argv[2]), null, 2));
  } catch (error) {
    console.error(JSON.stringify({ ok: false, failures: error.failures ?? [error.message] }, null, 2));
    process.exit(1);
  }
}
