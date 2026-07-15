#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';

function usage() {
  console.log(`Usage: node scripts/validate_artifact_cdn_headers.mjs [--url URL ...] [--release manifest.json] [options]\n\nOptions:\n  --url URL          Logical alias URL to validate against the current channel; may be repeated\n  --release PATH     Local V1/v2 release manifest whose hosted immutable release and physical representations must be validated\n  --artifact-base URL  Hosted artifact origin (default https://assets.0x88.app)\n  --limit N          Max release artifacts to validate\n  --range BYTES      Range probe length (default 1024)\n  --verify-bodies    Download and hash full identity and decoded Brotli bodies\n  --json             Print JSON only\n  -h, --help         Show help\n\nBy default the validator checks HEAD twice and a small Range GET without downloading\nfull artifacts. --release first verifies the exact hosted /releases/<releaseId>.json,\nthen probes that release's physical identity and Brotli representation URLs directly;\nRange always targets physical identity. --url remains a logical-alias canary for the\ncurrent channel and validates identity/Brotli negotiation. Physical release and artifact\nURLs require immutable one-year caching; logical aliases require short or\nrevalidation-safe caching. It never uploads, purges, or mutates channels.\n`);
}

function parseArgs(argv) {
  const args = { urls: [], rangeBytes: 1024, verifyBodies: false, json: false, artifactBase: 'https://assets.0x88.app' };
  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    const next = argv[i + 1];
    if (arg === '--url' && next) { args.urls.push(next); i += 1; continue; }
    if (arg === '--release' && next) { args.release = next; i += 1; continue; }
    if (arg === '--artifact-base' && next) { args.artifactBase = next; i += 1; continue; }
    if (arg === '--limit' && next) { args.limit = Number(next); i += 1; continue; }
    if (arg === '--range' && next) { args.rangeBytes = Number(next); i += 1; continue; }
    if (arg === '--verify-bodies') { args.verifyBodies = true; continue; }
    if (arg === '--json') { args.json = true; continue; }
    if (arg === '-h' || arg === '--help') { usage(); process.exit(0); }
    throw new Error(`Unknown argument: ${arg}`);
  }
  if (!args.urls.length && !args.release) throw new Error('Provide at least one --url or --release');
  return args;
}

function pickHeaders(headers) {
  const keys = ['cache-control', 'cdn-cache-control', 'cloudflare-cdn-cache-control', 'cf-cache-status', 'cache-status', 'age', 'etag', 'content-length', 'x-artifact-content-length', 'x-artifact-encoded-length', 'x-artifact-decoded-sha256', 'x-artifact-encoded-sha256', 'content-type', 'content-encoding', 'accept-ranges', 'content-range', 'vary', 'access-control-allow-origin', 'cross-origin-resource-policy', 'timing-allow-origin', 'access-control-expose-headers', 'set-cookie'];
  const out = {};
  for (const key of keys) {
    const value = headers.get(key);
    if (value !== null) out[key] = value;
  }
  return out;
}

function cachePolicyForUrl(url) {
  const path = new URL(url).pathname;
  return path.startsWith('/artifacts/sha256/') || path.startsWith('/releases/') ? 'immutable' : 'mutable';
}

function releaseTargets(release, limit, artifactBase) {
  const targets = [];
  for (const artifact of release.artifacts ?? []) {
    if (artifact.raw && Array.isArray(artifact.representations)) {
      const identity = artifact.representations.find((entry) => entry.encoding === 'identity');
      const br = artifact.representations.find((entry) => entry.encoding === 'br');
      if (!identity?.url) continue;
      const identityUrl = new URL(identity.url, artifactBase).href;
      targets.push({
        mode: 'release-physical',
        logicalUrl: artifact.logicalUrl ?? artifact.name,
        url: identityUrl,
        identityUrl,
        rangeUrl: identityUrl,
        brUrl: br?.url ? new URL(br.url, artifactBase).href : undefined,
        cachePolicy: 'immutable',
        expected: {
          raw: artifact.raw,
          identity,
          br,
        },
      });
      continue;
    }
    if (artifact.artifactUrl) {
      const url = new URL(artifact.artifactUrl, artifactBase).href;
      targets.push({
        mode: 'release-physical',
        logicalUrl: artifact.logicalUrl ?? artifact.name,
        url,
        identityUrl: url,
        rangeUrl: url,
        brUrl: url,
        cachePolicy: 'immutable',
      });
    }
  }
  return Number.isFinite(limit) ? targets.slice(0, limit) : targets;
}

function sha256Bytes(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

async function validateHostedRelease(path, artifactBase) {
  const localBytes = await readFile(path);
  const localRelease = JSON.parse(localBytes.toString('utf8'));
  if (typeof localRelease.releaseId !== 'string' || !/^[A-Za-z0-9._-]+$/.test(localRelease.releaseId)) {
    throw new Error('release.releaseId must contain only letters, numbers, dots, underscores, and hyphens');
  }
  const url = new URL(`/releases/${localRelease.releaseId}.json`, artifactBase).href;
  const response = await fetch(url, {
    headers: { 'Accept-Encoding': 'identity' },
    cache: 'no-store',
  });
  const headers = pickHeaders(response.headers);
  const hostedBytes = new Uint8Array(await response.arrayBuffer());
  const expectedSha256 = sha256Bytes(localBytes);
  const actualSha256 = sha256Bytes(hostedBytes);
  const failures = [];
  if (!response.ok) failures.push(`hosted release status ${response.status}`);
  const cacheControl = headers['cache-control'] ?? '';
  const directives = cacheControlDirectives(cacheControl);
  if (!directives.has('immutable') || directives.get('max-age') !== '31536000') {
    failures.push(`hosted release cache policy is not immutable: ${cacheControl || 'missing'}`);
  }
  if (hostedBytes.byteLength !== localBytes.byteLength || actualSha256 !== expectedSha256) {
    failures.push(`hosted release does not exactly match local manifest: got ${hostedBytes.byteLength} bytes sha256 ${actualSha256}, expected ${localBytes.byteLength} bytes sha256 ${expectedSha256}`);
  }
  return {
    localRelease,
    validation: {
      url,
      status: response.status,
      headers,
      bytes: hostedBytes.byteLength,
      sha256: actualSha256,
      expectedBytes: localBytes.byteLength,
      expectedSha256,
      ok: failures.length === 0,
      failures,
    },
  };
}

async function head(url, acceptEncoding) {
  const headers = acceptEncoding ? { 'Accept-Encoding': acceptEncoding } : undefined;
  const response = await fetch(url, { method: 'HEAD', headers, cache: 'no-store' });
  return { status: response.status, headers: pickHeaders(response.headers) };
}

async function rangeGet(url, rangeBytes) {
  const response = await fetch(url, { headers: { Range: `bytes=0-${rangeBytes - 1}`, 'Accept-Encoding': 'br' }, cache: 'no-store' });
  const headers = pickHeaders(response.headers);
  if (response.status !== 206) {
    await response.body?.cancel();
    return { status: response.status, headers, bodyBytes: 0, requestedBytes: rangeBytes };
  }
  const body = await response.arrayBuffer();
  return { status: response.status, headers, bodyBytes: body.byteLength, requestedBytes: rangeBytes };
}

async function hashGet(url, acceptEncoding) {
  const response = await fetch(url, {
    headers: { 'Accept-Encoding': acceptEncoding },
    cache: 'no-store',
  });
  const headers = pickHeaders(response.headers);
  const hash = createHash('sha256');
  let bodyBytes = 0;
  if (response.body) {
    const reader = response.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      hash.update(value);
      bodyBytes += value.byteLength;
    }
  }
  return {
    status: response.status,
    headers,
    bodyBytes,
    sha256: hash.digest('hex'),
  };
}

function hasExpectedBodyMetadata(expected, representation) {
  return /^[a-f0-9]{64}$/i.test(expected?.raw?.sha256 ?? '')
    && Number.isFinite(expected?.raw?.bytes)
    && /^[a-f0-9]{64}$/i.test(representation?.sha256 ?? '')
    && Number.isFinite(representation?.bytes);
}

function cacheControlDirectives(cacheControl) {
  const directives = new Map();
  for (const part of cacheControl.split(',')) {
    const [rawName, rawValue] = part.trim().split('=', 2);
    if (!rawName) continue;
    directives.set(rawName.toLowerCase(), rawValue?.replace(/^"|"$/g, '') ?? true);
  }
  return directives;
}

function positiveHeaderInteger(headers, names) {
  for (const name of names) {
    const raw = headers[name];
    if (raw === undefined) continue;
    if (!/^[1-9]\d*$/.test(raw)) return { name, raw, value: undefined };
    const value = Number(raw);
    return { name, raw, value: Number.isSafeInteger(value) ? value : undefined };
  }
  return undefined;
}

function validateCachePolicy(headers, policy, label, failures) {
  const cacheControl = headers['cache-control'] ?? '';
  const cacheDirectives = cacheControlDirectives(cacheControl);
  if (policy === 'immutable') {
    if (!cacheDirectives.has('immutable') || cacheDirectives.get('max-age') !== '31536000') {
      failures.push(`${label} cache policy is not immutable: ${cacheControl || 'missing'}`);
    }
    return;
  }
  const maxAgeValue = cacheDirectives.get('max-age');
  const maxAge = typeof maxAgeValue === 'string' && /^\d+$/.test(maxAgeValue) ? Number(maxAgeValue) : undefined;
  const revalidationSafe = cacheDirectives.has('no-cache')
    || cacheDirectives.has('no-store')
    || (Number.isFinite(maxAge) && maxAge <= 300);
  if (cacheDirectives.has('immutable') || !revalidationSafe) {
    failures.push(`${label} cache policy is not short or revalidation-safe: ${cacheControl || 'missing'}`);
  }
}

function validateRow(row) {
  const failures = [];
  if (row.firstHead.status < 200 || row.firstHead.status >= 400) failures.push(`first HEAD status ${row.firstHead.status}`);
  if (row.secondHead.status < 200 || row.secondHead.status >= 400) failures.push(`second HEAD status ${row.secondHead.status}`);
  if (row.firstHead.headers['set-cookie'] || row.secondHead.headers['set-cookie'] || row.range.headers['set-cookie']) failures.push('artifact response must not set cookies');
  const firstHeadLength = positiveHeaderInteger(row.firstHead.headers, ['x-artifact-encoded-length', 'content-length', 'x-artifact-content-length']);
  if (!firstHeadLength?.value) failures.push(`missing positive X-Artifact-Encoded-Length, Content-Length, or X-Artifact-Content-Length on HEAD${firstHeadLength ? ` (got ${firstHeadLength.name}: ${firstHeadLength.raw})` : ''}`);
  validateCachePolicy(
    row.firstHead.headers,
    row.cachePolicy,
    row.cachePolicy === 'immutable' ? 'physical artifact' : 'logical alias',
    failures,
  );
  if (!row.firstHead.headers.etag) failures.push('missing ETag on HEAD');
  if (!row.secondHead.headers.age) failures.push('missing Age on repeated HEAD');
  const cfCacheStatus = row.secondHead.headers['cf-cache-status']?.toUpperCase();
  const cacheStatus = row.secondHead.headers['cache-status']?.toLowerCase();
  if (!cfCacheStatus && !cacheStatus) failures.push('missing CDN cache status on repeated HEAD');
  if (cfCacheStatus && !['HIT', 'REVALIDATED', 'STALE', 'UPDATING'].includes(cfCacheStatus)) failures.push(`repeated HEAD CF-Cache-Status is not cache-hit-equivalent: ${cfCacheStatus}`);
  if (!cfCacheStatus && cacheStatus && !/\bhit\b/.test(cacheStatus)) failures.push(`repeated HEAD Cache-Status does not report a hit: ${cacheStatus}`);
  if (row.firstHead.headers['access-control-allow-origin'] !== '*') failures.push('missing Access-Control-Allow-Origin: *');
  if (row.firstHead.headers['cross-origin-resource-policy'] !== 'cross-origin') failures.push('missing Cross-Origin-Resource-Policy: cross-origin');
  const timingAllowOrigin = row.firstHead.headers['timing-allow-origin'] ?? '';
  if (!timingAllowOrigin) failures.push('missing Timing-Allow-Origin');
  if (timingAllowOrigin && timingAllowOrigin !== '*' && !timingAllowOrigin.split(',').map((value) => value.trim()).includes('https://0x88.app')) {
    failures.push('Timing-Allow-Origin does not include https://0x88.app');
  }
  const exposed = new Set((row.firstHead.headers['access-control-expose-headers']?.toLowerCase() ?? '').split(',').map((value) => value.trim()).filter(Boolean));
  for (const required of ['cf-cache-status', 'cache-status', 'age', 'etag', 'content-length', 'x-artifact-content-length', 'x-artifact-encoded-length']) {
    if (!exposed.has(required)) failures.push(`Access-Control-Expose-Headers missing ${required}`);
  }
  if (row.range.status !== 206) failures.push(`Range probe returned ${row.range.status}, expected 206`);
  if (row.range.headers['content-encoding']) failures.push(`Range probe must force identity, got Content-Encoding: ${row.range.headers['content-encoding']}`);
  const contentRange = row.range.headers['content-range'];
  if (!contentRange) failures.push('missing Content-Range on range response');
  const contentRangeMatch = contentRange?.match(/^bytes (\d+)-(\d+)\/(\d+)$/);
  if (contentRange && !contentRangeMatch) failures.push(`malformed Content-Range: ${contentRange}`);
  if (contentRangeMatch) {
    const start = Number(contentRangeMatch[1]);
    const end = Number(contentRangeMatch[2]);
    const total = Number(contentRangeMatch[3]);
    const expectedBodyBytes = end - start + 1;
    const headBytes = positiveHeaderInteger(row.identityHead.headers, ['x-artifact-content-length', 'x-artifact-encoded-length', 'content-length'])?.value;
    if (start !== 0) failures.push(`range response starts at ${start}, expected 0`);
    if (expectedBodyBytes !== row.range.bodyBytes) failures.push(`range body length ${row.range.bodyBytes} does not match Content-Range length ${expectedBodyBytes}`);
    if (Number.isFinite(headBytes) && total !== headBytes) failures.push(`Content-Range total ${total} does not match HEAD artifact length ${headBytes}`);
  }
  if (row.identityHead.status < 200 || row.identityHead.status >= 400) failures.push(`identity HEAD status ${row.identityHead.status}`);
  const identityEncoding = row.identityHead.headers['content-encoding'];
  if (identityEncoding && identityEncoding !== 'identity') failures.push(`identity probe returned Content-Encoding: ${identityEncoding}`);
  if (row.brHead.status < 200 || row.brHead.status >= 400) failures.push(`br HEAD status ${row.brHead.status}`);
  if (row.brUrl !== row.url) validateCachePolicy(row.brHead.headers, 'immutable', 'physical Brotli artifact', failures);
  if (row.expected) {
    const verifyIdentityBody = row.verifyBodies && hasExpectedBodyMetadata(row.expected, row.expected.identity);
    const verifyBrBody = row.verifyBodies && hasExpectedBodyMetadata(row.expected, row.expected.br);
    if (verifyIdentityBody && (row.identityBody.status < 200 || row.identityBody.status >= 400)) failures.push(`identity body status ${row.identityBody.status}`);
    if (verifyBrBody && (row.brBody.status < 200 || row.brBody.status >= 400)) failures.push(`br body status ${row.brBody.status}`);
    const expectedRawBytes = row.expected.raw?.bytes;
    const expectedRawSha256 = row.expected.raw?.sha256?.toLowerCase();
    const actualIdentityBytes = positiveHeaderInteger(row.identityHead.headers, ['x-artifact-content-length', 'x-artifact-encoded-length', 'content-length'])?.value;
    if (Number.isFinite(expectedRawBytes) && actualIdentityBytes !== expectedRawBytes) {
      failures.push(`identity decoded length ${actualIdentityBytes} does not match manifest ${expectedRawBytes}`);
    }
    const actualDecodedSha256 = row.identityHead.headers['x-artifact-decoded-sha256'];
    if (expectedRawSha256 && actualDecodedSha256 !== expectedRawSha256) {
      failures.push(`identity decoded SHA-256 ${actualDecodedSha256 ?? 'missing'} does not match manifest ${expectedRawSha256}`);
    }
    if (verifyIdentityBody && row.identityBody.bodyBytes !== expectedRawBytes) {
      failures.push(`identity body length ${row.identityBody.bodyBytes} does not match manifest ${expectedRawBytes}`);
    }
    if (verifyIdentityBody && row.identityBody.sha256 !== expectedRawSha256) {
      failures.push(`identity body SHA-256 ${row.identityBody.sha256} does not match manifest ${expectedRawSha256}`);
    }
    if (row.expected.br) {
      if (row.brHead.headers['content-encoding'] !== 'br') failures.push(`br probe returned Content-Encoding: ${row.brHead.headers['content-encoding'] ?? 'identity'}`);
      if (row.brNegotiated && !(row.brHead.headers.vary ?? '').toLowerCase().split(',').map((value) => value.trim()).includes('accept-encoding')) {
        failures.push('br probe missing Vary: Accept-Encoding');
      }
      const encodedSha256 = row.brHead.headers['x-artifact-encoded-sha256'];
      if (encodedSha256 !== row.expected.br.sha256?.toLowerCase()) {
        failures.push(`br encoded SHA-256 ${encodedSha256 ?? 'missing'} does not match manifest ${row.expected.br.sha256}`);
      }
      const encodedLength = positiveHeaderInteger(row.brHead.headers, ['x-artifact-encoded-length', 'content-length']);
      if (encodedLength?.value !== row.expected.br.bytes) {
        failures.push(`br encoded length ${encodedLength?.raw ?? 'missing'} does not match manifest ${row.expected.br.bytes}`);
      }
      if (verifyBrBody && row.brBody.bodyBytes !== expectedRawBytes) {
        failures.push(`br decoded body length ${row.brBody.bodyBytes} does not match manifest ${expectedRawBytes}`);
      }
      if (verifyBrBody && row.brBody.sha256 !== expectedRawSha256) {
        failures.push(`br decoded body SHA-256 ${row.brBody.sha256} does not match manifest ${expectedRawSha256}`);
      }
    }
  }
  return failures;
}

async function validateUrl(target, rangeBytes) {
  const { url, expected, verifyBodies } = target;
  const cachePolicy = target.cachePolicy ?? cachePolicyForUrl(url);
  const firstHead = await head(url);
  const secondHead = await head(url);
  const rangeUrl = target.rangeUrl ?? url;
  const identityUrl = target.identityUrl ?? url;
  const brUrl = target.brUrl ?? url;
  const range = await rangeGet(rangeUrl, rangeBytes);
  const identityHead = await head(identityUrl, 'identity');
  const brHead = await head(brUrl, 'br');
  const identityBody = verifyBodies && hasExpectedBodyMetadata(expected, expected?.identity) ? await hashGet(identityUrl, 'identity') : undefined;
  const brBody = verifyBodies && hasExpectedBodyMetadata(expected, expected?.br) ? await hashGet(brUrl, 'br') : undefined;
  const row = {
    url,
    mode: target.mode ?? 'logical-alias',
    logicalUrl: target.logicalUrl,
    cachePolicy,
    verifyBodies,
    rangeUrl,
    identityUrl,
    brUrl,
    brNegotiated: brUrl === identityUrl,
    ...(expected ? { expected } : {}),
    firstHead,
    secondHead,
    range,
    identityHead,
    brHead,
    identityBody,
    brBody,
  };
  const failures = validateRow(row);
  return { ...row, ok: failures.length === 0, failures };
}

async function main() {
  const args = parseArgs(process.argv);
  let hostedRelease;
  let physicalTargets = [];
  if (args.release) {
    const localRelease = JSON.parse(await readFile(args.release, 'utf8'));
    physicalTargets = releaseTargets(localRelease, args.limit, args.artifactBase);
    if (!physicalTargets.length) throw new Error('No artifact URLs to validate from release');
    const hosted = await validateHostedRelease(args.release, args.artifactBase);
    hostedRelease = hosted.validation;
  }
  const targets = [
    ...args.urls.map((url) => ({ url, mode: 'logical-alias', cachePolicy: cachePolicyForUrl(url) })),
    ...physicalTargets,
  ].map((target) => ({ ...target, verifyBodies: args.verifyBodies }));
  if (!targets.length) throw new Error('No artifact URLs to validate');
  const rows = [];
  for (const target of targets) rows.push(await validateUrl(target, args.rangeBytes));
  const result = {
    schema: 'lc0_browser.artifact_cdn_validation.v2',
    ok: (!hostedRelease || hostedRelease.ok) && rows.every((row) => row.ok),
    checked: rows.length,
    ...(hostedRelease ? { hostedRelease } : {}),
    rows,
  };
  console.log(JSON.stringify(result, null, 2));
  process.exit(result.ok ? 0 : 1);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
