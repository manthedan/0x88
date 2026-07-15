#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';

function usage() {
  console.log(`Usage: node scripts/validate_artifact_cdn_headers.mjs [--url URL ...] [--release manifest.json] [options]\n\nOptions:\n  --url URL          Artifact URL to validate; may be repeated\n  --release PATH     V1/v2 release manifest to validate\n  --artifact-base URL  Origin for v2 logical URLs (default https://assets.0x88.app)\n  --limit N          Max release artifacts to validate\n  --range BYTES      Range probe length (default 1024)\n  --json             Print JSON only\n  -h, --help         Show help\n\nThe validator checks HEAD twice, a small Range GET, and identity/br header probes.\nFor v2 releases it verifies representation negotiation, integrity metadata, and that\nRange requests force identity. It never uploads, purges, or mutates channels.\n`);
}

function parseArgs(argv) {
  const args = { urls: [], rangeBytes: 1024, json: false, artifactBase: 'https://assets.0x88.app' };
  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    const next = argv[i + 1];
    if (arg === '--url' && next) { args.urls.push(next); i += 1; continue; }
    if (arg === '--release' && next) { args.release = next; i += 1; continue; }
    if (arg === '--artifact-base' && next) { args.artifactBase = next; i += 1; continue; }
    if (arg === '--limit' && next) { args.limit = Number(next); i += 1; continue; }
    if (arg === '--range' && next) { args.rangeBytes = Number(next); i += 1; continue; }
    if (arg === '--json') { args.json = true; continue; }
    if (arg === '-h' || arg === '--help') { usage(); process.exit(0); }
    throw new Error(`Unknown argument: ${arg}`);
  }
  if (!args.urls.length && !args.release) throw new Error('Provide at least one --url or --release');
  return args;
}

function pickHeaders(headers) {
  const keys = ['cache-control', 'cdn-cache-control', 'cloudflare-cdn-cache-control', 'cf-cache-status', 'cache-status', 'age', 'etag', 'content-length', 'x-artifact-content-length', 'x-artifact-decoded-sha256', 'x-artifact-encoded-sha256', 'content-type', 'content-encoding', 'accept-ranges', 'content-range', 'vary', 'access-control-allow-origin', 'cross-origin-resource-policy', 'timing-allow-origin', 'access-control-expose-headers', 'set-cookie'];
  const out = {};
  for (const key of keys) {
    const value = headers.get(key);
    if (value !== null) out[key] = value;
  }
  return out;
}

async function urlsFromRelease(path, limit, artifactBase) {
  const release = JSON.parse(await readFile(path, 'utf8'));
  const targets = [];
  for (const artifact of release.artifacts ?? []) {
    if (artifact.raw && Array.isArray(artifact.representations)) {
      const identity = artifact.representations.find((entry) => entry.encoding === 'identity');
      const br = artifact.representations.find((entry) => entry.encoding === 'br');
      const rawUrl = artifact.logicalUrl ?? artifact.url ?? identity?.url;
      if (!rawUrl) continue;
      targets.push({
        url: new URL(rawUrl, artifactBase).href,
        expected: {
          raw: artifact.raw,
          identity,
          br,
        },
      });
      continue;
    }
    if (artifact.artifactUrl) targets.push({ url: new URL(artifact.artifactUrl, artifactBase).href });
  }
  return Number.isFinite(limit) ? targets.slice(0, limit) : targets;
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

function validateRow(row) {
  const failures = [];
  if (row.firstHead.status < 200 || row.firstHead.status >= 400) failures.push(`first HEAD status ${row.firstHead.status}`);
  if (row.secondHead.status < 200 || row.secondHead.status >= 400) failures.push(`second HEAD status ${row.secondHead.status}`);
  if (row.firstHead.headers['set-cookie'] || row.secondHead.headers['set-cookie'] || row.range.headers['set-cookie']) failures.push('artifact response must not set cookies');
  if (!row.firstHead.headers['content-length'] && !row.firstHead.headers['x-artifact-content-length']) failures.push('missing Content-Length or X-Artifact-Content-Length on HEAD');
  const cacheControl = row.firstHead.headers['cache-control'] ?? '';
  if (!/\bimmutable\b/i.test(cacheControl) || !/\bmax-age=31536000\b/i.test(cacheControl)) failures.push(`artifact cache policy is not immutable: ${cacheControl || 'missing'}`);
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
  for (const required of ['cf-cache-status', 'cache-status', 'age', 'etag', 'content-length', 'x-artifact-content-length']) {
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
    const headBytes = Number(row.firstHead.headers['x-artifact-content-length'] ?? row.firstHead.headers['content-length']);
    if (start !== 0) failures.push(`range response starts at ${start}, expected 0`);
    if (expectedBodyBytes !== row.range.bodyBytes) failures.push(`range body length ${row.range.bodyBytes} does not match Content-Range length ${expectedBodyBytes}`);
    if (Number.isFinite(headBytes) && total !== headBytes) failures.push(`Content-Range total ${total} does not match HEAD artifact length ${headBytes}`);
  }
  if (row.identityHead.status < 200 || row.identityHead.status >= 400) failures.push(`identity HEAD status ${row.identityHead.status}`);
  const identityEncoding = row.identityHead.headers['content-encoding'];
  if (identityEncoding && identityEncoding !== 'identity') failures.push(`identity probe returned Content-Encoding: ${identityEncoding}`);
  if (row.brHead.status < 200 || row.brHead.status >= 400) failures.push(`br HEAD status ${row.brHead.status}`);
  if (row.expected) {
    const verifyIdentityBody = hasExpectedBodyMetadata(row.expected, row.expected.identity);
    const verifyBrBody = hasExpectedBodyMetadata(row.expected, row.expected.br);
    if (verifyIdentityBody && (row.identityBody.status < 200 || row.identityBody.status >= 400)) failures.push(`identity body status ${row.identityBody.status}`);
    if (verifyBrBody && (row.brBody.status < 200 || row.brBody.status >= 400)) failures.push(`br body status ${row.brBody.status}`);
    const expectedRawBytes = row.expected.raw?.bytes;
    const expectedRawSha256 = row.expected.raw?.sha256?.toLowerCase();
    const actualIdentityBytes = Number(row.identityHead.headers['x-artifact-content-length'] ?? row.identityHead.headers['content-length']);
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
      if (!(row.brHead.headers.vary ?? '').toLowerCase().split(',').map((value) => value.trim()).includes('accept-encoding')) {
        failures.push('br probe missing Vary: Accept-Encoding');
      }
      const encodedSha256 = row.brHead.headers['x-artifact-encoded-sha256'];
      if (encodedSha256 !== row.expected.br.sha256?.toLowerCase()) {
        failures.push(`br encoded SHA-256 ${encodedSha256 ?? 'missing'} does not match manifest ${row.expected.br.sha256}`);
      }
      if (Number(row.brHead.headers['content-length']) !== row.expected.br.bytes) {
        failures.push(`br encoded length ${row.brHead.headers['content-length'] ?? 'missing'} does not match manifest ${row.expected.br.bytes}`);
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
  const { url, expected } = target;
  const firstHead = await head(url);
  const secondHead = await head(url);
  const range = await rangeGet(url, rangeBytes);
  const identityHead = await head(url, 'identity');
  const brHead = await head(url, 'br');
  const identityBody = hasExpectedBodyMetadata(expected, expected?.identity) ? await hashGet(url, 'identity') : undefined;
  const brBody = hasExpectedBodyMetadata(expected, expected?.br) ? await hashGet(url, 'br') : undefined;
  const row = { url, ...(expected ? { expected } : {}), firstHead, secondHead, range, identityHead, brHead, identityBody, brBody };
  const failures = validateRow(row);
  return { ...row, ok: failures.length === 0, failures };
}

async function main() {
  const args = parseArgs(process.argv);
  const releaseTargets = args.release ? await urlsFromRelease(args.release, args.limit, args.artifactBase) : [];
  const targets = [...args.urls.map((url) => ({ url })), ...releaseTargets];
  if (!targets.length) throw new Error('No artifact URLs to validate');
  const rows = [];
  for (const target of targets) rows.push(await validateUrl(target, args.rangeBytes));
  const result = {
    schema: 'lc0_browser.artifact_cdn_validation.v1',
    ok: rows.every((row) => row.ok),
    checked: rows.length,
    rows,
  };
  console.log(JSON.stringify(result, null, 2));
  process.exit(result.ok ? 0 : 1);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
