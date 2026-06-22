#!/usr/bin/env node
import { readFile } from 'node:fs/promises';

function usage() {
  console.log(`Usage: node scripts/validate_artifact_cdn_headers.mjs [--url URL ...] [--release manifest.json] [options]\n\nOptions:\n  --url URL          Artifact URL to validate; may be repeated\n  --release PATH     Release manifest containing artifactUrl entries\n  --limit N          Max release artifacts to validate\n  --range BYTES      Range probe length (default 1024)\n  --json             Print JSON only\n  -h, --help         Show help\n\nThe validator checks HEAD twice, a small Range GET, and identity/br header probes.\nIt expects immutable artifacts to expose Content-Length and valid 206 range behavior.\n`);
}

function parseArgs(argv) {
  const args = { urls: [], rangeBytes: 1024, json: false };
  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    const next = argv[i + 1];
    if (arg === '--url' && next) { args.urls.push(next); i += 1; continue; }
    if (arg === '--release' && next) { args.release = next; i += 1; continue; }
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
  const keys = ['cache-control', 'cdn-cache-control', 'cloudflare-cdn-cache-control', 'cf-cache-status', 'cache-status', 'age', 'etag', 'content-length', 'content-type', 'content-encoding', 'accept-ranges', 'content-range', 'vary', 'access-control-allow-origin', 'cross-origin-resource-policy', 'timing-allow-origin', 'access-control-expose-headers', 'set-cookie'];
  const out = {};
  for (const key of keys) {
    const value = headers.get(key);
    if (value !== null) out[key] = value;
  }
  return out;
}

async function urlsFromRelease(path, limit) {
  const release = JSON.parse(await readFile(path, 'utf8'));
  const urls = (release.artifacts ?? []).map((artifact) => artifact.artifactUrl).filter(Boolean);
  return Number.isFinite(limit) ? urls.slice(0, limit) : urls;
}

async function head(url, acceptEncoding) {
  const headers = acceptEncoding ? { 'Accept-Encoding': acceptEncoding } : undefined;
  const response = await fetch(url, { method: 'HEAD', headers, cache: 'no-store' });
  return { status: response.status, headers: pickHeaders(response.headers) };
}

async function rangeGet(url, rangeBytes) {
  const response = await fetch(url, { headers: { Range: `bytes=0-${rangeBytes - 1}` }, cache: 'no-store' });
  const headers = pickHeaders(response.headers);
  if (response.status !== 206) {
    await response.body?.cancel();
    return { status: response.status, headers, bodyBytes: 0, requestedBytes: rangeBytes };
  }
  const body = await response.arrayBuffer();
  return { status: response.status, headers, bodyBytes: body.byteLength, requestedBytes: rangeBytes };
}

function validateRow(row) {
  const failures = [];
  if (row.firstHead.status < 200 || row.firstHead.status >= 400) failures.push(`first HEAD status ${row.firstHead.status}`);
  if (row.secondHead.status < 200 || row.secondHead.status >= 400) failures.push(`second HEAD status ${row.secondHead.status}`);
  if (row.firstHead.headers['set-cookie'] || row.secondHead.headers['set-cookie'] || row.range.headers['set-cookie']) failures.push('artifact response must not set cookies');
  if (!row.firstHead.headers['content-length']) failures.push('missing Content-Length on HEAD');
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
  for (const required of ['cf-cache-status', 'cache-status', 'age', 'etag', 'content-length']) {
    if (!exposed.has(required)) failures.push(`Access-Control-Expose-Headers missing ${required}`);
  }
  if (row.range.status !== 206) failures.push(`Range probe returned ${row.range.status}, expected 206`);
  const contentRange = row.range.headers['content-range'];
  if (!contentRange) failures.push('missing Content-Range on range response');
  const contentRangeMatch = contentRange?.match(/^bytes (\d+)-(\d+)\/(\d+)$/);
  if (contentRange && !contentRangeMatch) failures.push(`malformed Content-Range: ${contentRange}`);
  if (contentRangeMatch) {
    const start = Number(contentRangeMatch[1]);
    const end = Number(contentRangeMatch[2]);
    const total = Number(contentRangeMatch[3]);
    const expectedBodyBytes = end - start + 1;
    const headBytes = Number(row.firstHead.headers['content-length']);
    if (start !== 0) failures.push(`range response starts at ${start}, expected 0`);
    if (expectedBodyBytes !== row.range.bodyBytes) failures.push(`range body length ${row.range.bodyBytes} does not match Content-Range length ${expectedBodyBytes}`);
    if (Number.isFinite(headBytes) && total !== headBytes) failures.push(`Content-Range total ${total} does not match HEAD Content-Length ${headBytes}`);
  }
  if (row.identityHead.status < 200 || row.identityHead.status >= 400) failures.push(`identity HEAD status ${row.identityHead.status}`);
  const identityEncoding = row.identityHead.headers['content-encoding'];
  if (identityEncoding && identityEncoding !== 'identity') failures.push(`identity probe returned Content-Encoding: ${identityEncoding}`);
  if (row.brHead.status < 200 || row.brHead.status >= 400) failures.push(`br HEAD status ${row.brHead.status}`);
  return failures;
}

async function validateUrl(url, rangeBytes) {
  const firstHead = await head(url);
  const secondHead = await head(url);
  const range = await rangeGet(url, rangeBytes);
  const identityHead = await head(url, 'identity');
  const brHead = await head(url, 'br');
  const row = { url, firstHead, secondHead, range, identityHead, brHead };
  const failures = validateRow(row);
  return { ...row, ok: failures.length === 0, failures };
}

async function main() {
  const args = parseArgs(process.argv);
  const releaseUrls = args.release ? await urlsFromRelease(args.release, args.limit) : [];
  const urls = [...args.urls, ...releaseUrls];
  if (!urls.length) throw new Error('No artifact URLs to validate');
  const rows = [];
  for (const url of urls) rows.push(await validateUrl(url, args.rangeBytes));
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
