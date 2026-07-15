import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { test } from 'node:test';
import { brotliCompressSync } from 'node:zlib';

function listen(server) {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => resolve(server.address().port));
  });
}

function runValidator(args) {
  return new Promise((resolve) => {
    const child = spawn(process.execPath, [
      'scripts/validate_artifact_cdn_headers.mjs',
      ...args,
    ], { cwd: process.cwd(), stdio: ['ignore', 'pipe', 'pipe'] });
    let stdout = '';
    let stderr = '';
    child.stdout.setEncoding('utf8');
    child.stderr.setEncoding('utf8');
    child.stdout.on('data', (chunk) => { stdout += chunk; });
    child.stderr.on('data', (chunk) => { stderr += chunk; });
    child.on('close', (status) => resolve({ status, stdout, stderr }));
  });
}

async function runV2BodyValidationCase({ identityBody, brDecodedBody }) {
  const { mkdtemp, writeFile } = await import('node:fs/promises');
  const { tmpdir } = await import('node:os');
  const { join } = await import('node:path');
  const expectedBody = Buffer.from('abcdefghijklmnopqrstuvwxyz');
  const rawSha = createHash('sha256').update(expectedBody).digest('hex');
  const brBody = brotliCompressSync(brDecodedBody);
  const brSha = createHash('sha256').update(brBody).digest('hex');
  const server = createServer((req, res) => {
    if (req.url !== '/model.onnx') {
      res.writeHead(404).end();
      return;
    }
    const wantsBr = req.headers['accept-encoding']?.split(',').some((entry) => entry.trim() === 'br') && !req.headers.range;
    const headers = {
      'Cache-Control': 'public, max-age=300, stale-while-revalidate=86400',
      'Content-Type': 'application/octet-stream',
      'Content-Length': String(wantsBr ? brBody.length : identityBody.length),
      'X-Artifact-Content-Length': String(expectedBody.length),
      'X-Artifact-Encoded-Length': String(wantsBr ? brBody.length : identityBody.length),
      'X-Artifact-Decoded-SHA256': rawSha,
      'X-Artifact-Encoded-SHA256': wantsBr ? brSha : rawSha,
      'Accept-Ranges': 'bytes',
      ETag: '"body-validation"',
      Age: '10',
      'CF-Cache-Status': 'HIT',
      Vary: 'Accept-Encoding',
      'Access-Control-Allow-Origin': '*',
      'Cross-Origin-Resource-Policy': 'cross-origin',
      'Timing-Allow-Origin': 'https://0x88.app',
      'Access-Control-Expose-Headers': 'CF-Cache-Status, Cache-Status, Age, ETag, Content-Length, X-Artifact-Content-Length, X-Artifact-Encoded-Length',
      ...(wantsBr ? { 'Content-Encoding': 'br' } : {}),
    };
    if (req.method === 'HEAD') {
      res.writeHead(200, headers).end();
      return;
    }
    if (req.headers.range) {
      const match = req.headers.range.match(/^bytes=(\d+)-(\d+)$/);
      const start = Number(match?.[1] ?? 0);
      const end = Math.min(Number(match?.[2] ?? expectedBody.length - 1), expectedBody.length - 1);
      res.writeHead(206, {
        ...headers,
        'Content-Length': String(end - start + 1),
        'Content-Range': `bytes ${start}-${end}/${expectedBody.length}`,
      });
      res.end(expectedBody.subarray(start, end + 1));
      return;
    }
    res.writeHead(200, headers);
    res.end(wantsBr ? brBody : identityBody);
  });
  const port = await listen(server);
  const root = await mkdtemp(join(tmpdir(), 'lc0-v2-cdn-corruption-'));
  const releasePath = join(root, 'release.json');
  await writeFile(releasePath, JSON.stringify({
    schema: 'lc0_browser.artifact_release_manifest.v2',
    releaseId: 'v2-corruption',
    artifacts: [{
      logicalUrl: '/model.onnx',
      raw: { sha256: rawSha, bytes: expectedBody.length },
      representations: [
        { encoding: 'identity', url: `/artifacts/sha256/${rawSha}/identity`, sha256: rawSha, bytes: expectedBody.length },
        { encoding: 'br', url: `/artifacts/sha256/${rawSha}/br/${brSha}`, sha256: brSha, bytes: brBody.length },
      ],
    }],
  }));
  try {
    return await runValidator([
      '--release', releasePath,
      '--artifact-base', `http://127.0.0.1:${port}`,
      '--range', '4',
      '--verify-bodies',
    ]);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
}

test('validate_artifact_cdn_headers accepts short-cache logical aliases without full body downloads', async () => {
  const body = Buffer.from('abcdefghijklmnopqrstuvwxyz');
  let fullBodyGets = 0;
  let cacheControl = 'public, max-age=300, stale-while-revalidate=86400';
  const server = createServer((req, res) => {
    if (req.url !== '/artifact.wasm') {
      res.writeHead(404).end();
      return;
    }
    const headers = {
      'Cache-Control': cacheControl,
      'Content-Type': 'application/wasm',
      'Content-Length': String(body.length),
      'Accept-Ranges': 'bytes',
      ETag: '"test-artifact"',
      Age: '10',
      'CF-Cache-Status': 'HIT',
      'Access-Control-Allow-Origin': '*',
      'Cross-Origin-Resource-Policy': 'cross-origin',
      'Timing-Allow-Origin': 'https://0x88.app',
      'Access-Control-Expose-Headers': 'CF-Cache-Status, Cache-Status, Age, ETag, Content-Length, X-Artifact-Content-Length, X-Artifact-Encoded-Length',
    };
    if (req.method === 'HEAD') {
      res.writeHead(200, headers).end();
      return;
    }
    const range = req.headers.range;
    if (range) {
      const match = range.match(/^bytes=(\d+)-(\d+)$/);
      const start = Number(match?.[1] ?? 0);
      const end = Math.min(Number(match?.[2] ?? body.length - 1), body.length - 1);
      res.writeHead(206, { ...headers, 'Content-Length': String(end - start + 1), 'Content-Range': `bytes ${start}-${end}/${body.length}` });
      res.end(body.subarray(start, end + 1));
      return;
    }
    fullBodyGets += 1;
    res.writeHead(200, headers).end(body);
  });
  const port = await listen(server);
  try {
    const result = await new Promise((resolve) => {
      const child = spawn(process.execPath, [
        'scripts/validate_artifact_cdn_headers.mjs',
        '--url', `http://127.0.0.1:${port}/artifact.wasm`,
        '--range', '4',
      ], { cwd: process.cwd(), stdio: ['ignore', 'pipe', 'pipe'] });
      let stdout = '';
      let stderr = '';
      child.stdout.setEncoding('utf8');
      child.stderr.setEncoding('utf8');
      child.stdout.on('data', (chunk) => { stdout += chunk; });
      child.stderr.on('data', (chunk) => { stderr += chunk; });
      child.on('close', (status) => resolve({ status, stdout, stderr }));
    });
    assert.equal(result.status, 0, result.stderr);
    const parsed = JSON.parse(result.stdout);
    assert.equal(parsed.ok, true);
    assert.equal(parsed.rows[0].range.status, 206);
    assert.equal(parsed.rows[0].firstHead.headers['cf-cache-status'], 'HIT');
    assert.equal(parsed.rows[0].identityBody, undefined);
    assert.equal(parsed.rows[0].brBody, undefined);
    assert.equal(fullBodyGets, 0);

    const { mkdtemp, writeFile } = await import('node:fs/promises');
    const { tmpdir } = await import('node:os');
    const { join } = await import('node:path');
    const root = await mkdtemp(join(tmpdir(), 'lc0-v1-cdn-release-'));
    const releasePath = join(root, 'release.json');
    await writeFile(releasePath, JSON.stringify({
      schema: 'lc0_browser.artifact_release_manifest.v1',
      releaseId: 'v1',
      artifacts: [{
        logicalUrl: '/artifact.wasm',
        artifactUrl: `http://127.0.0.1:${port}/artifact.wasm`,
        bytes: body.length,
        sha256: createHash('sha256').update(body).digest('hex'),
      }],
    }));
    const v1Result = await runValidator(['--release', releasePath, '--range', '4']);
    assert.equal(v1Result.status, 0, v1Result.stderr);
    assert.equal(fullBodyGets, 0);

    cacheControl = 'public, max-age=31536000, immutable';
    const rejected = await runValidator(['--url', `http://127.0.0.1:${port}/artifact.wasm`, '--range', '4']);
    assert.notEqual(rejected.status, 0);
    assert.ok(JSON.parse(rejected.stdout).rows[0].failures.some((failure) => /logical alias cache policy is not short or revalidation-safe/.test(failure)));
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});

test('validate_artifact_cdn_headers accepts worker artifact length when HEAD is compressed', async () => {
  const body = Buffer.from('abcdefghijklmnopqrstuvwxyz');
  const server = createServer((req, res) => {
    if (req.url !== '/compressed-artifact.json') {
      res.writeHead(404).end();
      return;
    }
    const headers = {
      'Cache-Control': 'public, max-age=300, stale-while-revalidate=86400',
      'Content-Type': 'application/json',
      'X-Artifact-Content-Length': String(body.length),
      'Accept-Ranges': 'bytes',
      ETag: '"test-compressed-artifact"',
      Age: '10',
      'CF-Cache-Status': 'HIT',
      'Access-Control-Allow-Origin': '*',
      'Cross-Origin-Resource-Policy': 'cross-origin',
      'Timing-Allow-Origin': 'https://0x88.app',
      'Access-Control-Expose-Headers': 'CF-Cache-Status, Cache-Status, Age, ETag, Content-Length, X-Artifact-Content-Length, X-Artifact-Encoded-Length',
    };
    if (req.method === 'HEAD') {
      const encodingHeaders = req.headers['accept-encoding'] === 'identity' ? {} : { 'Content-Encoding': 'br' };
      res.writeHead(200, { ...headers, ...encodingHeaders }).end();
      return;
    }
    const match = req.headers.range?.match(/^bytes=(\d+)-(\d+)$/);
    const start = Number(match?.[1] ?? 0);
    const end = Math.min(Number(match?.[2] ?? body.length - 1), body.length - 1);
    res.writeHead(206, { ...headers, 'Content-Length': String(end - start + 1), 'Content-Range': `bytes ${start}-${end}/${body.length}` });
    res.end(body.subarray(start, end + 1));
  });
  const port = await listen(server);
  try {
    const result = await new Promise((resolve) => {
      const child = spawn(process.execPath, [
        'scripts/validate_artifact_cdn_headers.mjs',
        '--url', `http://127.0.0.1:${port}/compressed-artifact.json`,
        '--range', '4',
      ], { cwd: process.cwd(), stdio: ['ignore', 'pipe', 'pipe'] });
      let stdout = '';
      let stderr = '';
      child.stdout.setEncoding('utf8');
      child.stderr.setEncoding('utf8');
      child.stdout.on('data', (chunk) => { stdout += chunk; });
      child.stderr.on('data', (chunk) => { stderr += chunk; });
      child.on('close', (status) => resolve({ status, stdout, stderr }));
    });
    assert.equal(result.status, 0, result.stderr);
    const parsed = JSON.parse(result.stdout);
    assert.equal(parsed.ok, true);
    assert.equal(parsed.rows[0].firstHead.headers['x-artifact-content-length'], String(body.length));
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});

test('validate_artifact_cdn_headers verifies v2 Brotli negotiation and Range identity forcing', async () => {
  const { mkdtemp, writeFile } = await import('node:fs/promises');
  const { tmpdir } = await import('node:os');
  const { join } = await import('node:path');
  const body = Buffer.from('abcdefghijklmnopqrstuvwxyz');
  const rawSha = createHash('sha256').update(body).digest('hex');
  const brBody = brotliCompressSync(body);
  const brSha = createHash('sha256').update(brBody).digest('hex');
  const brBytes = brBody.length;
  let fullBodyGets = 0;
  const server = createServer((req, res) => {
    if (req.url !== '/models/lc0/model.onnx') {
      res.writeHead(404).end();
      return;
    }
    const wantsBr = req.headers['accept-encoding']?.split(',').some((entry) => entry.trim() === 'br') && !req.headers.range;
    const headers = {
      'Cache-Control': 'public, max-age=300, stale-while-revalidate=86400',
      'Content-Type': 'application/octet-stream',
      'Content-Length': String(wantsBr ? brBytes : body.length),
      'X-Artifact-Content-Length': String(body.length),
      'X-Artifact-Encoded-Length': String(wantsBr ? brBytes : body.length),
      'X-Artifact-Decoded-SHA256': rawSha,
      'X-Artifact-Encoded-SHA256': wantsBr ? brSha : rawSha,
      'Accept-Ranges': 'bytes',
      ETag: '"v2-artifact"',
      Age: '10',
      'CF-Cache-Status': 'HIT',
      Vary: 'Accept-Encoding',
      'Access-Control-Allow-Origin': '*',
      'Cross-Origin-Resource-Policy': 'cross-origin',
      'Timing-Allow-Origin': 'https://0x88.app',
      'Access-Control-Expose-Headers': 'CF-Cache-Status, Cache-Status, Age, ETag, Content-Length, X-Artifact-Content-Length, X-Artifact-Encoded-Length',
      ...(wantsBr ? { 'Content-Encoding': 'br' } : {}),
    };
    if (req.method === 'HEAD') {
      res.writeHead(200, {
        ...headers,
        'Content-Length': wantsBr ? '0' : headers['Content-Length'],
      }).end();
      return;
    }
    if (req.headers.range) {
      const match = req.headers.range.match(/^bytes=(\d+)-(\d+)$/);
      const start = Number(match?.[1] ?? 0);
      const end = Math.min(Number(match?.[2] ?? body.length - 1), body.length - 1);
      res.writeHead(206, {
        ...headers,
        'Content-Length': String(end - start + 1),
        'Content-Range': `bytes ${start}-${end}/${body.length}`,
      });
      res.end(body.subarray(start, end + 1));
      return;
    }
    fullBodyGets += 1;
    res.writeHead(200, headers);
    res.end(wantsBr ? brBody : body);
  });
  const port = await listen(server);
  const root = await mkdtemp(join(tmpdir(), 'lc0-v2-cdn-release-'));
  const releasePath = join(root, 'release.json');
  await writeFile(releasePath, JSON.stringify({
    schema: 'lc0_browser.artifact_release_manifest.v2',
    releaseId: 'v2',
    artifacts: [{
      logicalUrl: '/models/lc0/model.onnx',
      raw: { sha256: rawSha, bytes: body.length },
      representations: [
        { encoding: 'identity', url: `/artifacts/sha256/${rawSha}/identity`, sha256: rawSha, bytes: body.length },
        { encoding: 'br', url: `/artifacts/sha256/${rawSha}/br/${brSha}`, sha256: brSha, bytes: brBytes },
      ],
    }],
  }));
  try {
    const result = await new Promise((resolve) => {
      const child = spawn(process.execPath, [
        'scripts/validate_artifact_cdn_headers.mjs',
        '--release', releasePath,
        '--artifact-base', `http://127.0.0.1:${port}`,
        '--range', '4',
      ], { cwd: process.cwd(), stdio: ['ignore', 'pipe', 'pipe'] });
      let stdout = '';
      let stderr = '';
      child.stdout.setEncoding('utf8');
      child.stderr.setEncoding('utf8');
      child.stdout.on('data', (chunk) => { stdout += chunk; });
      child.stderr.on('data', (chunk) => { stderr += chunk; });
      child.on('close', (status) => resolve({ status, stdout, stderr }));
    });
    assert.equal(result.status, 0, result.stderr || result.stdout);
    const parsed = JSON.parse(result.stdout);
    assert.equal(parsed.rows[0].brHead.headers['content-encoding'], 'br');
    assert.equal(parsed.rows[0].brHead.headers['content-length'], '0');
    assert.equal(parsed.rows[0].brHead.headers['x-artifact-encoded-length'], String(brBytes));
    assert.equal(parsed.rows[0].identityHead.headers['content-encoding'], undefined);
    assert.equal(parsed.rows[0].range.headers['content-encoding'], undefined);
    assert.equal(parsed.rows[0].cachePolicy, 'mutable');
    assert.equal(parsed.rows[0].identityBody, undefined);
    assert.equal(parsed.rows[0].brBody, undefined);
    assert.equal(fullBodyGets, 0);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});

test('validate_artifact_cdn_headers requires immutable caching for physical representation URLs', async () => {
  const body = Buffer.from('abcdefghijklmnopqrstuvwxyz');
  const rawSha = createHash('sha256').update(body).digest('hex');
  let cacheControl = 'public, max-age=31536000, immutable';
  const path = `/artifacts/sha256/${rawSha}/identity`;
  const server = createServer((req, res) => {
    if (req.url !== path) {
      res.writeHead(404).end();
      return;
    }
    const headers = {
      'Cache-Control': cacheControl,
      'Content-Type': 'application/octet-stream',
      'Content-Length': String(body.length),
      'Accept-Ranges': 'bytes',
      ETag: '"physical-artifact"',
      Age: '10',
      'CF-Cache-Status': 'HIT',
      'Access-Control-Allow-Origin': '*',
      'Cross-Origin-Resource-Policy': 'cross-origin',
      'Timing-Allow-Origin': 'https://0x88.app',
      'Access-Control-Expose-Headers': 'CF-Cache-Status, Cache-Status, Age, ETag, Content-Length, X-Artifact-Content-Length, X-Artifact-Encoded-Length',
    };
    if (req.method === 'HEAD') {
      res.writeHead(200, headers).end();
      return;
    }
    const match = req.headers.range?.match(/^bytes=(\d+)-(\d+)$/);
    const start = Number(match?.[1] ?? 0);
    const end = Math.min(Number(match?.[2] ?? body.length - 1), body.length - 1);
    res.writeHead(206, { ...headers, 'Content-Length': String(end - start + 1), 'Content-Range': `bytes ${start}-${end}/${body.length}` });
    res.end(body.subarray(start, end + 1));
  });
  const port = await listen(server);
  try {
    const url = `http://127.0.0.1:${port}${path}`;
    const accepted = await runValidator(['--url', url, '--range', '4']);
    assert.equal(accepted.status, 0, accepted.stderr || accepted.stdout);
    assert.equal(JSON.parse(accepted.stdout).rows[0].cachePolicy, 'immutable');

    cacheControl = 'public, max-age=300, stale-while-revalidate=86400';
    const rejected = await runValidator(['--url', url, '--range', '4']);
    assert.notEqual(rejected.status, 0);
    assert.ok(JSON.parse(rejected.stdout).rows[0].failures.some((failure) => /physical artifact cache policy is not immutable/.test(failure)));
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});

test('validate_artifact_cdn_headers rejects corrupt identity bodies despite valid headers', async () => {
  const expectedBody = Buffer.from('abcdefghijklmnopqrstuvwxyz');
  const result = await runV2BodyValidationCase({
    identityBody: Buffer.from('abcdefghijklmnopqrstuvwxzz'),
    brDecodedBody: expectedBody,
  });

  assert.notEqual(result.status, 0);
  const parsed = JSON.parse(result.stdout);
  assert.equal(parsed.ok, false);
  assert.ok(parsed.rows[0].failures.some((failure) => /identity body SHA-256/.test(failure)));
});

test('validate_artifact_cdn_headers rejects corrupt decoded Brotli bodies despite valid headers', async () => {
  const expectedBody = Buffer.from('abcdefghijklmnopqrstuvwxyz');
  const result = await runV2BodyValidationCase({
    identityBody: expectedBody,
    brDecodedBody: Buffer.from('abcdefghijklmnopqrstuvwxzz'),
  });

  assert.notEqual(result.status, 0);
  const parsed = JSON.parse(result.stdout);
  assert.equal(parsed.ok, false);
  assert.ok(parsed.rows[0].failures.some((failure) => /br decoded body SHA-256/.test(failure)));
});

test('validate_artifact_cdn_headers rejects empty release manifests', async () => {
  const { mkdtemp, writeFile } = await import('node:fs/promises');
  const { tmpdir } = await import('node:os');
  const { join } = await import('node:path');
  const root = await mkdtemp(join(tmpdir(), 'lc0-empty-release-'));
  const releasePath = join(root, 'release.json');
  await writeFile(releasePath, JSON.stringify({ schema: 'lc0_browser.artifact_release_manifest.v1', artifacts: [] }));
  const result = await new Promise((resolve) => {
    const child = spawn(process.execPath, [
      'scripts/validate_artifact_cdn_headers.mjs',
      '--release', releasePath,
    ], { cwd: process.cwd(), stdio: ['ignore', 'pipe', 'pipe'] });
    let stdout = '';
    let stderr = '';
    child.stdout.setEncoding('utf8');
    child.stderr.setEncoding('utf8');
    child.stdout.on('data', (chunk) => { stdout += chunk; });
    child.stderr.on('data', (chunk) => { stderr += chunk; });
    child.on('close', (status) => resolve({ status, stdout, stderr }));
  });
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /No artifact URLs to validate/);
});
