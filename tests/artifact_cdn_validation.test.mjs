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
  const release = {
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
  };
  const releaseBody = Buffer.from(JSON.stringify(release));
  const identityPath = release.artifacts[0].representations[0].url;
  const brPath = release.artifacts[0].representations[1].url;
  const server = createServer((req, res) => {
    if (req.url === '/releases/v2-corruption.json') {
      res.writeHead(200, {
        'Cache-Control': 'public, max-age=31536000, immutable',
        'Content-Type': 'application/json',
        'Content-Length': String(releaseBody.byteLength),
      });
      res.end(req.method === 'HEAD' ? undefined : releaseBody);
      return;
    }
    if (req.url !== identityPath && req.url !== brPath) {
      res.writeHead(404).end();
      return;
    }
    const wantsBr = req.url === brPath;
    const headers = {
      'Cache-Control': 'public, max-age=31536000, immutable',
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
  await writeFile(releasePath, releaseBody);
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
  let hostedV1ReleaseBody;
  const bodySha = createHash('sha256').update(body).digest('hex');
  const physicalPath = `/artifacts/sha256/${bodySha}/artifact.wasm`;
  const server = createServer((req, res) => {
    if (req.url === '/releases/v1.json') {
      if (!hostedV1ReleaseBody) {
        res.writeHead(404).end();
        return;
      }
      res.writeHead(200, {
        'Cache-Control': 'public, max-age=31536000, immutable',
        'Content-Type': 'application/json',
        'Content-Length': String(hostedV1ReleaseBody.byteLength),
      });
      res.end(req.method === 'HEAD' ? undefined : hostedV1ReleaseBody);
      return;
    }
    if (req.url !== '/artifact.wasm' && req.url !== physicalPath) {
      res.writeHead(404).end();
      return;
    }
    const targetCacheControl = req.url === physicalPath
      ? 'public, max-age=31536000, immutable'
      : cacheControl;
    const headers = {
      'Cache-Control': targetCacheControl,
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
    hostedV1ReleaseBody = Buffer.from(JSON.stringify({
      schema: 'lc0_browser.artifact_release_manifest.v1',
      releaseId: 'v1',
      artifacts: [{
        logicalUrl: '/artifact.wasm',
        artifactUrl: `http://127.0.0.1:${port}${physicalPath}`,
        bytes: body.length,
        sha256: bodySha,
      }],
    }));
    await writeFile(releasePath, hostedV1ReleaseBody);
    const v1Result = await runValidator([
      '--release', releasePath,
      '--artifact-base', `http://127.0.0.1:${port}`,
      '--range', '4',
    ]);
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

test('validate_artifact_cdn_headers validates the hosted release and its explicit physical v2 representations', async () => {
  const { mkdtemp, writeFile } = await import('node:fs/promises');
  const { tmpdir } = await import('node:os');
  const { join } = await import('node:path');
  const body = Buffer.from('abcdefghijklmnopqrstuvwxyz');
  const rawSha = createHash('sha256').update(body).digest('hex');
  const brBody = brotliCompressSync(body);
  const brSha = createHash('sha256').update(brBody).digest('hex');
  const brBytes = brBody.length;
  let fullBodyGets = 0;
  let brCacheControl = 'public, max-age=31536000, immutable';
  const release = {
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
  };
  const releaseBody = Buffer.from(JSON.stringify(release));
  const identityPath = release.artifacts[0].representations[0].url;
  const brPath = release.artifacts[0].representations[1].url;
  const server = createServer((req, res) => {
    if (req.url === '/releases/v2.json') {
      res.writeHead(200, {
        'Cache-Control': 'public, max-age=31536000, immutable',
        'Content-Type': 'application/json',
        'Content-Length': String(releaseBody.byteLength),
      });
      res.end(req.method === 'HEAD' ? undefined : releaseBody);
      return;
    }
    if (req.url !== identityPath && req.url !== brPath) {
      res.writeHead(404).end();
      return;
    }
    const wantsBr = req.url === brPath;
    const headers = {
      'Cache-Control': wantsBr ? brCacheControl : 'public, max-age=31536000, immutable',
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
        'Content-Length': '0',
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
  await writeFile(releasePath, releaseBody);
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
    assert.equal(parsed.hostedRelease.ok, true);
    assert.equal(parsed.hostedRelease.url, `http://127.0.0.1:${port}/releases/v2.json`);
    assert.equal(parsed.rows[0].url, `http://127.0.0.1:${port}${identityPath}`);
    assert.equal(parsed.rows[0].identityUrl, `http://127.0.0.1:${port}${identityPath}`);
    assert.equal(parsed.rows[0].rangeUrl, `http://127.0.0.1:${port}${identityPath}`);
    assert.equal(parsed.rows[0].brUrl, `http://127.0.0.1:${port}${brPath}`);
    assert.equal(parsed.rows[0].brHead.headers['content-encoding'], 'br');
    assert.equal(parsed.rows[0].brHead.headers['content-length'], '0');
    assert.equal(parsed.rows[0].brHead.headers['x-artifact-encoded-length'], String(brBytes));
    assert.equal(parsed.rows[0].identityHead.headers['content-length'], '0');
    assert.equal(parsed.rows[0].identityHead.headers['x-artifact-encoded-length'], String(body.length));
    assert.equal(parsed.rows[0].identityHead.headers['content-encoding'], undefined);
    assert.equal(parsed.rows[0].range.headers['content-encoding'], undefined);
    assert.equal(parsed.rows[0].cachePolicy, 'immutable');
    assert.equal(parsed.rows[0].brNegotiated, false);
    assert.equal(parsed.rows[0].identityBody, undefined);
    assert.equal(parsed.rows[0].brBody, undefined);
    assert.equal(fullBodyGets, 0);

    brCacheControl = 'public, max-age=300, stale-while-revalidate=86400';
    const rejected = await runValidator([
      '--release', releasePath,
      '--artifact-base', `http://127.0.0.1:${port}`,
      '--range', '4',
    ]);
    assert.notEqual(rejected.status, 0);
    assert.ok(JSON.parse(rejected.stdout).rows[0].failures.some((failure) => /physical Brotli artifact cache policy is not immutable/.test(failure)));
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});

test('validate_artifact_cdn_headers fails a pre-promotion release canary when the hosted release is absent or mismatched despite shared bodies', async () => {
  const { mkdtemp, writeFile } = await import('node:fs/promises');
  const { tmpdir } = await import('node:os');
  const { join } = await import('node:path');
  const body = Buffer.from('abcdefghijklmnopqrstuvwxyz');
  const rawSha = createHash('sha256').update(body).digest('hex');
  const identityPath = `/artifacts/sha256/${rawSha}/identity`;
  const release = {
    schema: 'lc0_browser.artifact_release_manifest.v2',
    releaseId: 'pre-promotion',
    artifacts: [{
      logicalUrl: '/models/lc0/model.onnx',
      raw: { sha256: rawSha, bytes: body.byteLength },
      representations: [{ encoding: 'identity', url: identityPath, sha256: rawSha, bytes: body.byteLength }],
    }],
  };
  const releaseBody = Buffer.from(JSON.stringify(release));
  let hostedState = 'absent';
  const server = createServer((req, res) => {
    if (req.url === '/releases/pre-promotion.json') {
      if (hostedState === 'absent') {
        res.writeHead(404, { 'Cache-Control': 'no-store' }).end('Not found');
      } else {
        const mismatched = Buffer.from(JSON.stringify({ ...release, generatedAt: 'different' }));
        res.writeHead(200, {
          'Cache-Control': 'public, max-age=31536000, immutable',
          'Content-Length': String(mismatched.byteLength),
        }).end(mismatched);
      }
      return;
    }
    if (req.url !== identityPath) {
      res.writeHead(404).end();
      return;
    }
    const headers = {
      'Cache-Control': 'public, max-age=31536000, immutable',
      'Content-Type': 'application/octet-stream',
      'Content-Length': String(body.byteLength),
      'X-Artifact-Content-Length': String(body.byteLength),
      'X-Artifact-Encoded-Length': String(body.byteLength),
      'X-Artifact-Decoded-SHA256': rawSha,
      'X-Artifact-Encoded-SHA256': rawSha,
      'Accept-Ranges': 'bytes',
      ETag: '"shared-body"',
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
    if (req.headers.range) {
      const end = Math.min(3, body.byteLength - 1);
      res.writeHead(206, {
        ...headers,
        'Content-Length': String(end + 1),
        'Content-Range': `bytes 0-${end}/${body.byteLength}`,
      }).end(body.subarray(0, end + 1));
      return;
    }
    res.writeHead(200, headers).end(body);
  });
  const port = await listen(server);
  const root = await mkdtemp(join(tmpdir(), 'lc0-pre-promotion-cdn-release-'));
  const releasePath = join(root, 'release.json');
  await writeFile(releasePath, releaseBody);
  try {
    for (const state of ['absent', 'mismatched']) {
      hostedState = state;
      const result = await runValidator([
        '--release', releasePath,
        '--artifact-base', `http://127.0.0.1:${port}`,
        '--range', '4',
      ]);
      assert.notEqual(result.status, 0);
      const parsed = JSON.parse(result.stdout);
      assert.equal(parsed.hostedRelease.ok, false);
      assert.equal(parsed.rows[0].ok, true, JSON.stringify(parsed.rows[0].failures));
      assert.ok(parsed.hostedRelease.failures.some((failure) => (
        state === 'absent'
          ? /hosted release status 404/.test(failure)
          : /does not exactly match local manifest/.test(failure)
      )));
    }
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
  assert.match(result.stderr, /No artifact URLs to validate from release/);
});
