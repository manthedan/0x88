import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { test } from 'node:test';

function listen(server) {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => resolve(server.address().port));
  });
}

test('validate_artifact_cdn_headers accepts cacheable ranged artifacts', async () => {
  const body = Buffer.from('abcdefghijklmnopqrstuvwxyz');
  const server = createServer((req, res) => {
    if (req.url !== '/artifact.wasm') {
      res.writeHead(404).end();
      return;
    }
    const headers = {
      'Cache-Control': 'public, max-age=31536000, immutable',
      'Content-Type': 'application/wasm',
      'Content-Length': String(body.length),
      'Accept-Ranges': 'bytes',
      ETag: '"test-artifact"',
      Age: '10',
      'CF-Cache-Status': 'HIT',
      'Access-Control-Allow-Origin': '*',
      'Cross-Origin-Resource-Policy': 'cross-origin',
      'Timing-Allow-Origin': 'https://0x88.app',
      'Access-Control-Expose-Headers': 'CF-Cache-Status, Cache-Status, Age, ETag, Content-Length, X-Artifact-Content-Length',
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
      'Cache-Control': 'public, max-age=31536000, immutable',
      'Content-Type': 'application/json',
      'X-Artifact-Content-Length': String(body.length),
      'Accept-Ranges': 'bytes',
      ETag: '"test-compressed-artifact"',
      Age: '10',
      'CF-Cache-Status': 'HIT',
      'Access-Control-Allow-Origin': '*',
      'Cross-Origin-Resource-Policy': 'cross-origin',
      'Timing-Allow-Origin': 'https://0x88.app',
      'Access-Control-Expose-Headers': 'CF-Cache-Status, Cache-Status, Age, ETag, Content-Length, X-Artifact-Content-Length',
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
