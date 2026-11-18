// Counting reverse proxy in front of the isolated static server. Each session gets its own port, hence
// its own origin (fresh storage partition), and every request is logged so a run can prove its cache
// state: a warm run must fetch zero model bytes from the network.
import { createServer, request as httpRequest } from 'node:http';
import { request as httpsRequest } from 'node:https';

export function buildOrigin(target) {
  const url = new URL(target);
  if (!['http:', 'https:'].includes(url.protocol) || url.href !== `${url.origin}/`) {
    throw Error('Proxy upstream must be an origin-only HTTP(S) URL');
  }
  return url;
}

export async function startCountingProxy(target, { htmlScript } = {}) {
  const upstream = buildOrigin(target);
  const request = upstream.protocol === 'https:' ? httpsRequest : httpRequest;
  // Channel-v2 model representations have no extension. Identify their raw hash
  // from app-origin model pins, rather than counting unrelated WASM/glue artifacts.
  const modelHashes = new Set();
  await Promise.all(
    ['lc0', 'maia3'].map(async (family) => {
      try {
        const response = await fetch(new URL(`/models/${family}/manifest.json`, upstream), { signal: AbortSignal.timeout(5000) });
        if (!response.ok) return;
        const manifest = await response.json();
        for (const entry of manifest.models ?? []) if (/^[a-f0-9]{64}$/i.test(entry.sha256 ?? '')) modelHashes.add(entry.sha256.toLowerCase());
      } catch {
        /* Legacy extension paths remain countable when a manifest is unavailable. */
      }
    }),
  );
  const isModel = (requestPath) => {
    const pathname = new URL(requestPath, 'http://proxy.local').pathname;
    if (pathname.endsWith('/manifest.json')) return false;
    if (/\.(onnx|bin|params|lc0web|br)$|\/models\//.test(pathname)) return true;
    const artifact = /^\/artifacts\/sha256\/([a-f0-9]{64})\/(?:identity|br\/[a-f0-9]{64})$/i.exec(pathname);
    return artifact !== null && modelHashes.has(artifact[1].toLowerCase());
  };
  const requests = [];
  const pending = new Set();
  let closing = false;
  const server = createServer((req, res) => {
    pending.add(res);
    const entry = { at: Date.now(), method: req.method, path: req.url, status: null, bytes: 0, complete: false, failure: null, clientAborted: false };
    requests.push(entry);
    let upstreamComplete = false;
    const fail = (reason) => {
      entry.failure ??= reason;
      if (res.headersSent) res.destroy();
      else if (!res.destroyed) res.writeHead(502).end('Upstream request failed');
    };
    res.on('finish', () => {
      entry.endedAt = Date.now();
      entry.complete = upstreamComplete && !entry.failure;
    });
    const up = request(
      {
        protocol: upstream.protocol,
        host: upstream.hostname,
        port: upstream.port,
        method: req.method,
        path: req.url,
        headers: { ...req.headers, host: upstream.host, ...(htmlScript ? { 'accept-encoding': 'identity' } : {}) },
      },
      (upRes) => {
        entry.status = upRes.statusCode;
        upRes.on('data', (chunk) => {
          entry.bytes += chunk.length;
        });
        upRes.on('end', () => {
          upstreamComplete = upRes.complete;
          if (!upstreamComplete) fail('Incomplete upstream response');
        });
        upRes.on('error', (error) => fail(`Upstream response: ${error.message}`));
        upRes.on('close', () => {
          if (!upRes.complete) fail('Truncated upstream response');
        });
        if (htmlScript && String(upRes.headers['content-type']).includes('text/html') && upRes.statusCode === 200) {
          // Route probes must run before hydration/worker creation, including on Safari.
          // Request identity HTML so we never inject into a compressed representation.
          const chunks = [];
          upRes.on('data', (chunk) => chunks.push(chunk));
          upRes.on('end', () => {
            if (!upstreamComplete || res.destroyed) return;
            const body = Buffer.from(Buffer.concat(chunks).toString().replace('<head>', `<head><script>${htmlScript}</script>`));
            const headers = { ...upRes.headers, 'content-length': body.length };
            delete headers['content-encoding'];
            delete headers['transfer-encoding']; // Buffered response is now length-delimited, never both.
            res.writeHead(200, headers);
            res.end(body);
          });
        } else {
          res.writeHead(upRes.statusCode ?? 502, upRes.headers);
          upRes.pipe(res);
        }
      },
    );
    up.on('error', (error) => fail(`Upstream request: ${error.message}`));
    res.on('close', () => {
      pending.delete(res);
      if (res.writableFinished) return;
      entry.endedAt = Date.now();
      if (!entry.failure) {
        entry.clientAborted = !closing;
        entry.failure = closing ? 'Proxy closed before response finished' : 'Browser disconnected before response finished';
      }
      up.destroy();
    });
    req.pipe(up);
  });
  return new Promise((resolveProxy) => {
    server.listen(0, '127.0.0.1', () => {
      const { port } = server.address();
      resolveProxy({
        url: `http://127.0.0.1:${port}`,
        requests,
        /** Requests for model-like assets (.onnx and friends) since `since` (epoch ms). */
        modelRequests: (since) => requests.filter((r) => r.at >= since && isModel(r.path ?? '')),
        close: async () => {
          closing = true;
          // server.close's callback can precede the pending responses' close events.
          const drained = [...pending].map((res) => new Promise((r) => res.once('close', r)));
          await Promise.all([
            ...drained,
            new Promise((r) => {
              server.close(() => r());
              server.closeAllConnections();
            }),
          ]);
        },
      });
    });
  });
}
