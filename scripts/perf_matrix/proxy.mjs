// Counting reverse proxy in front of the isolated static server. Each session gets its own port, hence
// its own origin (fresh storage partition), and every request is logged so a run can prove its cache
// state: a warm run must fetch zero model bytes from the network.
import { createServer, request as httpRequest } from 'node:http';

export function startCountingProxy(target) {
  const upstream = new URL(target);
  const requests = [];
  const server = createServer((req, res) => {
    const entry = { at: Date.now(), method: req.method, path: req.url, status: null, bytes: 0 };
    requests.push(entry);
    const up = httpRequest(
      { host: upstream.hostname, port: upstream.port, method: req.method, path: req.url, headers: { ...req.headers, host: upstream.host } },
      (upRes) => {
        entry.status = upRes.statusCode;
        res.writeHead(upRes.statusCode ?? 502, upRes.headers);
        upRes.on('data', (chunk) => {
          entry.bytes += chunk.length;
        });
        upRes.pipe(res);
      },
    );
    up.on('error', (error) => {
      entry.status = 502;
      res.writeHead(502);
      res.end(String(error));
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
        modelRequests: (since) =>
          requests.filter((r) => r.at >= since && /\.(onnx|bin|params|lc0web|br)(\?|$)|\/models\//.test(r.path ?? '') && !/manifest\.json/.test(r.path ?? '')),
        close: () =>
          new Promise((r) => {
            server.close(() => r());
            server.closeAllConnections();
          }),
      });
    });
  });
}
