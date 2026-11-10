import { ENGINE_ARTIFACT_PINS } from './engineArtifactPins.generated.ts';

export interface EngineArtifactPin {
  sha256: string;
  bytes: number;
}

/** Self-contained so classic blob workers can use exactly the same verifier. */
export function createEngineArtifactLoader(pins: Record<string, EngineArtifactPin>) {
  function pin(url: string): EngineArtifactPin | undefined {
    const path = new URL(url, typeof location === 'undefined' ? 'http://localhost/' : location.href).pathname;
    const hash = path.match(/^\/artifacts\/sha256\/([a-f0-9]{64})\//)?.[1];
    return hash ? Object.values(pins).find((entry) => entry.sha256 === hash) : pins[path];
  }
  function local(): boolean {
    if (typeof location === 'undefined') return true;
    const hostname = new URL(location.origin).hostname;
    return ['localhost', '127.0.0.1', '[::1]'].includes(hostname) || hostname.endsWith('.localhost');
  }
  function failure(url: string, detail: string): Error {
    return new Error(
      `Engine artifact integrity check failed for ${url}: ${detail}. Retry loading the engine; if it persists, reload the app or contact the site owner.`,
    );
  }
  function expected(url: string): EngineArtifactPin | undefined {
    const entry = pin(url);
    if (!entry) {
      if (!local()) throw failure(url, 'this app build has no trusted SHA-256 pin');
      console.warn(`Local engine artifact has no integrity pin: ${url}. Regenerate the engine manifest and integrity registry before publishing.`);
    }
    return entry;
  }
  async function verify(url: string, bytes: ArrayBuffer): Promise<ArrayBuffer> {
    const entry = expected(url);
    // Locally built, unregistered lab artifacts are allowed only on loopback.
    // A registered artifact MUST match even locally (including static dist).
    if (!entry) return bytes;
    if (bytes.byteLength !== entry.bytes) throw failure(url, `expected ${entry.bytes} bytes, received ${bytes.byteLength}`);
    const digest = await crypto.subtle.digest('SHA-256', bytes);
    const hash = Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('');
    if (hash !== entry.sha256) throw failure(url, 'SHA-256 mismatch');
    return bytes;
  }
  async function response(url: string): Promise<Response> {
    expected(url);
    // Unverified network bytes must never enter the HTTP cache. Workers retain
    // only verified modules/data; a failed load is evicted by each adapter.
    const result = await fetch(url, { mode: 'cors', credentials: 'omit', cache: 'no-store' });
    if (!result.ok) throw new Error(`Failed to fetch engine artifact ${url}: HTTP ${result.status}`);
    return result;
  }
  async function bytes(url: string): Promise<ArrayBuffer> {
    return verify(url, await (await response(url)).arrayBuffer());
  }
  return { pin, verify, response, bytes };
}

export const engineArtifacts = createEngineArtifactLoader(ENGINE_ARTIFACT_PINS);
export function engineArtifactWorkerScript(): string {
  return `const engineArtifacts = (${createEngineArtifactLoader.toString()})(${JSON.stringify(ENGINE_ARTIFACT_PINS)});`;
}
